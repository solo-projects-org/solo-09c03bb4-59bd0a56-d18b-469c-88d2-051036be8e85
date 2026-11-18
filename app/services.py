"""Transactional services: revision acceptance and package generation."""
from __future__ import annotations

import itertools
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from . import config, deid
from .csvutil import Row, canonical_hash, iter_parse_csv
from .database import get_conn, write_lock

log = logging.getLogger("deid.services")

# Identity lookups are batched well below SQLite's variable limit.
_IDENTITY_QUERY_BATCH = 500


class ConflictError(Exception):
    """The same revision number was already accepted with different content."""


class StaleRevisionError(Exception):
    """The package being finalised no longer belongs to the current revision."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _cohort_dir(cohort_id: str) -> Path:
    return config.UPLOAD_DIR / cohort_id


def accept_revision(cohort_id: str, revision_label: str, rows: list[Row], raw: bytes) -> dict:
    """Validate and accept a new revision.

    Rules implemented here:

    * same (cohort, revision) + same content hash -> idempotent no-op, the
      existing package keeps its current state;
    * same (cohort, revision) + different hash -> :class:`ConflictError`
      (HTTP 409), nothing is changed;
    * a new revision supersedes every non-terminal package of the cohort and
      creates a fresh ``pending`` package, so accepting a revision immediately
      withdraws the current eligibility of the old package;
    * new subjects receive stable pseudonyms and per-subject date shifts,
      existing subjects keep theirs across revisions.
    """
    content_hash = canonical_hash(rows)
    now = _utcnow()
    tmp_path: Path | None = None

    with write_lock:
        conn = get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT content_hash FROM revisions WHERE cohort_id = ? AND revision = ?",
                (cohort_id, revision_label),
            ).fetchone()
            if existing is not None:
                if existing["content_hash"] == content_hash:
                    conn.execute("COMMIT")
                    return {
                        "status": "duplicate",
                        "revision": revision_label,
                        "content_hash": content_hash,
                    }
                raise ConflictError(
                    f"修订 {revision_label} 已存在且内容不同；同修订号的内容冲突已拒绝"
                )

            cohort = conn.execute(
                "SELECT cohort_id, salt FROM cohorts WHERE cohort_id = ?",
                (cohort_id,),
            ).fetchone()
            if cohort is None:
                salt = deid.new_salt()
                conn.execute(
                    "INSERT INTO cohorts (cohort_id, salt, created_at) VALUES (?, ?, ?)",
                    (cohort_id, salt, now),
                )
            else:
                salt = cohort["salt"]

            # Assign stable identities to any subjects not seen before.
            max_idx_row = conn.execute(
                "SELECT COALESCE(MAX(CAST(SUBSTR(pseudonym, 2) AS INTEGER)), 0) AS m "
                "FROM subjects WHERE cohort_id = ?",
                (cohort_id,),
            ).fetchone()
            next_index = max_idx_row["m"] + 1
            for row in rows:
                known = conn.execute(
                    "SELECT 1 FROM subjects WHERE cohort_id = ? AND subject_id = ?",
                    (cohort_id, row.subject_id),
                ).fetchone()
                if known is None:
                    conn.execute(
                        "INSERT INTO subjects (cohort_id, subject_id, pseudonym, date_shift_days) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            cohort_id,
                            row.subject_id,
                            deid.pseudonym(next_index),
                            deid.subject_date_shift(salt, row.subject_id),
                        ),
                    )
                    next_index += 1

            raw_dir = _cohort_dir(cohort_id)
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"rev{revision_label}.csv"
            tmp_path = raw_path.with_suffix(".csv.tmp")
            tmp_path.write_bytes(raw)
            tmp_path.replace(raw_path)

            conn.execute(
                "INSERT INTO revisions (cohort_id, revision, content_hash, row_count, raw_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cohort_id, revision_label, content_hash, len(rows), str(raw_path), now),
            )

            # Withdraw current eligibility of every older package, including
            # one already published: accepting a new revision immediately
            # retires the previously downloadable package.
            conn.execute(
                "UPDATE packages SET status = 'superseded', updated_at = ? "
                "WHERE cohort_id = ? AND status IN ('pending', 'generating', 'failed', 'published')",
                (now, cohort_id),
            )
            conn.execute(
                "INSERT INTO packages (cohort_id, revision, status, created_at, updated_at) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (cohort_id, revision_label, now, now),
            )
            conn.execute(
                "UPDATE cohorts SET current_revision = ? WHERE cohort_id = ?",
                (revision_label, cohort_id),
            )
            conn.execute("COMMIT")
            return {
                "status": "accepted",
                "revision": revision_label,
                "content_hash": content_hash,
                "rows": len(rows),
            }
        except Exception:
            conn.execute("ROLLBACK")
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
        finally:
            conn.close()


def claim_pending_package() -> tuple[str, str] | None:
    """Atomically claim one due package for building (``pending`` or a
    ``failed`` package whose backoff has elapsed)."""
    now = _utcnow()
    with write_lock:
        conn = get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT cohort_id, revision FROM packages
                 WHERE status = 'pending'
                    OR (status = 'failed' AND (next_retry_at IS NULL OR next_retry_at <= ?))
                 ORDER BY updated_at
                 LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE packages SET status = 'generating', attempts = attempts + 1, "
                "updated_at = ? WHERE cohort_id = ? AND revision = ?",
                (now, row["cohort_id"], row["revision"]),
            )
            conn.execute("COMMIT")
            return row["cohort_id"], row["revision"]
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()


def finalize_package(cohort_id: str, revision_label: str) -> None:
    """Build the de-identified package segment by segment and publish it.

    The build streams the accepted revision from disk: per segment it looks
    up only that segment's identities and pushes the de-identified CSV through
    the zip compressor into a temp file, so memory stays bounded no matter how
    many visit rows the revision has or how large the cohort's subject history
    is. Before every segment the build re-checks that its package is still the
    cohort's current ``generating`` one; if a newer revision was accepted
    meanwhile, the build stops promptly (freeing the worker queue) and its
    temp file is removed. The zip is atomically renamed into place only when
    complete, and the final publish UPDATE is conditional, so a stale or
    half-written package can never become downloadable.
    """
    conn = get_conn()
    tmp_path: Path | None = None
    try:
        revision = conn.execute(
            "SELECT * FROM revisions WHERE cohort_id = ? AND revision = ?",
            (cohort_id, revision_label),
        ).fetchone()
        if revision is None:
            raise RuntimeError("revision row disappeared")

        # Snapshot the cohort subject count for the manifest (same value the
        # pre-refactor full-identity-map build reported).
        subject_count = conn.execute(
            "SELECT COUNT(*) AS n FROM subjects WHERE cohort_id = ?",
            (cohort_id,),
        ).fetchone()["n"]

        pkg_dir = config.PACKAGE_DIR / cohort_id
        pkg_dir.mkdir(parents=True, exist_ok=True)
        package_path = pkg_dir / f"rev{revision_label}_deid.zip"
        tmp_path = package_path.with_name(package_path.name + ".tmp")

        stats = {"rows": 0}

        def ensure_current() -> None:
            if not _is_current_generating(conn, cohort_id, revision_label):
                raise StaleRevisionError(
                    f"修订 {revision_label} 已被更新的修订取代，生成任务提前停止"
                )

        ensure_current()
        rows = iter_parse_csv(revision["raw_path"])
        try:
            segments = _deidentified_segments(conn, cohort_id, rows, stats, ensure_current)
            deid.write_package_streaming(
                tmp_path,
                segments,
                lambda: deid.build_manifest(stats["rows"], subject_count),
            )
        finally:
            rows.close()

        # Zip complete: atomically move into place, then publish. The UPDATE
        # is conditional on the package still being ``generating`` and the
        # revision still being current for the cohort. A late build for a
        # revision that was superseded while the worker was busy is discarded
        # instead of overwriting the newer revision's state.
        tmp_path.replace(package_path)
        tmp_path = None  # renamed away; nothing left to clean up on failure

        now = _utcnow()
        with write_lock:
            cur = conn.execute(
                """
                UPDATE packages
                   SET status = 'published',
                       package_path = ?,
                       last_error = NULL,
                       next_retry_at = NULL,
                       updated_at = ?
                 WHERE cohort_id = ? AND revision = ?
                   AND status = 'generating'
                   AND revision = (SELECT current_revision FROM cohorts WHERE cohort_id = ?)
                """,
                (str(package_path), now, cohort_id, revision_label, cohort_id),
            )
            if cur.rowcount == 0:
                # Late build: a newer revision superseded this one. Discard the
                # artifact so it can never be mistaken for the current package.
                package_path.unlink(missing_ok=True)
                state = conn.execute(
                    "SELECT status FROM packages WHERE cohort_id = ? AND revision = ?",
                    (cohort_id, revision_label),
                ).fetchone()
                raise StaleRevisionError(
                    f"修订 {revision_label} 的迟到生成结果已丢弃 (当前状态: "
                    f"{state['status'] if state else 'missing'})"
                )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        conn.close()


def _is_current_generating(conn, cohort_id: str, revision_label: str) -> bool:
    """True while the package may keep building: still ``generating`` and its
    revision is still the cohort's current one. Turns false as soon as a newer
    revision is accepted (the accept transaction supersedes this package)."""
    row = conn.execute(
        """
        SELECT 1 FROM packages
         WHERE cohort_id = ? AND revision = ? AND status = 'generating'
           AND revision = (SELECT current_revision FROM cohorts WHERE cohort_id = ?)
        """,
        (cohort_id, revision_label, cohort_id),
    ).fetchone()
    return row is not None


def _identity_for_subjects(conn, cohort_id: str, subject_ids: set[str]) -> dict[str, tuple[str, int]]:
    """Fetch pseudonym/shift pairs for one segment's subjects in batched
    point queries, instead of materialising the cohort's whole identity map."""
    identity: dict[str, tuple[str, int]] = {}
    ids = sorted(subject_ids)
    for i in range(0, len(ids), _IDENTITY_QUERY_BATCH):
        batch = ids[i : i + _IDENTITY_QUERY_BATCH]
        placeholders = ", ".join("?" for _ in batch)
        for row in conn.execute(
            "SELECT subject_id, pseudonym, date_shift_days FROM subjects "
            f"WHERE cohort_id = ? AND subject_id IN ({placeholders})",
            (cohort_id, *batch),
        ):
            identity[row["subject_id"]] = (row["pseudonym"], row["date_shift_days"])
    return identity


def _deidentified_segments(
    conn,
    cohort_id: str,
    rows: Iterator[Row],
    stats: dict,
    ensure_current: Callable[[], None],
) -> Iterator[bytes]:
    """Yield encoded CSV segments (header first, rows in upload order).

    Each segment covers ``config.BUILD_CHUNK_ROWS`` parsed rows: only that
    segment's rows and identities are held in memory at any time. The
    staleness check runs before every segment, so a build is abandoned within
    one segment once a newer revision is accepted.
    """
    chunk_size = max(1, int(config.BUILD_CHUNK_ROWS))
    first = True
    while True:
        chunk = list(itertools.islice(rows, chunk_size))
        if not chunk:
            return
        ensure_current()
        identity = _identity_for_subjects(conn, cohort_id, {r.subject_id for r in chunk})
        parts = [deid.CSV_HEADER] if first else []
        first = False
        for row in chunk:
            mapping = identity.get(row.subject_id)
            if mapping is None:
                raise RuntimeError("数据不一致：修订包含未登记的受试者")
            parts.append(deid.format_csv_line(row, mapping[0], mapping[1]))
        stats["rows"] += len(chunk)
        yield "".join(parts).encode("utf-8")


def cleanup_orphan_tmp_files() -> int:
    """Delete streaming temp files left behind by interrupted builds.

    Runs at startup together with the database recovery: a crash mid-build
    can leave ``*.zip.tmp`` files that are never referenced by the database
    and never downloadable; removing them is safe because the affected
    package is rebuilt from the stored raw revision.
    """
    removed = 0
    pkg_root = config.PACKAGE_DIR
    if not pkg_root.is_dir():
        return 0
    for tmp in pkg_root.rglob("*.zip.tmp"):
        try:
            tmp.unlink()
            removed += 1
        except OSError:
            log.warning("could not remove orphan temp package file: %s", tmp)
    return removed


def mark_failed(cohort_id: str, revision_label: str, error: str) -> None:
    """Record a build failure with capped exponential backoff.

    A failure stays visible as to-be-generated: the worker retries it, a
    restart retries it immediately, and it can never be downloaded.
    """
    delay = min(60, 2 ** min(_attempts(cohort_id, revision_label), 6))
    next_retry = datetime.now(timezone.utc).timestamp() + delay
    next_retry_text = datetime.fromtimestamp(next_retry, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    with write_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                UPDATE packages
                   SET status = 'failed',
                       last_error = ?,
                       next_retry_at = ?,
                       updated_at = ?
                 WHERE cohort_id = ? AND revision = ? AND status = 'generating'
                """,
                (error[:1000], next_retry_text, _utcnow(), cohort_id, revision_label),
            )
        finally:
            conn.close()


def _attempts(cohort_id: str, revision_label: str) -> int:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT attempts FROM packages WHERE cohort_id = ? AND revision = ?",
            (cohort_id, revision_label),
        ).fetchone()
        return row["attempts"] if row else 0
    finally:
        conn.close()
