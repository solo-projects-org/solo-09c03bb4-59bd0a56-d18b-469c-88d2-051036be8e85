"""De-identification primitives and package assembly.

Stable pseudonyms
-----------------
For each cohort, subjects receive sequential pseudonyms ``S000001``,
``S000002`` ... in order of first appearance. The mapping lives in the
``subjects`` table and is never exported, so later revisions reuse the same
pseudonym while the original subject id never leaves the system.

Stable, private date shifts
---------------------------
Each subject gets one fixed integer shift in [-180, 180] days derived from a
per-cohort random salt and the original subject id via HMAC-SHA256. Because the
salt never leaves the service:

* the same subject keeps the same shift across every revision (visit intervals
  are preserved exactly);
* different cohorts use independent salts, hence independent shifts;
* shifted dates cannot be correlated back to calendar dates without the salt.

The shift is materialised once in the ``subjects`` table and reused on every
subsequent revision.

Bounded-memory assembly
-----------------------
:func:`write_package_streaming` writes the zip member by member, segment by
segment, straight to a file on disk: the de-identified CSV is streamed through
the compressor instead of being materialised in memory. :func:`build_package`
keeps the same output rules for small, already in-memory inputs and serves as
the reference implementation in tests; both share :func:`format_csv_line`,
:func:`build_manifest` and :func:`manifest_bytes` so the two paths cannot
drift apart.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import secrets
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

from .csvutil import Row

SHIFT_RANGE = 361  # -180 .. 180 inclusive

CSV_HEADER = "pseudonym,visit_date_shifted,value\n"
CSV_MEMBER = "deidentified_visits.csv"
MANIFEST_MEMBER = "manifest.json"


def new_salt() -> str:
    return secrets.token_hex(32)


def subject_date_shift(salt: str, subject_id: str) -> int:
    digest = hmac.new(salt.encode("utf-8"), subject_id.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % SHIFT_RANGE - 180


def pseudonym(index: int) -> str:
    return f"S{index:06d}"


def format_csv_line(row: Row, pseudonym_str: str, shift_days: int) -> str:
    """One output line: pseudonym + shifted date + value, never the originals."""
    shifted = row.visit_date + timedelta(days=shift_days)
    return f"{pseudonym_str},{shifted.isoformat()},{_format_value(row.value)}\n"


def build_manifest(row_count: int, subject_count: int) -> dict:
    """Only structural metadata: no identifiers, no original dates, no salt."""
    return {
        "schema_version": 1,
        "columns": ["pseudonym", "visit_date_shifted", "value"],
        "row_count": row_count,
        "subject_count": subject_count,
        "date_note": (
            "visit_date_shifted = original visit date + per-subject constant shift; "
            "within-subject visit intervals are unchanged"
        ),
        "privacy": "stable cohort-scoped pseudonyms; original subject ids and dates are not included",
    }


def manifest_bytes(manifest: dict) -> bytes:
    return (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_package(
    rows: list[Row],
    identity: dict[str, tuple[str, int]],
) -> tuple[bytes, dict]:
    """Build the de-identified zip package in memory.

    ``identity`` maps original subject id -> (pseudonym, date_shift_days).
    Only pseudonyms, shifted dates and measurements are written; original
    identifiers and original dates are never serialised anywhere in the
    archive. Kept for small inputs and as the in-test reference for the
    streaming path (:func:`write_package_streaming`), which produces identical
    member contents without materialising the whole package.
    """
    csv_text = CSV_HEADER + "".join(
        format_csv_line(row, *identity[row.subject_id]) for row in rows
    )
    manifest = build_manifest(row_count=len(rows), subject_count=len(identity))

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(CSV_MEMBER, csv_text.encode("utf-8"))
        zf.writestr(MANIFEST_MEMBER, manifest_bytes(manifest))
    return out.getvalue(), manifest


def write_package_streaming(
    tmp_path: Path,
    csv_segments: Iterable[bytes],
    manifest: Callable[[], dict],
) -> None:
    """Write the package zip to ``tmp_path`` segment by segment.

    ``csv_segments`` yields encoded CSV chunks (header first, rows in upload
    order); each chunk is pushed through the zip compressor as it arrives, so
    memory stays bounded no matter how many visit rows the revision has.
    ``manifest`` is called only after the CSV member is complete, so counts
    observed while streaming can be included. The caller owns the atomic
    rename to the final path and temp-file cleanup on failure.
    """
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with zf.open(CSV_MEMBER, "w") as member:
            for segment in csv_segments:
                member.write(segment)
        zf.writestr(MANIFEST_MEMBER, manifest_bytes(manifest()))


def _format_value(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return repr(value)
