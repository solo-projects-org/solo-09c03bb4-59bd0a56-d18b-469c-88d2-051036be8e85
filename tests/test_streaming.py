"""Tests for streaming package generation: bounded memory, prompt preemption
when revisions interleave, and failure/restart recovery of the current package."""
from __future__ import annotations

import csv
import io
import json
import tracemalloc
import zipfile

import pytest

from app.csvutil import CSVValidationError, iter_parse_csv, parse_csv


def _csv_bytes(rows: list[tuple], header: str = "subject_id,visit_date,value") -> bytes:
    buf = io.StringIO()
    buf.write(header + "\n")
    for r in rows:
        buf.write(",".join(str(x) for x in r) + "\n")
    return buf.getvalue().encode("utf-8")


def _upload(client, cohort: str, revision: str, payload: bytes):
    return client.post(
        f"/api/cohorts/{cohort}/revisions/{revision}",
        files={"file": ("rev.csv", payload, "text/csv")},
    )


def _read_zip(resp_bytes: bytes) -> tuple[list[dict], dict]:
    zf = zipfile.ZipFile(io.BytesIO(resp_bytes))
    assert set(zf.namelist()) == {"deidentified_visits.csv", "manifest.json"}
    rows = list(csv.DictReader(io.StringIO(zf.read("deidentified_visits.csv").decode("utf-8"))))
    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    return rows, manifest


def _wait_published(app_env, cohort: str, max_ticks: int = 10):
    tick = app_env["tick"]
    for _ in range(max_ticks):
        status = app_env["client"].get(f"/api/cohorts/{cohort}").json()
        if status["current_package"]["status"] == "published":
            return status
        tick()
    raise AssertionError("package did not reach published state")


# ---------------------------------------------------------------------------
# Streaming CSV parser
# ---------------------------------------------------------------------------


def test_iter_parse_csv_matches_parse_csv(tmp_path):
    payload = "﻿受试者号,访视日期,测值\nP1,2026-01-01,1.5\nP2,2026-02-03,2\n\n\n".encode("utf-8")
    path = tmp_path / "rev1.csv"
    path.write_bytes(payload)
    streamed = list(iter_parse_csv(path))
    assert streamed == parse_csv(payload)


def test_iter_parse_csv_rejects_bad_encoding_and_empty_data(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"subject_id,visit_date,value\nP1,2026-01-01,1\nP2,2026-01-02,\xff\n")
    with pytest.raises(CSVValidationError):
        list(iter_parse_csv(bad))

    header_only = tmp_path / "empty.csv"
    header_only.write_bytes(b"subject_id,visit_date,value\n")
    with pytest.raises(CSVValidationError):
        list(iter_parse_csv(header_only))


# ---------------------------------------------------------------------------
# Bounded memory on large revisions
# ---------------------------------------------------------------------------


def test_large_revision_build_memory_stays_bounded(app_env):
    client, tick = app_env["client"], app_env["tick"]
    n_subjects, n_visits = 20_000, 10  # 200k visit rows, 20k cohort subjects
    subjects = [f"P{i:06d}" for i in range(n_subjects)]
    lines = ["subject_id,visit_date,value"]
    for v in range(n_visits):
        for s in subjects:
            lines.append(f"{s},2026-{v + 1:02d}-15,1.5")
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    resp = _upload(client, "cohort-big", "1", payload)
    assert resp.status_code == 202, resp.text

    tracemalloc.start()
    try:
        assert tick() is True
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    status = client.get("/api/cohorts/cohort-big").json()
    assert status["current_package"]["status"] == "published", status["current_package"]

    dl = client.get("/api/cohorts/cohort-big/package")
    assert dl.status_code == 200
    rows, manifest = _read_zip(dl.content)
    assert manifest["row_count"] == n_subjects * n_visits
    assert manifest["subject_count"] == n_subjects
    assert len(rows) == n_subjects * n_visits
    # Generation memory is bounded by the segment size, not by the number of
    # visit rows or cohort subjects: the previous all-in-memory build needed
    # well over 100 MB for this input (row list + identity map + zip bytes).
    assert peak < 32 * 1024 * 1024, f"build peak memory {peak / 2**20:.1f} MiB"


# ---------------------------------------------------------------------------
# Output identical to the original rules
# ---------------------------------------------------------------------------


def test_streaming_package_matches_reference_builder(app_env):
    from app import deid as deid_module

    client, tick = app_env["client"], app_env["tick"]
    rows_data = [
        ("P003", "2026-02-01", 9.9),
        ("P001", "2026-01-10", 12.5),
        ("P001", "2026-02-10", 13.0),
        ("P002", "2026-01-15", 7.2),
    ]
    payload = _csv_bytes(rows_data)
    assert _upload(client, "cohort-ref", "1", payload).status_code == 202
    assert tick() is True

    dl = client.get("/api/cohorts/cohort-ref/package")
    assert dl.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(dl.content))
    got_csv = zf.read("deidentified_visits.csv")
    got_manifest = zf.read("manifest.json")

    db = app_env["modules"]["database"].get_conn()
    try:
        identity = {
            r["subject_id"]: (r["pseudonym"], r["date_shift_days"])
            for r in db.execute(
                "SELECT subject_id, pseudonym, date_shift_days FROM subjects "
                "WHERE cohort_id = 'cohort-ref'"
            ).fetchall()
        }
    finally:
        db.close()

    expected_payload, _ = deid_module.build_package(parse_csv(payload), identity)
    ezf = zipfile.ZipFile(io.BytesIO(expected_payload))
    assert got_csv == ezf.read("deidentified_visits.csv")
    assert got_manifest == ezf.read("manifest.json")

    # Row order is exactly the upload order; pseudonyms follow first appearance.
    lines = got_csv.decode("utf-8").splitlines()
    assert lines[0] == "pseudonym,visit_date_shifted,value"
    assert [line.split(",")[0] for line in lines[1:]] == [
        "S000001",
        "S000002",
        "S000002",
        "S000003",
    ]


def test_manifest_subject_count_covers_cohort_history(app_env):
    client, tick = app_env["client"], app_env["tick"]
    _upload(client, "cohort-hist", "1", _csv_bytes([
        ("P001", "2026-01-10", 1.0),
        ("P002", "2026-01-11", 2.0),
        ("P003", "2026-01-12", 3.0),
    ]))
    assert tick() is True

    # Rev2 mentions only one old subject plus one new one; the other two
    # subjects stay in the cohort identity table.
    _upload(client, "cohort-hist", "2", _csv_bytes([
        ("P001", "2026-02-10", 1.5),
        ("P004", "2026-02-11", 4.0),
    ]))
    assert tick() is True

    dl = client.get("/api/cohorts/cohort-hist/package")
    assert dl.status_code == 200
    rows, manifest = _read_zip(dl.content)
    assert manifest["row_count"] == 2
    # Same rule as the original full-identity-map build: subject_count is the
    # cohort's total registered subjects, not just those in this revision.
    assert manifest["subject_count"] == 4
    assert {r["pseudonym"] for r in rows} == {"S000001", "S000004"}


# ---------------------------------------------------------------------------
# Interleaved revisions: the running build stops promptly
# ---------------------------------------------------------------------------


def test_new_revision_aborts_running_build_promptly(app_env, monkeypatch):
    client, tick = app_env["client"], app_env["tick"]
    services = app_env["modules"]["services"]
    config = app_env["modules"]["config"]

    rows1 = [(f"P{i:04d}", "2026-01-10", 1.0) for i in range(50)]
    assert _upload(client, "cohort-pre", "1", _csv_bytes(rows1)).status_code == 202
    assert services.claim_pending_package() == ("cohort-pre", 1)

    # Small segments + a hook that lands rev2 while rev1 is mid-build.
    monkeypatch.setattr(config, "BUILD_CHUNK_ROWS", 5)
    injected = {"done": False}
    real_identity = services._identity_for_subjects

    def inject_mid_build(conn, cohort_id, subject_ids):
        if not injected["done"]:
            injected["done"] = True
            rows2 = [(f"P{i:04d}", "2026-02-10", 2.0) for i in range(50)]
            assert _upload(client, "cohort-pre", "2", _csv_bytes(rows2)).status_code == 202
        return real_identity(conn, cohort_id, subject_ids)

    monkeypatch.setattr(services, "_identity_for_subjects", inject_mid_build)

    with pytest.raises(services.StaleRevisionError):
        services.finalize_package("cohort-pre", "1")

    # The old build stopped: no final package, no temp file, nothing downloadable.
    pkg_dir = config.PACKAGE_DIR / "cohort-pre"
    assert not (pkg_dir / "rev1_deid.zip").exists()
    assert not (pkg_dir / "rev1_deid.zip.tmp").exists()
    status = client.get("/api/cohorts/cohort-pre").json()
    states = {p["revision"]: p["status"] for p in status["packages"]}
    assert states["1"] == "superseded"
    assert states["2"] == "pending"
    assert client.get("/api/cohorts/cohort-pre/package").status_code == 409

    # The queue is free immediately: the next tick builds rev2, which then
    # becomes the only downloadable package.
    assert tick() is True
    _wait_published(app_env, "cohort-pre")
    dl = client.get("/api/cohorts/cohort-pre/package")
    assert dl.status_code == 200
    rows, manifest = _read_zip(dl.content)
    assert manifest["row_count"] == 50
    # Content is rev2's (value 2), never a mix with the preempted rev1 (value 1).
    assert {r["value"] for r in rows} == {"2"}


# ---------------------------------------------------------------------------
# Failure and restart recovery
# ---------------------------------------------------------------------------


def test_write_failure_and_restart_recovery(app_env, monkeypatch):
    client, tick = app_env["client"], app_env["tick"]
    services = app_env["modules"]["services"]
    config = app_env["modules"]["config"]
    database = app_env["modules"]["database"]
    from app import deid as deid_module

    # A second cohort whose published package must stay downloadable throughout.
    assert _upload(client, "cohort-ok", "1", _csv_bytes([("K1", "2026-01-05", 3.5)])).status_code == 202
    assert tick() is True
    assert client.get("/api/cohorts/cohort-ok/package").status_code == 200

    payload = _csv_bytes([("P1", "2026-01-01", 1.0), ("P2", "2026-01-02", 2.0)])
    assert _upload(client, "cohort-wf", "1", payload).status_code == 202

    # Disk write fails mid-stream (after a partial temp file exists).
    fail = {"on": True}
    real_stream = deid_module.write_package_streaming

    def flaky_stream(tmp_path, segments, manifest):
        if fail["on"]:
            with open(tmp_path, "wb") as fh:
                fh.write(b"PK\x03\x04 half-written")
            raise OSError("No space left on device")
        return real_stream(tmp_path, segments, manifest)

    monkeypatch.setattr(deid_module, "write_package_streaming", flaky_stream)

    assert tick() is True
    pkg = client.get("/api/cohorts/cohort-wf").json()["current_package"]
    assert pkg["status"] == "failed"
    assert "No space left on device" in pkg["last_error"]
    # Partial temp file cleaned up; nothing downloadable; other cohort fine.
    assert not list((config.PACKAGE_DIR / "cohort-wf").glob("*.zip.tmp"))
    assert client.get("/api/cohorts/cohort-wf/package").status_code == 409
    assert client.get("/api/cohorts/cohort-ok/package").status_code == 200

    # Disk recovers: manual retry re-arms the package, the rebuild succeeds.
    fail["on"] = False
    assert client.post("/api/cohorts/cohort-wf/retry").json()["status"] == "pending"
    assert tick() is True
    assert client.get("/api/cohorts/cohort-wf/package").status_code == 200

    # Simulate a crash mid-build: package stuck in 'generating' plus a stray
    # temp file. Startup recovery must re-arm it and sweep the temp file.
    conn = database.get_conn()
    conn.execute("UPDATE packages SET status='generating' WHERE cohort_id='cohort-wf'")
    conn.close()
    stray = config.PACKAGE_DIR / "cohort-wf" / "rev1_deid.zip.tmp"
    stray.write_bytes(b"half-written")

    assert database.reset_interrupted_packages() >= 1
    assert services.cleanup_orphan_tmp_files() == 1
    assert not stray.exists()
    pkg = client.get("/api/cohorts/cohort-wf").json()["current_package"]
    assert pkg["status"] == "pending"
    assert client.get("/api/cohorts/cohort-wf/package").status_code == 409
    assert client.get("/api/cohorts/cohort-ok/package").status_code == 200

    # The current package is rebuilt from the stored revision.
    assert tick() is True
    dl = client.get("/api/cohorts/cohort-wf/package")
    assert dl.status_code == 200
    rows, manifest = _read_zip(dl.content)
    assert manifest["row_count"] == 2
    assert client.get("/api/cohorts/cohort-ok/package").status_code == 200
