"""Shared triage guard, end to end against a running stack.

Submits three PMIDs: two SIBiLS has, one it does not. The missing one must come
back with score null and error "missing", the job as partial with the counts.
Runs once per API version (v1, v2, v3 share the code, each scores with its own
model).

Opt-in on purpose: BIOMOQA_BASE must name the API explicitly, e.g.
BIOMOQA_BASE=http://localhost:8501. There is no default because localhost
ports are often SSH tunnels to a production host, and this test creates jobs.
Skipped as a whole when BIOMOQA_BASE is unset or no API answers there.
"""
import os
import time

import httpx
import pytest

BASE = os.environ.get("BIOMOQA_BASE")
if not BASE:
    pytest.skip("BIOMOQA_BASE not set (e.g. http://localhost:8501); stack test is opt-in", allow_module_level=True)
POSITIVE = 28812590           # Nogue et al. 2017, island biodiversity conservation (readiness positive control)
NEGATIVE = 30967128           # Alghadir et al. 2019, knee osteoarthritis RCT (readiness negative control)
UNKNOWN = 999999999           # not a PubMed id
VERSIONS = os.environ.get("BIOMOQA_TEST_VERSIONS", "v1,v2,v3").split(",")
TIMEOUT_S = 300

try:
    httpx.get(f"{BASE}/healthz", timeout=5).raise_for_status()
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"no BioMoQA API at {BASE}: {exc}", allow_module_level=True)


def wait_for_job(version: str, job_id: str) -> dict:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        job = httpx.get(f"{BASE}/api/{version}/job/{job_id}", timeout=30).json()
        if job["status"] in ("done", "partial", "failed"):
            return job
        time.sleep(5)
    raise AssertionError(f"job {job_id} still {job['status']} after {TIMEOUT_S}s")


@pytest.mark.parametrize("version", VERSIONS)
def test_missing_pmid_gets_null_score_and_partial_job(version):
    r = httpx.post(f"{BASE}/api/{version}/job",
                   json={"article_set": [{"pmid": p} for p in (POSITIVE, NEGATIVE, UNKNOWN)], "use_fulltext": False},
                   timeout=30)
    assert r.status_code == 200
    job = wait_for_job(version, r.json()["job_id"])

    assert job["status"] == "partial", job
    assert job["error"] is None
    assert job["model_version"] == version
    assert (job["n_submitted"], job["n_scored"], job["n_no_text"], job["n_fetch_error"], job["n_no_prediction"]) == (3, 2, 1, 0, 0)
    assert job["sibils_version"] and job["collection_version"]
    assert str(UNKNOWN) in (job["sibils_warning"] or "")

    by_pmid = {a["pmid"]: a for a in job["article_set"]}
    missing = by_pmid[UNKNOWN]
    assert missing["score"] is None and missing["error"] == "missing" and missing["text_status"] == "missing"
    for p in (POSITIVE, NEGATIVE):
        a = by_pmid[p]
        assert a["error"] is None and a["text_status"] == "ok"
        assert isinstance(a["score"], float) and 0.0 <= a["score"] <= 1.0
    # The island-biodiversity paper must outrank the clinical trial; the classifier is not a coin.
    assert by_pmid[POSITIVE]["score"] > 0.5 > by_pmid[NEGATIVE]["score"]

    status = httpx.get(f"{BASE}/api/{version}/job/{job['id']}/status", timeout=30).json()
    assert status["status"] == "partial" and status["n_scored"] == 2 and status["n_no_text"] == 1
    assert status["model_version"] == version


def test_readyz_reports_sibils_and_worker():
    r = httpx.get(f"{BASE}/readyz", timeout=60)
    body = r.json()
    assert r.status_code == 200, body
    assert body["sibils"]["sibils_version"] and body["sibils"]["collection_version"]
    assert body["worker"]["ok"] is True
    for version, m in body["worker"]["model"]["versions"].items():
        assert m["positive"] > 0.5 > m["negative"], (version, m)
