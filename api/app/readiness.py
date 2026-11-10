"""Readiness gate for the inference worker (shared triage guard, §4).

Runs before the Celery worker starts (Dockerfile.worker CMD and the worker
`command` in docker-compose.yml); by hand:

    docker compose exec worker /app/.venv/bin/python -m app.readiness

It exits non-zero unless ALL of the following hold, for EVERY served model
version (BIOMOQA_READINESS_VERSIONS, default `v1,v2,v3`, the versions
/api/v1, /api/v2 and /api/v3 score with):

1. Model: every fold directory `<HF_MODEL_BASE_DIR[_Vn]>/<HF_MODEL_PREFIX>_fold-<i>`
   exists and holds a single-logit (BCE) sequence classifier: that is what
   model_infer.predict_batch assumes when it reads the pipeline's `score` as
   the positive-class probability.
2. Model behaviour: the two control texts in readiness_controls.json score as
   expected through the same ensemble the worker uses: positive > negative and
   positive >= BIOMOQA_READINESS_MIN (default 0.6). A generic or degraded model
   (the DisTriage incident: an SST-2 sentiment model, every score <= 0.5)
   fails here.
3. Fetch: SIBILS_URL returns a non-empty record for the positive control's
   PMID. The SIBiLS instance and collection versions are printed and recorded.
   A stale or unreachable text source (the CellTriage incident) fails here.
   BIOMOQA_READINESS_FETCH=0 skips this step, loudly, for offline dev only.

The outcome is written best-effort to the `readiness` collection in Mongo
(document `_id: worker`) so the API's /readyz can report it; a Mongo failure
does not change the exit code.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

REQUIRED_FILES = ("config.json", "tokenizer_config.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
CONTROLS_PATH = Path(__file__).with_name("readiness_controls.json")
DEFAULT_VERSIONS = "v1,v2,v3"


def load_controls() -> dict:
    with CONTROLS_PATH.open(encoding="utf-8") as fh:
        c = json.load(fh)
    for key in ("positive", "negative"):
        assert c[key]["title"] and c[key]["abstract"], f"{key} control needs title and abstract"
    assert int(c["positive"]["pmid"]) > 0
    return c


def readiness_versions() -> list[str]:
    raw = os.environ.get("BIOMOQA_READINESS_VERSIONS", DEFAULT_VERSIONS)
    return [v.strip() for v in raw.split(",") if v.strip()]


class ReadinessError(Exception):
    """One failed check; main() records it and exits 1."""


def fail(msg: str) -> NoReturn:
    raise ReadinessError(msg)


def check_model_dir(model_dir: str) -> dict:
    if not model_dir:
        fail("model directory is empty")
    if not model_dir.startswith("/"):
        fail(
            f"model directory {model_dir!r} is not an absolute path; Hub ids are "
            "not allowed in a deployment (the model is provided, never downloaded)"
        )
    if not os.path.isdir(model_dir):
        fail(f"model directory {model_dir!r} does not exist")
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        fail(f"model directory {model_dir} lacks {missing}")
    if not any(os.path.isfile(os.path.join(model_dir, w)) for w in WEIGHT_FILES):
        fail(f"model directory {model_dir} has no weights ({' or '.join(WEIGHT_FILES)})")
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    archs = cfg.get("architectures") or []
    if not any(a.endswith("ForSequenceClassification") for a in archs):
        fail(f"{model_dir}: config.json architectures {archs} is not a sequence classifier")
    id2label = cfg.get("id2label")
    n_labels = len(id2label) if id2label else 2  # Transformers' default is 2
    if n_labels != 1:
        # predict_batch reads the pipeline's `score` as P(positive); that only
        # holds for a single-logit BCE head (sigmoid). A 2-label softmax head
        # would silently return the top class's probability instead.
        fail(f"{model_dir}: expected a single-logit (BCE) classifier, config.json declares {n_labels} labels")
    return cfg


def model_dirs(version: str) -> list[str]:
    from app.config import settings
    from app.services.model_infer import MODEL_DIRS

    base = MODEL_DIRS.get(version)
    if base is None:
        fail(f"unknown model version {version!r}; known: {sorted(MODEL_DIRS)}")
    return [f"{base}/{settings.HF_MODEL_PREFIX}_fold-{i}" for i in range(1, settings.HF_NUM_FOLDS + 1)]


def check_model_version(version: str, controls: dict, min_pos: float) -> dict:
    """Steps 1 and 2 for one model version. Returns what was measured, or raises."""
    from app.services import model_infer

    t0 = time.monotonic()
    dirs = model_dirs(version)
    archs = None
    for d in dirs:
        archs = check_model_dir(d).get("architectures")
    print(f"{version}: {len(dirs)} fold dir(s) ok under {os.path.dirname(dirs[0])} (architectures={archs})")

    pos_c, neg_c = controls["positive"], controls["negative"]
    try:
        preds = model_infer.predict_batch(
            [pos_c["title"], neg_c["title"]], [pos_c["abstract"], neg_c["abstract"]], version=version
        )
    except Exception as exc:  # noqa: BLE001 - any load/inference error is a failure
        fail(f"{version}: inference raised {type(exc).__name__}: {exc}")
    finally:
        # The worker loads its own copy; do not keep 5 folds per version in this process.
        model_infer._CACHE.pop(version, None)

    if len(preds) != 2:
        fail(f"{version}: model returned {len(preds)} predictions for 2 texts")
    pos, neg = float(preds[0]["score"]), float(preds[1]["score"])
    print(f"{version}: positive-control={pos:.3f} negative-control={neg:.3f} ({time.monotonic() - t0:.1f}s)")
    if pos < min_pos:
        fail(f"{version}: positive control scored {pos:.3f} < {min_pos}: wrong or degraded model")
    if pos <= neg:
        fail(f"{version}: positive control ({pos:.3f}) does not outrank the negative control ({neg:.3f})")
    return {"model_dir": os.path.dirname(dirs[0]), "folds": len(dirs),
            "positive": round(pos, 4), "negative": round(neg, 4)}


def check_model(controls: dict) -> dict:
    min_pos = float(os.environ.get("BIOMOQA_READINESS_MIN", "0.6"))
    versions = readiness_versions()
    if not versions:
        fail("BIOMOQA_READINESS_VERSIONS is empty: at least one served model version must be checked")
    print(f"checking model versions {versions} (offline={os.environ.get('HF_HUB_OFFLINE', '0')}, min_positive={min_pos})")
    return {"versions": {v: check_model_version(v, controls, min_pos) for v in versions},
            "min_positive": min_pos}


async def fetch_control(pmid: int) -> dict:
    """Step 3, shared with the API's /readyz. Raises on any failure."""
    from app.config import settings
    from app.services.sibils_client import client_session, fetch_medline

    async with client_session() as session:
        res = await fetch_medline(session, [str(pmid)])
    record = res.found.get(str(pmid))
    doc = (record or {}).get("document") or {}
    if not record or not (doc.get("title") or doc.get("abstract")):
        raise RuntimeError(
            f"SIBiLS {settings.SIBILS_URL} returned no text for control PMID {pmid} "
            f"(warning={res.warning!r}, collection={res.collection_version})"
        )
    return {"url": settings.SIBILS_URL, "control_pmid": pmid, "sibils_version": res.sibils_version,
            "collection_version": res.collection_version, "warning": res.warning}


def check_fetch(controls: dict) -> dict | None:
    if os.environ.get("BIOMOQA_READINESS_FETCH", "1") == "0":
        print("WARNING: SIBiLS fetch check skipped (BIOMOQA_READINESS_FETCH=0); dev only", file=sys.stderr)
        return None
    pmid = int(controls["positive"]["pmid"])
    try:
        info = asyncio.run(fetch_control(pmid))
    except Exception as exc:  # noqa: BLE001
        fail(f"SIBiLS fetch check failed: {exc}")
    print(f"sibils ok: {info['url']} version={info['sibils_version']} collection={info['collection_version']}")
    return info


def record(result: dict) -> None:
    """Best-effort: store the outcome for /readyz. Never changes the exit code."""
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings

    async def _write():
        client = AsyncIOMotorClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        try:
            await client[settings.MONGO_DB]["readiness"].replace_one({"_id": "worker"}, result, upsert=True)
        finally:
            client.close()

    try:
        asyncio.run(_write())
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not record readiness in Mongo: {exc}", file=sys.stderr)


def main() -> int:
    controls = load_controls()
    result: dict = {"_id": "worker", "ok": False, "checked_at": datetime.now(timezone.utc), "error": None}
    try:
        result["model"] = check_model(controls)
        result["sibils"] = check_fetch(controls)
        result["ok"] = True
    except ReadinessError as exc:
        result["error"] = str(exc)
        record(result)
        print(f"READINESS FAIL: {exc}", file=sys.stderr)
        return 1
    record(result)
    print("READINESS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
