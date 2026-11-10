# BioMoQA-Triage API

A FastAPI + Celery + MongoDB + Redis pipeline for **BioMoQA-Triage** using an ensemble of fine-tuned RoBERTa models.  

---

## Features

- **Job-based submission** – Create jobs with a set of PMIDs
- **Two-Stage Pipeline**:
  1. **Ingress**: Fetch MEDLINE abstracts and, if available, PMC fulltext via SIBILS API
  2. **Inference**: Run fine-tuned RoBERTa model on title + abstract
- **Single Model Inference** – Default uses 1 model for ~3x faster inference (configurable up to 5-fold ensemble)
- **Batch processing** – Configurable batch sizes and wait times for ingress & inference
- **Job Status Tracking** – Poll `/api/v1/job/{job_id}` for progress
- **Results Retrieval** – Get scores, source (abstract/fulltext), and texts
- **MongoDB Backend** – Stores jobs & documents, enforces `(job_id, pmid)` uniqueness
- **Celery + Redis** – Distributed task queue
- **Flower UI** – Celery monitoring on port `5556`
- **Downloadable results** – Retrieve ranked results as JSON or export for downstream use

---

## Project Layout

```
api/
├── app/
│   ├── main.py              # FastAPI entrypoint, /healthz, /readyz; mounts the job router at /api/v1, /api/v2, /api/v3
│   ├── config.py            # Settings from the environment (no silent default for SIBiLS)
│   ├── db.py                # MongoDB connection (Beanie)
│   ├── routers/job.py       # Job endpoints (one implementation, three API versions)
│   ├── models/              # Beanie documents: Job, DocumentEntry
│   ├── services/
│   │   ├── sibils_client.py # SIBiLS fetch with per-id outcome, retries, versions
│   │   ├── job_summary.py   # n_* counts and done/partial/failed from the documents
│   │   ├── model_infer.py   # Transformers ensemble per model version
│   │   └── batching.py, text_utils.py
│   ├── readiness.py         # Model + SIBiLS gate before the worker starts (python -m app.readiness)
│   └── readiness_controls.json  # Control texts for the gate and /readyz
├── worker/                  # Celery app + tasks (copied to app/worker in the image)
├── tests/                   # Unit tests (job_summary, sibils parsing) and stack tests
├── Dockerfile.api
├── Dockerfile.worker
├── docker-compose.yml
├── docker-compose.dev.yml
├── pyproject.toml, uv.lock
└── README.md
```

---

## Configuration

Environment variables (via `.env` or `docker-compose`):

| Variable              | Default                                          | Description                   |
|-----------------------|--------------------------------------------------|-------------------------------|
| `MONGO_URI`           | `mongodb://mongo:27017/biomoqa-triage`           | MongoDB connection            |
| `MONGO_DB`            | `biomoqa-triage`                                 | Database name                 |
| `REDIS_URL`           | `redis://redis:6379/0`                           | Redis broker                  |
| `HF_MODEL_BASE_DIR`   | `/models/checkpoints`                            | Base directory for models     |
| `HF_MODEL_PREFIX`     | `best_model_cross_val_BCE_roberta-base`          | Model filename prefix         |
| `HF_NUM_FOLDS`        | `1`                                              | Number of cross-validation folds |
| `HF_DEVICE`           | `-1`                                             | Device (`-1` = CPU, `0` = GPU)|
| `MAX_TOKENS`          | `512`                                            | Maximum tokens for tokenizer  |
| `MAX_TEXT_CHARS`      | `5000`                                           | Maximum characters per doc    |
| `INGRESS_BATCH_SIZE`  | `64`                                             | PMIDs per ingress batch       |
| `INGRESS_MAX_WAIT_MS` | `5000`                                           | Max wait time for ingress batch (ms) |
| `INFER_BATCH_SIZE`    | `32`                                             | Docs per inference batch      |
| `INFER_MAX_WAIT_MS`   | `5000`                                           | Max wait time for inference batch (ms) |
| `SIBILS_URL`          | **none** (required)                              | SIBiLS API base URL, no trailing slash (the client appends `/fetch`). The process refuses to start without it; Compose sets it from `BIOMOQA_SIBILS_URL` (default `https://biodiversitypmc.sibils.org/api`) |
| `SIBILS_BATCH`        | `100`                                            | Batch size for SIBILS requests |
| `SIBILS_TIMEOUT`      | `30`                                             | Timeout for SIBILS API (seconds) |
| `SIBILS_RETRIES`      | `5`                                              | Attempts per SIBiLS fetch (backoff 2 s → 30 s, about one minute in all) before the batch is `fetch_error` |
| `BIOMOQA_READINESS_VERSIONS` | `v1,v2,v3`                                | Model versions `app.readiness` checks (every version the API serves) |
| `BIOMOQA_READINESS_MIN` | `0.6`                                          | Minimum score of the positive control in `app.readiness` |
| `BIOMOQA_READINESS_FETCH` | `1`                                          | `0` skips the SIBiLS part of `app.readiness` (offline development only) |
| `CORS_ORIGINS`        | `*`                                              | Allowed CORS origins          |

---

## Running

```bash
docker compose up --build
```

Services:
- **API** → [http://localhost:8501](http://localhost:8501)
- **MongoDB** → `localhost:27017`
- **Redis** → `localhost:6379`
- **Worker** → Celery worker (GPU-enabled for ensemble inference)
- **Flower** → [http://localhost:5556](http://localhost:5556)

---

## API Endpoints

### Create Job
**POST** `/api/v1/job`

Request:
```json
{
  "use_fulltext": true,
  "article_set": [
    { "pmid": 36585756 },
    { "pmid": 36564873 }
  ]
}
```

Response:
```json
{
  "job_id": "uuid"
}
```

---

### Job Details
**GET** `/api/v1/job/{job_id}` (same under `/api/v2` and `/api/v3`)

A score is only ever produced from text the worker actually fetched. An
article without a score has `score: null` and an `error` saying why:
`missing` (SIBiLS did not return the PMID), `empty` (record without title and
abstract), `fetch_error` (SIBiLS unreachable after retries) or `no_prediction`
(the model gave no usable score). Never `0.0` for "no text".

A finished job whose scores are all one single value (all 0.0, all 0.5) over 20
or more articles is `failed` with `error: "degenerate_scores"`: that is a wrong
or degraded model, never a result.

Job `status`: `done` when every article is scored, `partial` when some are not
(the scored ones are valid), `failed` when none is scored or when more than
half of a job of 20 or more articles has no text (`error: "sibils_fetch"`,
with the SIBiLS warning in `sibils_warning`).

Response:
```json
{
  "id": "uuid",
  "use_fulltext": false,
  "model_version": "v2",
  "status": "partial",
  "error": null,
  "n_submitted": 3,
  "n_scored": 2,
  "n_no_text": 1,
  "n_fetch_error": 0,
  "n_no_prediction": 0,
  "sibils_version": "v5.1.5.10",
  "collection_version": "med_r1_v5.10",
  "sibils_warning": "missing ids:999999999",
  "job_created_at": "2026-11-10T20:26:01.333Z",
  "process_start_at": "2026-11-10T20:26:01.333Z",
  "process_end_at": "2026-11-10T20:27:12.000Z",
  "process_time": 70.7,
  "article_set": [
    {
      "pmid": 28812590,
      "score": 0.97,
      "error": null,
      "text_status": "ok",
      "pmcid": null,
      "text_source": "abstract",
      "text": "Island biodiversity conservation needs palaeoecology. The discovery and colonization..."
    },
    {
      "pmid": 999999999,
      "score": null,
      "error": "missing",
      "text_status": "missing",
      "pmcid": null,
      "text_source": "abstract",
      "text": null
    }
  ]
}
```

---

### Job Status
**GET** `/api/v1/job/{job_id}/status` (same under `/api/v2` and `/api/v3`)

Same `status`, `error` and `n_*` counts as above, plus the progress counters.

Response:
```json
{
  "job_id": "uuid",
  "status": "running",
  "model_version": "v2",
  "error": null,
  "submitted_pmids": 200,
  "dedup_dropped": 3,
  "n_submitted": 197,
  "n_scored": 165,
  "n_no_text": 7,
  "n_fetch_error": 0,
  "n_no_prediction": 5,
  "ingress_queued": 10,
  "ingress_done": 180,
  "ingress_failed": 7,
  "infer_queued": 10,
  "infer_done": 165,
  "infer_failed": 5
}
```

---

### Download Results
**GET** `/api/v1/job/{job_id}?download=true`  

Returns the full job results as a downloadable JSON file (ranked by submission order + scores).  

---

## Model

The system uses a **fine-tuned RoBERTa model** (single fold by default) for robust predictions:

```bash
HF_MODEL_BASE_DIR=/models/checkpoints
HF_MODEL_PREFIX=best_model_cross_val_BCE_roberta-base
HF_NUM_FOLDS=5
```

Models are loaded from:
```
../model/checkpoints/best_model_cross_val_BCE_roberta-base_fold-1
../model/checkpoints/best_model_cross_val_BCE_roberta-base_fold-2
../model/checkpoints/best_model_cross_val_BCE_roberta-base_fold-3
../model/checkpoints/best_model_cross_val_BCE_roberta-base_fold-4
../model/checkpoints/best_model_cross_val_BCE_roberta-base_fold-5
```

Default uses 1 model for fast inference. Set HF_NUM_FOLDS=5 for ensemble averaging (slower but slightly more robust).

---

## Readiness (shared triage guard)

```bash
docker compose exec worker /app/.venv/bin/python -m app.readiness
curl -i http://127.0.0.1:8501/readyz
```

`app.readiness` runs before the Celery worker starts (`Dockerfile.worker` and
the worker `command` in `docker-compose.yml`). It exits non-zero unless, for
every version in `BIOMOQA_READINESS_VERSIONS` (default `v1,v2,v3`), all fold
directories exist and hold a single-logit sequence classifier, the two
control texts in `app/readiness_controls.json` score as expected through the
same ensemble the worker uses (positive ≥ `BIOMOQA_READINESS_MIN`, default
0.6, and above the negative one), and `SIBILS_URL` returns a record for the
positive control's PMID. It prints the SIBiLS instance and collection versions
and stores the outcome in the `readiness` collection. A worker that fails the
gate keeps restarting (`docker compose ps`), and `docker compose logs worker`
says which check failed.

`GET /readyz` on the API returns 200 with the live SIBiLS versions and the
worker's last readiness outcome, or 503 with a `problems` list. `GET /healthz`
stays a plain liveness check.

Why: two sister services published meaningless scores for weeks (one scored
with a generic model, every score ≤ 0.5; one fetched from a stale SIBiLS and
wrote 0.0 for every article) and nothing noticed. The model check makes the
first loud; the fetch check and the per-article `text_status` make the second.

### Tests

```bash
uv run --group dev pytest tests/test_scoring_guard_units.py                      # pure, no stack needed
BIOMOQA_BASE=http://localhost:8501 uv run --group dev pytest tests/test_scoring_guard.py   # needs the stack
```

`tests/test_scoring_guard.py` submits, per API version, two PMIDs SIBiLS has
and one it does not, and checks `score: null` + `error: "missing"` +
`status: partial` with the counts. It is opt-in: it skips itself unless
`BIOMOQA_BASE` names the API explicitly (no `localhost` default, because a
local port is often an SSH tunnel to a production host and the test creates
jobs), or when no API answers there.

---

## Development

Run locally:
```bash
pip install -r requirements.base.txt
uvicorn app.main:app --reload
celery -A app.worker.celery_app worker --loglevel=info -Q ingress,infer
```

---

## License
MIT License
