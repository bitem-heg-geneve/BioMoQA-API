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
│   ├── main.py              # FastAPI entrypoint
│   ├── api/                 # Routers (job endpoints)
│   ├── db/                  # MongoDB connection + models
│   ├── services/            # Job logic, inference, batching
│   └── worker/              # Celery workers
├── Dockerfile.api
├── Dockerfile.worker
├── docker-compose.yml
├── docker-compose.dev.yml
├── requirements.api.txt
├── requirements.worker.txt
├── requirements.base.txt
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
| `SIBILS_URL`          | `https://biodiversitypmc.sibils.org/api`         | SIBILS API endpoint           |
| `SIBILS_BATCH`        | `100`                                            | Batch size for SIBILS requests |
| `SIBILS_TIMEOUT`      | `30`                                             | Timeout for SIBILS API (seconds) |
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
**GET** `/api/v1/job/{job_id}`

Response:
```json
{
  "id": "uuid",
  "use_fulltext": true,
  "status": "running",
  "job_created_at": "2025-08-15T20:26:01.333Z",
  "process_start_at": "2025-08-15T20:30:00.000Z",
  "process_end_at": null,
  "process_time": null,
  "article_set": [
    {
      "pmid": 36585756,
      "score": 0.91,
      "pmcid": "PMC123456",
      "text_source": "fulltext",
      "text": "Title + abstract/fulltext snippet..."
    }
  ]
}
```

---

### Job Status
**GET** `/api/v1/job/{job_id}/status`

Response:
```json
{
  "job_id": "uuid",
  "status": "running",
  "submitted_pmids": 200,
  "dedup_dropped": 3,
  "ingress_queued": 10,
  "ingress_done": 180,
  "ingress_failed": 7,
  "infer_queued": 15,
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
