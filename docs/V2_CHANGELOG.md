# BioMoQA-Triage API — v2 Endpoint

**Date:** 2026-03-24

## Summary

Added a `/api/v2` endpoint that serves a retrained BioMoQA classifier designed to eliminate false positives on clinical and biomedical papers. The existing `/api/v1` endpoint remains unchanged.

## Problem

The v1 BioMoQA model was trained only with "environment NOT islands" negatives. It had no exposure to clinical content during training, causing it to misclassify papers about human disease (dengue, malaria, cancer epidemiology, clinical trials) as biodiversity-relevant — often with 100% confidence.

Production triage data showed ~9% of clinical papers scored above 0.5, with ~4% scoring above 0.9.

## Solution

The v2 model (RoBERTa-base, BCE loss, 5-fold CV) was retrained with an expanded negative set:

| Training Data | Count |
|---------------|-------|
| Original positives (island biodiversity) | 995 |
| Original negatives (environment) | 438 |
| Optional negatives (sampled) | 500 |
| Clinical hard negatives (curated from PubMed) | 1,392 |
| Mined false positives (from production triage exports) | 955 |

## Results

**On clinical papers (2,338 evaluated):**

| Metric | v1 | v2 |
|--------|-----|-----|
| False positives (score >= 0.5) | 206 (8.8%) | 0 (0.0%) |
| Mean P(biodiversity) | 0.118 | 0.002 |

**On original biodiversity test set (5-fold CV):**

| Metric | v1 | v2 |
|--------|-----|-----|
| F1 | 0.901 | 0.893 |
| Recall | 0.905 | 0.931 |
| ROC-AUC | 0.903 | 0.890 |

100% clinical false positive elimination with marginal (<1%) F1 trade-off.

## API Changes

### Endpoints

| Endpoint | Model | Behavior |
|----------|-------|----------|
| `POST /api/v1/job` | Original model | Unchanged |
| `GET /api/v1/job/{id}` | | Unchanged |
| `POST /api/v2/job` | Retrained model | Same request/response format |
| `GET /api/v2/job/{id}` | | Includes `model_version` field |

### Response Changes

The `GET /job/{id}` and `GET /job/{id}/status` responses now include a `model_version` field (`"v1"` or `"v2"`), indicating which model was used for scoring.

### Configuration

New environment variable:
- `HF_MODEL_BASE_DIR_V2` — path to v2 model checkpoints (default: `/models/checkpoints_v2`)

### Docker

The `docker-compose.yml` mounts both checkpoint directories:
- `../model/checkpoints:/models/checkpoints:ro` (v1)
- `../model/checkpoints_v2:/models/checkpoints_v2:ro` (v2)

### Model Download

```bash
./download_checkpoints.sh v1    # download v1 only
./download_checkpoints.sh v2    # download v2 only
./download_checkpoints.sh all   # download both
```

## S3 Layout

- v1: `s3://biomoqa-classifier/checkpoints/best_model_cross_val_BCE_roberta-base_fold-{1..5}/`
- v2: `s3://biomoqa-classifier/checkpoints_v2/best_model_cross_val_BCE_roberta-base_fold-{1..5}/`
