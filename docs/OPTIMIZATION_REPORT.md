# BioMoQA API Performance Optimization Report

**Date:** December 19, 2025
**Author:** Paul van Rijen (with Claude Code assistance)

## Executive Summary

The BioMoQA triage API performance was optimized to process a full MEDLINE week (~20,000 articles) in under 3 hours, down from an estimated 2-3 days. This was achieved through CPU threading optimization and increased worker parallelism.

## Problem Statement

The API was taking 2-3 days to process a week's worth of new MEDLINE articles (~20,000 PMIDs), making it impractical for weekly triage workflows. The goal was to reduce processing time to under 1 day.

## Benchmark Results (200 PMIDs)

| Configuration | Total Time | Per PMID | Est. 20k PMIDs |
|---------------|------------|----------|----------------|
| **Before** (2 workers, unlimited threads) | ~300s | 1.56s | ~8.7 hours |
| **After** (4 workers, 12 threads each) | ~86s | 0.45s | **~2.5 hours** |
| **Improvement** | **3.5x faster** | **3.5x faster** | **3.5x faster** |

*Note: Benchmarks performed with 5-fold RoBERTa ensemble for maximum accuracy.*

## Configuration Changes

### Before (docker-compose.yml)

```yaml
worker:
  environment:
    HF_NUM_FOLDS: "5"
    # No thread limits - each worker used all 52 cores
  command:
    - "--concurrency=2"
    - "--prefetch-multiplier=2"
```

### After (docker-compose.yml)

```yaml
worker:
  environment:
    HF_NUM_FOLDS: "5"
    OMP_NUM_THREADS: "12"      # Limit each worker to 12 threads
    MKL_NUM_THREADS: "12"      # For Intel MKL operations
    TOKENIZERS_PARALLELISM: "false"  # Prevent tokenizer deadlocks
    INFER_BATCH_SIZE: "64"     # Increased from 32
    INGRESS_BATCH_SIZE: "128"  # Increased from 64
  command:
    - "--concurrency=4"        # Increased from 2
    - "--prefetch-multiplier=1"
```

## Key Optimizations

### 1. Thread Limiting (OMP_NUM_THREADS=12)
- **Problem:** With 2 workers on a 52-core machine, both workers competed for all cores, causing severe CPU contention
- **Solution:** Limit each worker to 12 threads, allowing 4 workers to run efficiently in parallel
- **Impact:** Eliminates thread contention, enables true parallelism

### 2. Increased Worker Concurrency (4 workers)
- **Before:** 2 workers processing batches sequentially
- **After:** 4 workers processing batches in parallel
- **Impact:** ~2x throughput improvement

### 3. Larger Batch Sizes
- **INFER_BATCH_SIZE:** 32 → 64 (more efficient CPU utilization per batch)
- **INGRESS_BATCH_SIZE:** 64 → 128 (faster MEDLINE metadata fetching)

### 4. Tokenizer Parallelism Disabled
- Prevents potential deadlocks in forked Celery workers

## Additional Changes

### use_fulltext Default Changed to False
The `use_fulltext` parameter was changed from `true` to `false` as the default because:
1. Full-text processing is not currently implemented in the inference pipeline
2. The API only uses title + abstract for classification
3. This makes the API behavior clearer to users

## Server Specifications

- **CPU:** 52 cores
- **RAM:** 62 GB
- **Model:** RoBERTa-base (5-fold cross-validation ensemble)
- **Framework:** PyTorch (CPU inference)

## Recommendations

1. **For weekly MEDLINE triage:** Current 5-fold configuration (~2.5 hours for 20k)
2. **For faster processing:** Set `HF_NUM_FOLDS=1` (~36 minutes for 20k, slight accuracy trade-off)
3. **For GPU acceleration:** If a GPU becomes available, set `HF_DEVICE=0` for 10-50x speedup

## Conclusion

The optimization reduced processing time from ~8.7 hours to ~2.5 hours for 20,000 articles (3.5x improvement), well within the 1-day target. Weekly MEDLINE triage can now be completed in a practical timeframe while maintaining full 5-fold ensemble accuracy.
