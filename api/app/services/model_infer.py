# api/app/services/model_infer.py
from transformers import pipeline, AutoTokenizer
from ..config import settings
from typing import Optional

_CACHE: dict[str, Optional[list]] = {}

MODEL_DIRS = {
    "v1": settings.HF_MODEL_BASE_DIR,
    "v2": settings.HF_MODEL_BASE_DIR_V2,
}

def get_pipes(version: str = "v1"):
    """Load all cross-validation fold models for ensemble inference."""
    if version not in _CACHE:
        base_dir = MODEL_DIRS.get(version, settings.HF_MODEL_BASE_DIR)
        pipes = []
        for fold_idx in range(1, settings.HF_NUM_FOLDS + 1):
            model_path = f"{base_dir}/{settings.HF_MODEL_PREFIX}_fold-{fold_idx}"
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                model_max_length=settings.MAX_TOKENS,
                truncation_side="right",
            )
            pipe = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=tokenizer,
                device=settings.HF_DEVICE,
            )
            pipes.append(pipe)
        _CACHE[version] = pipes
    return _CACHE[version]

def predict_batch(titles: list[str], abstracts: list[str], version: str = "v1") -> list[dict]:
    """
    Ensemble inference across all cross-validation folds.
    Returns a dict per text with:
      - score: ensemble probability of the POSITIVE class (0..1)
      - all:   full list of {label, score} for debugging/inspection
    """
    pipes = get_pipes(version)

    # Combine titles and abstracts
    texts = [f"{title}. {abstract}".strip() for title, abstract in zip(titles, abstracts)]

    # Collect predictions from all folds
    # For single-label models (num_labels=1), pipeline returns sigmoid of logit as the score
    all_fold_predictions = []
    for pipe in pipes:
        outputs = pipe(
            texts,
            truncation=True,
            max_length=settings.MAX_TOKENS,
            padding=True,
        )
        all_fold_predictions.append(outputs)

    # Ensemble: average predictions across folds
    results = []
    for text_idx in range(len(texts)):
        fold_scores = []
        for fold_outputs in all_fold_predictions:
            # For single-label models, score field contains the positive probability
            output = fold_outputs[text_idx]
            pos_score = float(output.get("score", 0.0))
            fold_scores.append(pos_score)

        # Average across folds
        ensemble_score = sum(fold_scores) / len(fold_scores)

        results.append({
            "score": ensemble_score,
            "all": [{"label": "LABEL_0", "score": ensemble_score}]
        })

    return results
