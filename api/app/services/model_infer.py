# api/app/services/model_infer.py
from transformers import pipeline, AutoTokenizer
from ..config import settings
from typing import Optional

_CACHE: dict[str, Optional[list]] = {"pipes": None}

def get_pipes():
    """Load all cross-validation fold models for ensemble inference."""
    if _CACHE["pipes"] is None:
        pipes = []
        for fold_idx in range(1, settings.HF_NUM_FOLDS + 1):
            model_path = f"{settings.HF_MODEL_BASE_DIR}/{settings.HF_MODEL_PREFIX}_fold-{fold_idx}"
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
        _CACHE["pipes"] = pipes
    return _CACHE["pipes"]

def _pick_pos_score(all_scores: list[dict]) -> float:
    """
    Given Hugging Face return_all_scores output (list of {label, score}),
    return the POSITIVE-class probability.
    Tries common label names, then falls back to a best guess.
    """
    for key in ("POS", "Positive", "positive", "LABEL_1", "1"):
        for s in all_scores:
            if s.get("label") == key:
                return float(s.get("score", 0.0))

    candidates = [s for s in all_scores if "1" in str(s.get("label", ""))]
    if candidates:
        return float(candidates[0].get("score", 0.0))

    if len(all_scores) >= 2:
        return float(all_scores[1].get("score", 0.0))

    return float(all_scores[0].get("score", 0.0)) if all_scores else 0.0

def predict_batch(titles: list[str], abstracts: list[str]) -> list[dict]:
    """
    Ensemble inference across all cross-validation folds.
    Returns a dict per text with:
      - score: ensemble probability of the POSITIVE class (0..1)
      - all:   full list of {label, score} for debugging/inspection
    """
    pipes = get_pipes()

    # Combine titles and abstracts
    texts = [f"{title}. {abstract}".strip() for title, abstract in zip(titles, abstracts)]

    # Collect predictions from all folds
    all_fold_predictions = []
    for pipe in pipes:
        outputs = pipe(
            texts,
            return_all_scores=True,
            function_to_apply="softmax",
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
            pos_score = _pick_pos_score(fold_outputs[text_idx])
            fold_scores.append(pos_score)

        # Average across folds
        ensemble_score = sum(fold_scores) / len(fold_scores)

        results.append({
            "score": ensemble_score,
            "all": all_fold_predictions[0][text_idx]
        })

    return results