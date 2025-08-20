from typing import Dict

from evaluate import EvaluationModule
from huggingface_hub import (
    ModelCard,
    HfApi,
    ModelCardData,
    EvalResult,
)
from transformers import EvalPrediction, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def compute_bleu_metrics(
    pred: EvalPrediction,
    processor: WhisperProcessor,
    bleu: EvaluationModule,
    sacrebleu: EvaluationModule,
    normalizer: BasicTextNormalizer,
) -> Dict:
    """
    Compute BLEU and SacreBLEU metrics for a speech translation task.

    BLEU / SacreBLEU are standard MT metrics (higher is better). We additionally
    report a "normalized" variant where Whisper's BasicTextNormalizer is applied
    (lower‑casing, punctuation handling, etc.) prior to scoring. Normalization can
    reduce surface-form variance and give a complementary view of quality.

    Args:
        pred: EvalPrediction with model prediction token ids and label token ids.
        processor: WhisperProcessor used to decode token ids to text.
        bleu: evaluate.load("bleu") metric module (returns keys: bleu, precisions, ...).
        sacrebleu: evaluate.load("sacrebleu") metric module (returns keys: score, ...).
        normalizer: BasicTextNormalizer for optional normalized scoring.

    Returns:
        Dict containing raw and normalized BLEU / SacreBLEU scores:
            bleu, sacrebleu, bleu_norm, sacrebleu_norm
    """

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    # evaluate expects references as list[list[str]] for MT metrics
    references = [[ref] for ref in label_str]

    bleu_res = bleu.compute(predictions=pred_str, references=references)
    sacrebleu_res = sacrebleu.compute(predictions=pred_str, references=references)

    # Normalized versions
    pred_str_norm = [normalizer(p) for p in pred_str]
    label_str_norm = [normalizer(l) for l in label_str]
    # Keep only non-empty normalized references
    filtered_preds = []
    filtered_refs = []
    for p, r in zip(pred_str_norm, label_str_norm):
        if r.strip():
            filtered_preds.append(p)
            filtered_refs.append([r])  # still list[list[str]]

    if len(filtered_preds) == 0:
        # Avoid division by zero inside metrics; fall back to raw
        bleu_norm_res = {"bleu": bleu_res.get("bleu", 0.0)}
        sacrebleu_norm_res = {"score": sacrebleu_res.get("score", 0.0)}
    else:
        bleu_norm_res = bleu.compute(predictions=filtered_preds, references=filtered_refs)
        sacrebleu_norm_res = sacrebleu.compute(predictions=filtered_preds, references=filtered_refs)

    return {
        "bleu": bleu_res.get("bleu"),
        "sacrebleu": sacrebleu_res.get("score"),
        "bleu_precisions": bleu_res.get("precisions"),  # keep extra diagnostic data
        "bleu_norm": bleu_norm_res.get("bleu"),
        "sacrebleu_norm": sacrebleu_norm_res.get("score"),
    }


def get_hf_username() -> str:
    return HfApi().whoami()["name"]


def create_model_card(
    model_id: str,
    dataset_id: str,
    language_id: str,
    language: str,
    n_train_samples: int,
    n_eval_samples: int,
    baseline_eval_results: Dict,
    ft_eval_results: Dict,
) -> ModelCard:
    """Create a Model Card for a speech translation finetuned model using BLEU / SacreBLEU metrics."""

    # Prefer SacreBLEU as primary leaderboard metric
    sacrebleu_ft = round(ft_eval_results.get("eval_sacrebleu", 0.0), 3)
    card_metadata = ModelCardData(
        model_name=f"Finetuned {model_id} on {language}",
        base_model=model_id,
        datasets=[dataset_id.split("/")[-1]],
        language=language_id,
        license="apache-2.0",
        library_name="transformers",
        eval_results=[
            EvalResult(
                task_type="translation",
                task_name="Speech Translation",
                dataset_type="common_voice",
                dataset_name=f"Common Voice ({language})",
                metric_type="sacrebleu",
                metric_value=sacrebleu_ft,
            )
        ],
    )

    def fmt(res: Dict, key: str) -> str:
        return f"{res.get(key):.3f}" if key in res else "n/a"

    content = f"""
---
{card_metadata.to_yaml()}
---

# Finetuned {model_id} on {n_train_samples} {language} training audio samples from {dataset_id}.

This model was created from the Mozilla.ai Blueprint:
[speech-to-text-finetune](https://github.com/mozilla-ai/speech-to-text-finetune).

## Evaluation results on {n_eval_samples} audio samples of {language}:

### Baseline model (before finetuning) on {language}
- SacreBLEU: {fmt(baseline_eval_results, 'eval_sacrebleu')}
- SacreBLEU (Normalized): {fmt(baseline_eval_results, 'eval_sacrebleu_norm')}
- BLEU: {fmt(baseline_eval_results, 'eval_bleu')}
- BLEU (Normalized): {fmt(baseline_eval_results, 'eval_bleu_norm')}
- Loss: {fmt(baseline_eval_results, 'eval_loss')}

### Finetuned model (after finetuning) on {language}
- SacreBLEU: {fmt(ft_eval_results, 'eval_sacrebleu')}
- SacreBLEU (Normalized): {fmt(ft_eval_results, 'eval_sacrebleu_norm')}
- BLEU: {fmt(ft_eval_results, 'eval_bleu')}
- BLEU (Normalized): {fmt(ft_eval_results, 'eval_bleu_norm')}
- Loss: {fmt(ft_eval_results, 'eval_loss')}
"""

    return ModelCard(content)


def update_hf_model_card_with_fleurs_results(
    model_repo_id: str,
    language: str,
    ft_eval_results: Dict,
) -> None:
    """Append FLEURS evaluation (speech translation) BLEU / SacreBLEU scores to the Model Card."""
    model_card = ModelCard.load(model_repo_id)
    sacrebleu_val = ft_eval_results.get("eval_sacrebleu")
    sacrebleu_norm_val = ft_eval_results.get("eval_sacrebleu_norm")
    bleu_val = ft_eval_results.get("eval_bleu")
    bleu_norm_val = ft_eval_results.get("eval_bleu_norm")
    loss_val = ft_eval_results.get("eval_loss")
    n_samples = ft_eval_results.get("n_eval_samples", "n/a")
    model_card.content += f"""
### Finetuned model (after finetuning) on the {language} FLEURS test set (total of {n_samples} samples)
- SacreBLEU: {sacrebleu_val:.3f if sacrebleu_val is not None else 'n/a'}
- SacreBLEU (Normalized): {sacrebleu_norm_val:.3f if sacrebleu_norm_val is not None else 'n/a'}
- BLEU: {bleu_val:.3f if bleu_val is not None else 'n/a'}
- BLEU (Normalized): {bleu_norm_val:.3f if bleu_norm_val is not None else 'n/a'}
- Loss: {loss_val:.3f if loss_val is not None else 'n/a'}
"""
    model_card.push_to_hub(model_repo_id)
