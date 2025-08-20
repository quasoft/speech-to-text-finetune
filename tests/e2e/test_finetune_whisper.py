import shutil
from pathlib import Path

from speech_to_text_finetune.config import load_config
from speech_to_text_finetune.finetune_whisper import run_finetuning


def test_finetune_whisper_local(example_config_path):
    base_results, eval_results = run_finetuning(config_path=example_config_path)

    cfg = load_config(example_config_path)
    expected_dir_path = Path(f"artifacts/{cfg.repo_name}")
    assert expected_dir_path.exists()

    assert 0 < base_results["eval_loss"] < 10
    # BLEU / SacreBLEU scores are between 0 and 100 (higher is better)
    assert 0 <= base_results["eval_sacrebleu"] <= 100
    assert 0 <= base_results.get("eval_sacrebleu_norm", base_results["eval_sacrebleu"]) <= 100
    assert 0 <= base_results["eval_bleu"] <= 100
    assert 0 <= base_results.get("eval_bleu_norm", base_results["eval_bleu"]) <= 100
    assert 0 < eval_results["eval_loss"] < 10
    assert 0 <= eval_results["eval_sacrebleu"] <= 100
    assert 0 <= eval_results.get("eval_sacrebleu_norm", eval_results["eval_sacrebleu"]) <= 100
    assert 0 <= eval_results["eval_bleu"] <= 100
    assert 0 <= eval_results.get("eval_bleu_norm", eval_results["eval_bleu"]) <= 100

    shutil.rmtree(expected_dir_path)
