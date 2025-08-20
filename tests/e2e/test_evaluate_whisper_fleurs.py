import shutil
from pathlib import Path
from speech_to_text_finetune.evaluate_whisper_fleurs import evaluate_fleurs


def test_evaluate_fleurs_e2e():
    results = evaluate_fleurs(
        model_id="openai/whisper-tiny",
        lang_code="af_za",
        language="Afrikaans",
        eval_batch_size=16,
        n_test_samples=10,
        fp16=False,
    )

    expected_dir_path = Path("artifacts/af_za_google_fleurs")
    assert expected_dir_path.exists()

    assert 0 < results["eval_loss"] < 20
    # BLEU metrics existence and bounds
    assert 0 <= results["eval_sacrebleu"] <= 100
    assert 0 <= results.get("eval_sacrebleu_norm", results["eval_sacrebleu"]) <= 100
    assert 0 <= results["eval_bleu"] <= 100
    assert 0 <= results.get("eval_bleu_norm", results["eval_bleu"]) <= 100

    shutil.rmtree("artifacts")
