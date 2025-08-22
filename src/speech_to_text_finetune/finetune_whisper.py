import json
import os
from functools import partial

from transformers import (
    Seq2SeqTrainer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    GenerationConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import snapshot_download
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
import torch
from typing import Dict, Tuple
import evaluate
from loguru import logger

from speech_to_text_finetune.config import load_config
from speech_to_text_finetune.data_process import (
    DataCollatorSpeechSeq2SeqWithPadding,
    load_dataset_from_dataset_id,
    try_find_processed_version,
    process_dataset,
    load_subset_of_dataset,
)
from speech_to_text_finetune.utils import (
    get_hf_username,
    create_model_card,
    compute_wer_cer_metrics,
    compute_bleu_chrf_metrics,
    lowercase_normalizer,
)


def run_finetuning(
    config_path: str = "config.yaml",
) -> Tuple[Dict, Dict]:
    """
    Complete pipeline for preprocessing the Common Voice dataset and then finetuning a Whisper model on it.

    Args:
        config_path (str): yaml filepath that follows the format defined in config.py

    Returns:
        Tuple[Dict, Dict]: evaluation metrics from the baseline and the finetuned models
    """
    cfg = load_config(config_path)

    language_id = TO_LANGUAGE_CODE.get(cfg.language.lower())
    if not language_id:
        raise ValueError(
            f"\nThis language is not inherently supported by this Whisper model. However you can still “teach” Whisper "
            f"the language of your choice!\nVisit https://glottolog.org/, find which language is most closely "
            f"related to {cfg.language} from the list of supported languages below, and update your config file with "
            f"that language.\n{json.dumps(TO_LANGUAGE_CODE, indent=4, sort_keys=True)}."
        )

    if cfg.repo_name == "default":
        cfg.repo_name = f"{cfg.model_id.split('/')[1]}-{language_id}"
    local_output_dir = f"./artifacts/{cfg.repo_name}"

    logger.info(f"Finetuning starts soon, results saved locally at {local_output_dir}")
    hf_repo_name = ""
    if cfg.training_hp.push_to_hub:
        hf_username = get_hf_username()
        hf_repo_name = f"{hf_username}/{cfg.repo_name}"
        logger.info(
            f"Results will also be uploaded in HF at {hf_repo_name}. "
            f"Private repo is set to {cfg.training_hp.hub_private_repo}."
        )

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(
        f"Loading {cfg.model_id} on {device} and configuring it for task={cfg.task} target_language={cfg.language}."
    )
    processor = WhisperProcessor.from_pretrained(
        cfg.model_id, language=cfg.language, task=cfg.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id)
    # For non-English translation targets we optionally disable forced decoder ids (English bias)
    if cfg.task == "translate":
        # Let generation decide without English bias; user can re-enable by editing config
        model.config.forced_decoder_ids = None
    else:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=cfg.language, task=cfg.task
        )
    # disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False
    # Create and attach a default GenerationConfig so downstream scripts can load it directly
    # Values mirror our eval-time generation behavior and spot-test expectations
    gen_max_new_tokens = (
        getattr(cfg.training_hp, "generation_max_length", None) or 225
    )
    generation_config = GenerationConfig(
        num_beams=5,
        no_repeat_ngram_size=3,
        length_penalty=1.1,
        do_sample=False,
        max_new_tokens=gen_max_new_tokens,
    )
    # Keep forced ids consistent with model.config logic above
    if cfg.task == "translate":
        generation_config.forced_decoder_ids = None
    else:
        generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=cfg.language, task=cfg.task
        )
    # Provide Whisper language mapping so GenerationConfig.from_pretrained has it available
    # (spot-test.py relies on this)
    try:
        generation_config.lang_to_id = processor.tokenizer.lang_code_to_id  # type: ignore[attr-defined]
    except Exception:
        pass
    model.generation_config = generation_config
    # convenience partial for generation during eval/prediction
    model.generate = partial(
        model.generate,
        language=cfg.language.lower(),
        task=cfg.task,
        use_cache=True,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=local_output_dir,
        hub_model_id=hf_repo_name,
        report_to=["tensorboard"],
        **cfg.training_hp.model_dump(),
    )

    # If running in a fresh ephemeral environment (e.g., new Colab session) and pushing to hub, try to
    # reconstruct the local checkpoint directory from the remote repo so we can resume seamlessly.
    if (
        cfg.resume
        and cfg.training_hp.push_to_hub
        and hf_repo_name
        and not os.path.isdir(local_output_dir)
    ):
        try:
            logger.info(
                f"Local output dir '{local_output_dir}' not found. Attempting to download existing checkpoints from {hf_repo_name}."
            )
            snapshot_download(
                repo_id=hf_repo_name,
                local_dir=local_output_dir,
                local_dir_use_symlinks=False,
            )
            logger.info("Download complete. Local directory ready for resume.")
        except Exception as e:
            logger.warning(
                f"Could not download existing repo snapshot for resume (will start fresh if no local checkpoints): {e}"
            )

    if proc_dataset := try_find_processed_version(
        dataset_id=cfg.dataset_id, language_id=language_id
    ):
        logger.info(
            f"Loading processed dataset version of {cfg.dataset_id} and skipping processing."
        )
        dataset = proc_dataset
        dataset["train"] = load_subset_of_dataset(dataset["train"], cfg.n_train_samples)
        dataset["test"] = load_subset_of_dataset(dataset["test"], cfg.n_test_samples)
    else:
        logger.info(f"Loading {cfg.dataset_id}. Language selected {cfg.language}")
        dataset, save_proc_dataset_dir = load_dataset_from_dataset_id(
            dataset_id=cfg.dataset_id,
            language_id=language_id,
        )
        dataset["train"] = load_subset_of_dataset(dataset["train"], cfg.n_train_samples)
        dataset["test"] = load_subset_of_dataset(dataset["test"], cfg.n_test_samples)
        logger.info("Processing dataset...")
        dataset = process_dataset(
            dataset=dataset,
            processor=processor,
            batch_size=cfg.training_hp.per_device_train_batch_size,
            proc_dataset_path=save_proc_dataset_dir,
        )
        logger.info(
            f"Processed dataset saved at {save_proc_dataset_dir}. Future runs of {cfg.dataset_id} will "
            f"automatically use this processed version."
        )

    if cfg.task == "translate":
        bleu = evaluate.load("bleu")
        chrf = evaluate.load("chrf")
        compute_metrics_fn = partial(
            compute_bleu_chrf_metrics,
            processor=processor,
            bleu=bleu,
            chrf=chrf,
            normalizer=lowercase_normalizer if getattr(cfg, "eval_lowercase", False) else None,
        )
    else:
        wer = evaluate.load("wer")
        cer = evaluate.load("cer")
        compute_metrics_fn = partial(
            compute_wer_cer_metrics,
            processor=processor,
            wer=wer,
            cer=cer,
            normalizer=BasicTextNormalizer(),
        )

    # Optional: freeze encoder early to adapt decoder first
    if cfg.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen at start of training.")

    class EncoderUnfreezeCallback(TrainerCallback):
        def __init__(self, unfreeze_step: int):
            self.unfreeze_step = unfreeze_step
            self.unfroze = False

        def on_step_begin(self, args, state, control, **kwargs):  # type: ignore
            if not self.unfroze and state.global_step >= self.unfreeze_step:
                for p in model.model.encoder.parameters():
                    p.requires_grad = True
                self.unfroze = True
                logger.info(
                    f"[EncoderUnfreezeCallback] Encoder unfrozen automatically at step {state.global_step}."
                )
            return control

    callbacks = []
    if cfg.freeze_encoder and not cfg.freeze_encoder_keep_frozen and cfg.freeze_encoder_until_step > 0:
        callbacks.append(EncoderUnfreezeCallback(cfg.freeze_encoder_until_step))

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        processing_class=processor.feature_extractor,
        callbacks=callbacks or None,
    )

    processor.save_pretrained(training_args.output_dir)
    # Persist generation config alongside the model so clients can load with GenerationConfig.from_pretrained
    try:
        generation_config.save_pretrained(training_args.output_dir)
    except Exception as e:
        logger.warning(f"Failed to save generation_config: {e}")

    # Resume / baseline logic
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if cfg.resume else None
    baseline_metrics_path = f"{training_args.output_dir}/baseline_eval_results.json"
    if last_checkpoint:
        logger.info(f"Found existing checkpoint at {last_checkpoint}. Will resume training from this point.")
        # Try load previously stored baseline metrics if available
        if os.path.isfile(baseline_metrics_path):
            try:
                with open(baseline_metrics_path, "r", encoding="utf-8") as f:
                    baseline_eval_results = json.load(f)
                logger.info("Loaded stored baseline evaluation metrics.")
            except Exception as e:
                logger.warning(f"Could not load stored baseline metrics: {e}. Using empty placeholder.")
                baseline_eval_results = {}
        else:
            logger.warning("No stored baseline metrics file found; model card will omit baseline stats.")
            baseline_eval_results = {}
    else:
        logger.info(
            f"Before finetuning, run evaluation on the baseline model {cfg.model_id} to easily compare performance"
            f" before and after finetuning"
        )
        baseline_eval_results = trainer.evaluate()
        logger.info(f"Baseline evaluation complete. Results:\n\t {baseline_eval_results}")
        # Persist baseline metrics for future resumed runs
        try:
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(baseline_metrics_path, "w", encoding="utf-8") as f:
                json.dump(baseline_eval_results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write baseline metrics file: {e}")

    logger.info(
        f"Start finetuning job on {dataset['train'].num_rows} audio samples. Monitor training metrics in real time in "
        f"a local tensorboard server by running in a new terminal: tensorboard --logdir {training_args.output_dir}/runs"
    )
    try:
        if cfg.resume and last_checkpoint:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()
    except KeyboardInterrupt:
        logger.info("Stopping the finetuning job prematurely...")
    else:
        logger.info("Finetuning job complete.")

    logger.info(f"Start evaluation on {dataset['test'].num_rows} audio samples.")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation complete. Results:\n\t {eval_results}")
    model_card = create_model_card(
        model_id=cfg.model_id,
        dataset_id=cfg.dataset_id,
        language_id=language_id,
        language=cfg.language,
        n_train_samples=dataset["train"].num_rows,
        n_eval_samples=dataset["test"].num_rows,
        baseline_eval_results=baseline_eval_results,
        ft_eval_results=eval_results,
        task=cfg.task,
    )
    model_card.save(f"{local_output_dir}/README.md")

    if cfg.training_hp.push_to_hub:
        logger.info(f"Uploading model and eval results to HuggingFace: {hf_repo_name}")
        try:
            trainer.push_to_hub()
        except Exception as e:
            logger.info(f"Did not manage to upload final model. See: \n{e}")
        model_card.push_to_hub(hf_repo_name)

    logger.info(f"Find your final, best performing model at {local_output_dir}")
    return baseline_eval_results, eval_results


if __name__ == "__main__":
    run_finetuning(config_path="example_data/config.yaml")
