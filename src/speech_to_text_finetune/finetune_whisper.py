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
from speech_to_text_finetune.normalizer import BasicTextNormalizer as LocalBasicTextNormalizer
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
import torch
from typing import Dict, Tuple
import evaluate
from loguru import logger

from speech_to_text_finetune.config import load_config
from typing import Optional

# Optional PEFT for LoRA fine-tuning (no Unsloth dependency)
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        TaskType,
    )
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    PeftModel = None  # type: ignore
    TaskType = None  # type: ignore
from speech_to_text_finetune.data_process import (
    DataCollatorSpeechSeq2SeqWithPadding,
    load_dataset_from_dataset_id,
    try_find_processed_version,
    process_dataset,
    load_subset_of_dataset,
    upsample_films_and_interviews,
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
    # Optionally load base model with 8-bit/4-bit if requested (requires bitsandbytes)
    model_kwargs: Dict = {}
    if getattr(cfg, "load_in_8bit", False):
        model_kwargs["load_in_8bit"] = True
    if getattr(cfg, "load_in_4bit", False):
        model_kwargs["load_in_4bit"] = True
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id, **model_kwargs)
    # Do not rely on forced_decoder_ids; prefer prompt_ids in GenerationConfig
    model.config.forced_decoder_ids = None
    # disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False
    # Build GenerationConfig from pretrained to inherit model-specific defaults (suppress_tokens, etc.)
    gen_max_new_tokens = getattr(cfg.training_hp, "generation_max_length", None) or 225
    generation_config = GenerationConfig.from_pretrained(cfg.model_id)
    generation_config.num_beams = 5
    generation_config.no_repeat_ngram_size = 3
    generation_config.length_penalty = 1.1
    generation_config.do_sample = False
    generation_config.max_new_tokens = gen_max_new_tokens
    # Keep forced ids off; prompt_ids is the preferred mechanism
    generation_config.forced_decoder_ids = None

    # Decide decode behavior. decode_mode overrides task when provided.
    decode_mode = (cfg.decode_mode or cfg.task).lower()
    # Normalize decode_mode to allowed values
    if decode_mode not in {"transcribe", "translate", "neutral"}:
        logger.warning(f"Unknown decode_mode '{decode_mode}', defaulting to '{cfg.task}'.")
        decode_mode = cfg.task

    # Assign task/language and prompt_ids according to decode choice
    generation_config.language = cfg.language.lower()
    if decode_mode == "neutral":
        generation_config.task = None  # type: ignore[assignment]
        # No prompt to keep decoder language-neutral
        generation_config.prompt_ids = None  # type: ignore[attr-defined]
    else:
        generation_config.task = decode_mode
        # Stable API: returns (position, token_id) pairs
        dec_prompt = processor.get_decoder_prompt_ids(
            task=decode_mode, language=cfg.language, no_timestamps=True
        )
        prompt_ids = [tid for _, tid in dec_prompt] if dec_prompt is not None else None
        # For translate, Whisper biases toward English; choose decode_mode="transcribe" for non-English targets.
        generation_config.prompt_ids = prompt_ids  # type: ignore[attr-defined]

    # Provide Whisper language mapping so GenerationConfig consumers can access it
    try:
        generation_config.lang_to_id = processor.tokenizer.lang_code_to_id  # type: ignore[attr-defined]
    except Exception:
        pass
    model.generation_config = generation_config

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # If enabled, wrap model with LoRA adapters using PEFT
    if getattr(cfg, "use_lora", False):
        if LoraConfig is None or get_peft_model is None:
            raise ImportError(
                "PEFT is not installed. Install with `pip install peft` or disable use_lora in the config."
            )
        # Whisper is a seq2seq model; use SEQ_2_SEQ_LM task type
        lora_cfg = LoraConfig(
            r=int(cfg.lora_r),
            lora_alpha=int(cfg.lora_alpha),
            lora_dropout=float(cfg.lora_dropout),
            target_modules=list(getattr(cfg, "lora_target_modules", ["q_proj", "v_proj"])),
            bias=getattr(cfg, "lora_bias", "none"),
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_cfg)
        logger.info(
            f"Enabled LoRA: r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, dropout={lora_cfg.lora_dropout}, "
            f"targets={lora_cfg.target_modules}, bias={lora_cfg.bias}"
        )

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
        # Optional domain upsampling (only affects train, requires a 'domain' column)
        if getattr(cfg, "upsample_domains", False) and getattr(cfg, "upsample_domains_factor", 1) > 1:
            try:
                dataset["train"] = upsample_films_and_interviews(
                    dataset["train"], factor=int(cfg.upsample_domains_factor)
                )
                logger.info(
                    f"Applied domain upsampling (factor={cfg.upsample_domains_factor}) to train split."
                )
            except Exception as e:
                logger.warning(f"Could not upsample domains: {e}")
    else:
        logger.info(f"Loading {cfg.dataset_id}. Language selected {cfg.language}")
        dataset, save_proc_dataset_dir = load_dataset_from_dataset_id(
            dataset_id=cfg.dataset_id,
            language_id=language_id,
        )
        dataset["train"] = load_subset_of_dataset(dataset["train"], cfg.n_train_samples)
        dataset["test"] = load_subset_of_dataset(dataset["test"], cfg.n_test_samples)
        # Optional domain upsampling (only affects train, requires a 'domain' column)
        if getattr(cfg, "upsample_domains", False) and getattr(cfg, "upsample_domains_factor", 1) > 1:
            try:
                dataset["train"] = upsample_films_and_interviews(
                    dataset["train"], factor=int(cfg.upsample_domains_factor)
                )
                logger.info(
                    f"Applied domain upsampling (factor={cfg.upsample_domains_factor}) to train split."
                )
            except Exception as e:
                logger.warning(f"Could not upsample domains: {e}")
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

    if cfg.metric == "bleu":
        bleu = evaluate.load("bleu")
        chrf = evaluate.load("chrf")
        compute_metrics_fn = partial(
            compute_bleu_chrf_metrics,
            processor=processor,
            bleu=bleu,
            chrf=chrf,
            normalizer=(LocalBasicTextNormalizer(remove_diacritics=False).__call__)
            if getattr(cfg, "eval_lowercase", False)
            else None,
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
        metric=cfg.metric
    )
    model_card.save(f"{local_output_dir}/README.md")

    # Optionally merge LoRA adapters into the base model and save/push merged weights
    def _merge_and_save_if_needed(output_dir: str) -> Optional[str]:
        if not getattr(cfg, "use_lora", False) or not getattr(cfg, "merge_lora_on_save", False):
            return None
        if PeftModel is None or not isinstance(model, PeftModel):  # type: ignore[arg-type]
            logger.warning("merge_lora_on_save is set but model is not a PEFT model; skipping merge.")
            return None
        merged_dir = os.path.join(output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        base = model.merge_and_unload()  # returns a standard HF model with LoRA merged
        base.save_pretrained(merged_dir, safe_serialization=cfg.training_hp.save_safetensors)
        processor.save_pretrained(merged_dir)
        # Save generation config alongside merged model
        try:
            generation_config.save_pretrained(merged_dir)
        except Exception as e:
            logger.warning(f"Failed to save merged generation_config: {e}")
        logger.info(f"Merged model saved to {merged_dir}")
        # Optionally push to a separate repo
        if cfg.training_hp.push_to_hub and getattr(cfg, "push_merged_repo_id", None):
            from huggingface_hub import HfApi
            api = HfApi()
            try:
                api.create_repo(repo_id=cfg.push_merged_repo_id, private=cfg.training_hp.hub_private_repo, exist_ok=True)
            except Exception:
                pass
            try:
                api.upload_folder(repo_id=cfg.push_merged_repo_id, repo_type="model", folder_path=merged_dir)
                logger.info(f"Merged model pushed to {cfg.push_merged_repo_id}")
            except Exception as e:
                logger.warning(f"Failed to push merged model: {e}")
        return merged_dir

    merged_dir = _merge_and_save_if_needed(training_args.output_dir)

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
