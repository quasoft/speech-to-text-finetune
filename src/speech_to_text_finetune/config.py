import yaml
from pydantic import BaseModel


def load_config(config_path: str):
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return Config(**config_dict)


class TrainingConfig(BaseModel):
    """
    More info at https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    """

    push_to_hub: bool
    hub_private_repo: bool
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    gradient_checkpointing: bool
    fp16: bool
    eval_strategy: str
    per_device_eval_batch_size: int
    predict_with_generate: bool
    generation_max_length: int
    save_steps: int
    logging_steps: int
    load_best_model_at_end: bool
    save_total_limit: int
    metric_for_best_model: str
    greater_is_better: bool


class Config(BaseModel):
    """
    Store configuration used for finetuning

    Attributes:
        model_id: HF model id of a Whisper model used for finetuning
        dataset_id: HF dataset id of a Common Voice dataset version, ideally from the mozilla-foundation repo
        language: registered language string that is supported by the Common Voice dataset
        repo_name: used both for local dir and HF, "default" will create a name based on the model and language id
        n_train_samples: explicitly set how many samples to train+validate on. If -1, use all train+val data available
        n_test_samples: explicitly set how many samples to evaluate on. If -1, use all eval data available
        training_hp: store selective hyperparameter values from Seq2SeqTrainingArguments
    """

    model_id: str
    dataset_id: str
    language: str  # For translation this is the TARGET language (what the model should output)
    repo_name: str
    n_train_samples: int
    n_test_samples: int
    task: str = "transcribe"  # "transcribe" or "translate"
    # Optional encoder freezing strategy (useful for low-resource speech translation)
    freeze_encoder: bool = False  # if True, freeze entire encoder at start
    freeze_encoder_until_step: int = 0  # unfreeze after this global step (ignored if 0 or freeze_encoder False)
    freeze_encoder_keep_frozen: bool = False  # if True and freeze_encoder, never unfreeze (overrides until_step)
    # Control resume behavior
    resume: bool = True  # when False, start fresh: skip resume-from-checkpoint and allow overwriting output_dir
    training_hp: TrainingConfig


PROC_DATASET_DIR = "processed_version"
