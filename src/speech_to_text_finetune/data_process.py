import os
from pathlib import Path

from huggingface_hub.errors import HFValidationError

from speech_to_text_finetune.config import PROC_DATASET_DIR

import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple

from transformers import WhisperProcessor
from datasets import (
    load_dataset,
    DatasetDict,
    Audio,
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from loguru import logger


def try_find_processed_version(
    dataset_id: str, language_id: str | None = None
) -> DatasetDict | Dataset | None:
    """
    Try to load a processed version of the dataset if it exists locally. Check if:
    1. The dataset_id is a local path to an already processed dataset directory.
    or
    2. The dataset_id is a path to a local dataset, but a processed version already exists locally.
    or
    3. The dataset_id is a HuggingFace dataset ID, but a processed version already exists locally.
    """
    if Path(dataset_id).name == PROC_DATASET_DIR and Path(dataset_id).is_dir():
        if (
            Path(dataset_id + "/train").is_dir()
            and Path(dataset_id + "/test").is_dir()
            and Path(dataset_id + "/dataset_dict.json").is_file()
        ):
            return load_from_disk(dataset_id)
        else:
            raise FileNotFoundError("Processed dataset is incomplete.")

    proc_dataset_path = _get_local_proc_dataset_path(dataset_id)
    if Path(proc_dataset_path).is_dir():
        return load_from_disk(proc_dataset_path)

    hf_proc_dataset_path = _get_hf_proc_dataset_path(dataset_id, language_id)
    if Path(hf_proc_dataset_path).is_dir():
        logger.info(
            f"Found processed dataset version at {hf_proc_dataset_path} of HF dataset {dataset_id}. "
            f"Loading it directly and skipping processing again the original version."
        )
        return load_from_disk(hf_proc_dataset_path)

    return None


def _get_hf_proc_dataset_path(dataset_id: str, language_id: str) -> str:
    return (
        f"./artifacts/{language_id}_{dataset_id.replace('/', '_')}/{PROC_DATASET_DIR}"
    )


def _get_local_proc_dataset_path(dataset_id: str) -> str:
    return Path(dataset_id).resolve() / PROC_DATASET_DIR


def load_dataset_from_dataset_id(
    dataset_id: str,
    language_id: str | None = None,
) -> Tuple[DatasetDict, str]:
    """
    This function loads a dataset, based on the dataset_id and the content of its directory (if it is a local path).
    Possible cases:
    1. The dataset_id is a path to a local, Common Voice dataset directory.

    2. The dataset_id is a path to a local, custom dataset directory.

    3. The dataset_id is a HuggingFace dataset ID.

    Args:
        dataset_id: Path to a processed dataset directory or local dataset directory or HuggingFace dataset ID.
        language_id (Only used for the HF dataset case): Language identifier for the dataset (e.g., 'en' for English)

    Returns:
        DatasetDict: A processed dataset ready for training with train/test splits
        str: Path to save the processed directory

    Raises:
        ValueError: If the dataset cannot be found locally or on HuggingFace
    """
    try:
        dataset = _load_local_common_voice(dataset_id)
        return dataset, _get_local_proc_dataset_path(dataset_id)
    except FileNotFoundError:
        pass

    try:
        dataset = _load_custom_dataset(dataset_id)
        return dataset, _get_local_proc_dataset_path(dataset_id)
    except FileNotFoundError:
        pass

    try:
        dataset = _load_hf_common_voice(dataset_id, language_id)
        return dataset, _get_hf_proc_dataset_path(dataset_id, language_id)
    except HFValidationError:
        pass
    except FileNotFoundError:
        pass
    except ValueError as e:
        # This typically occurs when a language-specific BuilderConfig (e.g. 'bg') is not
        # present for the dataset. In that case we fall back to the generic HF loader which
        # expects ready-made 'train'/'test' splits. Re-raise any other unexpected ValueErrors.
        msg = str(e)
        if "BuilderConfig" in msg and "not found" in msg:
            logger.debug(
                "Falling back to generic HF dataset loader because language-specific BuilderConfig was not found: "
                f"{msg}"
            )
        else:
            raise

    # Generic HF dataset loader (expects "train" and "test" splits with columns including "audio" and "sentence")
    try:
        dataset = _load_generic_hf_audio_text_dataset(dataset_id)
        # Use provided language_id if any, else a generic tag to namespace cache directory
        return dataset, _get_hf_proc_dataset_path(dataset_id, language_id or "generic")
    except HFValidationError:
        pass
    except FileNotFoundError:
        pass

    raise ValueError(
        f"Could not find dataset {dataset_id}, neither locally nor at HuggingFace. "
        f"If its a private repo, make sure you are logged in locally."
    )


def _load_hf_common_voice(dataset_id: str, language_id: str) -> DatasetDict:
    """
    Load the default train+validation split used for finetuning and a test split used for evaluation.
    Args:
        dataset_id: official Common Voice dataset id from the mozilla-foundation organisation from Hugging Face
        language_id: a registered language identifier from Common Voice (most often in ISO-639 format)

    Returns:
        DatasetDict: HF Dataset dictionary that consists of two distinct Datasets
    """
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset(
        dataset_id,
        language_id,
        split="train+validation",
        trust_remote_code=True,
    )
    common_voice["test"] = load_dataset(
        dataset_id,
        language_id,
        split="test",
        trust_remote_code=True,
    )
    common_voice = common_voice.select_columns(["audio", "sentence"])

    return common_voice


def _load_generic_hf_audio_text_dataset(dataset_id: str) -> DatasetDict:
    """Load a generic Hugging Face dataset repo that already contains (at least) 'train' and 'test' splits
    with 'audio' and 'sentence' columns.

    This allows users to prepare & push their own multi-domain dataset (e.g. merged domains with a 'domain' column)
    and still reuse the existing processing / finetuning pipeline without mimicking the Common Voice structure.

    Expectations:
        - load_dataset(dataset_id) returns a DatasetDict with 'train' and 'test'.
        - Each split has at minimum columns: 'audio' (either a path string or an Audio feature) and 'sentence'.
        - Any extra metadata columns (e.g. 'domain') are preserved until processing (they will be removed later when
          process_dataset selects required columns, unless user customizes it).

    Raises:
        FileNotFoundError: if structure / columns are not present (so calling code can try other loaders)
        HFValidationError: if dataset_id is invalid on the Hub.
    """
    ds = load_dataset(dataset_id, trust_remote_code=True)

    # Ensure required splits
    if not isinstance(ds, DatasetDict) or not {"train", "test"}.issubset(ds.keys()):
        raise FileNotFoundError(
            "Generic HF dataset loader expects 'train' and 'test' splits."
        )

    # Check required columns
    required_cols = {"audio", "sentence"}
    for split in ["train", "test"]:
        cols = set(ds[split].column_names)
        if not required_cols.issubset(cols):
            raise FileNotFoundError(
                f"Split '{split}' missing required columns {required_cols - cols}."
            )

    # Optionally narrow columns to at least what downstream expects; keep others for potential later analysis.
    # Do not call select_columns here to avoid dropping user metadata prematurely; processing later will prune.
    return ds


def upsample_films_and_interviews(
    ds: Dataset,
    domain_col: str = "domain",
    factor: int = 3,
    seed: int = 42,
) -> Dataset:
    """Upsample only the "Films" and "Interviews" domains by a fixed factor.

    This duplicates examples from the target domains while keeping all other
    domains at their original count. The resulting dataset is shuffled.

    Args:
        ds: Input Hugging Face Dataset with a domain metadata column.
        domain_col: Column name that contains the domain label (default: "domain").
        factor: Replication factor for target domains (factor=1 returns the input unchanged).
        seed: RNG seed used for the final shuffle.

    Returns:
        A new Dataset with "Films" and "Interviews" samples upsampled.
    """
    if factor <= 1:
        return ds
    if domain_col not in ds.column_names:
        logger.warning(
            f"upsample_films_and_interviews: column '{domain_col}' not found; returning dataset unchanged."
        )
        return ds

    target_domains = {"Films", "Interviews"}

    try:
        # Partition dataset once to avoid accidental double counting
        others = ds.filter(lambda x: x[domain_col] not in target_domains)
        parts = [others]
        present_any = False

        for dom in sorted(target_domains):
            sub = ds.filter(lambda x, d=dom: x[domain_col] == d)
            if sub.num_rows == 0:
                continue
            # include original once plus (factor-1) extra copies
            parts.append(sub)
            for _ in range(factor - 1):
                parts.append(sub)
            present_any = True

        if not present_any:
            logger.info(
                "upsample_films_and_interviews: no 'Films'/'Interviews' rows found; returning dataset unchanged."
            )
            return ds

        upsampled = concatenate_datasets(parts).shuffle(seed=seed)
        return upsampled
    except Exception as e:
        logger.warning(f"upsample_films_and_interviews failed ({e}); returning dataset unchanged.")
        return ds


def _load_local_common_voice(cv_data_dir: str) -> DatasetDict:
    """
    Load a local Common Voice dataset (as downloaded from the official Common Voice website) into a DatasetDict.
    We only use the validated.tsv file to source the data to use for both training and testing.

    Args:
        cv_data_dir (str): path to the local Common Voice dataset directory

    Returns:
        DatasetDict: HF Dataset dictionary that consists of two distinct Datasets (train+validation and test)
    """
    cv_data_dir = Path(cv_data_dir)
    train_df = pd.read_csv(cv_data_dir / "train.tsv", sep="\t")
    test_df = pd.read_csv(cv_data_dir / "test.tsv", sep="\t")

    # Replace relative path with absolute
    train_df = train_df.rename(columns={"path": "audio"})
    train_df["audio"] = train_df["audio"].apply(
        lambda p: str(cv_data_dir / "clips" / p)
    )

    test_df = test_df.rename(columns={"path": "audio"})
    test_df["audio"] = test_df["audio"].apply(lambda p: str(cv_data_dir / "clips" / p))

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
        }
    )


def _get_audio_files_from_dir(dataset_dir: str) -> List[str]:
    return sorted(
        [
            f"{dataset_dir}/{f}"
            for f in os.listdir(f"{dataset_dir}")
            if f.endswith(".wav") or f.endswith(".mp3")
        ],
    )


def _load_custom_dataset(dataset_dir: str) -> DatasetDict:
    """
    Load sentences and accompanied recorded audio files into a pandas DataFrame, then split into train/test and finally
    load it into two distinct train Dataset and test Dataset.

    Sentences and audio files should be indexed like this: <index>: <sentence> should be accompanied by rec_<index>.wav

    Args:
        dataset_dir (str): path to the local dataset, expecting a text.csv and .wav files under the directory

    Returns:
        DatasetDict: HF Dataset dictionary that consists of two distinct Datasets (train+validation and test)
    """
    train_file = dataset_dir + "/train/text.csv"
    train_dir = dataset_dir + "/train/clips"
    test_file = dataset_dir + "/test/text.csv"
    test_dir = dataset_dir + "/test/clips"

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_df["audio"] = _get_audio_files_from_dir(train_dir)
    test_df["audio"] = _get_audio_files_from_dir(test_dir)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
        }
    )


def load_and_proc_hf_fleurs(
    language_id: str,
    n_test_samples: int,
    processor: WhisperProcessor,
    eval_batch_size: int,
) -> Dataset:
    """
    Load only the test split of fleurs on a specific language and process it for Whisper.
    Args:
        language_id (str): a registered language identifier from Fleurs
            (see https://huggingface.co/datasets/google/fleurs/blob/main/fleurs.py)
        n_test_samples (int): number of samples to use from the test split
        processor (WhisperProcessor): Processor from Whisper to process the dataset
        eval_batch_size (int): batch size to use for processing the dataset

    Returns:
        DatasetDict: HF Dataset
    """
    fleurs_dataset_id = "google/fleurs"
    if proc_dataset := try_find_processed_version(fleurs_dataset_id, language_id):
        return proc_dataset

    dataset = load_dataset(
        fleurs_dataset_id, language_id, trust_remote_code=True, split="test"
    )
    dataset = load_subset_of_dataset(dataset, n_test_samples)

    dataset = dataset.rename_column(
        original_column_name="raw_transcription", new_column_name="sentence"
    )
    dataset = dataset.select_columns(["audio", "sentence"])

    save_proc_dataset_path = _get_hf_proc_dataset_path(fleurs_dataset_id, language_id)
    logger.info("Processing dataset...")
    dataset = process_dataset(
        dataset=dataset,
        processor=processor,
        batch_size=eval_batch_size,
        proc_dataset_path=save_proc_dataset_path,
    )
    logger.info(
        f"Processed dataset saved at {save_proc_dataset_path}. Future runs of {fleurs_dataset_id} will "
        f"automatically use this processed version."
    )
    return dataset


def load_subset_of_dataset(dataset: Dataset, n_samples: int) -> Dataset:
    return dataset.select(range(n_samples)) if n_samples != -1 else dataset


def _is_audio_in_length_range(length: float, max_input_length: float = 30.0) -> bool:
    return 0 < length < max_input_length


def _are_labels_in_length_range(labels: List[int], max_label_length: int = 448) -> bool:
    return len(labels) < max_label_length


def process_dataset(
    dataset: DatasetDict | Dataset,
    processor: WhisperProcessor,
    batch_size: int,
    proc_dataset_path: str,
) -> DatasetDict | Dataset:
    """
    Process dataset to the expected format by a Whisper model and then save it locally for future use.
    """
    # Create a new column that consists of the resampled audio samples in the right sample rate for whisper
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate)
    )

    dataset = dataset.map(
        _process_inputs_and_labels_for_whisper,
        fn_kwargs={"processor": processor},
        remove_columns=dataset.column_names["train"]
        if "train" in dataset.column_names
        else None,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
    )

    dataset = dataset.filter(
        _is_audio_in_length_range,
        input_columns=["input_length"],
        fn_kwargs={"max_input_length": 30.0},
        num_proc=1,
    )
    dataset = dataset.filter(
        _are_labels_in_length_range,
        input_columns=["labels"],
        fn_kwargs={"max_label_length": 448},
        num_proc=1,
    )

    proc_dataset_path = Path(proc_dataset_path)
    Path.mkdir(proc_dataset_path, parents=True, exist_ok=True)
    dataset.save_to_disk(proc_dataset_path)
    return dataset


def _process_inputs_and_labels_for_whisper(
    batch: Dict, processor: WhisperProcessor
) -> Dict:
    """
    Use Whisper's feature extractor to transform the input audio arrays into log-Mel spectrograms
     and the tokenizer to transform the text-label into tokens. This function is expected to be called using
     the .map method in order to process the data batch by batch.
    """
    batched_audio = batch["audio"]

    batch = processor(
        audio=[audio["array"] for audio in batched_audio],
        sampling_rate=processor.feature_extractor.sampling_rate,
        text=batch["sentence"],
        return_attention_mask=True,  # ensure downstream model receives explicit attention mask
    )

    batch["input_length"] = [
        len(audio["array"]) / audio["sampling_rate"] for audio in batched_audio
    ]

    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data Collator class in the format expected by Seq2SeqTrainer used for processing
    input data and labels in batches while finetuning. More info here:
    """

    processor: WhisperProcessor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", return_attention_mask=True
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyway
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
