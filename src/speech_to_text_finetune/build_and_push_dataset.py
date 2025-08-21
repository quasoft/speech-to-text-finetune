"""Helper script to build a merged multi-domain audio/text dataset and push it to a private
Hugging Face Hub repository for Whisper finetuning.

Usage (from project root):

    python -m speech_to_text_finetune.build_and_push_dataset \
        --projects-json ./projects.json \
        --repo-id your-username/whisper-multidomain-rm \
        --language rm \
        --train-split 0.9 \
        --private

Expectations:
  - A JSON file (projects.json) with entries like:
        [
          {"title": "DomainOne", "language": "rm", "progress": 100, "task": "transcribe", "train": true},
          {"title": "DomainTwo", "language": "rm", "progress": 100, "task": "transcribe"}
        ]
  - Each entry's "title" matches a sibling directory next to projects.json, containing audio/text pairs:
        DomainOne/
            FILE_1.wav
            FILE_1.txt
            FILE_2.wav
            FILE_2.txt
        (Supports .wav or .mp3; transcript file must share the same stem and have .txt extension.)

Result:
  - Builds an in-memory DatasetDict with 'train' and 'test' splits.
  - Columns: 'audio' (absolute path), 'sentence' (transcript), 'domain' (folder name), plus optional 'language'.
  - Pushes DatasetDict to Hub (private if requested), retaining the 'domain' metadata until the repo's processing step.

Notes:
  - Only projects with "train": true are included. (If you want a default include behavior, pass --include-missing-train.)
  - Long audio (> 30s) will later be filtered by existing processing pipeline; we do not pre-filter here to allow
    you to inspect distribution.
  - Provide an HF token via environment HUGGINGFACE_HUB_TOKEN or CLI --token.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Iterable
from dataclasses import dataclass
import random
import sys

from loguru import logger
from datasets import Dataset, DatasetDict, Features, Value


@dataclass
class ProjectSpec:
    title: str
    language: str | None
    progress: int | None
    task: str | None
    train: bool

    @staticmethod
    def from_dict(d: Dict, include_missing_train: bool) -> "ProjectSpec | None":
        # If train key absent and include_missing_train flag is False -> skip.
        train_flag = d.get("train")
        if train_flag is None and not include_missing_train:
            return None
        return ProjectSpec(
            title=d.get("title"),
            language=d.get("language"),
            progress=d.get("progress"),
            task=d.get("task"),
            train=bool(train_flag) if train_flag is not None else True,
        )


def iter_audio_transcript_pairs(domain_dir: Path, exts: Iterable[str]) -> Iterable[Dict]:
    stem_to_audio = {}
    # Collect audio files
    for p in domain_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            stem_to_audio[p.stem] = p
    # For each audio, attempt transcript
    for stem, audio_path in stem_to_audio.items():
        txt_path = domain_dir / f"{stem}.txt"
        if not txt_path.is_file():
            logger.warning(f"Skipping {audio_path} (missing transcript {txt_path.name})")
            continue
        try:
            text = txt_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode transcript {txt_path}; skipping.")
            continue
        if not text:
            logger.warning(f"Empty transcript {txt_path}; skipping.")
            continue
        yield {
            "audio": str(audio_path.resolve()),
            "sentence": text,
        }


def build_datasetdict(
    projects_json: Path,
    train_split: float,
    seed: int,
    audio_exts: List[str],
    include_missing_train: bool,
) -> DatasetDict:
    root = projects_json.parent
    logger.info(f"Reading project specs from {projects_json}")
    specs_raw = json.loads(projects_json.read_text(encoding="utf-8"))
    specs: List[ProjectSpec] = []
    for entry in specs_raw:
        spec = ProjectSpec.from_dict(entry, include_missing_train)
        if spec and spec.train:
            if not spec.title:
                logger.warning(f"Skipping entry without title: {entry}")
                continue
            specs.append(spec)

    if not specs:
        raise ValueError("No projects selected for training. Check 'train': true flags or use --include-missing-train.")

    logger.info(f"Selected {len(specs)} project(s): {[s.title for s in specs]}")

    rows: List[Dict] = []
    total_audio = 0
    for spec in specs:
        domain_dir = root / spec.title
        if not domain_dir.is_dir():
            logger.warning(f"Domain directory missing: {domain_dir}; skipping.")
            continue
        domain_rows = list(iter_audio_transcript_pairs(domain_dir, exts=audio_exts))
        for r in domain_rows:
            r["domain"] = spec.title
            if spec.language:
                r["language"] = spec.language
        rows.extend(domain_rows)
        total_audio += len(domain_rows)
        logger.info(f"Collected {len(domain_rows)} samples from {spec.title}")

    if not rows:
        raise ValueError("No audio/text pairs found across selected domains.")

    random.seed(seed)
    random.shuffle(rows)

    k = int(len(rows) * train_split)
    train_rows = rows[:k]
    test_rows = rows[k:] if k < len(rows) else []
    logger.info(f"Dataset size: total={len(rows)} train={len(train_rows)} test={len(test_rows)}")

    # Define features explicitly to preserve types (audio paths stored as string, will be loaded as paths later)
    base_features = {
        "audio": Value("string"),
        "sentence": Value("string"),
        "domain": Value("string"),
    }
    if any("language" in r for r in rows):
        base_features["language"] = Value("string")

    # We keep audio as path (string). Processing pipeline will cast to Audio later.
    train_ds = Dataset.from_list(train_rows, features=Features(base_features))
    test_ds = Dataset.from_list(test_rows, features=Features(base_features)) if test_rows else Dataset.from_list([], features=Features(base_features))

    return DatasetDict({"train": train_ds, "test": test_ds})


def push_dataset(
    ds: DatasetDict,
    repo_id: str,
    private: bool,
    token: str | None,
    commit_message: str,
    create_pr: bool,
) -> None:
    logger.info(f"Pushing dataset to Hub: {repo_id} (private={private})")
    ds.push_to_hub(
        repo_id=repo_id,
        private=private,
        token=token,
        commit_message=commit_message,
        create_pr=create_pr,
    )
    logger.info("Push complete.")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and push merged multi-domain audio dataset.")
    p.add_argument("--projects-json", required=True, type=Path, help="Path to projects.json descriptor")
    p.add_argument("--repo-id", required=True, help="Target Hub dataset repo id (e.g. user/my-dataset)")
    p.add_argument("--language", required=False, help="Primary language code (added if missing in entries)")
    p.add_argument("--train-split", type=float, default=0.9, help="Fraction of data for train split")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    p.add_argument(
        "--audio-exts",
        nargs="*",
        default=[".wav", ".mp3"],
        help="Audio file extensions to include",
    )
    p.add_argument("--private", action="store_true", help="Create / push as private dataset")
    p.add_argument("--token", help="HF token (fallback to env HUGGINGFACE_HUB_TOKEN)")
    p.add_argument("--commit-message", default="Add/Update merged dataset")
    p.add_argument("--create-pr", action="store_true", help="Create a PR instead of direct push (if repo exists)")
    p.add_argument("--include-missing-train", action="store_true", help="Include entries without 'train' flag (default skip)")
    p.add_argument("--dry-run", action="store_true", help="Build locally but skip push")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    if args.train_split <= 0 or args.train_split >= 1:
        raise ValueError("--train-split must be in (0,1)")

    ds = build_datasetdict(
        projects_json=args.projects_json,
        train_split=args.train_split,
        seed=args.seed,
        audio_exts=[ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.audio_exts],
        include_missing_train=args.include_missing_train,
    )

    # If some rows lack a language and a global language is provided, fill.
    if args.language:
        def add_lang(example):
            if "language" not in example or not example["language"]:
                example["language"] = args.language
            return example
        ds = ds.map(add_lang)

    logger.info("Sample record from train split:")
    logger.info(ds["train"][0])

    if args.dry_run:
        logger.info("Dry run complete; skipping push.")
        return

    token = args.token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        logger.warning("No token provided; relying on cached auth (huggingface-cli login).")

    push_dataset(
        ds=ds,
        repo_id=args.repo_id,
        private=bool(args.private),
        token=token,
        commit_message=args.commit_message,
        create_pr=args.create_pr,
    )

if __name__ == "__main__":  # pragma: no cover
    main()
