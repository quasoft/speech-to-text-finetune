"""
Romani to Bulgarian spot-check aligned with finetune/eval settings.

Key changes vs previous version:
- Do NOT force decoder prompt ids. Clear any saved forced_decoder_ids on both config and generation_config.
- Explicitly pass task and target language to model.generate (like the trainer does), so outputs match BLEU eval.
"""

import os
import random
import torch
from datasets import load_dataset, Audio
from huggingface_hub import login
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig

# Model/checkpoint and data
model_id = "quasoft2/whisper-large-v2-rm"   # or a local checkpoint dir
dataset_id = "quasoft2/voxrom"

# Use target language NAME (as in training config), not the code.
# Whisper internals will map names to codes.
target_language_name = "Bulgarian"
task = "translate"  # keep consistent with training
num_samples = 20

hf_token = os.environ.get("HF", "").strip() or None
if hf_token:
    login(token=hf_token, add_to_git_credential=False)

processor = WhisperProcessor.from_pretrained(model_id, token=hf_token)

# Explicitly load and update the generation config
generation_config = GenerationConfig.from_pretrained(model_id)
# Add the missing attribute based on the processor's language mappings
generation_config.lang_to_id = processor.tokenizer.lang_code_to_id
model = WhisperForConditionalGeneration.from_pretrained(model_id, generation_config=generation_config, token=hf_token).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure no forced English bias or stale prompt ids
if hasattr(model.config, "forced_decoder_ids"):
    model.config.forced_decoder_ids = None
model.generation_config.forced_decoder_ids = None


# Load dataset and enforce Whisper SR
ds = load_dataset(dataset_id, **({"token": hf_token} if hf_token else {}))
split = "test" if "test" in ds else ("validation" if "validation" in ds else "train")
test = ds[split].cast_column(
    "audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate)
)

@torch.inference_mode()
def generate_text(example):
    audio = example["audio"]
    inputs = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs["input_features"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Mirror trainer's eval-time generation kwargs
    gen_ids = model.generate(
        input_features,
        attention_mask=attention_mask,
        language=target_language_name.lower(),
        task=task,
        use_cache=True,
        num_beams=5,
        no_repeat_ngram_size=3,
        length_penalty=1.1,
        do_sample=False,
        max_new_tokens=225,
    )
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

# Deterministic sampling for easier repro
random.seed(0)
indices = random.sample(range(len(test)), k=min(num_samples, len(test)))
for i in indices:
    ref = test[i].get("sentence", "")
    pred = generate_text(test[i])
    print(f"\n# Sample {i}\nPRED: {pred}\nREF : {ref}")
