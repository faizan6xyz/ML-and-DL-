"""
=============================================================
  FINE-TUNING A LANGUAGE MODEL — REQUIREMENTS & FULL PROCESS
=============================================================

This file walks through every stage of fine-tuning a pre-trained
language model using Hugging Face Transformers + PEFT (LoRA).

Sections
--------
1.  Requirements (pip installs + hardware)
2.  Imports
3.  Configuration dataclass
4.  Dataset preparation & tokenisation
5.  Model loading (base + LoRA adapters)
6.  Training with SFTTrainer (TRL)
7.  Evaluation
8.  Saving & merging weights
9.  Inference with the fine-tuned model
10. Entry-point

Run:
    python fine_tuning_guide.py

Minimum hardware:
    - 16 GB GPU VRAM  (e.g. NVIDIA A10 / RTX 3090) for a 7 B-param model
    - 32 GB RAM
    - 50 GB free disk space

Install dependencies:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets peft trl accelerate bitsandbytes evaluate rouge_score
"""

# ─────────────────────────────────────────────
# 1. REQUIREMENTS  (comment block for reference)
# ─────────────────────────────────────────────
REQUIREMENTS = """
Hardware Requirements
---------------------
• GPU         : NVIDIA GPU with ≥ 16 GB VRAM (CUDA 11.8+)
                  - 7 B  model  → 16 GB  (with 4-bit QLoRA)
                  - 13 B model  → 24 GB  (with 4-bit QLoRA)
                  - 70 B model  → 80 GB+ (multi-GPU)
• RAM         : ≥ 32 GB system RAM
• Disk        : ≥ 50 GB free (model weights + checkpoints)
• Python      : 3.9 – 3.11
• CUDA        : 11.8 or 12.x

Software Packages
-----------------
torch              >= 2.1.0
transformers       >= 4.38.0
datasets           >= 2.17.0
peft               >= 0.9.0       # LoRA / QLoRA adapters
trl                >= 0.8.0       # SFTTrainer (supervised fine-tuning)
accelerate         >= 0.27.0      # multi-GPU / mixed-precision
bitsandbytes       >= 0.43.0      # 4-bit / 8-bit quantisation
evaluate           >= 0.4.0       # metrics
rouge_score        >= 0.1.2       # ROUGE evaluation

Data Requirements
-----------------
• Format         : JSONL  {"prompt": "...", "completion": "..."}
• Minimum size   : ~500 examples (1 000 – 10 000 recommended)
• Quality        : clean, consistent, representative of target task
• Train/val/test : 80 % / 10 % / 10 % split
"""

# ─────────────────────────────────────────────
# 2. IMPORTS
# ─────────────────────────────────────────────
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from trl import SFTTrainer
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 3. CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class FineTuneConfig:
    """Central config — adjust these before training."""

    # ── Model ─────────────────────────────────
    base_model_name: str = "mistralai/Mistral-7B-v0.1"
    # Other popular choices:
    #   "meta-llama/Llama-2-7b-hf"
    #   "google/gemma-7b"
    #   "microsoft/phi-2"

    # ── Data ──────────────────────────────────
    dataset_name: Optional[str] = "yahma/alpaca-cleaned"   # HF Hub id …
    local_data_path: Optional[str] = None                  # … or local JSONL
    prompt_column: str = "instruction"
    completion_column: str = "output"
    max_seq_length: int = 1024       # tokens per example
    train_split: float = 0.8
    val_split: float = 0.1

    # ── LoRA (PEFT) ────────────────────────────
    lora_r: int = 16                 # rank (higher = more params, more expressive)
    lora_alpha: int = 32             # scaling factor (typically 2 × lora_r)
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    # For GPT-2 style models use: ["c_attn"]
    # For Falcon use:             ["query_key_value"]

    # ── Quantisation (QLoRA) ───────────────────
    use_4bit: bool = True            # QLoRA — cuts VRAM in half
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # ── Training ──────────────────────────────
    output_dir: str = "./fine_tuned_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4   # effective batch = 4 × 4 = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    fp16: bool = True
    bf16: bool = False               # set True on Ampere GPUs (A100, RTX 30/40xx)
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    report_to: str = "none"          # "wandb" | "tensorboard" | "none"
    seed: int = 42


# ─────────────────────────────────────────────
# 4. DATASET PREPARATION
# ─────────────────────────────────────────────

def load_and_prepare_dataset(cfg: FineTuneConfig) -> DatasetDict:
    """
    Load data from Hugging Face Hub or a local JSONL file,
    apply a chat-style prompt template, and split into
    train / validation / test sets.
    """
    logger.info("Loading dataset …")

    if cfg.local_data_path:
        raw = load_dataset("json", data_files=cfg.local_data_path, split="train")
    else:
        raw = load_dataset(cfg.dataset_name, split="train")

    logger.info("Raw dataset size: %d examples", len(raw))

    # ── Build a unified 'text' column ──────────
    def format_example(example):
        prompt = example.get(cfg.prompt_column, "")
        completion = example.get(cfg.completion_column, "")
        # Alpaca-style template — adjust to your task
        text = (
            f"### Instruction:\n{prompt}\n\n"
            f"### Response:\n{completion}"
        )
        return {"text": text}

    dataset = raw.map(format_example, remove_columns=raw.column_names)

    # ── Train / val / test split ───────────────
    test_size = 1.0 - cfg.train_split - cfg.val_split
    split1 = dataset.train_test_split(test_size=(cfg.val_split + test_size), seed=cfg.seed)
    split2 = split1["test"].train_test_split(
        test_size=test_size / (cfg.val_split + test_size), seed=cfg.seed
    )

    splits = DatasetDict({
        "train":      split1["train"],
        "validation": split2["train"],
        "test":       split2["test"],
    })

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(splits["train"]), len(splits["validation"]), len(splits["test"]),
    )
    return splits


# ─────────────────────────────────────────────
# 5. MODEL LOADING
# ─────────────────────────────────────────────

def load_tokenizer(cfg: FineTuneConfig):
    logger.info("Loading tokeniser for %s …", cfg.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name,
        trust_remote_code=True,
    )
    # Most causal LMs need a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(cfg: FineTuneConfig):
    """Load the base model, optionally in 4-bit (QLoRA)."""
    logger.info("Loading base model %s …", cfg.base_model_name)

    bnb_config = None
    if cfg.use_4bit:
        compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,   # nested quantisation
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",           # automatically place layers across GPUs
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Disable KV-cache during training (incompatible with gradient checkpointing)
    model.config.use_cache = False
    model.config.pretraining_tp = 1  # for Llama models

    logger.info("Base model loaded. Parameters: %s", f"{model.num_parameters():,}")
    return model


def apply_lora(model, cfg: FineTuneConfig):
    """Wrap the base model with LoRA adapters via PEFT."""
    logger.info("Applying LoRA (r=%d, alpha=%d) …", cfg.lora_r, cfg.lora_alpha)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Example output:
    #   trainable params: 4,194,304 || all params: 3,756,122,112 || trainable%: 0.11
    return model


# ─────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────

def build_training_args(cfg: FineTuneConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        report_to=cfg.report_to,
        seed=cfg.seed,
        gradient_checkpointing=True,   # trade speed for lower VRAM
        optim="paged_adamw_32bit",     # memory-efficient optimiser (bitsandbytes)
        group_by_length=True,          # batch similar-length sequences together
    )


def train(cfg: FineTuneConfig, model, tokenizer, datasets: DatasetDict):
    logger.info("Starting fine-tuning …")

    training_args = build_training_args(cfg)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=False,              # set True to pack short sequences for speed
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ──────────────────────────────────
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # ── Save adapter weights ───────────────────
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("Adapter weights saved to %s", cfg.output_dir)

    return trainer


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(trainer, datasets: DatasetDict):
    """Run evaluation on the held-out test set and log metrics."""
    logger.info("Evaluating on test set …")

    metrics = trainer.evaluate(eval_dataset=datasets["test"])
    logger.info("Test metrics: %s", metrics)

    perplexity = torch.exp(torch.tensor(metrics["eval_loss"])).item()
    logger.info("Perplexity: %.4f", perplexity)

    return metrics


# ─────────────────────────────────────────────
# 8. SAVING & MERGING
# ─────────────────────────────────────────────

def merge_and_save(cfg: FineTuneConfig, base_model, tokenizer):
    """
    Merge the LoRA adapters back into the base model weights
    to produce a single stand-alone model (no PEFT dependency at inference).
    """
    merged_dir = os.path.join(cfg.output_dir, "merged")
    logger.info("Merging LoRA adapters into base model …")

    # Re-load base model in full precision for the merge
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    merged_model = PeftModel.from_pretrained(base, cfg.output_dir)
    merged_model = merged_model.merge_and_unload()

    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    logger.info("Merged model saved to %s", merged_dir)

    return merged_dir


# ─────────────────────────────────────────────
# 9. INFERENCE
# ─────────────────────────────────────────────

def run_inference(model_dir: str, prompt: str, max_new_tokens: int = 256) -> str:
    """
    Load the fine-tuned (or merged) model and generate a response.
    Call this after training, or anytime to test the saved model.
    """
    logger.info("Loading fine-tuned model from %s …", model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    formatted = (
        f"### Instruction:\n{prompt}\n\n"
        f"### Response:\n"
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ─────────────────────────────────────────────
# 10. ENTRY-POINT
# ─────────────────────────────────────────────

def main():
    # ── Step 0: Print requirements ─────────────
    print(REQUIREMENTS)

    # ── Step 1: Build config ───────────────────
    cfg = FineTuneConfig()
    logger.info("Config: %s", cfg)

    # ── Step 2: Prepare data ───────────────────
    datasets = load_and_prepare_dataset(cfg)

    # ── Step 3: Load tokeniser & model ─────────
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)

    # ── Step 4: Train ──────────────────────────
    trainer = train(cfg, model, tokenizer, datasets)

    # ── Step 5: Evaluate ───────────────────────
    evaluate_model(trainer, datasets)

    # ── Step 6: Merge & save full model ────────
    merged_dir = merge_and_save(cfg, model, tokenizer)

    # ── Step 7: Test inference ─────────────────
    test_prompt = "Explain the concept of gradient descent in simple terms."
    logger.info("Running test inference …")
    response = run_inference(merged_dir, test_prompt)
    print("\n" + "─" * 60)
    print(f"Prompt   : {test_prompt}")
    print(f"Response : {response}")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()