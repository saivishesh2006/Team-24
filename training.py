#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py

Training script using 4-bit + LoRA for sequence classification on DailyDialog.
If --use-prepared is provided, load tokenized datasets & tokenizer from prepared_dir
and skip re-tokenization/oversampling.
Supports weighted cross-entropy and focal loss via --loss-type.
"""
import os
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

from dataset import prepare_daily_dialog_from_splits

# ----------------------- CLI -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model-dir", type=str, default="./falcon-rw-1b/tiiuae/falcon-rw-1b")
parser.add_argument("--loss-type", type=str, choices=["weighted_ce", "focal"], default="weighted_ce",
                    help="Loss to use: weighted_ce or focal")
parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (only for focal)")
parser.add_argument("--output-dir", type=str, default="./outputs")
parser.add_argument("--num-epochs", type=int, default=4)
parser.add_argument("--per-device-train-batch-size", type=int, default=3)
parser.add_argument("--grad-accum-steps", type=int, default=8)
parser.add_argument("--seed-resample", type=int, default=42)
parser.add_argument("--cuda-device", type=str, default=None,
                    help="If provided, will set CUDA_VISIBLE_DEVICES to this value (e.g. '0' or '0,1').")
parser.add_argument("--fp16", action="store_true", help="Use fp16 training if available")
parser.add_argument("--use-prepared", action="store_true", help="Load tokenized datasets from prepared_dir and skip tokenization/oversampling")
parser.add_argument("--prepared-dir", type=str, default="./prepared_data_full", help="Directory with pre-tokenized datasets and tokenizer (used when --use-prepared)")
args = parser.parse_args()

# Optionally restrict visible CUDA devices (only when explicitly provided)
if args.cuda_device:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

# Reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ─── Helper: build prompts (same as before) ───────────────────────────────────
LABELS = ["neutral","angry","disgust","fear","happy","sad","surprised"]

def build_samples(raw_split):
    samples = []
    for dialog, emotions in zip(raw_split["dialog"], raw_split["emotion"]):
        for i, emo in enumerate(emotions):
            utt = dialog[i]
            spk = i % 2
            ctx = "\n".join(
                f'Speaker_{j%2}: "{dialog[j]}"'
                for j in range(max(0, i-2), i)
            )
            prompt = (
                "Instruction:\n"
                "You are an expert in analyzing emotions in multi-party dialogues.\n\n"
                f"Historical Content:\n{ctx.strip()}\n\n"
                "Question:\n"
                f'What is the emotional label of the utterance by Speaker_{spk}: "{utt}" '
                f'from the following options: {", ".join(LABELS)}?'
            )
            samples.append({"text": prompt, "label": int(emo)})
    return samples

# ─── 1) Load raw splits (we still load them to compute original distribution for class weights) ─
print(" Loading DailyDialog splits (for original distribution / fallback)")
train_raw, val_raw, test_raw = prepare_daily_dialog_from_splits()
train_samples_raw = build_samples(train_raw)
val_samples_raw   = build_samples(val_raw)
test_samples_raw  = build_samples(test_raw)

orig_counts = Counter(s["label"] for s in train_samples_raw)
total_orig = sum(orig_counts.values())
print(f"[info] original train distribution: {dict(orig_counts)}")

# ---------------- Compute class weights from original distribution ------------
num_classes = len(LABELS)
class_weights = []
for i in range(num_classes):
    cnt = orig_counts.get(i, 0)
    if cnt <= 0:
        w = 0.0
    else:
        w = total_orig / (num_classes * cnt)
    class_weights.append(w)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"[info] class weights (computed from original distribution): {class_weights}")

# ---------------- Prepare datasets/tokenizer ---------------------------------
if args.use_prepared:
    prepared_dir = os.path.realpath(args.prepared_dir)
    if not os.path.isdir(prepared_dir):
        raise FileNotFoundError(f"prepared_dir not found: {prepared_dir}")

    print(f" Loading pre-tokenized datasets from {prepared_dir}/train_tok, val_tok, test_tok")
    train_tok = load_from_disk(os.path.join(prepared_dir, "train_tok"))
    val_tok   = load_from_disk(os.path.join(prepared_dir, "val_tok"))
    test_tok  = load_from_disk(os.path.join(prepared_dir, "test_tok"))

    # load tokenizer saved during prepare_tokens
    tokenizer_dir = os.path.join(prepared_dir, "tokenizer")
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer folder not found in prepared_dir: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.pad_token = tokenizer.eos_token

    print(" Loaded prepared tokenized datasets and tokenizer. Skipping oversampling & tokenization.")
else:
    # Build, oversample, and tokenize as earlier
    print("  Building prompts from raw splits and performing oversampling + tokenization")
    train_samples = build_samples(train_raw)
    val_samples   = build_samples(val_raw)
    test_samples  = build_samples(test_raw)

    # Oversampling by duplication
    indices = np.arange(len(train_samples)).reshape(-1, 1)
    labels  = np.array([s["label"] for s in train_samples])
    ros = RandomOverSampler(random_state=args.seed_resample)
    inds_res, y_res = ros.fit_resample(indices, labels)
    n_orig = len(train_samples)
    new_samples = []
    for idx_arr, lbl in zip(inds_res[n_orig:], y_res[n_orig:]):
        orig_idx = idx_arr[0]
        cp = train_samples[orig_idx].copy()
        cp["label"] = int(lbl)
        new_samples.append(cp)
    train_samples += new_samples
    print(f" Random duplication added {len(new_samples)} samples")

    # Convert to HF Datasets and tokenize
    train_ds = Dataset.from_list(train_samples)
    val_ds   = Dataset.from_list(val_samples)
    test_ds  = Dataset.from_list(test_samples)

    # Tokenizer from model dir
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.pad_token = tokenizer.eos_token

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    def preprocess(batch):
        tok = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )
        tok["labels"] = batch["label"]
        return tok

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=["text","label"])
    val_tok   = val_ds.map(preprocess, batched=True, remove_columns=["text","label"])
    test_tok  = test_ds.map(preprocess, batched=True, remove_columns=["text","label"])

    print(" Tokenization completed (on-the-fly).")

# quick sanity: ensure 'labels' column exists
if "labels" not in train_tok.column_names and "label" not in train_tok.column_names:
    raise RuntimeError("[error] train_tok missing 'labels' column")

# ---------------- Tokenizer & model setup (4-bit + LoRA + seq-cls head) -------
print(" Loading model (4-bit + LoRA) and preparing PEFT")
MODEL_DIR = args.model_dir

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Robust device_map: integer GPU index when CUDA available, otherwise "cpu"
if torch.cuda.is_available():
    device_map = {"": 0}   # map entire model to GPU index 0 (respects CUDA_VISIBLE_DEVICES)
else:
    device_map = {"": "cpu"}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb,
    num_labels=len(LABELS),
    device_map=device_map
)

# ensure tokenizer vocabulary sizing
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)
model.config.id2label = { i: lab for i, lab in enumerate(LABELS) }
model.config.label2id = { lab: i for i, lab in enumerate(LABELS) }

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.0,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_cfg)

# Freeze all non-lora parameters robustly
for n, p in model.named_parameters():
    if "lora" not in n.lower():
        p.requires_grad = False

# ---------------- Collator ---------------------------------------------------
collator = DataCollatorWithPadding(tokenizer)

# ---------------- Custom Trainer to use weighted CE or focal loss -------------
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, loss_type="weighted_ce", focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = class_weights.clone().detach().float()

    # accept extra kwargs that Trainer may pass (e.g., num_items_in_batch)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels missing from inputs in compute_loss")

        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        device = logits.device
        labels = labels.to(device=device, dtype=torch.long)

        if self.class_weights is not None:
            cw = self.class_weights.to(device=device, dtype=torch.float32)
        else:
            cw = None

        if self.loss_type == "weighted_ce":
            loss_fct = nn.CrossEntropyLoss(weight=cw)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            ce_loss = F.cross_entropy(logits, labels, reduction="none", weight=cw)
            probs = F.softmax(logits, dim=-1)
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
            focal_factor = (1.0 - pt) ** self.focal_gamma
            loss = (focal_factor * ce_loss).mean()

        return (loss, outputs) if return_outputs else loss

# ---------------- TrainingArguments ----------------
print("  Preparing TrainingArguments")
train_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.grad_accum_steps,
    fp16=args.fp16 and torch.cuda.is_available(),
    bf16=False,
    logging_strategy="steps",
    logging_steps=200,
    eval_strategy="no",
    save_strategy="no",
    report_to=None,
    run_name="falcon-lora-erc",
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    no_cuda=not torch.cuda.is_available(),
)

# ---------------- Trainer & Train ----------------
print("  Initializing Trainer and starting training")
trainer = CustomTrainer(
    model=model,
    args=train_args,
    train_dataset=train_tok,
    data_collator=collator,
    class_weights=class_weights_tensor,
    loss_type=args.loss_type,
    focal_gamma=args.focal_gamma,
)

# optional: set label_names on the trainer instance to avoid the info/warning
trainer.label_names = LABELS

trainer.train()
trainer.save_model(train_args.output_dir)

# ─── Final Evaluation ─────────────────────────────────────────────────────
print("=== Final Evaluation ===")
eval_results = trainer.evaluate(test_tok)
print(eval_results)

preds_out = trainer.predict(test_tok)
preds = np.argmax(preds_out.predictions, axis=-1)
labels = np.array(test_tok["labels"]).reshape(-1)

report_dict = classification_report(labels, preds, digits=4, output_dict=True)
report_df   = pd.DataFrame(report_dict).transpose()
csv_path    = os.path.join(args.output_dir, "classification_report.csv")
report_df.to_csv(csv_path, index=True)
print(f" Classification report saved to {csv_path}")
print(report_df)
