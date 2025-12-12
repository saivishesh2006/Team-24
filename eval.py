#!/usr/bin/env python3
# eval.py — Evaluate a LoRA adapter on prepared tokenized test set and save reports.
# Robust to bitsandbytes / cublas runtime faults via automatic fallbacks.

import os
import argparse
import json
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Optional BitsAndBytes support
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

def safe_preds_to_array(preds):
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if isinstance(preds, dict) and "logits" in preds:
        preds = preds["logits"]
    return np.asarray(preds)

def infer_num_labels_from_dataset(ds):
    if "labels" not in ds.features:
        return None
    lab_feature = ds.features["labels"]
    if hasattr(lab_feature, "num_classes") and lab_feature.num_classes is not None:
        return int(lab_feature.num_classes)
    if hasattr(lab_feature, "names") and lab_feature.names is not None:
        return int(len(lab_feature.names))
    try:
        vals = list(set(ds["labels"]))
        return int(max(vals) + 1)
    except Exception:
        return None

def decode_texts_from_dataset(ds, tokenizer):
    decoded = []
    if "input_ids" in ds.column_names:
        col = "input_ids"
    else:
        col = None
        for c in ds.column_names:
            if c in ("attention_mask", "labels"):
                continue
            sample = ds[0][c]
            if isinstance(sample, list) and len(sample) and isinstance(sample[0], int):
                col = c
                break
    if col is None:
        return [""] * len(ds)
    for ids in ds[col]:
        if isinstance(ids, list) and len(ids) and isinstance(ids[0], list):
            ids = ids[0]
        try:
            txt = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            txt = " ".join(map(str, ids if isinstance(ids, list) else [ids]))
        decoded.append(txt)
    return decoded

def load_base_sequence_classification(model_dir, num_labels, device_map, use_bnb, bnb_compute_dtype, device):
    if use_bnb:
        if not BNB_AVAILABLE:
            raise RuntimeError("bitsandbytes not available")
        bnb_conf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_compute_dtype
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            device_map=device_map,
            quantization_config=bnb_conf
        )
    else:
        base = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            device_map=device_map
        )
    return base

def load_try_variants(model_dir, adapter_dir, tokenizer, num_labels, device, requested_use_bnb):
    """
    Attempt to load/apply adapter with several strategies:
      1) use_bnb with compute_dtype=float16 (fast) [if requested]
      2) use_bnb with compute_dtype=float32 (fallback)
      3) without bnb (full precision) — may OOM
      4) CPU inference w/o bnb
    Returns: model (PEFT-wrapped), used_backend string
    Raises: RuntimeError if all attempts fail
    """
    device_map_gpu = {"": 0} if device.type == "cuda" else {"": "cpu"}
    tok_size = len(tokenizer)
    tried = []

    # helper to attempt a single load and adapter apply
    def attempt(use_bnb, bnb_dtype, map_to):
        desc = f"bnb={use_bnb},dtype={bnb_dtype},device_map={map_to}"
        tried.append(desc)
        print(f"[attempt] loading base with {desc}")
        base = load_base_sequence_classification(model_dir, num_labels, device_map=map_to, use_bnb=use_bnb, bnb_compute_dtype=bnb_dtype, device=device)
        # ensure vocab match
        model_vocab = base.get_input_embeddings().weight.size(0)
        if model_vocab != tok_size:
            print(f"[info] resize base embeddings {model_vocab} -> {tok_size}")
            base.resize_token_embeddings(tok_size)
        print("[attempt] applying adapter...")
        peft_model = PeftModel.from_pretrained(base, adapter_dir, device_map=map_to)
        return peft_model

    # Try 1: user's requested path (fast)
    if requested_use_bnb:
        try:
            model = attempt(True, torch.float16 if device.type == "cuda" else torch.float32, device_map_gpu)
            return model, "bnb-fp16-gpu"
        except Exception as e:
            print(f"[warn] bnb-fp16-gpu attempt failed: {e}")
            traceback.print_exc()

        # try bnb with float32 compute dtype
        try:
            model = attempt(True, torch.float32, device_map_gpu)
            return model, "bnb-fp32-gpu"
        except Exception as e:
            print(f"[warn] bnb-fp32-gpu attempt failed: {e}")
            traceback.print_exc()

    # Try 2: no bitsandbytes (full precision) on GPU
    if device.type == "cuda":
        try:
            model = attempt(False, None, device_map_gpu)
            return model, "no-bnb-gpu"
        except Exception as e:
            print(f"[warn] no-bnb-gpu attempt failed: {e}")
            traceback.print_exc()

    # Try 3: CPU fallback (guaranteed to run, slower)
    try:
        print("[info] Falling back to CPU load (this will be slower but safer).")
        model = attempt(False, None, {"": "cpu"})
        return model, "cpu"
    except Exception as e:
        print(f"[error] All loads failed, attempts: {tried}")
        traceback.print_exc()
        raise RuntimeError("Unable to load model+adapter with any fallback.") from e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-bnb", action="store_true", help="Try bitsandbytes 4-bit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    prepared_dir = os.path.realpath(args.prepared_dir)
    adapter_dir = os.path.realpath(args.adapter_dir)
    model_dir = os.path.realpath(args.model_dir)
    output_dir = os.path.realpath(args.output_dir) if args.output_dir else adapter_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"[info] device = {device}, use_bnb requested = {args.use_bnb}")

    # load test tokenized set
    test_tok = load_from_disk(os.path.join(prepared_dir, "test_tok"))
    print(f"[info] test_tok loaded: {len(test_tok)} examples; columns={test_tok.column_names}")

    # load tokenizer (prefer adapter tokenizer)
    tok_candidates = [adapter_dir, os.path.join(prepared_dir, "tokenizer"), model_dir]
    tokenizer = None
    for cand in tok_candidates:
        if os.path.isdir(cand):
            try:
                tokenizer = AutoTokenizer.from_pretrained(cand, use_fast=False)
                print(f"[info] tokenizer loaded from {cand} (vocab_size={len(tokenizer)})")
                break
            except Exception as e:
                print(f"[warn] failed to load tokenizer from {cand}: {e}")
    if tokenizer is None:
        raise FileNotFoundError("Tokenizer not found under adapter/prepared/model dirs")

    num_labels = infer_num_labels_from_dataset(test_tok) or 7
    print(f"[info] inferred num_labels = {num_labels}")

    # load model+adapter with automatic fallbacks
    model, backend = load_try_variants(model_dir, adapter_dir, tokenizer, num_labels, device, args.use_bnb)
    print(f"[info] loaded model using backend = {backend}")

    # Fix: Falcon has no pad_token → assign EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()

    # Use a Trainer configured for evaluation
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=False,
        do_predict=True,
        logging_strategy="no",
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)

    # run prediction but catch runtime errors (like cublas issues) and retry fallback if needed
    try:
        print("[info] Running prediction...")
        preds_out = trainer.predict(test_tok)
    except RuntimeError as e:
        msg = str(e).lower()
        print(f"[warn] Prediction failed with RuntimeError: {e}")
        # look for cublas / cublaslt signature and attempt fallback to safer backend
        if "cublas" in msg or "cublaslt" in msg or "cublas_status_not_supported" in msg:
            print("[warn] Detected cublas runtime failure during inference. Retrying with safer backend (no-bnb or cpu).")
            # try reload with no-bnb GPU then CPU
            try:
                model, backend2 = load_try_variants(model_dir, adapter_dir, tokenizer, num_labels, device, requested_use_bnb=False)
                print(f"[info] Retried load succeeded with backend={backend2}. Running prediction again.")
                trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)
                preds_out = trainer.predict(test_tok)
            except Exception as e2:
                print("[error] Retry after cublas failure also failed. Falling back to CPU full-precision.")
                try:
                    model, backend3 = load_try_variants(model_dir, adapter_dir, tokenizer, num_labels, torch.device("cpu"), requested_use_bnb=False)
                    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)
                    preds_out = trainer.predict(test_tok)
                except Exception as e3:
                    print("[fatal] All retries failed. See traces:")
                    traceback.print_exc()
                    raise
        else:
            # unknown runtime error — re-raise
            traceback.print_exc()
            raise

    preds_arr = safe_preds_to_array(preds_out.predictions)

    # interpret predictions (logits vs indices)
    if preds_arr.ndim == 1:
        pred_labels = preds_arr.astype(int)
        probs = None
    elif preds_arr.ndim == 2:
        exps = np.exp(preds_arr - np.max(preds_arr, axis=-1, keepdims=True))
        probs = exps / exps.sum(axis=-1, keepdims=True)
        pred_labels = np.argmax(probs, axis=-1)
    elif preds_arr.ndim == 3:
        if preds_arr.shape[1] == 1:
            logits2d = preds_arr[:, 0, :]
        else:
            logits2d = preds_arr[:, -1, :]
        exps = np.exp(logits2d - np.max(logits2d, axis=-1, keepdims=True))
        probs = exps / exps.sum(axis=-1, keepdims=True)
        pred_labels = np.argmax(probs, axis=-1)
    else:
        raise RuntimeError(f"Unexpected prediction shape: {preds_arr.shape}")

    true_labels = np.array(test_tok["labels"]).reshape(-1)
    decoded_texts = decode_texts_from_dataset(test_tok, tokenizer)

    # save predictions csv
    rows = []
    for i in range(len(true_labels)):
        pr = probs[i].tolist() if (probs is not None) else None
        prob_pred = float(pr[int(pred_labels[i])]) if pr is not None else None
        rows.append({
            "idx": i,
            "text": decoded_texts[i],
            "true_label": int(true_labels[i]),
            "pred_label": int(pred_labels[i]),
            "pred_prob": prob_pred,
            "probs": json.dumps(pr) if pr is not None else ""
        })
    df_preds = pd.DataFrame(rows)
    df_preds.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    print(f"[info] predictions.csv saved to {output_dir}")

    # classification report
    report_dict = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    with open(os.path.join(output_dir, "classification_report.json"), "w") as fh:
        json.dump(report_dict, fh, indent=2)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(output_dir, "classification_report.csv"))
    print(f"[info] classification report saved to {output_dir}")

    # confusion matrix plot
    labels_sorted = sorted(list(set(true_labels.tolist() + pred_labels.tolist())))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels_sorted)), yticks=np.arange(len(labels_sorted)),
           xticklabels=labels_sorted, yticklabels=labels_sorted,
           ylabel="True label", xlabel="Predicted label", title="Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"[info] confusion_matrix.png saved to {output_dir}")

    # overall metrics
    overall = {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(true_labels, pred_labels, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "n_samples": int(len(true_labels))
    }
    with open(os.path.join(output_dir, "overall_metrics.json"), "w") as fh:
        json.dump(overall, fh, indent=2)
    print(f"[info] overall metrics saved to {output_dir}")
    print("[info] Evaluation complete.")

if __name__ == "__main__":
    main()
