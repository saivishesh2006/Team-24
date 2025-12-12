#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_tokens.py (fully offline)

New pipeline:
  1) Load DailyDialog
  2) Build instruct-style prompts (same as before)
  3) For every train original, create candidate set = {original} U {back-translation via each pivot}
  4) (Optional) SBERT filtering applied to BT candidates only
  5) For each class, random-sample from its candidate pool until class-size == target_count
     where target_count = largest ORIGINAL class count
  6) Tokenize and save
"""
import os
import sys
import argparse
import json
import time
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer as HFTokenizer

# CTranslate2 + SentencePiece (offline)
import ctranslate2
import sentencepiece as spm

# Try SBERT; will load ONLY from local path provided
SBERT_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except Exception:
    SBERT_AVAILABLE = False

# ----------------------------- CLI / Config -----------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Prepare tokenized DailyDialog with offline multi-pivot back-translation (full-augment-then-sample)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prepared-dir", type=str, default="./prepared_data", help="Output folder")
    p.add_argument("--helsinki-base", type=str, default="./helsinki-nlp-ctranslate2",
                   help="Base folder containing en-xx and xx-en CTranslate2 models")
    p.add_argument("--pivots", type=str, default="de,fr,es,ru,zh", help="Comma-separated pivots to use")
    p.add_argument("--model-dir", type=str, default="./falcon-rw-1b/tiiuae/falcon-rw-1b",
                   help="Local Falcon tokenizer dir (no internet)")
    p.add_argument("--sbert-dir", type=str, default="./all-MiniLM-L6-v2-hf",
                   help="Local SBERT model dir; leave as-is to enable filtering; pass --no-sbert to disable")
    p.add_argument("--no-sbert", action="store_true", help="Disable SBERT similarity filtering")
    p.add_argument("--sbert-threshold", type=float, default=0.75,
                   help="Keep BT sample only if cosine(original, BT) >= threshold. Set 0 to skip filtering")
    p.add_argument("--max-length", type=int, default=256, help="Tokenizer max_length")
    p.add_argument("--bt-batch-size", type=int, default=64, help="Batch size for CT2 translate_batch")
    p.add_argument("--bt-num-beams", type=int, default=5, help="Beam size for CT2 decoding")
    p.add_argument("--use-fp16-ct2", action="store_true", help="Use float16 compute_type on CUDA")
    p.add_argument("--save-preview", action="store_true", help="Save a preview JSONL of first ~2000 train samples")
    p.add_argument("--dry-run", action="store_true", help="Skip heavy BT; only plan and exit after reporting")
    p.add_argument("--device", type=str, default=None, help="Force device for CT2 (cuda/cpu). Default: auto")
    return p

args = build_argparser().parse_args()

# Offline-friendly env (prevents HF calls)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Reproducibility
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device for CT2
BT_DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16_CT2 = bool(args.use_fp16_ct2)
COMPUTE_TYPE = "float16" if (BT_DEVICE.startswith("cuda") and USE_FP16_CT2) else "default"

PREPARED_DIR = args.prepared_dir
HELSINKI_BASE = args.helsinki_base
PIVOTS_REQ = [p.strip() for p in args.pivots.split(",") if p.strip()]
SBERT_DIR = None if args.no_sbert else args.sbert_dir
SBERT_THRESHOLD = None if (args.no_sbert or args.sbert_threshold <= 0) else float(args.sbert_threshold)
MAX_LEN = int(args.max_length)
BT_BATCH = int(args.bt_batch_size)
BT_BEAMS = int(args.bt_num_beams)
DRY_RUN = bool(args.dry_run)

print(f"[info] BT device={BT_DEVICE}, compute_type={COMPUTE_TYPE}, batch={BT_BATCH}, beams={BT_BEAMS}")
print(f"[info] pivots_requested={PIVOTS_REQ}")
print(f"[info] SBERT_DIR={SBERT_DIR}, SBERT_THRESHOLD={SBERT_THRESHOLD}, SBERT_AVAILABLE={SBERT_AVAILABLE}")
print(f"[info] prepared_dir={PREPARED_DIR}, helsinki_base={HELSINKI_BASE}, dry_run={DRY_RUN}")

os.makedirs(PREPARED_DIR, exist_ok=True)

# --------------------------- Load DailyDialog ---------------------------------
print(" Loading DailyDialog splits from dataset.prepare_daily_dialog_from_splits()")
try:
    from dataset import prepare_daily_dialog_from_splits
except Exception as e:
    print(f"[fatal] Cannot import prepare_daily_dialog_from_splits: {e}")
    sys.exit(1)

train_raw, val_raw, test_raw = prepare_daily_dialog_from_splits()

# --------------------------- Build prompts ------------------------------------
LABELS = ["neutral","angry","disgust","fear","happy","sad","surprised"]

def build_samples(raw_split):
    """Same prompt style you used in your training script."""
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

print(" Building prompts for train/val/test")
train_samples = build_samples(train_raw)
val_samples   = build_samples(val_raw)
test_samples  = build_samples(test_raw)

def print_distribution(name, samples_like):
    if hasattr(samples_like, "column_names"):
        if "labels" in samples_like.column_names:
            labs = samples_like["labels"]
        elif "label" in samples_like.column_names:
            labs = samples_like["label"]
        else:
            labs = []
    elif isinstance(samples_like, list):
        if len(samples_like)==0:
            labs=[]
        elif isinstance(samples_like[0], dict):
            labs=[int(s["label"]) for s in samples_like]
        else:
            labs=[int(x) for x in samples_like]
    else:
        labs=[int(x) for x in samples_like]
    c = Counter(labs)
    total = sum(c.values())
    print(f"\n {name} distribution:")
    for i, lab in enumerate(LABELS):
        print(f"  {lab:10s}: {c.get(i,0):6d}")
    print(f"  total      : {total}\n")

print_distribution("Train (orig)", train_samples)
print_distribution("Val   (orig)", val_samples)
print_distribution("Test  (orig)", test_samples)

# ------------------- Detect available pivots on disk --------------------------
def pivot_available(p: str) -> bool:
    return os.path.isdir(os.path.join(HELSINKI_BASE, f"en-{p}")) and \
           os.path.isdir(os.path.join(HELSINKI_BASE, f"{p}-en"))

PIVOTS = [p for p in PIVOTS_REQ if pivot_available(p)]
if not PIVOTS:
    print(f"[fatal] No usable pivots under {HELSINKI_BASE} for requested {PIVOTS_REQ}")
    sys.exit(1)
print(f"[info] pivots_used={PIVOTS}")

# -------------------------- Back-translation helpers --------------------------
def load_spm(folder: str) -> spm.SentencePieceProcessor:
    cands = ("source.spm", "target.spm", "sentencepiece.model", "spm.model", "model.spm")
    for name in cands:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            sp = spm.SentencePieceProcessor()
            sp.load(path)
            return sp
    raise FileNotFoundError(f"No SentencePiece model found in {folder} (tried {cands})")

translators_en_to: Dict[str, ctranslate2.Translator] = {}
translators_to_en: Dict[str, ctranslate2.Translator] = {}
sp_en_src: Dict[str, spm.SentencePieceProcessor]     = {}
sp_en_tgt: Dict[str, spm.SentencePieceProcessor]     = {}
sp_pv_src: Dict[str, spm.SentencePieceProcessor]     = {}
sp_pv_tgt: Dict[str, spm.SentencePieceProcessor]     = {}

def ensure_pivot_loaded(pivot: str):
    if pivot in translators_en_to:
        return
    en_p_dir = os.path.join(HELSINKI_BASE, f"en-{pivot}")
    p_en_dir = os.path.join(HELSINKI_BASE, f"{pivot}-en")
    if not (os.path.isdir(en_p_dir) and os.path.isdir(p_en_dir)):
        raise FileNotFoundError(f"Missing CT2 dirs for pivot={pivot}: {en_p_dir}, {p_en_dir}")

    print(f"[bt] Loading CT2 translators and SP models for pivot '{pivot}'")
    translators_en_to[pivot] = ctranslate2.Translator(en_p_dir, device=BT_DEVICE, compute_type=COMPUTE_TYPE)
    translators_to_en[pivot] = ctranslate2.Translator(p_en_dir, device=BT_DEVICE, compute_type=COMPUTE_TYPE)

    sp_en_src[pivot] = load_spm(en_p_dir)   # encode EN to pieces for en->pivot
    sp_pv_src[pivot] = load_spm(p_en_dir)   # encode PIVOT to pieces for pivot->en
    sp_en_tgt[pivot] = load_spm(p_en_dir)   # decode EN pieces from pivot->en outputs
    sp_pv_tgt[pivot] = load_spm(en_p_dir)   # decode PIVOT pieces from en->pivot outputs (rarely used)

def decode_with_sp(sp: spm.SentencePieceProcessor, tokens) -> str:
    if tokens is None or len(tokens) == 0:
        return ""
    try:
        return sp.decode_pieces(tokens) if hasattr(sp, "decode_pieces") else sp.decode(tokens)
    except Exception:
        return " ".join(tokens if isinstance(tokens, list) else [str(tokens)]) 

def backtranslate_pivot_batch(pivot: str, src_texts: List[str]) -> List[str]:
    ensure_pivot_loaded(pivot)
    tr_fwd = translators_en_to[pivot]
    tr_bwd = translators_to_en[pivot]
    sp_src_en = sp_en_src[pivot]
    sp_tgt_en = sp_en_tgt[pivot]

    # EN -> pivot
    mid_tokens_all: List[List[str]] = []
    for start in range(0, len(src_texts), BT_BATCH):
        batch = src_texts[start:start+BT_BATCH]
        pieces = [sp_src_en.encode(text, out_type=str) for text in batch]
        try:
            outs = tr_fwd.translate_batch(pieces, beam_size=BT_BEAMS, max_decoding_length=256)
        except Exception as e:
            size = len(pieces)
            small = max(1, size // 2)
            outs = []
            for i in range(0, size, small):
                sub = pieces[i:i+small]
                outs.extend(tr_fwd.translate_batch(sub, beam_size=BT_BEAMS, max_decoding_length=256))
        for o in outs:
            hyp = o.hypotheses[0] if o.hypotheses else []
            mid_tokens_all.append(hyp)

    # pivot -> EN
    final_texts: List[str] = []
    for start in range(0, len(mid_tokens_all), BT_BATCH):
        batch_mid = mid_tokens_all[start:start+BT_BATCH]
        try:
            outs2 = tr_bwd.translate_batch(batch_mid, beam_size=BT_BEAMS, max_decoding_length=256)
        except Exception as e:
            size = len(batch_mid)
            small = max(1, size // 2)
            outs2 = []
            for i in range(0, size, small):
                sub = batch_mid[i:i+small]
                outs2.extend(tr_bwd.translate_batch(sub, beam_size=BT_BEAMS, max_decoding_length=256))
        for o2 in outs2:
            hyp_tokens = o2.hypotheses[0] if o2.hypotheses else []
            final_texts.append(decode_with_sp(sp_tgt_en, hyp_tokens))

    if len(final_texts) != len(src_texts):
        raise RuntimeError(f"[bt] size mismatch for pivot={pivot}: got {len(final_texts)} vs {len(src_texts)}")
    return final_texts

# -------------------------- SBERT (local) load - robust -----------------------
sbert_model = None
if SBERT_THRESHOLD is not None:
    if not SBERT_AVAILABLE:
        print("[warn] SBERT not importable; disabling similarity filtering.")
        SBERT_THRESHOLD = None
    else:
        if not SBERT_DIR:
            print("[warn] SBERT dir not provided; disabling similarity filtering.")
            SBERT_THRESHOLD = None
        else:
            candidate = os.path.realpath(SBERT_DIR)
            if not os.path.isdir(candidate):
                print(f"[warn] Provided SBERT_DIR '{SBERT_DIR}' resolved to '{candidate}' which does not exist.")
                found = None
                for root, dirs, files in os.walk('.', topdown=True):
                    for d in dirs:
                        if d.startswith('all-MiniLM-L6') or d == os.path.basename(SBERT_DIR):
                            found = os.path.realpath(os.path.join(root, d))
                            break
                    if found:
                        break
                if found:
                    print(f"[info] Auto-discovered candidate SBERT folder: {found}")
                    candidate = found
                else:
                    print("[warn] Could not find a local SBERT folder; disabling similarity filtering.")
                    SBERT_THRESHOLD = None
                    candidate = None

            if candidate:
                print(f"[info] Attempting to load SBERT locally from: {candidate}")
                try:
                    try:
                        print("[info] top-level listing:", sorted(os.listdir(candidate))[:50])
                    except Exception:
                        pass
                    sbert_model = SentenceTransformer(candidate, device=("cuda" if torch.cuda.is_available() else "cpu"))
                    print("[info] SBERT loaded successfully from:", candidate)
                except Exception as e:
                    print(f"[warn] Failed to load SBERT from {candidate}: {e}")
                    nested_try = None
                    for cand_name in ("0_BERT", "1_Pooling", "bert", "transformer"):
                        path2 = os.path.join(candidate, cand_name) if candidate else None
                        if path2 and os.path.isdir(path2):
                            nested_try = path2
                            break
                    if nested_try:
                        try:
                            print(f"[info] Trying nested fallback: {nested_try}")
                            sbert_model = SentenceTransformer(nested_try, device=("cuda" if torch.cuda.is_available() else "cpu"))
                            print("[info] SBERT loaded from nested path:", nested_try)
                        except Exception as e2:
                            print(f"[warn] Nested load also failed: {e2}; disabling SBERT filtering.")
                            sbert_model = None
                            SBERT_THRESHOLD = None
                    else:
                        print("[warn] No nested fallback found; disabling SBERT filtering.")
                        sbert_model = None
                        SBERT_THRESHOLD = None

# -------------------------- Full augmentation pass ---------------------------
n_train = len(train_samples)
train_counts = Counter(s["label"] for s in train_samples)
target_count = max(train_counts.values()) if train_counts else 0
print(f"[info] original train counts: {dict(train_counts)}; target_count = {target_count}")

# If dry-run, we won't perform heavy BT; just create a preview plan and exit.
if DRY_RUN:
    preview_map = os.path.join(PREPARED_DIR, "mapping_preview_fullaugment.jsonl")
    with open(preview_map, "w", encoding="utf-8") as fh:
        for idx, s in enumerate(train_samples[:2000]):
            fh.write(json.dumps({
                "orig_idx": idx,
                "label": int(s["label"]),
                "text_snippet": s["text"][:256],
                "pivots_will_be_used_for_bt": PIVOTS
            }, ensure_ascii=False) + "\n")
    print(f"[dry-run] full-augment plan preview written to {preview_map}")
    sys.exit(0)

# Build candidate pools per label. Each candidate is a dict:
# { "orig_idx": int, "text": str, "pivot": str ("orig" or pivot), "_bt_sbert_sim": Optional[float] }
candidates_by_label: Dict[int, List[Dict]] = defaultdict(list)

# Add originals as candidates
for idx, s in enumerate(train_samples):
    candidates_by_label[s["label"]].append({
        "orig_idx": idx,
        "text": s["text"],
        "pivot": "orig",
        "_bt_sbert_sim": None
    })

# For each pivot, backtranslate ALL train samples (batched) and add to corresponding label pool
all_src_texts = [s["text"] for s in train_samples]  # aligned by index

for pivot in PIVOTS:
    print(f"[bt] Backtranslating ALL train originals via pivot='{pivot}'")
    bt_texts = backtranslate_pivot_batch(pivot, all_src_texts)  # length == n_train

    if SBERT_THRESHOLD is None or sbert_model is None:
        # no sbert filtering: add all
        for i, bt in enumerate(bt_texts):
            lab = train_samples[i]["label"]
            candidates_by_label[lab].append({
                "orig_idx": i,
                "text": bt,
                "pivot": pivot,
                "_bt_sbert_sim": None
            })
    else:
        # compute SBERT similarities between original and bt candidates (per-pivot batch)
        with torch.no_grad():
            batch_orig = [train_samples[i]["text"] for i in range(n_train)]
            emb_o = sbert_model.encode(batch_orig, convert_to_tensor=True, normalize_embeddings=True)
            emb_a = sbert_model.encode(bt_texts, convert_to_tensor=True, normalize_embeddings=True)
            sims = sbert_util.cos_sim(emb_o, emb_a).diag().float().cpu().tolist()
        kept = 0
        for i, (bt, sim) in enumerate(zip(bt_texts, sims)):
            if sim >= SBERT_THRESHOLD:
                lab = train_samples[i]["label"]
                candidates_by_label[lab].append({
                    "orig_idx": i,
                    "text": bt,
                    "pivot": pivot,
                    "_bt_sbert_sim": float(sim)
                })
                kept += 1
        print(f"[bt][{pivot}] kept {kept}/{n_train} after SBERT >= {SBERT_THRESHOLD}")

# Report pool sizes
pool_sizes = {lab: len(lst) for lab, lst in candidates_by_label.items()}
print(f"[info] candidate pool sizes per label (after full-augment & SBERT filtering): {pool_sizes}")

# ----------------------- Random sampling to equalize classes -----------------
final_train_samples: List[Dict] = []
for lab in range(len(LABELS)):
    pool = candidates_by_label.get(lab, [])
    if not pool:
        print(f"[warn] label {lab} ({LABELS[lab]}) has empty candidate pool -> skipping (will remain empty)")
        continue

    if len(pool) >= target_count:
        chosen = random.sample(pool, target_count)
    else:
        # sample all pool, then sample with replacement to fill to target_count
        chosen = list(pool)  # copy
        while len(chosen) < target_count:
            chosen.append(random.choice(pool))
    # convert chosen entries to sample dicts used later
    for c in chosen:
        final_train_samples.append({
            "text": c["text"],
            "label": int(lab),
            "_bt_sbert_sim": c.get("_bt_sbert_sim", None),
            "_bt_pivot": c.get("pivot", "orig"),
            "_orig_idx": int(c.get("orig_idx", -1))
        })

print(f"[info] final training samples after sampling: {len(final_train_samples)} (expected = {target_count * len(LABELS)})")
# sanity distribution
print_distribution("Train (post-sample)", final_train_samples)

# ---------------------- Append augmented to training (we replaced train_samples) --
# We will replace train_samples with final_train_samples for tokenization/training
train_samples = final_train_samples

# -------------------------- HF Datasets + Tokenize ----------------------------
print(" Converting to HuggingFace Datasets")
train_ds = Dataset.from_list(train_samples)
val_ds   = Dataset.from_list(val_samples)
test_ds  = Dataset.from_list(test_samples)

print(" Loading Falcon tokenizer (local only)")
model_dir_real = os.path.realpath(args.model_dir)
if not os.path.isdir(model_dir_real):
    raise FileNotFoundError(f"Falcon tokenizer dir not found: {args.model_dir} -> tried {model_dir_real}")
tok = HFTokenizer.from_pretrained(model_dir_real, use_fast=False)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": tok.eos_token})
    tok.pad_token = tok.eos_token

tok.save_pretrained(os.path.join(PREPARED_DIR, "tokenizer"))

def preprocess(batch):
    out = tok(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )
    out["labels"] = batch["label"]
    if "_bt_sbert_sim" in batch:
        out["_bt_sbert_sim"] = batch["_bt_sbert_sim"]
    return out

print(" Tokenizing datasets (this may take a bit)")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
train_tok = train_ds.map(preprocess, batched=True, remove_columns=["text","label","_bt_pivot","_orig_idx"], batch_size=BT_BATCH)
val_tok   = val_ds.map(preprocess,   batched=True, remove_columns=["text","label"], batch_size=BT_BATCH)
test_tok  = test_ds.map(preprocess,  batched=True, remove_columns=["text","label"], batch_size=BT_BATCH)

if ("labels" not in train_tok.column_names) and ("label" not in train_tok.column_names):
    raise RuntimeError("[error] tokenized train missing 'labels' column")
print_distribution("Train_tok", train_tok)

# ------------------------------- Save -----------------------------------------
print(f" Saving tokenized datasets to {PREPARED_DIR}")
train_tok.save_to_disk(os.path.join(PREPARED_DIR, "train_tok"))
val_tok.save_to_disk(os.path.join(PREPARED_DIR, "val_tok"))
test_tok.save_to_disk(os.path.join(PREPARED_DIR, "test_tok"))

meta = {
    "labels": LABELS,
    "num_labels": len(LABELS),
    "prepared_on": time.strftime("%Y-%m-%d %H:%M:%S"),
    "backtranslation": {
        "pivots_requested": PIVOTS_REQ,
        "pivots_used": PIVOTS,
        "batch_size": BT_BATCH,
        "num_beams": BT_BEAMS,
        "device": BT_DEVICE,
        "compute_type": COMPUTE_TYPE,
        "sbert_dir": SBERT_DIR,
        "sbert_threshold": SBERT_THRESHOLD,
    },
    "tokenizer_dir": args.model_dir,
    "helsinki_base": HELSINKI_BASE,
    "max_length": MAX_LEN
}
with open(os.path.join(PREPARED_DIR, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=2)

if args.save_preview:
    preview_path = os.path.join(PREPARED_DIR, "train_preview.jsonl")
    with open(preview_path, "w", encoding="utf-8") as fh:
        for s in train_samples[:2000]:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[info] preview saved to {preview_path}")

print(" prepare_tokens.py finished. You can now train on ./prepared_data/*")
