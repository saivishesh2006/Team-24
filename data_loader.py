from datasets import load_from_disk
from transformers import AutoTokenizer
from collections import Counter
import os

# Paths
DATA_PATH = "./prepared_data_raw"   # your pre-tokenized split location
SAVE_PATH = "./prepared_data"       # where tokenized data goes
MODEL_NAME = "./falcon_tokenizer"   # adjust if needed

print("Loading prepared dataset from", DATA_PATH)
dataset_dict = load_from_disk(DATA_PATH)

print("\nPre-tokenization counts:")
for split in dataset_dict:
    labels = dataset_dict[split]["label"]
    cnt = Counter(labels)
    print(f"{split} size = {len(labels)} → {cnt}")

# Load tokenizer
print("\n Loading tokenizer:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize_fn(batch):
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

# Apply tokenization
print("\n Tokenizing...")
tokenized_datasets = {}
for split in dataset_dict:
    print(f"  → Processing {split}...")
    tokenized_split = dataset_dict[split].map(tokenize_fn, batched=True)
    tokenized_datasets[split] = tokenized_split
    
    # Post-tokenization check
    labels_after = tokenized_split["label"]
    print(f"    {split} size after tokenization: {len(labels_after)}")
    print(f"    Label distribution: {Counter(labels_after)}")
    print(f"    First 5 labels: {labels_after[:5]}")

# Save tokenized datasets
print(f"\n Saving tokenized datasets to {SAVE_PATH} ...")
os.makedirs(SAVE_PATH, exist_ok=True)
tokenized_datasets["train"].save_to_disk(os.path.join(SAVE_PATH, "train"))
tokenized_datasets["validation"].save_to_disk(os.path.join(SAVE_PATH, "validation"))
tokenized_datasets["test"].save_to_disk(os.path.join(SAVE_PATH, "test"))

print("\n Tokenization complete.")
