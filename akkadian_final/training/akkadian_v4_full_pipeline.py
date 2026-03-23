# ============================================================================
# AKKADIAN TRANSLATION V4 - FULL PIPELINE
# ============================================================================
# Uses your existing augmented data (train_complete.csv with ~17K pairs)
# + preprocessing + optimized training + competition metric
#
# Key improvements over previous versions:
#   1. Input preprocessing (clean scribal notations, normalize gaps/determinatives)
#   2. Uses your 17K augmented dataset (not the raw 1.5K documents)
#   3. Dual-metric optimization: sqrt(BLEU × chrF++) = competition score
#   4. Clean inference (no aggressive repetition/ngram penalties)
#   5. Post-processing for BLEU/chrF++ boost
#
# Expected: 28-38 competition score (up from 20)
# ============================================================================

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
!pip install evaluate sacrebleu
!pip install -U bitsandbytes

# ============================================================================
# CELL 2: Imports & Setup
# ============================================================================
import bitsandbytes as bnb
import pandas as pd
import torch
import os
import re
import shutil
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "google/byt5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ===================== KAGGLE PATHS =====================
# Your augmented dataset (uploaded as a Kaggle dataset)
AUGMENTED_CSV = "/kaggle/input/datasets/ushreyas14/akkadian-tokens/train_complete.csv"
TOKENIZED_INPUT = "/kaggle/input/datasets/ushreyas14/akkadian-tokens/tokenized_data"

# Competition data
TEST_PATH = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"

# Working directories
WORKING_TOKENIZED = "/kaggle/working/fixed_tokenized_data"
OUTPUT_DIR = "/kaggle/working/akkadian_v4"

print(f"Augmented CSV: {AUGMENTED_CSV}")
print(f"Test path: {TEST_PATH}")


# ============================================================================
# CELL 3: Preprocessing Functions
# ============================================================================
def preprocess_transliteration(text):
    """
    Clean Akkadian transliteration for model input.
    Removes scribal notations, normalizes gaps, determinatives, etc.
    """
    if not isinstance(text, str):
        return str(text) if pd.notna(text) else ""

    s = text

    # 1. Remove scribal notations (modern editorial marks)
    s = re.sub(r'[!?]', '', s)
    s = re.sub(r'/', ' ', s)
    s = re.sub(r'(?<!\w):(?!\w)', ' ', s)  # colon not inside words

    # 2. Standardize gaps and breaks
    s = re.sub(r'\[x+\]', '<gap>', s)
    s = re.sub(r'\[\.\.\.\]', '<gap>', s)
    s = re.sub(r'\[…\]', '<gap>', s)
    s = re.sub(r'\.\.\.', '<gap>', s)
    s = re.sub(r'…', '<gap>', s)

    # 3. Remove brackets and parentheses (restoration marks)
    s = re.sub(r'\[([^\]]*)\]', r'\1', s)
    s = re.sub(r'\(([^)]*)\)', r'\1', s)

    # 4. Normalize determinatives
    for det in ['d', 'f', 'm', 'ki', 'URU', 'KUR', 'DINGIR', 'LU', 'LU2',
                'MUNUS', 'GIS', 'TUG', 'KU3', 'AN', 'NA4']:
        s = re.sub(rf'\({det}\)', '{' + det + '}', s, flags=re.IGNORECASE)

    # 5. Normalize subscripts
    subscript_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    s = s.translate(subscript_map)

    # 6. Normalize superscripts
    superscript_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
    s = s.translate(superscript_map)

    # 7. Collapse whitespace
    s = re.sub(r'(<gap>\s*)+', '<gap> ', s)
    s = re.sub(r'\s+', ' ', s)

    return s.strip()


def preprocess_translation(text):
    """Clean English translation."""
    if not isinstance(text, str):
        return str(text) if pd.notna(text) else ""
    s = text.strip()
    s = re.sub(r'\[?\.\.\.\]?', '...', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def postprocess_prediction(text):
    """Post-process model output for better BLEU/chrF++."""
    if not isinstance(text, str):
        return str(text) if text else ""
    s = text.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s([.,;:!?])', r'\1', s)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s.strip()


print("✅ Preprocessing functions defined")


# ============================================================================
# CELL 4: Load & Preprocess Data
# ============================================================================
# Strategy A: Use train_complete.csv (your augmented dataset) — PREFERRED
# Strategy B: Fall back to pre-tokenized data if CSV not available

USE_CSV = os.path.exists(AUGMENTED_CSV)

if USE_CSV:
    print(f"\n📊 Loading augmented CSV: {AUGMENTED_CSV}")
    train_df = pd.read_csv(AUGMENTED_CSV)
    print(f"   Raw rows: {len(train_df):,}")
    print(f"   Columns: {list(train_df.columns)}")

    # Apply preprocessing to transliterations
    print("🔧 Applying preprocessing to transliterations...")
    train_df['transliteration'] = train_df['transliteration'].apply(preprocess_transliteration)
    train_df['translation'] = train_df['translation'].apply(preprocess_translation)

    # Drop rows with empty text
    train_df = train_df[
        (train_df['transliteration'].str.len() > 2) &
        (train_df['translation'].str.len() > 2)
    ].reset_index(drop=True)
    print(f"   After cleaning: {len(train_df):,} rows")

    # Train/eval split (90/10)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.9 * len(train_df))
    eval_df = train_df.iloc[split_idx:].reset_index(drop=True)
    train_df = train_df.iloc[:split_idx].reset_index(drop=True)
    print(f"   Train: {len(train_df):,}, Eval: {len(eval_df):,}")

    # Tokenize
    print("\n🤖 Loading tokenizer & tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    PREFIX = "translate Akkadian to English: "
    MAX_INPUT_LEN = 256
    MAX_TARGET_LEN = 256

    def tokenize_df(df):
        inputs = [PREFIX + str(t) for t in df['transliteration']]
        targets = [str(t) for t in df['translation']]

        model_inputs = tokenizer(
            inputs, max_length=MAX_INPUT_LEN,
            padding="max_length", truncation=True,
        )
        labels = tokenizer(
            targets, max_length=MAX_TARGET_LEN,
            padding="max_length", truncation=True,
        )
        model_inputs["labels"] = [
            [-100 if t == tokenizer.pad_token_id else t for t in ids]
            for ids in labels["input_ids"]
        ]
        return model_inputs

    train_dataset = Dataset.from_dict(tokenize_df(train_df))
    eval_dataset = Dataset.from_dict(tokenize_df(eval_df))
    print(f"✅ Tokenized: {len(train_dataset):,} train, {len(eval_dataset):,} eval")

else:
    # Strategy B: Use pre-tokenized data
    print(f"\n📊 CSV not found. Using pre-tokenized data: {TOKENIZED_INPUT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    PREFIX = "translate Akkadian to English: "
    MAX_INPUT_LEN = 256
    MAX_TARGET_LEN = 256

    if not os.path.exists(WORKING_TOKENIZED):
        print("📁 Copying tokenized dataset...")
        shutil.copytree(TOKENIZED_INPUT, WORKING_TOKENIZED)
        # Fix Kaggle's renamed files
        for split in ["train", "test"]:
            split_dir = os.path.join(WORKING_TOKENIZED, split)
            if os.path.exists(split_dir):
                for fn in os.listdir(split_dir):
                    if "(1)" in fn:
                        os.rename(
                            os.path.join(split_dir, fn),
                            os.path.join(split_dir, fn.replace(" (1)", "")),
                        )

    tokenized_datasets = load_from_disk(WORKING_TOKENIZED)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]
    print(f"✅ Loaded: {len(train_dataset):,} train, {len(eval_dataset):,} eval")


# ============================================================================
# CELL 5: Load Model + LoRA
# ============================================================================
print("\n🤖 Loading ByT5-base with LoRA...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.enable_input_require_grads()

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules="all-linear",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# ============================================================================
# CELL 6: Metrics (BLEU + chrF++ + Competition Score)
# ============================================================================
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(preds, np.ndarray) and len(preds.shape) == 3:
        preds = np.argmax(preds, axis=-1)
    elif isinstance(preds, (list, tuple)):
        preds_array = np.array(preds)
        if len(preds_array.shape) == 3:
            preds = np.argmax(preds_array, axis=-1)

    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    preds = np.where(preds > 1114114, tokenizer.pad_token_id, preds)
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels_flat = [label.strip() for label in decoded_labels]
    decoded_labels_nested = [[label] for label in decoded_labels_flat]

    bleu_result = bleu_metric.compute(
        predictions=decoded_preds, references=decoded_labels_nested
    )
    chrf_result = chrf_metric.compute(
        predictions=decoded_preds, references=decoded_labels_flat,
        word_order=2,
    )

    bleu_score = bleu_result["score"]
    chrf_score = chrf_result["score"]
    comp_score = np.sqrt(max(bleu_score, 0) * max(chrf_score, 0))

    result = {
        "bleu": round(bleu_score, 4),
        "chrf": round(chrf_score, 4),
        "competition_score": round(comp_score, 4),
    }
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = round(np.mean(prediction_lens), 2)
    return result


# ============================================================================
# CELL 7: Training Arguments
# ============================================================================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    dataloader_num_workers=0,

    predict_with_generate=True,
    generation_max_length=256,
    generation_num_beams=4,

    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="competition_score",
    greater_is_better=True,
    save_total_limit=3,

    num_train_epochs=10,
    label_smoothing_factor=0.05,

    optim="adafactor",
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.10,
    weight_decay=0.01,

    fp16=True,
    max_grad_norm=1.0,
    ddp_find_unused_parameters=False,
    eval_accumulation_steps=1,
    logging_steps=50,
    report_to="none",
)


# ============================================================================
# CELL 8: Train
# ============================================================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print("\n🚀 Starting V4 training...")
print(f"   Dataset: {len(train_dataset):,} train, {len(eval_dataset):,} eval")
print(f"   Model: ByT5-base + LoRA r=32")
print(f"   Metric: competition_score = sqrt(BLEU × chrF++)")
torch.cuda.empty_cache()
trainer.train()


# ============================================================================
# CELL 9: Save Best Model
# ============================================================================
SAVE_PATH = "/kaggle/working/akkadian_v4_model"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"✅ Model saved to {SAVE_PATH}")


# ============================================================================
# CELL 10: Generate Predictions
# ============================================================================
print("\n📊 Loading test data...")
test_df = pd.read_csv(TEST_PATH)
print(f"   Test samples: {len(test_df):,}")

model.eval()
predictions = []
BATCH_SIZE = 8

print("\n🔄 Generating translations...")
for i in tqdm(range(0, len(test_df), BATCH_SIZE), desc="Translating"):
    batch = test_df.iloc[i : i + BATCH_SIZE]

    # Apply SAME preprocessing as training!
    inputs = [
        PREFIX + preprocess_transliteration(str(x))
        for x in batch["transliteration"].values
    ]

    encoded = tokenizer(
        inputs, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_INPUT_LEN,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=256,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            # NO repetition_penalty — hurts low-resource translation
            # NO no_repeat_ngram_size — blocks legitimate repeats
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded = [postprocess_prediction(p) for p in decoded]
    predictions.extend(decoded)

    del encoded, outputs
    torch.cuda.empty_cache()


# ============================================================================
# CELL 11: Show Samples & Save Submission
# ============================================================================
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS (V4 Full Pipeline)")
print("=" * 80)

for i in range(min(10, len(test_df))):
    print(f"\n--- Sample {i+1} ---")
    raw = str(test_df.iloc[i]['transliteration'])[:80]
    cleaned = preprocess_transliteration(raw)[:80]
    print(f"  Raw:       {raw}")
    print(f"  Cleaned:   {cleaned}")
    print(f"  Predicted: {predictions[i][:120]}")

submission = pd.DataFrame({"id": test_df["id"], "translation": predictions})
submission.to_csv("/kaggle/working/submission.csv", index=False)

print(f"\n✅ Submission saved: /kaggle/working/submission.csv ({len(submission):,} rows)")
print(f"   V4: preprocessing + augmented data ({len(train_dataset):,} pairs)")
print(f"   Optimized for: sqrt(BLEU × chrF++)")

pred_lens = [len(p.split()) for p in predictions]
print(f"\n📈 Prediction stats:")
print(f"   Mean length: {np.mean(pred_lens):.1f} words")
print(f"   Min/Max: {np.min(pred_lens)}/{np.max(pred_lens)} words")
print(f"   Empty: {sum(1 for p in predictions if len(p.strip()) == 0)}")
