# Akkadian Translation — Final Model (Score: 34.4)

**Competition:** Deep Past Initiative — Machine Translation (Akkadian → English)  
**Metric:** `√(BLEU × chrF++)`  
**Best Score:** 34.4

---

## Folder Structure

```
akkadian_final/
├── README.md                          ← You are here
├── docs/
│   └── walkthrough.md                 ← Detailed model walkthrough with code explanations
├── training/
│   └── akkadian_v4_full_pipeline.py   ← Training script (ByT5-base + LoRA r=32)
├── inference/
│   ├── akkadian-35-ensemble.ipynb     ← Ensemble submission notebook (2-model + MBR)
│   └── ensemble_config_v5.py          ← Ensemble configuration (EnsembleConfig dataclass)
└── model/
    ├── model.safetensors              ← Model weights (2.3 GB)
    ├── config.json                    ← Model architecture config
    ├── generation_config.json         ← Generation parameters
    ├── tokenizer_config.json          ← ByT5 tokenizer config
    ├── special_tokens_map.json        ← Special token definitions
    └── added_tokens.json              ← Additional tokens
```

---

## What's What

### `training/akkadian_v4_full_pipeline.py`
The full training pipeline that produces the model. Run on Kaggle with GPU:
- Loads ~17K augmented Akkadian-English pairs from `train_complete.csv`
- Preprocesses Akkadian transliterations (removes scholarly markup, normalizes gaps/determinatives)
- Fine-tunes **ByT5-base** with **LoRA r=32** using Adafactor optimizer
- Evaluates using the competition metric: `√(BLEU × chrF++)`
- Saves the best checkpoint based on competition score

### `inference/akkadian-35-ensemble.ipynb`
The ensemble submission notebook that produces the 34.4 score:
- Loads **2 models** (Model A: byt5-akkadian-optimized-34x, Model B: byt5-akkadian-mbr-v2)
- Each model generates 3 candidates per sample (2 beam search + 1 nucleus sample)
- **MBR decoding** selects the best translation using competition-aware utility scoring
- Postprocesses output (de-duplicates, removes artifacts, fixes punctuation)

### `inference/ensemble_config_v5.py`
Configuration dataclass for the ensemble (beam settings, MBR parameters, model paths).

### `model/`
The pre-trained model weights and configs. This is the **byt5-akkadian-optimized-34x** model used as Model A in the ensemble.

### `docs/walkthrough.md`
Complete detailed walkthrough explaining every component with code snippets.

---

## How to Reproduce

1. **Training** (optional — the model is already trained):
   - Upload `train_complete.csv` to Kaggle as a dataset
   - Create a Kaggle notebook with GPU, paste `training/akkadian_v4_full_pipeline.py`
   - Run all cells

2. **Submission** (to get the 34.4 score):
   - Upload `model/` folder to Kaggle as a dataset
   - Attach Model B (`mattiaangeli/byt5-akkadian-mbr-v2`) as a Kaggle model
   - Attach the competition dataset
   - Create a notebook from `inference/akkadian-35-ensemble.ipynb`
   - Submit

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Base model | ByT5-base (byte-level) | Handles Akkadian diacritics without OOV issues |
| Fine-tuning | LoRA r=32 | Parameter-efficient; avoids overfitting on 17K examples |
| Ensemble | 2-model MBR | Independent models make different errors; consensus improves quality |
| Candidate selection | Competition-aware MBR | Uses same metric as leaderboard for optimal selection |
| Preprocessing | Regex normalization | Critical for train/inference consistency |
