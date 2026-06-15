# Project Documentation — Multilingual Media-Sentiment Benchmark

**Author:** Marwane ElBaraka  
**Context:** Stage d'initiation — Week 1  
**Goal:** Benchmark open-source AI tools for sentiment analysis (positive / neutral / negative) on transcribed speech (SRT files) across four languages.

---

## Table of Contents

1. [What problem are we solving?](#1-what-problem-are-we-solving)
2. [The four languages and why they are hard](#2-the-four-languages-and-why-they-are-hard)
3. [Non-negotiable methodology rules](#3-non-negotiable-methodology-rules)
4. [Repository structure](#4-repository-structure)
5. [File-by-file code walkthrough](#5-file-by-file-code-walkthrough)
   - [src/utils.py](#srcutilspy)
   - [src/build_test_sets.py](#srcbuild_test_setspy)
   - [src/label_maps.py](#srclabel_mapspy)
   - [src/harness.py](#srcharnesspy)
   - [src/run_models.py](#srcrun_modelspy)
   - [src/finetune.py](#srcfinetunepy)
   - [src/atlas_chat.py](#srcatlas_chatpy)
   - [src/srt_utils.py](#srcsrt_utilspy)
   - [src/aggregate.py](#srcaggregatepy)
   - [src/smoke_test.py](#srcsmoke_testpy)
6. [Datasets — what they are, why chosen, what we found](#6-datasets--what-they-are-why-chosen-what-we-found)
   - [MAC — Moroccan Arabic Corpus](#mac--moroccan-arabic-corpus-darija_ar)
   - [Allociné](#allocin--francais)
   - [ASTD — Arabic Sentiment Tweets Dataset](#astd--arabic-sentiment-tweets-dataset-msa)
   - [MYC — Moroccan YouTube Comments](#myc--moroccan-youtube-comments-arabizi)
7. [Models — architecture, motivation, pros/cons](#7-models--architecture-motivation-proscons)
   - [xlm-t (XLM-RoBERTa twitter)](#xlm-t--cardiffnlptwitter-xlm-roberta-base-sentiment)
   - [camelbert-da](#camelbert-da--camel-labbertbasearabiccamelbert-da-sentiment)
   - [distilcamembert](#distilcamembert--cmarkea distilcamembert-base-sentiment)
   - [DarijaBERT (fine-tune)](#darijabert--si2m-labdarijabert)
   - [DarijaBERT-arabizi (fine-tune)](#darijabert-arabizi--si2m-labdarijabert-arabizi)
   - [MARBERTv2 (fine-tune)](#marbertv2--ubc-nlpmarbertv2)
   - [QARIB (fine-tune)](#qarib--qaribbert-base-qarib)
   - [Atlas-Chat (LLM zero-shot)](#atlas-chat--mbzuai-parisatlas-chat-2b-and-9b)
8. [Step-by-step execution log](#8-step-by-step-execution-log)
   - [Step 1 — Environment and smoke test](#step-1--environment-and-smoke-test)
   - [Step 2 — Building the frozen test sets](#step-2--building-the-frozen-test-sets)
   - [Step 3 — Ready-made model evaluation](#step-3--ready-made-model-evaluation)
9. [Results analysis](#9-results-analysis)
10. [Key concepts glossary](#10-key-concepts-glossary)
11. [What comes next](#11-what-comes-next)

---

## 1. What problem are we solving?

The client (HACA — Haute Autorité de la Communication Audiovisuelle) receives large volumes of broadcast media as audio. Those are first transcribed to text using a speech-to-text system that outputs SRT files (subtitle format — one timestamped caption at a time). The goal is then to automatically classify the **sentiment** (tone) of those captions as **positive, neutral, or negative**.

This matters for media regulation: regulators need to measure whether coverage of a topic is systematically biased (e.g., consistently negative about a political party or consistently positive about an institution).

The challenge is that Moroccan broadcast media uses **four distinct linguistic codes**, sometimes within the same programme:

| Code | Script | Example |
|---|---|---|
| Modern Standard Arabic (MSA) | Arabic | الحكومة تعلن عن إصلاحات |
| French | Latin | Le gouvernement annonce des réformes |
| Moroccan Darija (Arabic script) | Arabic | الحكومة كتعلن على إصلاحات |
| Moroccan Darija Arabizi (Latin script) | Latin + digits | lhoukoma katlan 3la lislahate |

No single off-the-shelf model handles all four well. This benchmark identifies which models to use for which language and whether the cost (GPU time, latency, memory) justifies the accuracy gain over a cheap multilingual baseline.

---

## 2. The four languages and why they are hard

### Modern Standard Arabic (MSA)
MSA is the formal written register of Arabic used in news broadcasts, official speeches, and subtitles. It is standardised, grammar-regulated, and has the most NLP resources. However it is not the spoken language of Moroccan people — it is learned in school. Models trained on MSA (like those from Egypt or the Gulf) may struggle with Moroccan pronunciation or syntax when it leaks into subtitles.

### French
Moroccan broadcast media uses standard French, mostly indistinguishable from metropolitan French. French NLP is mature, with large corpora and strong pre-trained models. The main challenge is domain adaptation: movie review models (like Allociné-trained ones) may not generalise perfectly to political/regulatory speech.

### Moroccan Darija — Arabic script
Darija is the spoken dialect of Morocco. When written in Arabic script, it mixes Arabic roots with Berber and French loanwords and does not follow MSA grammar rules. NLP for Darija is extremely under-resourced. The vocabulary is non-standard (no official orthography), which means the same word can be spelled dozens of different ways by different writers. Most Arabic NLP models — even those designed for "dialectal Arabic" — focus on Egyptian, Gulf, or Levantine dialects, which differ significantly from Moroccan Darija.

### Moroccan Darija — Arabizi (Latin script with digits)
Arabizi is an informal romanisation of Arabic. Moroccan Arabizi replaces Arabic letters that have no Latin equivalent with digits: **3** = ع (ayn), **7** = ح (ha), **9** = ق (qaf). So "3ayndak" means "you have" (عندك). This appears heavily in YouTube comments and social media. No standard spelling exists. Models pre-trained on English or French text will fail entirely because Arabizi looks like gibberish to them. Even Arabic models fail because they expect Arabic script.

---

## 3. Non-negotiable methodology rules

These rules were defined up front to ensure the benchmark is valid, reproducible, and gradeable.

### Rule 1 — Seed 42 everywhere
Every source of randomness in the project is fixed to seed 42:
- Python's `random` module
- NumPy
- PyTorch (CPU and GPU)

This means that if you run the code twice on the same machine with the same data, you get byte-for-byte identical results. This is required for academic reproducibility.

### Rule 2 — Build test sets once, then freeze them
The test sets (the files used to measure model performance) are built once, hashed with SHA-256, and made read-only on disk. They are never rebuilt. This prevents **data leakage**: if we rebuilt the test set after training a model, we might accidentally pick different examples that the model happens to do better on, giving a falsely optimistic result.

SHA-256 is a cryptographic hash — a fingerprint of the file. If a single character changes, the hash changes completely, which lets us verify that no one tampered with the test set.

### Rule 3 — Evaluate per language, not globally
Averaging performance across all four languages into a single number would hide important differences. A model might score 95% on French and 50% on Arabizi. A global average of 72.5% would make it look decent, but it would be useless for Arabizi content. We always report per-language results.

The primary metric is **macro-F1**: the average F1 score across all classes, treating each class equally regardless of its frequency in the test set. This is important because sentiment datasets are often imbalanced (more positive than negative), and macro averaging prevents the majority class from dominating the score.

### Rule 4 — Always inspect id2label before interpreting outputs
HuggingFace models declare their output labels in `model.config.id2label`. The documentation on the Hub or in papers may be outdated or wrong. We always read this dictionary from the actual loaded model and use it as the ground truth for label interpretation.

This rule saved us in practice: `camelbert-da` was documented as binary (positive/negative only) but its actual `id2label` showed three classes: `{0: positive, 1: negative, 2: neutral}`.

### Rule 5 — Central label mapping table
Every model uses its own label names. `xlm-t` outputs `negative/neutral/positive`. `distilcamembert` outputs `1 star` through `5 stars`. The ASTD dataset uses `POS/NEG/OBJ`. The LLM outputs French words `positif/neutre/negatif`. All of these must be normalised to the same canonical set `{neg, neu, pos}` before we can compare results. This normalisation lives in one single file (`label_maps.py`) to avoid errors from having the same mapping defined in multiple places with potential discrepancies.

---

## 4. Repository structure

```
benchmark/
│
├── src/                    ← all Python source files
│   ├── utils.py            ← seed helper (used by every other file)
│   ├── build_test_sets.py  ← Step 2: build + freeze the 4 test CSVs
│   ├── label_maps.py       ← Step 3: central label normalisation table
│   ├── harness.py          ← Step 3: benchmark harness (metrics, JSON, CSV)
│   ├── run_models.py       ← Step 3: run the 3 ready-made models
│   ├── finetune.py         ← Step 4: fine-tune encoders with Trainer API
│   ├── atlas_chat.py       ← Step 5: LLM zero-shot inference (gated)
│   ├── srt_utils.py        ← Step 6: SRT parser + language router
│   ├── aggregate.py        ← Step 7: plots + weighted scoring grid
│   └── smoke_test.py       ← Step 1: environment verification
│
├── data/                   ← git-ignored (too large / raw data)
│   ├── raw/
│   │   ├── MAC/            ← MAC corpus.csv (downloaded from GitHub)
│   │   ├── ASTD/           ← Tweets.txt (downloaded from GitHub)
│   │   └── MYC/            ← DATA_CLEANED.csv (downloaded from GitHub)
│   └── test_sets/          ← the 4 frozen CSVs (committed, read-only)
│       ├── darija_ar.csv
│       ├── francais.csv
│       ├── msa.csv
│       └── arabizi.csv
│
├── results/                ← one JSON per (model × language) + summary.csv
│   ├── figs/               ← heatmap, scatter plot, radar chart
│   ├── xlm-t_darija_ar.json
│   ├── ...
│   └── summary.csv
│
├── checkpoints/            ← git-ignored (fine-tuned model weights, multi-GB)
├── report/                 ← this file + final written report
├── notebooks/              ← exploratory Jupyter notebooks
│
├── requirements.txt        ← pinned Python dependencies
├── Dockerfile              ← reproducible container (python:3.11-slim)
├── .gitignore              ← excludes data/, checkpoints/, __pycache__
└── README.md               ← quickstart guide
```

**Why is `data/` git-ignored but `data/test_sets/` committed?**
Raw datasets are large (MYC alone is 20k rows) and have their own licences. We don't redistribute them. But the frozen test sets (1000 rows each, ~100KB) are small, and committing them means anyone who clones the repo gets identical evaluation data without having to re-download and re-sample — which is the reproducibility guarantee.

---

## 5. File-by-file code walkthrough

---

### src/utils.py

```python
import random
import numpy as np

SEED = 42

def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
```

**Purpose:** A single function that every other script calls at startup to lock all random number generators to seed 42.

**Why three separate seeds?**
Python, NumPy, and PyTorch each maintain their own independent random number generator. Setting one does not affect the others. If we only set `random.seed(42)` but forgot NumPy, then NumPy operations like `train_test_split` would produce different splits on each run.

**Why the try/except around torch?**
On a CPU-only machine without PyTorch installed (e.g. a lightweight analysis environment), the seed function still works for Python and NumPy. It gracefully skips the PyTorch seeding rather than crashing.

**Why `torch.cuda.manual_seed_all()`?**
`torch.manual_seed()` seeds the CPU random generator. When a GPU is available, PyTorch uses a separate CUDA random generator for operations on the GPU. `manual_seed_all()` seeds all available GPUs.

---

### src/build_test_sets.py

This is the most important setup script. It runs once and is never run again.

**High-level flow:**
1. Load raw data for each language from `data/raw/`
2. Normalise labels to `{neg, neu, pos}`
3. Sample up to 1000 rows (stratified by label, seed 42)
4. Save to `data/test_sets/<lang>.csv`
5. Make the file read-only (`chmod 444`)
6. Print SHA-256 hash + label distribution

**`sha256_file(path)`**
Reads the file in 64KB chunks and feeds them to a SHA-256 hasher. Chunking avoids loading the whole file into memory at once (important for large files).

**`freeze(path)`**
Calls `os.chmod` to set the file permissions to read-only for owner, group, and others (`S_IREAD | S_IRGRP | S_IROTH`). After this, any attempt to write to the file will raise a `PermissionError`.

**`sample_and_save(df, lang)`**
Uses `sklearn.model_selection.train_test_split` in a tricky way: it passes `test_size=N_SAMPLE` (1000) to select a stratified sample of 1000 rows. Stratification means it preserves the original label distribution — if the raw data has 60% positive, the sample will also have ~60% positive. This avoids accidentally sampling all-positive or all-negative sets.

**`load_mac()` — MAC dataset**
```python
df = pd.read_csv(path, on_bad_lines="skip")
df = df.rename(columns={"tweets": "text", "type": "label"})
```
The MAC CSV has three columns: `tweets` (the tweet text), `type` (the sentiment label: positive/negative/neutral/mixed), and `class` (the Arabic dialect type: standard/dialectal). We rename `tweets→text` and `type→label`. The `class` column is discarded — it describes the dialect of the tweet, not its sentiment, and we don't need it. Mixed-sentiment rows are dropped because they can't be unambiguously assigned to one class.

**`load_allocine()` — Allociné dataset**
```python
ds = load_dataset("tblard/allocine", split="test")
```
Allociné is loaded directly from HuggingFace's dataset hub, so no manual download is needed. Important note: the original `"allocine"` identifier no longer resolves in newer versions of the `datasets` library — it must be `"tblard/allocine"`. This was discovered during execution and documented in the loader. The dataset is already split into train/validation/test; we use only the `"test"` split to avoid using training data. Labels are `ClassLabel` integers (0=neg, 1=pos) that we map to strings.

**`load_astd()` — ASTD dataset**
```python
df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"], ...)
df["label"] = df["label"].astype(str).str.strip().str.upper()
label_map = {"POS": "pos", "NEG": "neg", "OBJ": None, "MIX": None}
```
ASTD's `Tweets.txt` is tab-separated with NO header row. The format is `tweet_text\tLABEL`. The labels use short codes: `POS`, `NEG`, `OBJ` (objective/neutral reporting), `MIX` (mixed sentiment). Per the plan, we drop `OBJ` and `MIX`. This means our MSA test set ends up binary (only `neg` and `pos`), which is an honest reflection of the ASTD labelling scheme — the dataset simply does not annotate "neutral" the same way we define it.

**`load_myc()` — MYC dataset**
```python
df = pd.read_csv(path, encoding="utf-16")
df = df.rename(columns={"sentence": "text", "polarity": "label"})
label_map = {1: "pos", -1: "neg"}
df = df[~df["text"].astype(str).apply(lambda t: bool(ARABIC_RE.search(t)))]
```
The MYC file is encoded in UTF-16 (starts with a byte-order mark `0xFF 0xFE`), not the more common UTF-8. The columns are `sentence` and `polarity` (integer: 1 or -1). MYC contains both Arabic-script Darija and Arabizi mixed together. We keep only the Arabizi rows by filtering out any row that contains at least one Arabic Unicode character (Unicode range U+0600–U+06FF, which is the Arabic block).

---

### src/label_maps.py

This file is the single source of truth for how every model's output labels map to the canonical set `{neg, neu, pos}`.

**Why centralise mappings?**
If the mapping for `xlm-t` were defined inside `run_models.py` and also referenced in `finetune.py` for comparison, a future edit to one file would silently break the other. Having one file means there is one place to check and update.

**`XLM_T_MAP`**
```python
XLM_T_MAP = {
    "negative": "neg",
    "neutral":  "neu",
    "positive": "pos",
}
```
Verified locally from `model.config.id2label`. Note lowercase — the Hub documentation sometimes shows title-case (`Negative`) but the actual model uses lowercase. This difference would cause a `KeyError` if we trusted the docs instead of inspecting the model.

**`CAMELBERT_DA_MAP`**
```python
CAMELBERT_DA_MAP = {
    "positive": "pos",
    "negative": "neg",
    "neutral":  "neu",
}
CAMELBERT_NEU_THRESHOLD = None  # model already predicts neutral natively
```
The original plan assumed this model was binary (positive/negative only) based on the paper. When we actually loaded the model and inspected `id2label`, it had three classes. The neutral class is already natively predicted, so the threshold heuristic described in the plan (predict neutral if confidence < 0.65) is not needed.

**`DISTILCAMEMBERT_MAP`**
```python
DISTILCAMEMBERT_MAP = {
    "1 star":  "neg",
    "2 stars": "neg",
    "3 stars": "neu",
    "4 stars": "pos",
    "5 stars": "pos",
}
```
This model predicts star ratings (1–5) from French text, like a review rating system. The mapping collapses them: very negative reviews (1–2 stars) → neg, ambiguous middle (3 stars) → neu, positive reviews (4–5 stars) → pos. This is a reasonable approximation but not perfect — a 3-star review might lean slightly positive or negative.

**`FINETUNED_MAP`**
When we fine-tune a model with our own 3-class head, we control the label names during training: `{0: "neg", 1: "neu", 2: "pos"}`. But HuggingFace's pipeline sometimes returns `LABEL_0`, `LABEL_1`, `LABEL_2` if the `id2label` config is not set correctly. We handle both variants.

**`apply_map(raw_label, mapping)`**
A strict lookup function. If a model returns a label not in the mapping, it raises a `KeyError` immediately with a clear message. This is intentional — silent errors (returning a default) would corrupt metrics without warning.

---

### src/harness.py

The benchmark harness is the engine that takes any model (expressed as a `predict_fn`) and any language test set, runs inference, and computes a standard set of metrics.

**`free_gpu(model)`**
```python
def free_gpu(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
```
On a T4 GPU with 16GB VRAM, loading a 278M-parameter model (xlm-t) uses ~1.1GB. If we load the next model without freeing the first, we may run out of memory. This function:
1. Deletes the Python reference to the model object (`del model`)
2. Runs the garbage collector to free the Python memory (`gc.collect()`)
3. Tells CUDA to release its memory pool back to the OS (`empty_cache()`)

**`evaluate_model(model_name, lang, predict_fn, ...)`**

This is the central function. Its signature accepts:
- `model_name`: a short string identifier (e.g., `"xlm-t"`)
- `lang`: which language test set to load (`darija_ar`, `francais`, `msa`, `arabizi`)
- `predict_fn`: a **function** that takes a list of text strings and returns a list of canonical labels. This is the key design choice — the harness doesn't care how the model works internally. Any model can be plugged in by wrapping it in this function signature.
- `model_obj`: the model object itself, used only to count parameters
- `extra_meta`: any additional fields to store in the JSON (e.g., GPU training time for fine-tuned models, non-response rate for the LLM)

**Warm-up (`predict_fn(texts[:5])`)**
The first few inferences on a model are slower than subsequent ones because of JIT compilation and memory caching. We run 5 "warm-up" examples and discard their timing so that the measured latency reflects steady-state performance, not startup overhead.

**Latency measurement**
```python
t0 = time.perf_counter()
y_pred = predict_fn(texts)   # all 1000 texts at once
elapsed = time.perf_counter() - t0
latency_ms = (elapsed / len(texts)) * 1000
```
`perf_counter()` is the most precise timer in Python (nanosecond resolution). We run all 1000 texts in a single call (the pipeline handles batching internally) and then divide by count to get the per-utterance latency.

**Macro-F1 computation**
```python
classes = sorted(set(y_true))
macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
```
`labels=classes` is critical. It tells scikit-learn to evaluate F1 only on the classes that appear in the ground truth. For binary test sets (francais, msa, arabizi — which have only `neg` and `pos`), this means we don't penalise a model for never predicting `neu` — because there are no neutral examples to miss in those sets.

**Output**
For each `(model, lang)` pair, the harness writes:
- `results/<model>_<lang>.json` — full detail (classification report, confusion matrix, latency, VRAM, param count)
- A row in `results/summary.csv` — one-line summary, aggregated across all runs

The JSON is idempotent: if you re-run the same `(model, lang)` pair, the old JSON is overwritten and the CSV row is replaced (not duplicated).

---

### src/run_models.py

Orchestrates the three ready-made models and plugs them into the harness.

**`make_pipeline_predict(pipe, label_map, neu_threshold=None)`**
```python
def predict(texts):
    results = pipe(texts, batch_size=BATCH_SIZE, truncation=True, max_length=512)
    preds = []
    for item in results:
        scores = item if isinstance(item, list) else [item]
        best = max(scores, key=lambda x: x["score"])
        ...
    return preds
```
HuggingFace pipelines with `top_k=None` return a list of score dictionaries per input text. The outer `for item in results` loop iterates over texts; each `item` is a list of `{label, score}` dicts (one per class). We find the dict with the highest score (`max(..., key=lambda x: x["score"])`), look up its canonical label in the mapping, and return the result.

**`truncation=True, max_length=512`**
XLM-RoBERTa has a maximum sequence length of 512 tokens. Allociné movie reviews can be thousands of characters long. Without truncation, the model would crash on long inputs. With just `truncation=True` but no `max_length`, some versions of the tokenizer still fail. Setting both explicitly is the safe approach.

**`top_k=None`**
This is the modern replacement for the deprecated `return_all_scores=True`. It returns scores for all classes, not just the top-1. We need all scores to apply the camelbert-da neutral threshold (when needed) and to do proper label mapping.

---

### src/finetune.py

Handles Step 4: fine-tuning pre-trained language models to add a sentiment classification head.

**The fundamental concept — transfer learning**
A model like `DarijaBERT` has been pre-trained on large amounts of Moroccan Darija text. It has learned rich representations of that language — it knows which words appear together, what different syntactic structures look like, and so on. However, it was not trained to predict sentiment.

Fine-tuning adds a small classification layer (a linear projection from the model's hidden state to 3 output classes) and trains the whole system (base model + classifier) on labelled sentiment data. Because the base model already understands the language, we only need a few thousand examples and a few training epochs to get strong results — rather than training from scratch on millions of examples.

**Training configuration (per plan)**
```
fp16       = True          # half-precision floats → halves VRAM usage
batch_size = 16            # texts per gradient update
max_len    = 128 tokens    # truncate longer texts (most social media fits)
lr         = 2e-5          # learning rate (small to avoid catastrophic forgetting)
epochs     = 3             # enough to converge without overfitting
eval       = per epoch     # compute validation macro-F1 after each epoch
best model = highest val macro-F1  # save the checkpoint with the best validation score
```

**Why fp16?**
Floating-point 16 (half precision) uses half the memory of standard fp32 (full precision). A T4 GPU has 16GB VRAM. A BERT-base model in fp32 takes ~1.7GB; in fp16 it takes ~0.9GB. With a batch of 16 sequences, activations add another ~1–2GB. fp16 makes this feasible on a single T4 and roughly doubles training throughput.

**Why `ignore_mismatched_sizes=True`?**
The pre-trained model has no sentiment head — it was trained for language modelling (predicting masked tokens). When we add a 3-class classifier head, the new head's weights are randomly initialised and don't match any weights in the checkpoint. This flag tells HuggingFace to ignore that mismatch and load only the base model weights, leaving the classifier head randomly initialised to be trained from scratch.

**Avoiding test set leakage**
```python
test_df = pd.read_csv(os.path.join(DATA_DIR, f"{lang}.csv"))
test_texts = set(test_df["text"].tolist())
df = df[~df["text"].isin(test_texts)]
```
Before splitting training/validation data from the raw dataset, we remove any row that appears in the frozen test set. This prevents the model from being trained (even inadvertently) on examples it will later be evaluated on, which would give falsely high results.

---

### src/atlas_chat.py

Handles Step 5: zero-shot sentiment classification using a large language model.

**What is zero-shot classification?**
The fine-tuned encoders were explicitly trained on labelled sentiment data. Atlas-Chat has never seen labelled sentiment examples during our pipeline. Instead, we write a natural language **prompt** that describes the task and asks the model to output one word from a closed vocabulary. This is called zero-shot because there are zero task-specific training examples.

**The prompt (in Arabic)**
```
أنت نظام تحليل مشاعر. صنّف النص التالي إلى فئة واحدة فقط من:
positif, neutre, negatif.
النص: {text}
الإجابة (كلمة واحدة فقط):
```
("You are a sentiment analysis system. Classify the following text into exactly one of: positif, neutre, negatif. Text: {text}. Answer (one word only):")

The vocabulary deliberately uses French words (`positif/neutre/negatif`) even in the Arabic prompt because Atlas-Chat was trained on code-switched Moroccan content and handles this naturally. Parseable French labels also make post-processing simpler.

**4-bit quantisation (BitsAndBytes)**
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```
Atlas-Chat-2B has 2 billion parameters. In fp32, that is 8GB just for weights — more than half the T4's VRAM before we even process a single text. 4-bit quantisation reduces each weight from 32 bits to 4 bits, compressing the model to ~1GB. `nf4` (NormalFloat4) is a quantisation format designed by HuggingFace's researchers that minimises accuracy loss compared to naive 4-bit rounding.

**Non-response rate**
Unlike classifiers, the LLM might output anything: a full sentence, a different word, or nothing useful. We parse the output with a regex looking for `positif|neutre|negatif`. If no match is found, we record it as `NON_REPONSE`. This rate is an important cost metric — it tells us how often the LLM is "confused" by the input and gives a non-parseable answer.

---

### src/srt_utils.py

The "real input" pipeline — how we go from an SRT file to sentiment-labelled utterances.

**What is an SRT file?**
SRT (SubRip Subtitle) is a plain text format where each subtitle entry has:
```
42
00:03:15,000 --> 00:03:17,500
هذا المشروع سيساهم في التنمية.
```
(Index number, start→end timestamps, then the text. Blank line separates entries.)

**`parse_srt(path)`**
Tries to open the file with multiple encodings (UTF-8, UTF-8 with BOM, Windows-1256 for Arabic, Latin-1). Arabic text in older Windows software is often saved in CP1256. If it can't decode any way, it raises an error rather than silently producing garbage. Primary parsing uses `pysrt`, a dedicated library. If that fails (malformed file), a regex fallback manually parses the SRT structure.

**`cues_to_utterances(cues)`**
Subtitle cues are short fragments split by the subtitle editor's line-length constraints, not by linguistic sentence boundaries. We merge all cue texts into a single stream and then re-split on sentence-ending punctuation: `.`, `!`, `?`, the Arabic question mark (U+061F ؟), and the Arabic full stop (U+06D4 ۔). The result is a list of complete utterances suitable for sentiment analysis.

**`detect_lang(text)`**
```python
def detect_lang(text: str) -> str:
    if ARABIC_RE.search(text):
        return "arabe"
    tokens = re.findall(r"\w+", text)
    arabizi_hits = sum(1 for t in tokens if ARABIZI_RE.match(t))
    if arabizi_hits / max(len(tokens), 1) > 0.10:
        return "arabizi"
    return "francais"
```
A three-way heuristic:
1. If any Arabic Unicode character is present → `arabe` (could be MSA or Darija-Arabic — distinguished later by camel-tools DialectIdentifier)
2. If more than 10% of word tokens contain the digits 3, 7, or 9 → `arabizi`
3. Otherwise → `francais`

This heuristic is simple and fast. We validate it on 100 hand-labelled utterances and report a confusion matrix (its own "accuracy" metric). The 10% digit threshold was chosen to avoid false positives from numbers in French text (e.g., "le rapport 2024" contains a digit but is not Arabizi).

---

### src/aggregate.py

Final step: synthesises all results into visualisations and a ranked recommendation.

**Macro-F1 heatmap**
A grid where rows are models, columns are languages, and cell colour encodes macro-F1 (red=low, green=high). Immediately shows which model-language combinations are strong and which are weak.

**Cost-precision scatter plot**
X-axis: median latency per utterance (log scale, because latencies span orders of magnitude from 90ms to 1000ms+). Y-axis: mean macro-F1 across languages. Bubble size: square root of parameter count. This visualises the classic precision/cost tradeoff: a model in the upper-left (high F1, low latency) is ideal; one in the lower-right (low F1, high latency) should be rejected.

**Weighted scoring grid**
Four criteria, each normalised 0–1 using min-max scaling:
- **Precision (50%):** average macro-F1 across all languages the model was evaluated on
- **Integration (20%):** placeholder = 1.0 for all models (all expose the same HuggingFace API — no integration cost difference at this stage)
- **Cost (20%):** inverted combination of latency (60%) and VRAM (40%) — lower cost = higher score
- **Coverage (10%):** number of languages evaluated / 4

The weighted sum gives a single ranked score that reflects the real deployment priorities (accuracy is most important, then integration, then cost, then language coverage).

**Radar chart**
Shows the four criteria as axes of a polygon, with each finalist model as a coloured polygon. Makes it easy to see multi-criteria profiles: e.g., an LLM might score high on language coverage but low on cost.

---

### src/smoke_test.py

Step 1 verification script. Run this after setting up the environment to confirm everything is installed correctly.

**What it checks:**
1. Python version
2. PyTorch version and CUDA availability
3. Transformers version
4. Downloads `cardiffnlp/twitter-xlm-roberta-base-sentiment` and runs it on three sentences (French, MSA, Arabizi)
5. Prints the actual `id2label` from the loaded model config
6. Shows the predicted label and confidence for each sentence

**Results observed on this machine:**
```
id2label: {0: 'negative', 1: 'neutral', 2: 'positive'}

[French]  Ce film est absolument magnifique → positive (0.952)
[MSA]     هذا الفيلم رائع جداً             → positive (0.950)
[Arabizi] hada film zwine bzaf, 7bito ktir → neutral  (0.342)
```
The Arabic and French sentences are confident and correct. The Arabizi sentence gets a very low-confidence `neutral` (0.342 is close to random for 3 classes) — confirming that the multilingual baseline struggles with Arabizi. This motivates the Arabizi-specific fine-tune.

---

## 6. Datasets — what they are, why chosen, what we found

---

### MAC — Moroccan Arabic Corpus (`darija_ar`)

| Property | Value |
|---|---|
| Source | github.com/LeMGarouani/MAC |
| File | `MAC corpus.csv` |
| Total rows | 18,087 |
| Classes | positive, negative, neutral, mixed |
| Mixed rows dropped | 643 |
| Our test sample | 1,000 rows (stratified, seed 42) |
| Distribution | pos=567, neu=232, neg=201 |

**What it is:** 18,000 tweets in Moroccan Darija (Arabic script), collected from Twitter/X and manually labelled by native Moroccan annotators. Inter-annotator agreement was IAA=0.90, which is considered high quality. It is the largest open Moroccan Darija sentiment corpus.

**Why chosen:** It is the only large-scale, open, manually-labelled corpus for Moroccan Darija Arabic-script sentiment. There is essentially no alternative for this language.

**Column discovery:** The CSV has three columns — `tweets` (text), `type` (the sentiment label), and `class` (the Arabic dialect category: standard/dialectal). At first glance `class` sounds like the label column, but it is actually a linguistic metadata field. `type` is the sentiment label. This was discovered by inspecting the actual column values, not by assuming from column names.

**Class imbalance:** Positive (56.7%) >> Neutral (23.2%) >> Negative (20.1%). Moroccan Twitter tends to be positive-skewed (social sharing culture). This imbalance is why macro-F1 is preferable to accuracy — accuracy would be 56.7% just by always predicting positive.

---

### Allociné — (`francais`)

| Property | Value |
|---|---|
| Source | HuggingFace Hub: `tblard/allocine` |
| Original | French movie review website allocine.fr |
| Total rows (test split) | 20,000 |
| Classes | neg, pos (BINARY — no neutral) |
| Our test sample | 1,000 rows (stratified, seed 42) |
| Distribution | neg=520, pos=480 |

**What it is:** French movie reviews scraped from allocine.fr, a major French cinema database. Reviews are paired with numeric ratings; sentiment labels (positive/negative) were derived from the rating. Standard French, no code-switching.

**Why chosen:** The de facto standard French sentiment benchmark. Well-balanced, clean, widely used in French NLP papers. Readily available on HuggingFace.

**Dataset ID change:** The legacy ID `"allocine"` no longer resolves in the current `datasets` library (v2.18+). The correct ID is `"tblard/allocine"`. This is a library versioning issue, not a data quality issue.

**Binary limitation:** The dataset has no neutral class — reviews are either positive or negative. This is appropriate for movie reviews (people rarely feel neutral about a film) but may not match broadcast speech, where neutral/informational content is common. The benchmark evaluates models only on the classes present in the ground truth, so this is handled correctly.

---

### ASTD — Arabic Sentiment Tweets Dataset (`msa`)

| Property | Value |
|---|---|
| Source | github.com/mahmoudnabil/ASTD |
| File | `data/Tweets.txt` |
| Total rows | ~10,000 |
| Original classes | POS, NEG, OBJ (objective), MIX (mixed) |
| Dropped | OBJ, MIX |
| Remaining | POS, NEG → binary |
| Our test sample | 1,000 rows (stratified, seed 42) |
| Distribution | neg=679, pos=321 |

**What it is:** 10,000 Arabic tweets collected from Egypt-focused Twitter accounts, focused on Egyptian politics (Morsi era, 2011–2013). Annotated by humans into four classes. The `OBJ` (objective) class corresponds to neutral factual reporting.

**Why chosen:** It is the most widely cited open MSA sentiment dataset and is commonly used as a benchmark in Arabic NLP papers, making results comparable to the literature.

**File format discovery:** The file is tab-separated with NO header and format `text\tLABEL` (text column first, label column second). The initial assumption was `label\ttext`, which would have put the tweet texts into the label column. This was caught by inspecting the first 5 raw lines.

**MSA vs Egyptian Arabic:** ASTD was collected from Egyptian political Twitter. Egyptian Arabic is close to MSA but not identical. For a Moroccan MSA benchmark, this introduces a small domain mismatch. However, no open Moroccan MSA sentiment dataset exists, so ASTD is the best available proxy.

**Negative skew (67.9% neg):** This reflects the political climate of the collection period — tweets about Egyptian politics during 2012–2013 were predominantly negative (protests, political instability). Models evaluated on this set should not be expected to generalise to balanced news coverage.

---

### MYC — Moroccan YouTube Comments (`arabizi`)

| Property | Value |
|---|---|
| Source | github.com/MouadJb/MYC |
| File | `DATA_CLEANED.csv` |
| Encoding | UTF-16 (BOM) |
| Total rows | 19,991 |
| Classes | 1 (positive), -1 (negative) — BINARY |
| Arabic-script rows | ~14,652 (filtered out) |
| Latin-script rows kept | 5,339 |
| Our test sample | 1,000 rows (stratified, seed 42) |
| Distribution | pos=657, neg=343 |

**What it is:** ~20,000 YouTube comments about Moroccan music and entertainment, split roughly half-half between Arabic-script Darija and Arabizi (Latin-script Darija). Labelled as positive or negative.

**Why chosen:** It is the only open Arabizi-specific dataset for Moroccan Arabic. The paper introducing it (Jbel et al., 2023, arXiv:2303.15987) is specifically about bridging Arabic and Latin-script Darija sentiment, which aligns exactly with our use case.

**UTF-16 discovery:** The file starts with byte `0xFF 0xFE`, which is the UTF-16 Little Endian byte-order mark (BOM). This is unusual — most text files today are UTF-8. The discovery was made when `pandas.read_csv()` crashed with `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0`. Reading with `encoding="utf-16"` resolved it.

**Latin-only filter:**
```python
df = df[~df["text"].astype(str).apply(lambda t: bool(ARABIC_RE.search(t)))]
```
The regex `[؀-ۿ]` matches any character in the Unicode Arabic block (U+0600–U+06FF). A row is kept only if it contains zero Arabic characters. This correctly separates Arabizi from Arabic-script Darija.

**Positive skew (65.7% pos):** YouTube comments on entertainment content tend to be positive (people comment more on content they like, and dislikes are less likely to be typed out).

---

## 7. Models — architecture, motivation, pros/cons

---

### xlm-t — `cardiffnlp/twitter-xlm-roberta-base-sentiment`

**Architecture:** XLM-RoBERTa base (355M parameters as base, 278M in this fine-tuned version) with a 3-class classification head.

**What is XLM-RoBERTa?**
RoBERTa is an optimised BERT-style transformer pre-trained on large English text. XLM-RoBERTa ("Cross-Lingual Language Model RoBERTa") extends this to 100 languages by pre-training on 2.5 terabytes of multilingual Common Crawl web text. It learns shared representations across languages, allowing knowledge to transfer between them. The `twitter` variant was further fine-tuned on 198 million tweets in 100+ languages, which is important because tweet language differs significantly from formal text (abbreviations, hashtags, emoji, code-switching).

**Labels (verified):** `{0: negative, 1: neutral, 2: positive}`

**Languages in benchmark:** ALL FOUR (darija_ar, francais, msa, arabizi)

| Pros | Cons |
|---|---|
| Single model for all languages | Not specialised for any one language |
| Strong multilingual transfer | Moroccan Darija is underrepresented in CC-100 |
| Twitter training → handles informal text | Tweet-style ≠ broadcast speech style |
| Well-maintained, widely used baseline | 278M params → slower than smaller models |
| Open, commercially usable licence | |

**Results:**
| Language | Macro-F1 |
|---|---|
| darija_ar | 0.723 |
| francais  | 0.772 |
| msa       | 0.813 |
| arabizi   | 0.762 |

**Interpretation:** Solid and consistent. MSA is strongest (Arabic Twitter is well-represented in its training data). French is lower than the specialist French model (0.772 vs 0.949 for distilcamembert) — the language-specialist wins when available. Arabizi at 0.762 is surprisingly decent given that Arabizi is not a standard language with defined training data.

---

### camelbert-da — `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`

**Architecture:** BERT-base with an Arabic-focused pre-training, fine-tuned for sentiment. CAMeL Lab (Computer, Arabic, Machine-Learning Lab, NYU Abu Dhabi) developed a family of Arabic BERT models trained on different Arabic registers.

**What is CAMeLBERT-DA?**
The "DA" stands for Dialectal Arabic. It was pre-trained on a corpus of dialectal Arabic text (tweets, forums, news comments) covering multiple Arab countries. The sentiment fine-tuning was done on top of this dialectal pre-training.

**Labels (verified):** `{0: positive, 1: negative, 2: neutral}` — 3-class (contrary to what the plan assumed)

**Languages in benchmark:** darija_ar, msa

| Pros | Cons |
|---|---|
| Trained on dialectal Arabic — closer to Darija than MSA models | Trained on many dialects, not specifically Moroccan |
| Only 109M params — faster than xlm-t | Does not handle Latin script (Arabizi) |
| Strong on MSA (0.924) | Weakest on Darija neutral class (recall 0.418) |
| Commercially usable licence | Egyptian/Gulf Darija ≠ Moroccan Darija |

**Lesson learned:** The plan assumed binary labels based on an older paper. The actual Hub model has 3 classes. Always verify with `id2label`.

**Results:**
| Language | Macro-F1 |
|---|---|
| darija_ar | 0.701 |
| msa       | 0.924 |

**Interpretation:** Excellent on MSA (0.924) — camelbert-da appears to have been trained on substantial MSA data alongside dialects. On Darija, the neutral class has poor recall (41.8% — misses most neutral tweets, classifying them as positive or negative). This is a known challenge: neutral Arabic text is hard to distinguish from slightly positive or slightly negative text, especially for models not specifically trained on Moroccan content.

---

### distilcamembert — `cmarkea/distilcamembert-base-sentiment`

**Architecture:** DistilCamemBERT with a 5-class classification head (1–5 stars). CamemBERT is a BERT-style model pre-trained specifically on French text (OSCAR corpus, 138GB of French). DistilCamemBERT is a distilled (smaller, faster) version: 68M parameters vs 110M for the base.

**What is distillation?**
Knowledge distillation trains a smaller "student" model to mimic the outputs of a larger "teacher" model. The student learns not just the correct labels but the full probability distribution over classes produced by the teacher. This soft supervision transfers more information than hard labels and results in a small model that performs nearly as well as the large one.

**Labels (verified):** `{0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}`

**Languages in benchmark:** francais only

| Pros | Cons |
|---|---|
| Highest F1 of all models (0.949) on French | French only |
| Smallest model (68M params) | Star ratings require remapping |
| Fast (420ms/1000 texts on CPU) | Trained on reviews — domain gap to broadcast speech |
| Commercially usable | 3→2-class mapping (3★→neu) may not suit all contexts |

**Results:**
| Language | Macro-F1 |
|---|---|
| francais | 0.949 |

**Interpretation:** Outstanding. The F1 advantage over xlm-t (+17.7 points: 0.949 vs 0.772) shows the value of language specialisation. The confusion matrix shows near-perfect separation: only 15 negative reviews classified as positive and 5 positive reviews classified as negative out of 1000. For a French-only deployment, this model should be the first choice by a wide margin.

---

### DarijaBERT — `SI2M-Lab/DarijaBERT`

**Architecture:** BERT-base pre-trained from scratch on Moroccan Darija text.

**What is SI2M Lab?**
SI2M (Information Systems and Intelligent Systems and Mathematical Modelling) is a research lab at Mohammed V University in Rabat, Morocco. They collected a large corpus of Moroccan Darija text (social media, forums, news comments in Arabic script) and pre-trained BERT on it.

**Why this is significant:** Standard BERT, RoBERTa, and even CAMeLBERT start from multilingual pre-training that treats Moroccan Darija as one dialect among many and underrepresents it. DarijaBERT was pre-trained *only* on Moroccan Darija, so its vocabulary and internal representations are specifically tuned to Moroccan Arabic word patterns.

**Fine-tune plan:** Add a 3-class sentiment head and train on MAC → evaluate on `darija_ar` test set.

| Pros | Cons |
|---|---|
| Only model pre-trained specifically on Moroccan Darija | Smaller pre-training corpus than multilingual models |
| Best vocabulary coverage for Moroccan lexicon | No Arabizi support (Arabic script only) |
| Expected to beat xlm-t on darija_ar | Results pending (Step 4) |

---

### DarijaBERT-arabizi — `SI2M-Lab/DarijaBERT-arabizi`

**Architecture:** Extension of DarijaBERT with vocabulary and pre-training extended to Latin-script Moroccan Arabizi.

**Why it exists:** The same SI2M lab recognised that a large portion of Moroccan digital communication is in Arabizi. Standard models trained on Arabic or Latin-script languages cannot handle the digit substitutions (3/7/9) or the mixed Latin-Arabic vocabulary. DarijaBERT-arabizi was pre-trained on a corpus of Arabizi text collected from Moroccan social media.

**Fine-tune plan:** Add a 3-class sentiment head and train on MYC-Latin → evaluate on `arabizi` test set.

| Pros | Cons |
|---|---|
| Only model pre-trained specifically on Moroccan Arabizi | Very small pre-training corpus (Arabizi is sparse) |
| Handles digit substitutions natively | No Arabic-script support |
| Expected to be the best Arabizi model | Results pending (Step 4) |

---

### MARBERTv2 — `UBC-NLP/MARBERTv2`

**Architecture:** Multi-Arabic BERT v2, pre-trained on 1 billion Arabic words covering both MSA and 25 Arabic dialects.

**What is MARBERT?**
Developed by the UBC Natural Language Processing group at the University of British Columbia. The "Multi" in MARBERT refers to covering multiple Arabic dialects. The training corpus was assembled specifically to be dialect-inclusive: Egyptian, Gulf, Levantine, Maghrebi (including Moroccan), and others.

**Fine-tune plan:** Add a 3-class sentiment head and train on MAC → evaluate on both `darija_ar` and `msa`.

⚠️ **RESEARCH-ONLY licence.** MARBERTv2 is released under a non-commercial research licence. It cannot be used in production without permission from UBC-NLP. This restriction is documented in the benchmark grid.

| Pros | Cons |
|---|---|
| Large diverse Arabic pre-training corpus | RESEARCH-ONLY — not for commercial use |
| Dialect-inclusive, including Maghrebi | No Arabizi support |
| Strong baseline in Arabic NLP literature | Licence blocks production deployment |
| Results pending (Step 4) | |

---

### QARIB — `qarib/bert-base-qarib`

**Architecture:** BERT-base pre-trained on 420 million Gulf and MSA Arabic words.

**Why chosen:** Bourahouat et al. (2024) — the literature most closely related to this project — benchmarks QARIB as a strong Arabic sentiment baseline and notes its good generalisation across Arabic dialects. This makes it a useful comparison point for situating our results in the literature.

**Fine-tune plan:** Add a 3-class sentiment head and train on MAC (and/or ASTD) → evaluate on `darija_ar` and `msa`.

| Pros | Cons |
|---|---|
| Strong performance in recent Arabic NLP papers | Gulf Arabic focus — less Moroccan |
| Open licence | Smaller corpus than MARBERTv2 |
| Good for MSA | Results pending (Step 4) |

---

### Atlas-Chat — `MBZUAI-Paris/Atlas-Chat-2B` (and 9B)

**Architecture:** Gemma-2B/9B large language model (decoder-only transformer) fine-tuned by the MBZUAI-Paris research centre specifically on Moroccan Darija conversational data.

**What is a decoder-only LLM vs an encoder?**
All other models in this benchmark are **encoders**: they read the full input text bidirectionally and produce a fixed-size representation that feeds into a classifier. They are fast and efficient but can only do classification.

Atlas-Chat is a **decoder**: it generates text token-by-token, conditioned on the input prompt. It is much larger (2 billion or 9 billion parameters vs ~100–280M for encoders) and can do any task expressible in language — but it is far slower and more expensive per inference.

**GATED model:** Atlas-Chat requires accepting the Gemma licence agreement on HuggingFace before downloading. `huggingface-cli login` is required.

**4-bit quantisation:** Required to fit on a 16GB T4. Without quantisation, Atlas-Chat-2B would require ~8GB in fp16 — already tight with inference overhead. 4-bit brings it to ~1GB.

| Pros | Cons |
|---|---|
| Zero-shot — no training data needed | GATED — requires licence acceptance |
| Natively trained on Moroccan Darija | 2–9B params: expensive to run |
| Can explain predictions in natural language | 4-bit quantisation reduces accuracy |
| Handles code-switching naturally | Non-response rate (parsing failures) |
| Flexible — prompt can be changed | Latency: ~5-20x slower than encoders |
| | Gated access may change or be revoked |

**Decision threshold:** Per the plan, if the fine-tuned encoder achieves macro-F1 ≥ 0.75 on Darija, or if the LLM-encoder F1 gap is less than 5 points, we should prefer the encoder (lower cost, same accuracy, no gating). This threshold will be applied once both Step 4 and Step 5 are complete.

---

## 8. Step-by-step execution log

---

### Step 1 — Environment and smoke test

**Goal:** Verify the environment is correctly set up.

**Environment discovered:**
```
Python  : 3.14.3
PyTorch : 2.12.0+cu130
CUDA    : not available on this machine (CPU-only dev environment)
Transformers: 5.12.0
```

**Bugs encountered and fixed:**
1. `return_all_scores=True` is deprecated in Transformers 5.x → replaced with `top_k=None`
2. The pipeline with `top_k=None` returns a flat list of dicts for a single input but a list of lists for a batch. Added handling for both shapes.
3. `id2label` uses lowercase (`negative/neutral/positive`), not title-case (`Negative`) as in the documentation. Updated `XLM_T_MAP` accordingly.

**Smoke test results:**
```
[French]  Ce film est absolument magnifique → positive (0.952) ✓
[MSA]     هذا الفيلم رائع جداً             → positive (0.950) ✓
[Arabizi] hada film zwine bzaf, 7bito ktir → neutral  (0.342) (low confidence — expected)
```

---

### Step 2 — Building the frozen test sets

**Goal:** Download the three manual datasets, build four 1000-row test CSVs, freeze them.

**Datasets downloaded automatically via GitHub CLI:**
```bash
gh api repos/LeMGarouani/MAC/contents/MAC%20corpus.csv --jq '.download_url' | xargs curl -sL -o ...
gh api repos/mahmoudnabil/ASTD/contents/data/Tweets.txt ...
gh api repos/MouadJb/MYC/contents/DATA_CLEANED.csv ...
```

**Allociné:** loaded via HuggingFace `load_dataset("tblard/allocine", split="test")`.

**Bugs encountered and fixed:**
1. **MAC column names:** `type` is the label (not `class`). `class` is dialect metadata.
2. **ASTD format:** `text\tlabel` (text first), not `label\ttext` as initially assumed.
3. **ASTD labels:** short codes `POS/NEG/OBJ/MIX` in uppercase, not full words.
4. **MYC encoding:** UTF-16, not UTF-8.
5. **MYC columns:** `sentence` (not `text`), `polarity` (not `label`), integer values `1`/`-1`.
6. **Allociné ID:** `"allocine"` no longer resolves; use `"tblard/allocine"`.

**Final hashes and distributions:**
| Language | n | sha256 (first 16 hex chars) | Distribution |
|---|---|---|---|
| darija_ar | 1000 | `82ac82925422afc8` | pos=567, neu=232, neg=201 |
| francais  | 1000 | `3cd37b448dc7e8db` | neg=520, pos=480 |
| msa       | 1000 | `de6f6753e70ffc45` | neg=679, pos=321 |
| arabizi   | 1000 | `857ef2614b95943b` | pos=657, neg=343 |

**All files are now read-only (chmod 444). They must never be regenerated.**

---

### Step 3 — Ready-made model evaluation

**Goal:** Run the three ready-made models and populate `results/`.

**Bugs encountered and fixed:**
1. **Sequence too long:** Allociné reviews exceed 512 tokens. Added `max_length=512` to pipeline calls.
2. **camelbert-da label error:** Model has a native `neutral` class not in our initial `CAMELBERT_DA_MAP`. Updated the map and removed the threshold heuristic.

**All 7 result files produced:**
```
results/xlm-t_darija_ar.json       results/xlm-t_francais.json
results/xlm-t_msa.json             results/xlm-t_arabizi.json
results/camelbert-da_darija_ar.json results/camelbert-da_msa.json
results/distilcamembert_francais.json
```

---

## 9. Results analysis

### Full results table (Steps 1–3 complete)

| Model | darija\_ar | francais | msa | arabizi | Params |
|---|---|---|---|---|---|
| xlm-t | 0.723 | 0.772 | 0.813 | 0.762 | 278M |
| camelbert-da | 0.701 | — | **0.924** | — | 109M |
| distilcamembert | — | **0.949** | — | — | 68M |

### Per-class breakdown

**xlm-t on darija_ar** (3-class):
```
neg: P=0.635, R=0.841, F1=0.724
neu: P=0.652, R=0.517, F1=0.577   ← neutral is hardest
pos: P=0.882, R=0.855, F1=0.868
macro-F1 = 0.723
```
Positive is well-learned (F1=0.868). Neutral has the weakest recall (51.7% — nearly half of neutral tweets are classified as something else). This is consistent across all Arabic models: "neutral" Arabic content is hard to distinguish from slightly positive or slightly negative.

**camelbert-da on darija_ar** (3-class):
```
neg: P=0.581, R=0.905, F1=0.708
neu: P=0.708, R=0.418, F1=0.526   ← neutral recall even worse
pos: P=0.882, R=0.855, F1=0.868
macro-F1 = 0.701
```
camelbert-da has high negative recall (90.5%) — it catches almost all negative texts, but at the cost of many false positives (precision 0.581). Many neutral tweets are predicted as negative.

**camelbert-da on msa** (binary, no neutral):
```
neg: P=0.979, R=0.895, F1=0.935
pos: P=0.938, R=0.888, F1=0.912
macro-F1 = 0.924
```
Excellent. The absence of the neutral class removes the hardest part of the classification. MSA Twitter sentiment (positive vs negative) is a clean binary problem for a model trained on Arabic.

**distilcamembert on francais** (binary):
```
neg: P=0.989, R=0.898, F1=0.942
pos: P=0.968, R=0.944, F1=0.956
macro-F1 = 0.949
```
Near-perfect. Only 20 errors out of 1000. The high precision on neg (0.989) means almost everything predicted as negative truly is negative. This model is clearly the right choice for French.

**xlm-t on arabizi** (binary):
```
neg: P=0.921, R=0.676, F1=0.780
pos: P=0.933, R=0.619, F1=0.745
macro-F1 = 0.762
```
Recall is low for both classes (67.6% neg, 61.9% pos) despite high precision. This suggests the model often abstains to "neutral" even though there are no neutral examples in this test set — it is hedging rather than committing. The fine-tuned Arabizi model (Step 4) should correct this.

### Key takeaways

1. **Language specialists dominate**: distilcamembert (+17.7 F1 over xlm-t on French) and camelbert-da (+11.1 F1 over xlm-t on MSA) show that language-specific pre-training is worth it when a good specialist exists.

2. **Neutral is universally hard**: Across all models and languages, the neutral class has the lowest F1. Neutral content often lacks strong sentiment signals and borders on both positive and negative.

3. **Binary is easier than 3-class**: MSA and French (binary) show much higher F1 than Darija (3-class), partly because removing the neutral class eliminates the hardest confusions.

4. **Arabizi is the frontier**: No model was specifically designed for Arabizi. xlm-t's 0.762 is the current baseline and the fine-tuned DarijaBERT-arabizi (Step 4) is expected to improve on it.

5. **Model size ≠ performance**: distilcamembert (68M) outperforms xlm-t (278M) on French by a wide margin. Specialist knowledge beats raw scale for well-resourced languages.

---

## 10. Key concepts glossary

**Macro-F1:** The average F1 score computed independently for each class and then averaged, giving equal weight to each class regardless of frequency. Contrast with micro-F1 (weighted by class frequency). Macro-F1 penalises models that ignore minority classes.

**F1 score:** The harmonic mean of precision and recall: `F1 = 2 × (P × R) / (P + R)`. Balances both measures.

**Precision:** Of all the times the model predicted class X, what fraction were actually class X? High precision = few false positives.

**Recall:** Of all the true examples of class X, what fraction did the model correctly find? High recall = few false negatives.

**Confusion matrix:** A square table where row = true class, column = predicted class. Diagonal cells are correct predictions. Off-diagonal cells are errors.

**Transformer:** The neural network architecture underlying all modern NLP models. Uses self-attention to model relationships between all pairs of words in a sentence simultaneously, regardless of distance. Both BERT-style (encoder) and GPT-style (decoder) models are transformers.

**BERT / RoBERTa:** Encoder transformer models pre-trained by predicting masked words (BERT) or through improved training recipes (RoBERTa). The pre-trained model produces contextual word representations; a classification head on top predicts labels.

**Fine-tuning:** Continuing training of a pre-trained model on a smaller, task-specific labelled dataset. The model's pre-learned language knowledge is preserved and adapted to the target task.

**fp16:** 16-bit floating-point numbers (half precision). Use half the memory and run faster on modern GPUs compared to fp32 (standard 32-bit), with minimal accuracy loss for most tasks.

**Quantisation (4-bit):** Compressing model weights from 32-bit floats to 4-bit integers to fit large models in GPU memory. The BitsAndBytes library implements NF4 (NormalFloat4), which minimises quantisation error.

**Transfer learning:** Using a model trained on one task as a starting point for a different task. In NLP, pre-trained language models are the standard starting point for virtually all downstream tasks.

**SRT:** SubRip Subtitle format. A plain text file with sequentially numbered cues, each containing a time range and subtitle text.

**id2label:** A dictionary in a HuggingFace model's configuration that maps integer output indices (0, 1, 2...) to human-readable label names. Always inspect this before interpreting model outputs.

**Arabizi:** Informal romanisation of Arabic using Latin characters and digit substitutions for sounds not in the Latin alphabet. Widely used in North African social media and SMS.

---

## 11. What comes next

### Step 4 — Fine-tuning (pending)

Run these on a GPU (Kaggle T4 or Google Colab):
```bash
PYTHONPATH=src python src/finetune.py --model darijabert
PYTHONPATH=src python src/finetune.py --model darijabert-arabizi
PYTHONPATH=src python src/finetune.py --model marbertv2
PYTHONPATH=src python src/finetune.py --model qarib
```
Each run takes ~15–30 minutes of GPU time. Save checkpoints immediately to Drive or Kaggle output after each run. Log GPU time for the cost metric.

### Step 5 — Atlas-Chat zero-shot (pending, GATED)

Requires `huggingface-cli login` with a Gemma licence-accepted account.
```bash
PYTHONPATH=src python src/atlas_chat.py --model 2b
```
Measures non-response rate, latency, and macro-F1 on `darija_ar` and `arabizi`.

### Step 6 — Domaine réel test set (pending)

You will provide actual SRT files from Moroccan broadcast media. We will:
1. Parse with `srt_utils.py`
2. Route to the right model by language
3. Annotate ~200 utterances manually (with written annotation rules)
4. Evaluate models on this real-world set

### Step 7 — Plots, weighted grid, final report

Once all models are evaluated:
```bash
PYTHONPATH=src python src/aggregate.py
```
Produces:
- `results/figs/heatmap_f1.png` — F1 heatmap (model × language)
- `results/figs/scatter_cost_precision.png` — latency vs F1
- `results/figs/radar_finalists.png` — radar chart of top models
- `results/weighted_grid.csv` — final ranked table
- Written report in `report/` with per-scenario recommendations
