# Fine-tuning — Complete Guide

**What this document covers:** Why we fine-tune, what models we fine-tune, on which datasets, how the training process works, and a cell-by-cell walkthrough of the Kaggle notebook.

---

## Table of Contents

1. [The core idea — why fine-tune at all?](#1-the-core-idea--why-fine-tune-at-all)
2. [Transfer learning explained](#2-transfer-learning-explained)
3. [What happens inside a BERT model](#3-what-happens-inside-a-bert-model)
4. [The four models we fine-tune](#4-the-four-models-we-fine-tune)
5. [The training datasets](#5-the-training-datasets)
6. [Training configuration — every parameter explained](#6-training-configuration--every-parameter-explained)
7. [The LOAD REPORT — UNEXPECTED and MISSING keys](#7-the-load-report--unexpected-and-missing-keys)
8. [Kaggle notebook — cell by cell](#8-kaggle-notebook--cell-by-cell)
9. [What to watch during training](#9-what-to-watch-during-training)
10. [How to use the checkpoint after training](#10-how-to-use-the-checkpoint-after-training)

---

## 1. The core idea — why fine-tune at all?

In Step 3 we ran three "ready-made" models — models that someone else had already trained for sentiment analysis and published online. They worked, but with limitations:

| Ready-made model | Problem |
|---|---|
| xlm-t | Generic multilingual — not specialised for Moroccan Darija |
| camelbert-da | Trained on multi-dialect Arabic — not specifically Moroccan |
| distilcamembert | French only — perfect for French, useless for everything else |

For Moroccan Darija and Arabizi specifically, there are **no ready-made sentiment models**. Nobody has published a model already trained to classify sentiment in Moroccan Arabic or Moroccan Arabizi. We have to build those ourselves.

Fine-tuning is how we build them: we take a model that already understands the Moroccan Darija **language** (its vocabulary, grammar, word associations) and we teach it the additional task of **classifying sentiment**.

---

## 2. Transfer learning explained

The concept that makes fine-tuning possible is called **transfer learning**. It works like this:

Imagine you want to teach someone to be a Moroccan Arabic movie critic. You have two options:

**Option A — Train from scratch:** Find someone who has never heard any language before (a newborn). Teach them Moroccan Arabic from scratch (years of exposure), then teach them about movies, then teach them what positive and negative reviews look like. This requires enormous time and resources.

**Option B — Transfer learning / Fine-tuning:** Find someone who already speaks Moroccan Arabic fluently (DarijaBERT, which was pre-trained on millions of Darija texts). They already know vocabulary, grammar, and how words relate to each other. You only need to teach them the sentiment task — give them 15,000 labelled examples and a few hours of study. Done.

Option B is obviously faster and requires far less data. This is why almost all practical NLP today uses pre-trained models fine-tuned for specific tasks.

**The transfer:** What transfers is the model's internal "understanding" of the language — encoded in hundreds of millions of numerical weights (parameters). These weights are the output of pre-training on large corpora. When we fine-tune, we start with those weights and nudge them slightly to also encode sentiment information.

---

## 3. What happens inside a BERT model

Before explaining fine-tuning in detail, you need a mental model of what BERT actually does.

### Pre-training: Masked Language Modelling

DarijaBERT was pre-trained using a task called **Masked Language Modelling (MLM)**. The idea:

1. Take a sentence: `هذا الفيلم رائع جداً`
2. Randomly hide 15% of the words: `هذا [MASK] رائع جداً`
3. Ask the model: what word was hidden?
4. The model guesses `الفيلم`
5. If wrong, adjust weights slightly. Repeat millions of times.

After training on millions of sentences this way, the model has learned rich representations of what words mean and how they relate to each other — because you can only predict masked words well if you understand context.

### The architecture: Layers and hidden states

BERT is a stack of **transformer layers** (12 layers for BERT-base). Each layer takes the representations from the previous layer and refines them. The final layer produces a vector (a list of 768 numbers) for each word in the input. These vectors encode the meaning of each word **in context** (the same word gets a different vector depending on the surrounding words).

For classification, we use a special `[CLS]` token that is prepended to every input. The 768-number vector at the `[CLS]` position in the final layer is treated as a summary of the entire input sentence. This is the "sentence embedding".

### The classification head

A classification head is a simple linear layer that sits on top of the pre-trained BERT:

```
Input text
    ↓
Tokenizer (text → integer IDs)
    ↓
BERT (12 transformer layers)
    ↓
[CLS] vector (768 numbers)
    ↓
Linear layer: 768 → 3 (one score per class)
    ↓
Softmax (convert scores to probabilities)
    ↓
Output: [neg=0.1, neu=0.2, pos=0.7] → predict "pos"
```

The **pre-trained weights** are everything up to and including the `[CLS]` vector. The **new weights** we add are the linear layer (768×3 = 2304 numbers) plus 3 bias terms. These new weights start randomly initialised and are trained from the labelled data.

### Why only 3 epochs and not 30?

During fine-tuning, we update ALL weights — both the pre-trained BERT weights and the new classifier weights. The pre-trained weights already encode language knowledge. If we update them too aggressively (too many epochs, too high learning rate), we **overwrite** that knowledge with random noise from our small dataset. This is called **catastrophic forgetting**.

3 epochs with lr=2e-5 is a well-established recipe for BERT fine-tuning that gives the classifier time to learn the task without destroying the pre-trained representations.

---

## 4. The four models we fine-tune

### DarijaBERT — `SI2M-Lab/DarijaBERT`

**What it is:** BERT-base (12 layers, 768 hidden size, 110M parameters) pre-trained from scratch on Moroccan Darija text in Arabic script. Built by the SI2M laboratory at Mohammed V University in Rabat, Morocco.

**Pre-training data:** A large corpus of Moroccan Darija text scraped from social media (Facebook, Twitter/X, YouTube comments) and Moroccan news websites, all in Arabic script. Hundreds of millions of tokens.

**Why we fine-tune it for this task:**
- It is the **only model specifically pre-trained on Moroccan Darija** Arabic script
- Its vocabulary (tokeniser) was built from Moroccan Darija text, so common Darija words are represented as single tokens rather than being broken into meaningless sub-pieces
- A model that understands the language well at pre-training will learn the sentiment task faster and with less data
- Expected to outperform the multilingual xlm-t (0.723) on Darija because it has native Moroccan vocabulary coverage

**What we train it on:** MAC dataset (17,444 rows after dropping mixed — training split after excluding the frozen 1000-row test set)

**What we evaluate it on:** `darija_ar` test set (the 1000 frozen rows from MAC)

**Licence:** Open, commercially usable.

---

### DarijaBERT-arabizi — `SI2M-Lab/DarijaBERT-arabizi`

**What it is:** An extension of DarijaBERT with a vocabulary and pre-training corpus that includes Latin-script Moroccan Arabizi (digits 3/7/9 substitutions). Also from the SI2M lab.

**Pre-training data:** A mixture of the original DarijaBERT corpus plus Arabizi text collected from Moroccan social media. Arabizi is rarer than Arabic-script Darija on the web, so the corpus is smaller.

**Why we fine-tune it for this task:**
- There is **no existing Arabizi sentiment model anywhere** — this is the frontier
- Arabizi is not a standard language with a fixed orthography. A general multilingual model sees words like "zwine" or "7bito" as nonsense. DarijaBERT-arabizi's vocabulary was built from actual Arabizi text, so it recognises common patterns
- The current best result on Arabizi is xlm-t at 0.762. A specialist model should do better

**What we train it on:** MYC-Latin dataset (4,339 Latin-script rows after excluding the frozen test set)

**What we evaluate it on:** `arabizi` test set (the 1000 frozen Latin-script rows from MYC)

**Licence:** Open, commercially usable.

---

### MARBERTv2 — `UBC-NLP/MARBERTv2`

**What it is:** Multi-dialect Arabic BERT, version 2. Pre-trained by the Natural Language Processing group at the University of British Columbia on **1 billion Arabic words** covering 25 Arab dialects including Moroccan (Maghrebi), plus MSA. Significantly larger pre-training corpus than DarijaBERT.

**Pre-training data:** 1 billion tokens assembled from Twitter, web-crawled dialect content, and MSA sources across 25 dialects. The Maghrebi/Moroccan component is explicitly included.

**Why we fine-tune it for this task:**
- The largest and most diverse Arabic dialect pre-training corpus available openly
- The 2024 Arabic NLP benchmark paper (Bourahouat et al., which this benchmark references) shows MARBERTv2 as a strong multi-dialect baseline
- It covers both Moroccan Darija AND MSA, so fine-tuning on MAC lets us evaluate on both `darija_ar` and `msa`
- Provides a comparison point: does a large diverse Arabic corpus beat a smaller Morocco-specific one?

**What we train it on:** MAC dataset (same as DarijaBERT)

**What we evaluate it on:** `darija_ar` AND `msa` test sets

**⚠️ Licence: RESEARCH-ONLY.** MARBERTv2 is released under a non-commercial research licence from UBC-NLP. It cannot be deployed in production without explicit permission from the authors. This must be recorded in the final benchmark grid and the report.

---

### QARIB — `qarib/bert-base-qarib`

**What it is:** BERT-base pre-trained on 420 million words of Gulf Arabic and MSA. QARIB stands for "Question Answering for Arabic Is Becoming Better" — it was originally developed for Arabic question answering but has since been used as a general Arabic encoder baseline.

**Pre-training data:** Gulf Arabic social media and MSA news articles, 420M tokens. Less diverse than MARBERTv2 but strong on MSA.

**Why we fine-tune it for this task:**
- Bourahouat et al. (2024) — the literature most directly relevant to this project — uses QARIB as a baseline and reports competitive results
- Including it lets us replicate (or challenge) their results and situate our benchmark in the literature
- It provides a third Arabic model to compare against DarijaBERT (Morocco-specific) and MARBERTv2 (broad dialectal), filling out the Arabic model landscape

**What we train it on:** MAC dataset

**What we evaluate it on:** `darija_ar` AND `msa` test sets

**Licence:** Open, commercially usable.

---

## 5. The training datasets

### MAC — for DarijaBERT, MARBERTv2, QARIB

**Why MAC for Darija training?**
MAC (Moroccan Arabic Corpus) is the only large open dataset of Moroccan Darija sentiment labels. There is no alternative. All three Arabic-script models train on it. The 17,444 labelled tweets (after dropping 643 mixed-sentiment rows) provide enough signal for 3-epoch fine-tuning of a BERT-base model.

**Train/test split:**
- The 1,000 frozen test rows are permanently excluded from training
- The remaining ~16,444 rows are split 90/10 into train and validation sets (seed 42)
- Training set: ~14,800 rows
- Validation set: ~1,644 rows (used to select the best checkpoint)

**Label distribution in training data:**
The raw MAC is ~56% positive, ~22% neutral, ~18% negative. This imbalance is preserved in the train/val split (stratified split). The model will be somewhat biased toward predicting positive — this is expected and is why we use macro-F1 (which penalises ignoring minority classes) rather than accuracy.

### MYC-Latin — for DarijaBERT-arabizi

**Why MYC for Arabizi training?**
MYC (Moroccan YouTube Comments) is the only open Arabizi-specific dataset with sentiment labels. The Latin-script subset (5,339 rows) is what we use. This is a smaller training set than MAC, which means the DarijaBERT-arabizi model may not converge as strongly — but it is the only available data.

**Note on binary labels:**
MYC only has positive/negative labels — no neutral. This means DarijaBERT-arabizi is fine-tuned with only 2 classes (`neg`, `pos`), but we gave `finetune.py` a 3-class head (`LABEL2ID = {neg: 0, neu: 1, pos: 2}`). In practice, since the training data has no neutral examples, the model will rarely or never predict neutral. We evaluate on the arabizi test set which is also binary, so this is consistent.

**Train/test split:**
- 1,000 frozen test rows excluded
- Remaining ~4,339 rows split 90/10 into train/val
- Training set: ~3,905 rows
- Validation set: ~434 rows

---

## 6. Training configuration — every parameter explained

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    seed=42,
)
```

### `num_train_epochs=3`
One **epoch** = one complete pass through the entire training dataset. With 3 epochs, the model sees every training example 3 times. This is the standard for BERT fine-tuning.

Too few epochs → the model hasn't learned enough from the data (underfitting).
Too many epochs → the model memorises the training examples and stops generalising (overfitting).

3 is the empirically validated sweet spot for BERT-scale models on typical NLP tasks.

### `per_device_train_batch_size=16`
A **batch** is a group of training examples processed together in one forward + backward pass. Size 16 means the model sees 16 tweets simultaneously, averages the loss over them, and updates weights once.

**Why not batch 1?** Processing one example at a time is noisy — a single tweet might be unusual and push weights in the wrong direction. Averaging over 16 gives a more stable gradient signal.

**Why not batch 128?** On a T4 GPU with 16GB VRAM, a BERT-base model in fp16 with max_length=128 fits comfortably at batch 16. Larger batches would require more VRAM. Also, very large batches have been shown to sometimes hurt generalisation.

`per_device_eval_batch_size=32` can be larger than train batch because evaluation doesn't need to store gradients (saves memory).

### `learning_rate=2e-5`
The learning rate controls how much the weights change after each batch. `2e-5` means 0.00002.

This is very small — deliberately. The pre-trained weights are already good. We want to nudge them slightly toward the sentiment task without overwriting their language knowledge. Standard BERT fine-tuning recommendations (from the original BERT paper) suggest 1e-5 to 5e-5. We use 2e-5.

If you set lr=0.01 (100× larger), the model would be catastrophically updated in the first few batches and lose all pre-trained knowledge.

### `fp16=True`
**fp16 = 16-bit floating point (half precision).**

Normally, neural network weights are stored as 32-bit floating point numbers (fp32). A 110M-parameter model in fp32 uses 110M × 4 bytes = ~440MB just for weights. Activations and gradients during training add several times more.

fp16 stores each number in 2 bytes instead of 4, halving memory usage. The T4 also has dedicated hardware for fp16 operations that runs roughly 2× faster than fp32. The small precision loss is negligible for fine-tuning (the pre-trained weights are already approximate).

Without fp16, training would be 2× slower and potentially run out of VRAM at batch 16.

### `eval_strategy="epoch"` and `save_strategy="epoch"`
After each of the 3 epochs, the model is evaluated on the validation set and a checkpoint is saved to disk. This gives us 3 checkpoints: epoch-1, epoch-2, epoch-3.

### `load_best_model_at_end=True` and `metric_for_best_model="macro_f1"`
After all 3 epochs complete, the Trainer automatically loads whichever checkpoint had the **best validation macro-F1** — not necessarily the final epoch. This is important because:

- Epoch 1 might be the best if the model starts overfitting after that
- Or epoch 3 might be best if performance steadily improves
- Without this, we'd always take the last epoch even if an earlier one was better

### `logging_steps=50`
Print the training loss to the console every 50 batches. Lets you watch progress and check that the loss is decreasing.

### `seed=42`
Seeds the Trainer's internal randomness (shuffling training data between epochs, etc.) for reproducibility.

---

## 7. The LOAD REPORT — UNEXPECTED and MISSING keys

When you load a pre-trained model for a different task, HuggingFace prints this report. You saw it for DarijaBERT:

```
Key                                        | Status
cls.predictions.decoder.bias               | UNEXPECTED
cls.predictions.decoder.weight             | UNEXPECTED
cls.predictions.transform.LayerNorm.weight | UNEXPECTED
...
classifier.weight                          | MISSING
bert.pooler.dense.bias                     | MISSING
bert.pooler.dense.weight                   | MISSING
classifier.bias                            | MISSING
```

### UNEXPECTED keys — safe to ignore

`cls.predictions.*` is the **Masked Language Modelling head** — the layer DarijaBERT used during pre-training to predict masked words. It looks like this at the end of the network:

```
[CLS] vector (768 numbers)
    ↓
MLM Transform (dense layer + LayerNorm)
    ↓
MLM Decoder (768 → vocab_size, ~32,000 classes)
    ↓
"What masked word was here?"
```

We're loading DarijaBERT into a `BertForSequenceClassification` model, which has no MLM head. So these weights from the checkpoint have "nowhere to go" — they're unexpected. HuggingFace simply discards them. This is exactly the right behaviour: we don't want the MLM head, we're replacing it with a classification head.

**Think of it like this:** You hired someone who worked as a cook (MLM head). You're giving them a new job as a sommelier (classifier head). Their cooking skills are irrelevant to the new role and you don't take them into account.

### MISSING keys — the new weights we train

`classifier.weight`, `classifier.bias`: The 3-class linear layer (768 inputs → 3 outputs). This is the NEW layer added on top of BERT for our sentiment task. It doesn't exist in the original DarijaBERT checkpoint (which was only pre-trained for MLM, never fine-tuned for classification). These weights are **randomly initialised** by PyTorch.

`bert.pooler.dense.weight`, `bert.pooler.dense.bias`: The pooler is a small dense layer that projects the `[CLS]` vector into a 768-dimensional space with a `tanh` activation. It is used by the classifier head. It was technically present in the original BERT architecture but not used during DarijaBERT's MLM pre-training, so it exists in the checkpoint in an untrained state. It's marked MISSING in some model variants.

**These are the weights that fine-tuning will train.** They start random and will be updated to learn sentiment.

### The `ignore_mismatched_sizes=True` parameter

In `finetune.py`:
```python
model = AutoModelForSequenceClassification.from_pretrained(
    hub_id,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)
```

Without `ignore_mismatched_sizes=True`, HuggingFace would raise an exception when it finds unexpected or missing keys. This flag tells it: "I know the keys don't match, I'm intentionally changing the task — load what you can and initialise the rest randomly."

---

## 8. Kaggle notebook — cell by cell

### Cell 1 — Install extra dependencies

```python
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "bitsandbytes>=0.43.0", "pysrt>=1.1.2"])
```

**Why `subprocess.check_call` instead of `!pip install`?**
Kaggle notebooks support the `!pip install` shell shortcut, but using `subprocess.check_call([sys.executable, "-m", "pip", ...])` is more reliable: it guarantees the package is installed into the same Python interpreter that's running the notebook (sometimes they differ in environment setups).

**`bitsandbytes`:** The library that enables 4-bit quantisation (used later for Atlas-Chat). Not in Kaggle's default image.

**`pysrt`:** SRT file parser for the real-input pipeline (used later in Step 6). Not in Kaggle's default image.

**The RAPIDS conflict warnings:** Kaggle's image includes `dask-cuda`, `cuml`, `cudf` (GPU-accelerated data science libraries from NVIDIA). These have pinned dependencies on specific `numba` versions. When we install `bitsandbytes`, it pulls a newer `numba` version that conflicts. This does NOT affect our code — we never use RAPIDS. The warning is pip's way of saying "I installed what you asked for, but your environment has other packages with conflicting version requirements." Always harmless for us.

---

### Cell 2 — Clone the repo

```python
if not os.path.isdir(REPO):
    subprocess.check_call(["git", "clone", GITHUB_REPO, REPO])
else:
    subprocess.check_call(["git", "-C", REPO, "pull"])
```

Clones the GitHub repo (or pulls updates if already cloned) into `/kaggle/working/benchmark`. Then:

```python
os.chdir(REPO)
sys.path.insert(0, f"{REPO}/src")
```

`os.chdir(REPO)` changes the working directory to the repo root, so relative paths in our scripts resolve correctly. `sys.path.insert(0, ...)` adds `src/` to Python's module search path, allowing `import finetune`, `import harness`, etc. to find the files.

**Why `sys.path.insert(0, ...)` and not `append`?**
`insert(0, ...)` puts our `src/` at the START of the search path, so it takes priority over any installed packages with the same name. If Kaggle happened to have a package called `utils` installed, `insert(0, ...)` ensures Python finds our `src/utils.py` first.

---

### Cell 3 — Download raw datasets

```python
r = requests.get(url, timeout=60)
r.raise_for_status()
with open(dest, "wb") as f:
    f.write(r.content)
```

Downloads MAC, ASTD, and MYC directly from GitHub's raw content CDN. The files are not in the repo (because `data/raw/` is gitignored — raw data can be 100MB+), so we re-download them on Kaggle.

`r.raise_for_status()` immediately raises an exception if the HTTP response is an error (404, 403, 500, etc.) rather than silently writing a zero-byte file or an HTML error page.

The `if not os.path.exists(dest)` check means if you restart the kernel without resetting the environment, the files don't get re-downloaded (saves time and bandwidth on re-runs).

---

### Cell 4 — Verify GPU and seeds

```python
import torch
from src.utils import set_seeds
set_seeds()

print(f"CUDA available : {torch.cuda.is_available()}")
```

`set_seeds()` locks random seeds to 42 in Python, NumPy, and PyTorch. This must happen early, before any data loading or model initialisation, otherwise some operations before the seed call will be non-reproducible.

The GPU check prints the GPU name and VRAM to confirm Kaggle assigned a GPU. If it prints `CUDA available: False`, you need to go to Settings → Accelerator → GPU T4.

---

### Cell 5 — Verify frozen test sets

```python
h = hashlib.sha256(open(path, "rb").read()).hexdigest()
```

Reads each frozen test CSV and computes its SHA-256 hash. Cross-check these against the hashes printed during Step 2 locally:

| Language | Expected sha256 (first 16) |
|---|---|
| darija_ar | `82ac82925422afc8` |
| francais  | `3cd37b448dc7e8db` |
| msa       | `de6f6753e70ffc45` |
| arabizi   | `857ef2614b95943b` |

If a hash doesn't match, the test set file was corrupted or accidentally modified. Stop and investigate before training — evaluating on corrupted test data would give meaningless results.

---

### Cell 6 — Choose the model

```python
MODEL_KEY = "darijabert"
```

This is the only thing you change between sessions. The valid values and what they do:

| `MODEL_KEY` | Pre-trained model | Train data | Eval languages |
|---|---|---|---|
| `"darijabert"` | SI2M-Lab/DarijaBERT | MAC | darija_ar |
| `"darijabert-arabizi"` | SI2M-Lab/DarijaBERT-arabizi | MYC-Latin | arabizi |
| `"marbertv2"` | UBC-NLP/MARBERTv2 | MAC | darija_ar, msa |
| `"qarib"` | qarib/bert-base-qarib | MAC | darija_ar, msa |

---

### Cell 7 — Run fine-tuning

```python
from src.finetune import finetune
finetune(MODEL_KEY)
```

This single call triggers the entire training pipeline. Here is what happens inside `finetune()` in order:

**Step A — Load the tokenizer**
```python
tokenizer = AutoTokenizer.from_pretrained(hub_id)
```
The tokenizer converts raw text into integer token IDs that the model understands. Each model has its own tokenizer trained on its own vocabulary. For DarijaBERT, the vocabulary was built from Moroccan Darija text, so Darija words are tokenised efficiently.

**Step B — Build train/val splits**
```python
train_df, val_df = load_train_split(train_lang)
```
Loads the raw dataset (MAC or MYC), removes the frozen test rows, and splits 90/10 into training and validation sets (stratified, seed 42).

**Step C — Tokenize and encode**
```python
train_ds = encode(train_df, tokenizer)
val_ds   = encode(val_df, tokenizer)
```
Runs every text through the tokenizer and converts the labels to integers. The output is a HuggingFace `Dataset` object with tensors ready for the model. Each example becomes:
- `input_ids`: list of token integer IDs, padded to 128
- `attention_mask`: 1 for real tokens, 0 for padding
- `labels`: integer (0=neg, 1=neu, 2=pos)

**Step D — Load the model**
```python
model = AutoModelForSequenceClassification.from_pretrained(
    hub_id, num_labels=3, ...
)
```
Downloads and loads the pre-trained weights from HuggingFace Hub. This is where the LOAD REPORT appears (UNEXPECTED MLM head, MISSING classifier — both expected, as explained in Section 7).

**Step E — Set up the Trainer**
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)
```
HuggingFace's `Trainer` handles the entire training loop: batching, forward passes, loss computation, backpropagation, weight updates, evaluation, and checkpointing. You don't write the training loop manually.

**Step F — Train**
```python
trainer.train()
```
Runs 3 epochs. After each epoch, evaluates on the validation set and saves a checkpoint.

**Step G — Save**
```python
trainer.save_model(ckpt_path)
tokenizer.save_pretrained(ckpt_path)
```
Saves the best checkpoint (the epoch with the highest validation macro-F1) to `checkpoints/<model_key>/`. The tokenizer is saved alongside so the checkpoint is fully self-contained.

**Step H — Evaluate through the harness**
```python
evaluate_model(model_key, lang, predict_fn, ...)
```
Uses our standard benchmark harness to evaluate the fine-tuned model on the frozen test set. This produces a JSON result file in `results/` with the same format as the ready-made model results, making everything directly comparable.

---

### Cell 8 — Show results

Prints the macro-F1, latency, and VRAM usage for the just-trained model. Quick sanity check before downloading.

---

### Cell 9 — Package for download

```python
shutil.make_archive(zip_dest, "zip", ckpt_src)
```

Zips the checkpoint directory into a single `.zip` file in `/kaggle/working/`, which is where Kaggle's Output tab shows downloadable files.

A BERT-base checkpoint is ~420MB (model weights in fp16 + tokenizer vocabulary files). The zip makes it a single download instead of hundreds of small files.

---

### Cell 10 — Run all four (optional)

A commented-out loop that runs all four fine-tunes sequentially in one session (~2 hours on T4). Only recommended if you have quota headroom and are confident the session won't time out.

---

## 9. What to watch during training

### Progress bars
The Trainer shows progress bars like:
```
Epoch 1/3:  100%|████████████| 925/925 [08:23<00:00, 1.84it/s]
```
- `925` = number of batches per epoch (≈14,800 examples ÷ 16 per batch)
- `it/s` = iterations (batches) per second — around 1.5–2 on T4 is normal

### Loss and eval metrics
Every 50 batches (logging_steps=50) you'll see:
```
{'loss': 0.8234, 'learning_rate': 1.9e-05, 'epoch': 0.54}
```
The loss should **decrease** over training. If it stays flat or increases from epoch 1 onward, something is wrong.

After each epoch:
```
{'eval_loss': 0.412, 'eval_macro_f1': 0.743, 'epoch': 1.0}
```
The `eval_macro_f1` is the validation macro-F1 after that epoch. You want this to increase epoch-over-epoch (or at least not collapse). The best epoch's checkpoint will be loaded at the end.

### What good training looks like

| Epoch | Train loss | Val macro-F1 | Status |
|---|---|---|---|
| 1 | ~0.8 → ~0.5 | ~0.70 | Learning |
| 2 | ~0.4 → ~0.3 | ~0.75 | Improving |
| 3 | ~0.3 → ~0.25 | ~0.77 | Converging |

### Warning signs

| Sign | Likely cause |
|---|---|
| Val F1 drops epoch 2→3 | Overfitting (the 3-epoch recipe prevents this usually) |
| Loss stays at ~1.1 throughout | Learning rate too high, or data labels are wrong |
| Loss goes to 0, val F1 very low | Model memorised training data (overfitting) |
| `nan` loss | Learning rate too high, or fp16 overflow |

---

## 10. How to use the checkpoint after training

After downloading `checkpoint_darijabert.zip`, unzip it into `checkpoints/darijabert/` locally. The checkpoint directory contains:
```
checkpoints/darijabert/
    config.json          ← model architecture + id2label
    model.safetensors    ← the actual weights (fp32 or fp16)
    tokenizer.json       ← vocabulary
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
```

To evaluate locally (no GPU needed for BERT-base on 1000 rows — takes ~3 minutes on your CPU):

```bash
PYTHONPATH=src python3 -c "
from transformers import pipeline
from label_maps import FINETUNED_MAP, apply_map
from harness import evaluate_model

pipe = pipeline('text-classification', model='checkpoints/darijabert',
                tokenizer='checkpoints/darijabert', device=-1, top_k=None)

def predict(texts):
    results = pipe(texts, batch_size=16, truncation=True, max_length=128)
    preds = []
    for item in results:
        scores = item if isinstance(item, list) else [item]
        best = max(scores, key=lambda x: x['score'])
        preds.append(apply_map(best['label'], FINETUNED_MAP))
    return preds

evaluate_model('darijabert', 'darija_ar', predict)
"
```

The result is written to `results/darijabert_darija_ar.json` and appended to `results/summary.csv`, ready for the final aggregation step.

---

## Summary table

| | DarijaBERT | DarijaBERT-arabizi | MARBERTv2 | QARIB |
|---|---|---|---|---|
| Hub ID | SI2M-Lab/DarijaBERT | SI2M-Lab/DarijaBERT-arabizi | UBC-NLP/MARBERTv2 | qarib/bert-base-qarib |
| Params | ~110M | ~110M | ~163M | ~110M |
| Pre-train data | Moroccan Darija (AR) | Moroccan Darija + Arabizi | 1B words, 25 dialects | 420M Gulf Arabic + MSA |
| Licence | Open | Open | Research-only ⚠️ | Open |
| Fine-tune data | MAC (~14.8k rows) | MYC-Latin (~3.9k rows) | MAC (~14.8k rows) | MAC (~14.8k rows) |
| Eval languages | darija_ar | arabizi | darija_ar, msa | darija_ar, msa |
| Baseline to beat | xlm-t 0.723 | xlm-t 0.762 | xlm-t 0.723 / camelbert 0.924 | xlm-t 0.723 |
| GPU time (T4) | ~25 min | ~15 min | ~25 min | ~25 min |

---

## Step 5 — Atlas-Chat-2B zero-shot (cells 13–16)

Atlas-Chat is not fine-tuned — it is a **causal language model** evaluated zero-shot. This section explains how it differs from the fine-tuned encoders above, walks through the new Kaggle cells, and summarises the results for comparison.

### What makes Atlas-Chat different

Fine-tuned encoders (Steps 1–4) follow this pattern:
```
Input text → tokeniser → BERT encoder → [CLS] vector → linear classification head → {neg, neu, pos}
```
The classification head is trained on labelled data. The model outputs probabilities directly.

Atlas-Chat follows a completely different pattern:
```
Input text → prompt template → causal LM → generated tokens → regex parse → {neg, neu, pos}
```
There is no classification head. The model is asked to *write* the label as its next tokens. We parse the output with a regex looking for `positif|neutre|negatif`.

**Practical consequence:** If the model generates anything other than one of those three words, it is a non-response (counted separately). The model cannot be batched efficiently — each `generate()` call is sequential.

### The quantisation (4-bit NF4)

Atlas-Chat-2B at full precision (fp32) would require ~6.4 GB just for weights (1.6B × 4 bytes). On a 16GB T4, that leaves no room for activations during generation.

4-bit NF4 quantisation (via `bitsandbytes`) compresses the weights to ~0.5 bytes per parameter: 1.6B × 0.5 bytes ≈ **0.8 GB** for weights, leaving ample room. The quality loss from quantisation is small for generation tasks, especially when the task is as constrained as single-word classification.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — better than int4 for LLMs
    bnb_4bit_use_double_quant=True,       # quantise the quantisation constants too
    bnb_4bit_compute_dtype=torch.float16, # dequantise to fp16 for matrix ops
)
```

### Kaggle notebook cells 13–16

**Cell 13 — Verify bitsandbytes:**
Checks that `bitsandbytes` is installed (it was already installed in cell 1). If it is missing (e.g., you started a fresh session without running cell 1), it installs it now. `accelerate` is also needed for `device_map="auto"`.

**Cell 14 — Choose model size:**
```python
ATLAS_SIZE = "2b"  # or "9b"
```
The 2B model fits on T4 with ~4 GB VRAM. The 9B model requires ~10 GB — possible on T4 but leaves very little headroom. Use 2B unless you have a specific reason to test 9B.

**Cell 15 — Run inference:**
```python
from atlas_chat import run_atlas
run_atlas(ATLAS_SIZE)
```
This loads the model, runs a 5-sample warm-up per language, then times each `generate()` call individually for 1000 samples. Results are written to:
- `results/atlas-chat-2b_darija_ar.json`
- `results/atlas-chat-2b_arabizi.json`
- `results/summary.csv` (appended)

Expect ~60 minutes total for the 2B model (1000 × 2 languages × ~18 seconds per sample at ~185ms/utt).

**Cell 16 — Show results:**
Prints macro-F1, median latency, peak VRAM, and non-response rate for each language. Cross-check against the values below before downloading.

### Results (2B model)

| Language | Macro-F1 | Latency | Non-response | Verdict vs encoder |
|---|---|---|---|---|
| darija_ar | 0.727 | 186 ms | 0.5% | Encoder dominates (+11.7 pts for MARBERTv2) |
| arabizi | 0.978 | 184 ms | 1.2% | Near-tie (gap = 0.5 pts vs DarijaBERT-arabizi) |

The starkest finding: on 3-class darija_ar, Atlas-Chat's **neutral recall is 32.8%** (only 76 of 232 neutral tweets correctly identified). The LLM cannot reliably detect the absence of sentiment. On binary arabizi, where there is no neutral class, it excels.

### Results (9B model) — scale reversal

| Language | Macro-F1 | Latency | Non-response | Neutral abstention | Verdict vs encoder |
|---|---|---|---|---|---|
| darija_ar | **0.833** | 451 ms | 0.0% | — | Gap = 1.1 pts vs MARBERTv2 |
| arabizi | 0.754 | 451 ms | 0.0% | **42.4%** (424/1000) | Worse than xlm-t baseline |

The 9B model inverts the 2B pattern entirely:

- **darija_ar:** Neutral F1 jumps from 0.480 (2B) to **0.746** (9B), nearly matching MARBERTv2's 0.751. The 9B reaches 0.833 macro-F1 — only 1.1 points below the best fine-tuned encoder, with no training whatsoever.
- **arabizi:** The same improved neutral calibration becomes a liability. The 9B predicts "neutre" for 424 of 1000 arabizi inputs despite this being a binary test set. Positive recall collapses to 0.417. The model scores 0.754 — worse than xlm-t (0.762).

The non-response rate is 0.0% for the 9B: it always outputs one of the three valid labels. The abstention is not a parsing failure — the model genuinely predicts "neutre" for ambiguous Arabizi content that the 2B rounds to a polar class.

### LLM vs fine-tuned encoders — final comparison

| | Fine-tuned encoder | Atlas-Chat-2B | Atlas-Chat-9B |
|---|---|---|---|
| Training required | Yes (~15–25 min on T4) | No | No |
| Inference mode | Batched (pipeline) | Sequential (generate) | Sequential (generate) |
| Latency on T4 | 2–4 ms/utt | ~185 ms/utt | ~451 ms/utt |
| darija_ar macro-F1 | 0.844 (MARBERTv2) | 0.727 | **0.833** |
| arabizi macro-F1 | 0.983 (DarijaBERT-arabizi) | **0.978** | 0.754 |
| Neutral F1 (darija_ar) | 0.751 (MARBERTv2) | 0.480 | **0.746** |
| Arabizi neutral abstention | 0% | 1.2% | **42.4%** |
| VRAM (T4) | 2.2–3.0 GB | ~4 GB (4-bit) | ~10 GB (4-bit) |

**Key insight:** The right LLM size depends on the task. Use 2B for binary arabizi; use 9B for 3-class Darija. In both cases, the fine-tuned encoder is still recommended for production — it is more accurate and 45–180× faster. The LLM sizes are useful as zero-shot baselines and for deployment contexts where no GPU is available for fine-tuned inference.

---

## Step 6 — French HACA fine-tune (`camembert-haca`, `xlm-r-haca`)

The Arabic/Darija pipeline had fine-tuned in-domain encoders, but **French** ran on an
off-the-shelf Hub model. On the French gold (`data/test_sets/francais_haca_gold.csv`, 90 real
broadcast utterances), the off-the-shelf models are weak — especially on the **neutral** class,
which is the bulk of factual broadcast speech:

| Model | macro-F1 | neutral F1 (recall) | note |
|---|---|---|---|
| `distilcamembert` (5★ reviews) | 0.324 | **0.000** (0.00) | never predicts neutral — 3★ is too rare |
| `xlm-sentiment` (Twitter 3-class) | 0.453 | 0.167 (0.10) | tweet-trained, under-produces neutral |

So we fine-tune a French encoder on HACA-style broadcast French, exactly like the Arabic models.

**The dataset (self-annotated, no LLM API).** There is no real French training pool (the only
real French SRT is the frozen gold), so the training data is **hand-authored** by Claude in
`src/synthetic_haca_fr.py` — 143 broadcast-register French utterances (63 pos / 40 neg / 40 neu)
across HACA topics (économie, santé, fiscalité, corruption, Sahara, éducation, sport…), following
the same content-valence rubric as the Arabic v3 set. Unlike the Arabic flow there is no MAC pool
to mix in, so the classes are kept roughly balanced (not neu-heavy) so a fresh head learns all
three. The gold is hand-labelled real data (`src/build_francais_gold.py`), kept strictly out of
training.

**The two bases.** Per the plan we try both:
- `camembert-haca` — `almanach/camembert-base`, the canonical French encoder (fresh 3-class head);
- `xlm-r-haca` — `cardiffnlp/twitter-xlm-roberta-base-sentiment`, already 3-class, adapted from
  tweets to broadcast.

Both use class-weighted focal loss (`focal_gamma=2.0`), 8 epochs (the set is small), lr 2e-5.

**Run it — easiest is the dedicated Kaggle notebook** `notebooks/kaggle_finetune_francais.ipynb`
(GPU T4, Internet ON): it clones the repo, trains both models, and prints the baseline-vs-fine-tune
comparison on the gold. Or from a shell on any GPU box:
```bash
python src/synthetic_haca_fr.py        # build the training CSV (already committed)
python src/build_francais_gold.py      # build the gold CSV (already committed)
python src/finetune.py --model camembert-haca     # ~5 min on T4 → checkpoints/camembert-haca/
python src/finetune.py --model xlm-r-haca         # ~6 min on T4 → checkpoints/xlm-r-haca/
# measure the gain on the real gold:
python src/eval_francais_gold.py --models xlm-sentiment camembert-haca xlm-r-haca
```
Once `checkpoints/camembert-haca/` exists, the dashboard's **Auto** mode and the French picker use
it automatically (`haca_pipeline.pick_model_for_lang("francais")` prefers the fine-tune, falling
back to `xlm-sentiment`). The baseline to beat is **macro-F1 0.453**, and in particular the near-
zero neutral recall.

> **Kaggle disk note.** The Trainer saves a checkpoint every epoch, and a full checkpoint includes
> the optimizer state (~2× the model size). Across 8 epochs and two models this overflows Kaggle's
> ~20 GB `/kaggle/working` quota, surfacing mid-training as
> `RuntimeError: [enforce fail at inline_container.cc] unexpected pos …` inside `torch.save`.
> `finetune.py` therefore sets `save_total_limit=1` (keep only the best checkpoint) and
> `save_only_model=True` (don't persist optimizer/scheduler) — final model selection is unaffected.
> If you still hit it, free space between runs: `rm -rf checkpoints/*/checkpoint-*`.

> **CamemBERT tokenizer note.** On some `transformers`/`tokenizers` versions (e.g. Kaggle's), the
> CamemBERT *fast* tokenizer fails to load with `argument 'vocab': 'str' object cannot be converted
> to 'PyTuple'`. Both `finetune.py` and `haca_pipeline.load_encoder` catch this and fall back to the
> slow SentencePiece tokenizer (needs `sentencepiece`, installed by the notebook), so training and
> inference proceed normally. If you saw `distilcamembert — FAILED …` in the baseline cell on an
> older checkout, that was the same cause and is harmless — the eval skips that one model and
> continues (the `xlm-sentiment` baseline is the one that matters).
