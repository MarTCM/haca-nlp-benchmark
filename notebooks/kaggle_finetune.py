# Kaggle notebook — HACA Benchmark Fine-tuning
#
# HOW TO USE:
#   1. On Kaggle: New Notebook → Settings → Accelerator: GPU T4 x2 (or T4 x1)
#   2. Paste each cell block below (separated by # %%) into a new notebook cell
#   3. Set GITHUB_REPO to your repo URL before running
#   4. Run all cells top to bottom
#   5. After training: File → Save & Run All → the checkpoints appear in Output
#      Download them via the Output tab, or link them as a Kaggle dataset
#
# GPU quota cost: ~20-30 min per model × 4 models = ~1.5-2 h total
# Run one model per session if you want to checkpoint safely.
#
# ─────────────────────────────────────────────────────────────────────────────

# %% [markdown]
# # HACA Benchmark — Fine-tuning on Kaggle T4
# Models: DarijaBERT, DarijaBERT-arabizi, MARBERTv2, QARIB
# Each model gets a 3-class sentiment head trained with HuggingFace Trainer API.

# %% 1 — Install extra dependencies not in Kaggle's default image
# (transformers, datasets, accelerate are already in Kaggle; bitsandbytes is not)
# Run this cell once, then restart the kernel if it says "restart required"

import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip("bitsandbytes>=0.43.0", "pysrt>=1.1.2", "seaborn>=0.13.0")

# %% 2 — Clone the project repo (replace with your actual GitHub repo URL)
GITHUB_REPO = "https://github.com/MarTCM/haca-nlp-benchmark.git"
# ↑ CHANGE THIS before running

import os, subprocess

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "benchmark")

if not os.path.isdir(REPO):
    subprocess.check_call(["git", "clone", GITHUB_REPO, REPO])
else:
    subprocess.check_call(["git", "-C", REPO, "pull"])

os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
print("Repo ready at", REPO)

# %% 3 — Download the three raw datasets from their GitHub repos
# (Allociné is fetched automatically by the datasets library)

import requests, os

RAW = os.path.join(REPO, "data", "raw")
os.makedirs(os.path.join(RAW, "MAC"),  exist_ok=True)
os.makedirs(os.path.join(RAW, "ASTD"), exist_ok=True)
os.makedirs(os.path.join(RAW, "MYC"),  exist_ok=True)

DOWNLOADS = [
    (
        "MAC/MAC corpus.csv",
        "https://raw.githubusercontent.com/LeMGarouani/MAC/master/MAC%20corpus.csv",
    ),
    (
        "ASTD/Tweets.txt",
        "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/Tweets.txt",
    ),
    (
        "MYC/DATA_CLEANED.csv",
        "https://raw.githubusercontent.com/MouadJb/MYC/main/DATA_CLEANED.csv",
    ),
]

for rel_path, url in DOWNLOADS:
    dest = os.path.join(RAW, rel_path)
    if os.path.exists(dest):
        print(f"  [skip] {rel_path} already exists")
        continue
    print(f"  Downloading {rel_path} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    print(f"  ✓ {rel_path}  ({len(r.content)//1024} KB)")

print("All raw data ready.")

# %% 4 — Verify GPU and seeds
import torch
from src.utils import set_seeds

set_seeds()

print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% 5 — Build / verify the frozen test sets
# The test CSVs are committed in the repo (data/test_sets/).
# This cell just checks they are present and prints their hashes.

import hashlib, os, pandas as pd

TEST_DIR = os.path.join(REPO, "data", "test_sets")
for lang in ["darija_ar", "francais", "msa", "arabizi"]:
    path = os.path.join(TEST_DIR, f"{lang}.csv")
    if not os.path.exists(path):
        print(f"[MISSING] {path} — check that data/test_sets/ is committed in the repo")
        continue
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    df = pd.read_csv(path)
    dist = df["label"].value_counts().to_dict()
    print(f"  [{lang}] n={len(df)}  sha256={h[:16]}…  {dist}")

# %% 6 — ⚙️  CHOOSE WHICH MODEL TO FINE-TUNE
# Change MODEL_KEY to run a different model.
# Recommended order:
#   Session 1: darijabert
#   Session 2: darijabert-arabizi
#   Session 3: marbertv2
#   Session 4: qarib

MODEL_KEY = "darijabert"   # ← change this each session

# Available keys and their GPU time estimates on T4:
MODEL_INFO = {
    "darijabert":         "~20 min — DarijaBERT on MAC → eval darija_ar",
    "darijabert-arabizi": "~15 min — DarijaBERT-arabizi on MYC-Latin → eval arabizi",
    "marbertv2":          "~25 min — MARBERTv2 on MAC → eval darija_ar + msa (RESEARCH-ONLY licence)",
    "qarib":              "~25 min — QARIB on MAC → eval darija_ar + msa",
    # French HACA fine-tunes (train on src/synthetic_haca_fr.py; eval after with
    # src/eval_francais_gold.py on data/test_sets/francais_haca_gold.csv):
    "camembert-haca":     "~5 min — CamemBERT-base on synthetic French broadcast (143 rows, 8 epochs)",
    "xlm-r-haca":         "~6 min — XLM-R sentiment on synthetic French broadcast (143 rows, 8 epochs)",
}
print(f"Selected: {MODEL_KEY}")
print(f"Estimate: {MODEL_INFO[MODEL_KEY]}")

# %% 7 — Run fine-tuning
# Checkpoints saved to /kaggle/working/benchmark/checkpoints/<model_key>/
# Results JSON saved to /kaggle/working/benchmark/results/

import time
from src.finetune import finetune

t0 = time.time()
finetune(MODEL_KEY)
print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")

# %% 8 — Show results
import json, os

RESULTS = os.path.join(REPO, "results")
print("\n=== Results for this model ===")
for fname in sorted(os.listdir(RESULTS)):
    if fname.startswith(MODEL_KEY) and fname.endswith(".json"):
        d = json.load(open(os.path.join(RESULTS, fname)))
        print(f"\n{fname}")
        print(f"  macro-F1 : {d['macro_f1']}")
        print(f"  latency  : {d['latency_ms_per_utt']} ms/utt")
        print(f"  VRAM     : {d['peak_vram_mb']} MB")

# %% 9 — Package checkpoints for download
# Zip saved to /kaggle/working/ (Jupyter server root) so FileLink URLs resolve correctly.

import shutil

WORK      = "/kaggle/working"
ckpt_src  = os.path.join(REPO, "checkpoints", MODEL_KEY)
zip_dest  = os.path.join(WORK, f"checkpoint_{MODEL_KEY}")   # no .zip — make_archive adds it
zip_final = zip_dest + ".zip"

if os.path.isdir(ckpt_src):
    shutil.make_archive(zip_dest, "zip", ckpt_src)
    size_mb = os.path.getsize(zip_final) / 1024 ** 2
    print(f"Checkpoint zipped : {zip_final}  ({size_mb:.0f} MB)")
    print("Run cell 10 for clickable download links.")
else:
    print("[WARN] Checkpoint directory not found — fine-tuning may have failed.")

# %% 10 — Clickable download links for this model's outputs
#
# FileLink has two constraints that must BOTH be satisfied:
#   1. os.path.exists(path) must be True  — checked relative to CWD
#   2. The generated href must point to a real file on the Jupyter server
#      — served from /kaggle/working/ regardless of CWD
#
# Fix: temporarily chdir to /kaggle/working/ so both constraints use the same root.

import glob, shutil
from IPython.display import display, FileLink

WORK = "/kaggle/working"
_cwd = os.getcwd()          # save current dir (/kaggle/working/benchmark)
os.chdir(WORK)              # move to Jupyter server root for FileLink

print(f"=== Download links for: {MODEL_KEY} ===\n")

# ── Checkpoint zip ─────────────────────────────────────────────────────────
zip_name = f"checkpoint_{MODEL_KEY}.zip"   # relative to WORK
if os.path.exists(zip_name):
    size_mb = os.path.getsize(zip_name) / 1024 ** 2
    display(FileLink(zip_name,
                     result_html_prefix=f"<b>Checkpoint zip</b> ({size_mb:.0f} MB) — "))
else:
    print(f"[WARN] {zip_name} not found — run cell 9 first.")

print()

# ── Results JSON files ─────────────────────────────────────────────────────
# Copy JSONs to WORK so they're also reachable from the Jupyter server root.
results_dir = os.path.join(_cwd, "results")
json_files  = sorted(glob.glob(os.path.join(results_dir, f"{MODEL_KEY}_*.json")))
if json_files:
    for jf in json_files:
        fname = os.path.basename(jf)
        shutil.copy2(jf, os.path.join(WORK, fname))   # copy to WORK
        display(FileLink(fname,
                         result_html_prefix=f"<b>Results</b> ({fname}) — "))
else:
    print(f"[WARN] No results JSON found for '{MODEL_KEY}' — run cell 7 first.")

os.chdir(_cwd)              # restore original working directory

# %% 11 — Free space: delete the zip AND the unzipped checkpoint weights
# Run ONLY after confirming the zip download completed.

import shutil

freed_mb = 0

# ── 1. Checkpoint zip (/kaggle/working/checkpoint_<model>.zip) ─────────────
zip_path = os.path.join("/kaggle/working", f"checkpoint_{MODEL_KEY}.zip")
if os.path.exists(zip_path):
    mb = os.path.getsize(zip_path) / 1024 ** 2
    os.remove(zip_path)
    freed_mb += mb
    print(f"Deleted zip     : {zip_path}  ({mb:.0f} MB)")
else:
    print(f"Zip not found   : {zip_path}")

# ── 2. Unzipped weights (benchmark/checkpoints/<model>/) ───────────────────
ckpt_dir = os.path.join(REPO, "checkpoints", MODEL_KEY)
if os.path.isdir(ckpt_dir):
    mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(ckpt_dir)
        for f in files
    ) / 1024 ** 2
    shutil.rmtree(ckpt_dir)
    freed_mb += mb
    print(f"Deleted weights : {ckpt_dir}  ({mb:.0f} MB)")
else:
    print(f"Weights not found: {ckpt_dir}")

print(f"\nTotal freed     : {freed_mb:.0f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Atlas-Chat zero-shot (separate session recommended; ~1–2 h on T4)
# ─────────────────────────────────────────────────────────────────────────────
# Run cells 1–5 first (install deps, clone repo, verify test sets).
# Then run cells 13–16 below.  Do NOT run a fine-tuning session at the same
# time — the 4-bit model needs the full 16 GB.

# %% 13 — Verify bitsandbytes is available
import importlib
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes {bnb.__version__} — OK")
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "bitsandbytes>=0.43.0", "accelerate>=0.26.0"])
    print("bitsandbytes installed — restart kernel if prompted")

# %% 14 — Choose Atlas-Chat model size
# 2B fits comfortably on T4 (16 GB) with 4-bit quantisation (~4 GB).
# 9B requires ~10 GB; possible on T4 but tight — use only if quota allows.

ATLAS_SIZE = "2b"   # ← change to "9b" if you have quota

print(f"Selected: Atlas-Chat-{ATLAS_SIZE.upper()}")
print(f"Targets : darija_ar, arabizi  (zero-shot, no training)")
print(f"Est. time: {'~60 min' if ATLAS_SIZE == '2b' else '~120 min'} on T4")

# %% 15 — Run Atlas-Chat zero-shot inference
# Predictions + metrics written to:
#   results/atlas-chat-2b_darija_ar.json
#   results/atlas-chat-2b_arabizi.json
#   results/summary.csv  (appended)
#
# Progress bar: each dot = 1 sample.  1000 samples × ~3-4 s/sample on T4.

import sys, os
sys.path.insert(0, os.path.join(REPO, "src"))

from atlas_chat import run_atlas
run_atlas(ATLAS_SIZE)

# %% 16 — Show Atlas-Chat results
import json, os

RESULTS = os.path.join(REPO, "results")
print(f"\n=== Atlas-Chat-{ATLAS_SIZE.upper()} results ===\n")
for lang in ["darija_ar", "arabizi"]:
    fname = f"atlas-chat-{ATLAS_SIZE}_{lang}.json"
    path  = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        print(f"[MISSING] {fname}")
        continue
    d = json.load(open(path))
    print(f"{lang}:")
    print(f"  macro-F1         : {d['macro_f1']}")
    print(f"  latency (median) : {d['latency_ms_per_utt']:.0f} ms/utt")
    print(f"  peak VRAM        : {d['peak_vram_mb']:.0f} MB")
    print(f"  non-response rate: {d.get('non_response_rate', 'N/A')}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# OPTION 2 — Broadcast-aware fine-tuning (target ≥0.70 on domaine_reel_v2)
#
# Two strategies (choose one per session):
#
#   A. marbertv2-mixed      — MAC + 500 broadcast utterances, fresh from Hub.
#                              Requires broadcast_train_raw.csv committed in repo.
#                              Est. ~30 min on T4.
#
#   B. marbertv2-broadcast  — broadcast-only, starting from the MAC checkpoint.
#                              Requires checkpoints/marbertv2/ uploaded to this session.
#                              Lower LR (5e-6), 5 epochs, smaller batch.
#                              Est. ~5 min on T4 (dataset is only ~400 examples).
#
# Step 0 (local, done once): annotate 500 utterances with Gemini, commit the CSV.
#   export GEMINI_API_KEY="AIza..."
#   python src/annotate_gemini.py         # writes data/test_sets/broadcast_train_raw.csv
#   git add data/test_sets/broadcast_train_raw.csv && git commit -m "feat: add broadcast annotations"
#   git push
# Then clone/pull on Kaggle (cells 1-2 already handle this).
# ─────────────────────────────────────────────────────────────────────────────

# %% 20 — [Option 2 / Strategy A] Fine-tune marbertv2-mixed (MAC + broadcast)
# Requires: broadcast_train_raw.csv committed to the repo (pull handled by cell 2).

import os, sys, time
sys.path.insert(0, os.path.join(REPO, "src"))

from src.finetune import finetune

STRATEGY = "marbertv2-mixed"    # ← change to "marbertv2-broadcast" for Strategy B

print(f"Option 2 / Strategy: {STRATEGY}")
t0 = time.time()
finetune(STRATEGY)
print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")


# %% 21 — [Option 2 / Strategy B setup] Upload the MAC checkpoint
# Strategy B starts from checkpoints/marbertv2/ (already fine-tuned on MAC).
# Upload the checkpoint zip to Kaggle as a Dataset, then mount it here.
# OR: if the checkpoints/ folder is in the repo via Git LFS, this cell is a no-op.

import os, shutil

WORK = "/kaggle/working"
CKPT_SRC = "/kaggle/input/haca-marbertv2-checkpoint/marbertv2"   # ← adjust dataset path
CKPT_DST = os.path.join(REPO, "checkpoints", "marbertv2")

if os.path.isdir(CKPT_SRC) and not os.path.isdir(CKPT_DST):
    shutil.copytree(CKPT_SRC, CKPT_DST)
    print(f"Checkpoint copied: {CKPT_DST}")
elif os.path.isdir(CKPT_DST):
    print(f"Checkpoint already present: {CKPT_DST}")
else:
    print(f"[WARN] Checkpoint not found at {CKPT_SRC}. Strategy B will fail.\n"
          "       Upload the checkpoint as a Kaggle dataset or switch to Strategy A.")


# %% 22 — [Option 2] Show domaine_reel_v2 result
# Both strategies evaluate on domaine_reel_v2 automatically after training.

import json, os

RESULTS = os.path.join(REPO, "results")
TARGET  = "domaine_reel_v2"      # frozen broadcast test set

print(f"\n=== Option 2 results on {TARGET} ===\n")
for strategy in ["marbertv2-mixed", "marbertv2-broadcast"]:
    fname = f"{strategy}_{TARGET}.json"
    path  = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    mf1 = d["macro_f1"]
    status = "✓ TARGET MET" if mf1 >= 0.70 else f"✗ {0.70 - mf1:.3f} below target"
    print(f"{strategy}:")
    print(f"  macro-F1 on v2 : {mf1:.4f}  {status}")
    print(f"  n              : {d['n']}")
    cr = d.get("classification_report", {})
    for c in ["neg", "neu", "pos"]:
        if c in cr:
            print(f"  {c}  P={cr[c]['precision']:.3f}  R={cr[c]['recall']:.3f}  "
                  f"F1={cr[c]['f1-score']:.3f}  n={int(cr[c]['support'])}")
    print()

# Also check regression on darija_ar benchmark
print("=== Regression check on darija_ar ===\n")
for strategy in ["marbertv2-mixed", "marbertv2-broadcast"]:
    fname = f"{strategy}_darija_ar.json"
    path  = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    print(f"{strategy}  darija_ar macro-F1: {d['macro_f1']:.4f}  "
          f"(baseline marbertv2: 0.8441)")


# %% 23 — [Option 2] Package Option 2 checkpoints for download

import shutil, os

WORK = "/kaggle/working"
for strategy in ["marbertv2-mixed", "marbertv2-broadcast"]:
    ckpt_src = os.path.join(REPO, "checkpoints", strategy)
    if not os.path.isdir(ckpt_src):
        print(f"[SKIP] {strategy} — checkpoint not found")
        continue
    zip_dest  = os.path.join(WORK, f"checkpoint_{strategy}")
    zip_final = zip_dest + ".zip"
    shutil.make_archive(zip_dest, "zip", ckpt_src)
    size_mb = os.path.getsize(zip_final) / 1024 ** 2
    print(f"Zipped {strategy}: {zip_final}  ({size_mb:.0f} MB)")


# %% 12 — (Optional) Run all four models in sequence
# Uncomment to run everything in one long session (~2 h on T4).
# Only do this if you have enough quota and the session won't time out.
# After the loop, run cells 10 and 11 once per MODEL_KEY to download and clean up.

# for key in ["darijabert", "darijabert-arabizi", "marbertv2", "qarib"]:
#     print(f"\n{'='*60}")
#     print(f"Fine-tuning {key}")
#     print(f"{'='*60}")
#     finetune(key)
#     ckpt = os.path.join(REPO, "checkpoints", key)
#     if os.path.isdir(ckpt):
#         shutil.make_archive(os.path.join(WORK, f"checkpoint_{key}"), "zip", ckpt)
#         print(f"Checkpoint saved: checkpoint_{key}.zip")
