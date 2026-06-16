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
GITHUB_REPO = "https://github.com/YOUR_USERNAME/haca-benchmark.git"
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

# %% 11 — Free space: delete the checkpoint zip after downloading
# Run ONLY after confirming the download completed.
# The unzipped weights in benchmark/checkpoints/<model>/ are NOT touched —
# only the zip copy created for downloading is removed.

zip_path = os.path.join("/kaggle/working", f"checkpoint_{MODEL_KEY}.zip")

if os.path.exists(zip_path):
    size_mb = os.path.getsize(zip_path) / 1024 ** 2
    os.remove(zip_path)
    print(f"Deleted: {zip_path}")
    print(f"Freed  : {size_mb:.0f} MB")
else:
    print(f"Nothing to delete — {zip_path} does not exist.")

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
