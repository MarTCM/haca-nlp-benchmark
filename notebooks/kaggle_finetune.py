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
# Zip the checkpoint directory so it appears as a single file in Kaggle Output.

import shutil

ckpt_src = os.path.join(REPO, "checkpoints", MODEL_KEY)
zip_dest = os.path.join(WORK, f"checkpoint_{MODEL_KEY}")

if os.path.isdir(ckpt_src):
    shutil.make_archive(zip_dest, "zip", ckpt_src)
    zip_path = zip_dest + ".zip"
    size_mb = os.path.getsize(zip_path) / 1024**2
    print(f"\nCheckpoint zipped: {zip_path}  ({size_mb:.0f} MB)")
    print("Download from Kaggle Output tab, or add as a Kaggle dataset.")
else:
    print("[WARN] Checkpoint directory not found — fine-tuning may have failed.")

# %% 10 — Upload results + checkpoint back to GitHub
#
# Results JSON  → committed and pushed directly to the repo (small text file, ~10 KB)
# Checkpoint    → uploaded to a GitHub Release asset (supports files up to 2 GB)
#
# Requirements:
#   - If repo is PRIVATE: add a Kaggle secret named GITHUB_TOKEN with your
#     GitHub personal access token (Settings → Secrets in the Kaggle notebook UI).
#   - If repo is PUBLIC: no secret needed; git push works without auth over HTTPS
#     once the remote was already set in cell 2 with the token embedded in the URL.
#
# One-time setup on GitHub (do this before running):
#   Your repo on GitHub → Releases → "Draft a new release" is NOT required —
#   this cell creates the release automatically via the API.

import glob, json, requests, shutil
from kaggle_secrets import UserSecretsClient

# ── Config — change GITHUB_OWNER/GITHUB_REPO_NAME to match your repo ──────
GITHUB_OWNER     = "MarTCM"
GITHUB_REPO_NAME = "haca-nlp-benchmark"
# ──────────────────────────────────────────────────────────────────────────

# Grab the GitHub token from Kaggle Secrets (optional for public repos)
try:
    gh_token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    AUTH_HEADERS = {
        "Authorization": f"token {gh_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    # Re-embed token in remote URL so git push works without interactive prompt
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=REPO
    ).decode().strip()
    if "@" not in remote:   # token not yet in URL
        authed = remote.replace("https://", f"https://{gh_token}@")
        subprocess.run(["git", "remote", "set-url", "origin", authed], cwd=REPO)
    print("GitHub token loaded from Kaggle Secrets.")
except Exception:
    AUTH_HEADERS = {"Accept": "application/vnd.github.v3+json"}
    print("No GITHUB_TOKEN secret found — assuming public repo.")

# ── Part 1: push results JSON(s) to the repo ──────────────────────────────
subprocess.run(["git", "config", "user.email", "kaggle-bot@haca.ma"], cwd=REPO, check=True)
subprocess.run(["git", "config", "user.name",  "Kaggle Training Bot"],  cwd=REPO, check=True)

files_to_stage = (
    glob.glob(f"{REPO}/results/{MODEL_KEY}_*.json")
    + [f"{REPO}/results/summary.csv"]
)

for f in files_to_stage:
    if os.path.exists(f):
        subprocess.run(["git", "add", f], cwd=REPO, check=True)

staged = subprocess.check_output(
    ["git", "diff", "--cached", "--name-only"], cwd=REPO
).decode().strip()

if staged:
    subprocess.run(
        ["git", "commit", "-m", f"results: {MODEL_KEY} fine-tune evaluation"],
        cwd=REPO, check=True,
    )
    subprocess.run(["git", "push"], cwd=REPO, check=True)
    print(f"\nResults pushed to GitHub:\n{staged}")
else:
    print("No new result files to push (already up to date).")

# ── Part 2: zip checkpoint ─────────────────────────────────────────────────
CKPT_DIR = f"{REPO}/checkpoints/{MODEL_KEY}"
ZIP_PATH = f"/kaggle/working/checkpoint_{MODEL_KEY}.zip"

if not os.path.isdir(CKPT_DIR):
    raise FileNotFoundError(f"Checkpoint directory not found: {CKPT_DIR}")

if not os.path.exists(ZIP_PATH):
    print(f"Zipping {CKPT_DIR} ...")
    shutil.make_archive(ZIP_PATH.replace(".zip", ""), "zip", CKPT_DIR)

zip_size_mb = os.path.getsize(ZIP_PATH) / 1024 ** 2
print(f"Checkpoint zip: {ZIP_PATH}  ({zip_size_mb:.0f} MB)")

# ── Part 3: create a GitHub Release and upload the zip ────────────────────
TAG = f"checkpoint-{MODEL_KEY}"
API  = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO_NAME}"

# Delete existing release with same tag if present (so re-runs are idempotent)
existing = requests.get(f"{API}/releases/tags/{TAG}", headers=AUTH_HEADERS)
if existing.status_code == 200:
    rel_id = existing.json()["id"]
    requests.delete(f"{API}/releases/{rel_id}", headers=AUTH_HEADERS)
    requests.delete(f"{API}/git/refs/tags/{TAG}", headers=AUTH_HEADERS)
    print(f"Deleted existing release '{TAG}' to re-upload.")

# Create the release
release_payload = {
    "tag_name":         TAG,
    "name":             f"Checkpoint — {MODEL_KEY}",
    "body":             (
        f"Fine-tuned checkpoint for `{MODEL_KEY}`.\n\n"
        f"Unzip into `checkpoints/{MODEL_KEY}/` in the local repo.\n\n"
        f"Results JSON is committed directly to `results/`."
    ),
    "draft":      False,
    "prerelease": True,    # marks it as a pre-release so it doesn't clutter the main releases page
}
r = requests.post(f"{API}/releases", headers=AUTH_HEADERS, json=release_payload)
r.raise_for_status()
release = r.json()
print(f"Release created: {release['html_url']}")

# Upload the zip as a release asset
upload_url = (
    release["upload_url"].split("{")[0]          # strip the {?name,label} template
    + f"?name=checkpoint_{MODEL_KEY}.zip"
)
print(f"Uploading {zip_size_mb:.0f} MB checkpoint — this may take a minute ...")
with open(ZIP_PATH, "rb") as fh:
    up = requests.post(
        upload_url,
        headers={**AUTH_HEADERS, "Content-Type": "application/zip"},
        data=fh,
        timeout=600,
    )
up.raise_for_status()
print(f"\nCheckpoint uploaded successfully.")
print(f"Download URL: {up.json()['browser_download_url']}")
print(f"\nTo download locally:\n  curl -L '{up.json()['browser_download_url']}' -o checkpoint_{MODEL_KEY}.zip")

# %% 11 — (Optional) Run all four models in one session (~2 h on T4)
# Only do this if you have enough quota and the session won't time out.

# for key in ["darijabert", "darijabert-arabizi", "marbertv2", "qarib"]:
#     print(f"\n{'='*60}")
#     print(f"Fine-tuning {key}")
#     print(f"{'='*60}")
#     finetune(key)
#     ckpt = os.path.join(REPO, "checkpoints", key)
#     if os.path.isdir(ckpt):
#         shutil.make_archive(os.path.join(WORK, f"checkpoint_{key}"), "zip", ckpt)
#         print(f"Checkpoint saved: checkpoint_{key}.zip")
