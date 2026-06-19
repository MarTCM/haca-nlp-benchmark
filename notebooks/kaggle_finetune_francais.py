# Kaggle notebook — HACA French tonality fine-tune
# Paired source for notebooks/kaggle_finetune_francais.ipynb (see report/FINETUNING.md §6).
# Settings: GPU T4, Internet ON. Set GITHUB_REPO, then run all cells top to bottom.

# %% [markdown]
# # HACA — French tonality fine-tune (Kaggle T4)
#
# Trains two French encoders on hand-authored broadcast French and evaluates them on the
# **frozen real French gold** (`emission_francaise.srt`, 90 hand-labelled utterances).
#
# Models: `camembert-haca` (almanach/camembert-base) and `xlm-r-haca`
# (cardiffnlp/twitter-xlm-roberta-base-sentiment).
#
# Full write-up: **report/FINETUNING.md §6**. Baseline to beat on the gold: macro-F1 **0.453**
# (`xlm-sentiment`); `distilcamembert` gets neutral-F1 **0.000**.
#
# **How to use:** New Notebook → Settings → Accelerator **GPU T4**, Internet **ON** → set
# `GITHUB_REPO` below → Run all. Each model trains in ~5–6 min.

# %% 1 — install deps (sentencepiece is REQUIRED for the CamemBERT / XLM-R tokenizers)
import subprocess, sys
def pip(*a): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *a])
pip("sentencepiece>=0.1.99", "pysrt>=1.1.2")
# transformers / datasets / accelerate ship in the Kaggle image already
print("deps installed")

# %% 2 — clone the project repo (CHANGE GITHUB_REPO if yours differs)
GITHUB_REPO = "https://github.com/MarTCM/haca-nlp-benchmark.git"
BRANCH      = "feat/haca-sent-v3"

import os, subprocess, sys
WORK = "/kaggle/working"; REPO = os.path.join(WORK, "benchmark")
if not os.path.isdir(REPO):
    subprocess.check_call(["git", "clone", "--branch", BRANCH, GITHUB_REPO, REPO])
else:
    subprocess.check_call(["git", "-C", REPO, "fetch", "origin", BRANCH])
    subprocess.check_call(["git", "-C", REPO, "checkout", BRANCH])
    subprocess.check_call(["git", "-C", REPO, "pull"])
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))
print("Repo ready at", REPO)

# %% 3 — verify GPU and set seeds
import torch
from src.utils import set_seeds
set_seeds()
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))
    print("VRAM           : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))

# %% 4 — verify / build the French datasets
# Training pool = src/synthetic_haca_fr_large.py (thousands of templated + ASR-noise-augmented
# rows, deterministic) + the 143 curated clean rows. Gold = hand-labelled emission_francaise.srt.
# The large set is safe to (re)build from source on Kaggle (no raw SRT needed); the gold CSV is
# committed and must NOT be rebuilt here (its source SRT is gitignored).
import os, subprocess, sys, pandas as pd
if not os.path.exists("data/test_sets/synthetic_haca_fr_large.csv"):
    subprocess.check_call([sys.executable, "src/synthetic_haca_fr_large.py"])
for csv in ["data/test_sets/synthetic_haca_fr_large.csv",
            "data/test_sets/synthetic_haca_fr.csv",
            "data/test_sets/francais_haca_gold.csv"]:
    assert os.path.exists(csv), f"MISSING {csv} — is data/test_sets/ committed?"
    d = pd.read_csv(csv)
    print(f"{csv}: n={len(d)}  {d['label'].value_counts().to_dict()}")

# %% 5 — BASELINE on the gold (off-the-shelf French models, before fine-tuning)
# These are the numbers to beat. Downloads the two Hub models (needs Internet ON).
import subprocess, sys
subprocess.run([sys.executable, "src/eval_francais_gold.py",
                "--models", "xlm-sentiment", "distilcamembert"], check=True)

# %% 6 — fine-tune CamemBERT  ->  checkpoints/camembert-haca/   (~5 min on T4)
import time
from src.finetune import finetune
t0 = time.time(); finetune("camembert-haca")
print("camembert-haca done in %.1f min" % ((time.time() - t0) / 60))

# %% 7 — fine-tune XLM-R  ->  checkpoints/xlm-r-haca/   (~6 min on T4)
import time
from src.finetune import finetune
t0 = time.time(); finetune("xlm-r-haca")
print("xlm-r-haca done in %.1f min" % ((time.time() - t0) / 60))

# %% 8 — EVALUATE both fine-tunes vs the baselines on the frozen gold
import subprocess, sys
subprocess.run([sys.executable, "src/eval_francais_gold.py", "--models",
                "xlm-sentiment", "distilcamembert", "camembert-haca", "xlm-r-haca"], check=True)

# %% 9 — package the final model for download (Output tab -> *.zip)
# Store (no compression): model weights are incompressible, so zip-deflate just wastes minutes.
# Archive only the root model files, not the checkpoint-*/ resume dir.
import os, glob, zipfile
for key in ["camembert-haca", "xlm-r-haca"]:
    src = os.path.join("checkpoints", key)
    if not os.path.isdir(src):
        print("[skip] no checkpoint for", key); continue
    out = f"/kaggle/working/{key}.zip"
    files = [f for f in glob.glob(os.path.join(src, "*")) if os.path.isfile(f)]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, arcname=os.path.join(key, os.path.basename(f)))
    print("zipped:", out, "%.0f MB" % (os.path.getsize(out) / 1e6))

# %% 10 — Clickable download links for the trained models
# FileLink needs both os.path.exists(path) (relative to CWD) and an href served from
# /kaggle/working — so chdir there first (same trick as the Arabic notebook).
import os
from IPython.display import display, FileLink

WORK = "/kaggle/working"
_cwd = os.getcwd()
os.chdir(WORK)
for key in ["camembert-haca", "xlm-r-haca"]:
    zip_name = f"{key}.zip"
    if os.path.exists(zip_name):
        size_mb = os.path.getsize(zip_name) / 1024 ** 2
        display(FileLink(zip_name, result_html_prefix=f"<b>{key}</b> ({size_mb:.0f} MB) — "))
    else:
        print(f"[WARN] {zip_name} not found — run cell 9 first.")
os.chdir(_cwd)

# %% [markdown]
# ## After training
#
# 1. Download `camembert-haca.zip` / `xlm-r-haca.zip` from the **Output** tab.
# 2. Unzip into `checkpoints/<key>/` in the repo, then commit (checkpoints/ is gitignored — link
#    them as a Kaggle dataset or copy to the deployment box).
# 3. The dashboard's **Auto** mode then uses `camembert-haca` for French automatically
#    (`haca_pipeline.pick_model_for_lang("francais")`), falling back to `xlm-sentiment` if absent.
# 4. Record the gold macro-F1 from cell 8 in report/FINETUNING.md §6 (replace the baseline row).

