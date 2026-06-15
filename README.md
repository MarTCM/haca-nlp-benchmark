# Multilingual Media-Sentiment Benchmark

Benchmark of open-source AI tools for media tonality analysis (positive / neutral / negative)
on transcribed speech (SRT), across four languages:
- **MSA** — Modern Standard Arabic
- **French** — standard French
- **Darija AR** — Moroccan Darija (Arabic script)
- **Darija Arabizi** — Moroccan Darija (Latin script, digits 3/7/9)

## Quickstart

```bash
# 1. Create a virtualenv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Smoke-test the environment
python src/smoke_test.py
```

## Reproducing the benchmark

```bash
# Build and freeze test sets (run once — files become read-only after)
python src/build_test_sets.py

# Run ready-made models
python src/run_models.py --models xlm-t camelbert-da distilcamembert

# Fine-tune
python src/finetune.py --model darijabert
python src/finetune.py --model darijabert-arabizi
python src/finetune.py --model marbertv2
python src/finetune.py --model qarib

# Aggregate results and produce plots
python src/aggregate.py
```

## Repo layout

```
src/              Python source files
data/             Raw downloads + test sets (git-ignored, data/test_sets/ committed)
results/          JSON per (model,language) + summary.csv
results/figs/     Plots (heatmap, scatter, radar)
checkpoints/      Fine-tuned model checkpoints (git-ignored)
report/           Markdown report + état de l'art
notebooks/        Exploratory notebooks
```

## Reproducibility

- Seed 42 everywhere (random / numpy / torch / torch.cuda).
- `pip freeze > requirements.txt` after every new dependency.
- Docker: `docker build -t haca-bench . && docker run --rm haca-bench python src/smoke_test.py`
