"""
HACA tonality dashboard — simple web UI.

Upload an SRT broadcast transcript → see the programme verdict, the neg/neu/pos percentages,
the per-segment timeline, and per-segment detail. Reuses src/haca_pipeline.py end to end.

Run:
    pip install streamlit
    streamlit run src/dashboard_app.py

Notes:
- The "stub (demo)" classifier runs anywhere (no model needed) — good for a quick demo.
- The encoder option needs a trained checkpoint in checkpoints/<model>/ (+ optional
  results/thresholds_<model>.json) and transformers/torch installed.
"""

import os
import sys
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import haca_pipeline as hp   # noqa: E402

COL = {"neg": "#d9534f", "neu": "#bdbdbd", "pos": "#5cb85c"}
LABEL_FR = {"neg": "NÉGATIF", "neu": "NEUTRE", "pos": "POSITIF"}

st.set_page_config(page_title="Tonalité HACA", page_icon="📺", layout="wide")


@st.cache_resource(show_spinner=False)
def get_classifier(choice: str):
    return hp.load_stub() if choice == "stub" else hp.load_encoder(choice)


@st.cache_resource(show_spinner=False)
def get_topic_detector(mode: str, ollama_model: str = "gemma2"):
    import topic_detect as td
    if mode.startswith("Ollama"):
        return td.load_ollama_topic(ollama_model)
    if mode.startswith("Atlas"):
        return td.load_atlas_topic("2b")
    return td.load_keyword_topic()


def proportions_bar(props):
    fig, ax = plt.subplots(figsize=(7, 0.9))
    left = 0.0
    for c in ("neg", "neu", "pos"):
        ax.barh(0, props[c], left=left, color=COL[c], edgecolor="white")
        if props[c] > 0.04:
            ax.text(left + props[c] / 2, 0, f"{props[c]:.0%}", va="center", ha="center",
                    color="white", fontsize=10, fontweight="bold")
        left += props[c]
    ax.set_xlim(0, 1); ax.axis("off")
    return fig


def timeline_fig(segments, floor):
    x = list(range(len(segments)))
    neg = [s["proportions"]["neg"] for s in segments]
    pos = [s["proportions"]["pos"] for s in segments]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(x, neg, color=COL["neg"], marker="o", ms=4, label="négatif")
    ax.plot(x, pos, color=COL["pos"], marker="o", ms=4, label="positif")
    ax.axhline(floor, ls="--", color="#999", lw=1, label=f"seuil {floor:.2f}")
    ax.set_ylim(0, 1); ax.set_xlabel("segment"); ax.set_ylabel("part des énoncés")
    ax.legend(fontsize=8); ax.set_title("Tonalité au fil des segments")
    return fig


# ── sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.header("Réglages")
model_choice = st.sidebar.selectbox(
    "Classifieur (tonalité)",
    ["stub", "marbertv2-haca", "darijabert-haca"],
    help="« stub » = démo sans modèle. Les encoders nécessitent un checkpoint entraîné.")
topic_mode = st.sidebar.selectbox(
    "Détection du sujet",
    ["mots-clés (rapide)", "Ollama (LLM local)", "Atlas-Chat-2B (CUDA)", "aucune"],
    help="« mots-clés » = instantané, partout. « Ollama » = ton LLM local (GPU intégré, rapide). "
         "« Atlas-Chat-2B » = transformers 4-bit (CUDA seulement).")
ollama_model = st.sidebar.text_input(
    "Modèle Ollama", "gemma2",
    help="Le modèle que tu as pull (ex: gemma2, aya, qwen2.5:7b). Nécessite `ollama serve`.") \
    if topic_mode.startswith("Ollama") else "gemma2"
floor = st.sidebar.slider("Seuil non-neutre (sensibilité)", 0.05, 0.50, hp.NONNEU_FLOOR, 0.05,
                          help="Part minimale de négatif/positif pour basculer le verdict. "
                               "Plus bas = plus sensible.")
window = st.sidebar.slider("Énoncés par segment", 5, 30, hp.WINDOW, 1)

# ── header ───────────────────────────────────────────────────────────────────
st.title("📺 Tableau de bord — Tonalité broadcast HACA")
st.caption("Charger un fichier SRT → verdict par émission, pourcentages, et tonalité par segment. "
           "Le système filtre l'ASR illisible et signale les cas incertains.")

up = st.file_uploader("Fichier SRT", type=["srt"])

if up is None:
    st.info("⬆️ Charger un fichier .srt pour démarrer. Astuce : essayer `data/raw/srt/8.srt` "
            "(documentaire corruption → négatif) ou `data/raw/srt/4.srt` (explicatif → neutre).")
    st.stop()

# ── run the pipeline on the upload ───────────────────────────────────────────
hp.NONNEU_FLOOR = floor      # the slider drives the lean rule
hp.WINDOW = window
with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tf:
    tf.write(up.read()); tmp = tf.name

topic = None
try:
    with st.spinner(f"Analyse de la tonalité avec « {model_choice} »…"):
        predict_proba, thr = get_classifier(model_choice)
        rep = hp.process_file(tmp, predict_proba, thr)
    if topic_mode != "aucune":
        _, rows = hp.segment_srt(tmp)
        clean_rows = [r["text"] for r in rows if r["clean"]]
        full_text = " ".join(clean_rows)
        # LLM backends: opening (intros state the subject) + a spread, short prefill.
        # Keyword: the full text (more keyword evidence = better).
        if topic_mode.startswith(("Ollama", "Atlas")) and clean_rows:
            head = clean_rows[:4]
            rest = clean_rows[4:]
            step = max(1, len(rest) // 8)
            src = " ".join(head + rest[::step][:8])[:1500]
        else:
            src = full_text
        if full_text:
            try:
                with st.spinner(f"Détection du sujet ({topic_mode})… "
                                "(le 1er appel charge le modèle, ça peut prendre un moment)"):
                    topic = get_topic_detector(topic_mode, ollama_model)(src)
            except Exception as e:
                st.warning(f"Détection du sujet indisponible ({topic_mode}) : {e}\n\n"
                           "Pistes : 1er appel = chargement du modèle (relancer) ; essayer un "
                           "modèle plus petit (`qwen2.5:3b`) ; vérifier `ollama serve` ; ou "
                           "« mots-clés ». `ollama ps` doit montrer le modèle sur GPU, pas 100% CPU.")
                topic = get_topic_detector("mots-clés", ollama_model)(full_text)
finally:
    os.unlink(tmp)

p = rep["programme"]
tone = p["tone"]

# ── headline: subject + verdict ──────────────────────────────────────────────
if topic:
    st.markdown(f"### 🏷️ Sujet : `{topic}`")
st.markdown(f"### Verdict émission : "
            f"<span style='color:{COL[tone]}'>{LABEL_FR[tone]}</span> "
            f"<span style='color:#888;font-size:0.7em'>({p['tone_label']})</span>",
            unsafe_allow_html=True)
if p["flag_review"]:
    st.warning(f"⚠ À revoir par un humain — {p['review_reason']}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Négatif", f"{p['proportions']['neg']:.0%}")
c2.metric("Neutre",  f"{p['proportions']['neu']:.0%}")
c3.metric("Positif", f"{p['proportions']['pos']:.0%}")
c4.metric("Couverture", f"{p['coverage']:.0%}", help=f"{p['n_clean']}/{p['n_total']} énoncés intelligibles")
c5.metric("Confiance", f"{p['confidence']:.0%}")

st.pyplot(proportions_bar(p["proportions"]), use_container_width=True)

# ── segment timeline ─────────────────────────────────────────────────────────
st.subheader("Tonalité par segment")
st.pyplot(timeline_fig(rep["segments"], floor), use_container_width=True)

# ── segment table ────────────────────────────────────────────────────────────
rows = []
for i, s in enumerate(rep["segments"]):
    pr = s["proportions"]
    rows.append({"segment": f"{s['window'][0]}-{s['window'][1]}", "tonalité": LABEL_FR[s["tone"]],
                 "neg": pr["neg"], "neu": pr["neu"], "pos": pr["pos"],
                 "couverture": s["coverage"], "confiance": s["confidence"],
                 "à_revoir": s["flag_review"]})
df = pd.DataFrame(rows)
st.subheader("Détail des segments")
st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button("⬇️ Télécharger le CSV (tableau de bord)",
                   df.to_csv(index=False).encode("utf-8"),
                   file_name=f"tonalite_{up.name}.csv", mime="text/csv")
