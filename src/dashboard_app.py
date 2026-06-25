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
LANG_FR = {"arabe": "🇲🇦 Arabe / Darija", "arabizi": "🔤 Arabizi (Darija latin)",
           "francais": "🇫🇷 Français"}

# Tonality classifier choices shown in the sidebar (key -> human label).
CLASSIFIER_CHOICES = {
    "auto": "🪄 Auto (langue → modèle)",
    "stub": "stub (démo, sans modèle)",
    "marbertv2-haca": "Arabe / Darija — MARBERTv2 (HACA) ★",
    "darijabert-haca": "Darija — DarijaBERT (HACA)",
    "qarib": "Arabe (MSA) — QARiB",
    "marbertv2": "Arabe — MARBERTv2",
    "darijabert-arabizi": "Arabizi — DarijaBERT",
    "xlm-sentiment": "Français — off-the-shelf Hub (xlm-sentiment) ★",
    "xlm-r-haca": "Français — fine-tune HACA (xlm-r-haca)",
    "camembert-haca": "Français — fine-tune HACA (camembert-haca)",
    "ensemble-fr": "Français — ensemble (xlm-r-haca + xlm-sentiment)",
    "distilcamembert": "Français — off-the-shelf Hub (distilcamembert, 5★)",
    "api": "API Cloud (Z.ai / OpenAI compatible)",
}

st.set_page_config(page_title="Tonalité HACA", page_icon="📺", layout="wide")


@st.cache_resource(show_spinner=False)
def get_classifier(choice: str):
    """Return (predict_proba, thr) for non-API models (cached)."""
    if choice == "stub":
        return hp.load_stub()
    return hp.load_encoder(choice)


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
    list(CLASSIFIER_CHOICES),
    format_func=lambda k: CLASSIFIER_CHOICES[k],
    help="« Auto » détecte la langue du SRT et choisit le modèle. « stub » = démo sans modèle. "
         "« API Cloud » = LLM via API compatible OpenAI (Z.ai, OpenAI, Groq…).")

# API cloud options
api_key = ""
api_url = "https://api.z.ai/api/paas/v4/chat/completions"
api_model = "glm-5.2"
use_api_topic = False
if model_choice == "api":
    api_key = st.sidebar.text_input("Clé API", type="password",
                                    help="Ta clé API pour le fournisseur choisi.")
    api_url = st.sidebar.text_input("URL de l'API", "https://api.z.ai/api/paas/v4/chat/completions",
                                    help="Endpoint compatible OpenAI Chat Completions.")
    api_model = st.sidebar.text_input("Modèle", "glm-5.2",
                                      help="Nom du modèle (ex: glm-5.2, gpt-4o, gemma2…).")
    use_api_topic = st.sidebar.checkbox(
        "Utiliser le même modèle pour le sujet (économie de tokens)",
        help="Cochez pour que le même appel API détermine aussi le sujet de l'émission.")

# Topic detection — disabled when the API classifier handles both
topic_disabled = (model_choice == "api" and use_api_topic)
topic_mode = st.sidebar.selectbox(
    "Détection du sujet",
    ["mots-clés (rapide)", "Ollama (LLM local)", "Atlas-Chat-2B (CUDA)", "aucune"],
    disabled=topic_disabled,
    help="« mots-clés » = instantané, partout. « Ollama » = ton LLM local (GPU intégré, rapide). "
         "« Atlas-Chat-2B » = transformers 4-bit (CUDA seulement).")
ollama_model = st.sidebar.text_input(
    "Modèle Ollama", "gemma2",
    help="Le modèle que tu as pull (ex: gemma2, aya, qwen2.5:7b). Nécessite `ollama serve`.") \
    if (not topic_disabled and topic_mode.startswith("Ollama")) else "gemma2"
floor = st.sidebar.slider("Seuil non-neutre (sensibilité)", 0.05, 0.50, hp.NONNEU_FLOOR, 0.05,
                          help="Part minimale de négatif/positif pour basculer le verdict. "
                               "Plus bas = plus sensible.")
window = st.sidebar.slider("Énoncés par segment", 5, 30, hp.WINDOW, 1)
per_speaker = st.sidebar.checkbox(
    "Analyse par locuteur (SRT diarisé)", value=False,
    help="Si le SRT est diarisé (étiquettes « [SPEAKER_XX] » en début de cue), ajoute "
         "une tonalité par locuteur. Sans diarisation, cette option est sans effet.")

# ── header ───────────────────────────────────────────────────────────────────
st.title("📺 Tableau de bord — Tonalité broadcast HACA")
st.caption("Configure les réglages à gauche, charge un SRT, puis clique sur « Lancer l'analyse ».")

# ── file upload + run button ────────────────────────────────────────────────
with st.form("analysis_form"):
    up = st.file_uploader("Fichier SRT", type=["srt"])
    run = st.form_submit_button("Lancer l'analyse", type="primary")

if not run:
    if up is None:
        st.info("⬆️ Charger un fichier .srt puis cliquer sur « Lancer l'analyse ». "
                "Astuce : essayer `data/raw/srt/8.srt` (documentaire corruption → négatif) "
                "ou `data/raw/srt/4.srt` (explicatif → neutre).")
    st.stop()

# ── run the pipeline (only when the button is clicked) ──────────────────────
hp.NONNEU_FLOOR = floor
hp.WINDOW = window

with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tf:
    tf.write(up.read()); tmp = tf.name

topic = None
rep = None
try:
    _, rows = hp.segment_srt(tmp)
    clean_rows = [r["text"] for r in rows if r["clean"]]
    full_text = " ".join(clean_rows) or " ".join(r["text"] for r in rows)
    lang = hp.detect_lang(full_text) if full_text else "arabe"
    resolved_model = hp.pick_model_for_lang(lang) if model_choice == "auto" else model_choice

    include_topic = (model_choice == "api" and use_api_topic)

    with st.spinner(f"Analyse de la tonalité avec "
                     f"« {CLASSIFIER_CHOICES.get(resolved_model, resolved_model)} »…"):
        if resolved_model == "api":
            predict_proba, thr = hp.load_api_classifier(
                api_model, api_key, api_url, include_topic=include_topic)
        else:
            predict_proba, thr = get_classifier(resolved_model)
        rep = hp.process_file(tmp, predict_proba, thr, by_speaker=per_speaker)

    if include_topic:
        topic = predict_proba.topic
    elif topic_mode != "aucune":
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

    # Warn if the API classifier couldn't parse many responses.
    if resolved_model == "api":
        rate = predict_proba.fallback_rate
        if rate > 0.5:
            msg = f"⚠ L'API n'a pas pu interpréter **{rate:.0%}** des réponses du modèle."
            if predict_proba.first_raw:
                msg += f" Exemple de réponse inattendue : « {predict_proba.first_raw} »"
            st.warning(msg)
        elif rate > 0.0:
            st.info(f"ℹ {rate:.0%} des réponses API étaient inattendues "
                     "(utilisées comme neutre par défaut).")

    # Store results in session_state so they survive sidebar changes.
    st.session_state.rep = rep
    st.session_state.topic = topic
    st.session_state.lang = lang
    st.session_state.resolved_model = resolved_model

finally:
    os.unlink(tmp)

# ── display results (from session_state) ────────────────────────────────────
if "rep" not in st.session_state:
    st.stop()
rep = st.session_state.rep
topic = st.session_state.get("topic")
lang = st.session_state.lang
resolved_model = st.session_state.resolved_model

p = rep["programme"]
tone = p["tone"]

# Always show which model produced the verdict.
model_label = CLASSIFIER_CHOICES.get(resolved_model, resolved_model)
if resolved_model == "api":
    model_label = f"API Cloud ({api_model})"
model_note = f" · modèle : `{model_label}`" + (
    " (auto)" if model_choice == "auto" else "")
st.markdown(f"### 🗣️ Langue détectée : {LANG_FR.get(lang, lang)}{model_note}")
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

# ── per-speaker breakdown (diarized SRTs) ────────────────────────────────────
speakers = rep.get("speakers") or {}
if speakers:
    st.subheader("🗣️ Tonalité par locuteur")
    st.caption("Le SRT est diarisé : chaque locuteur est analysé séparément "
               "(énoncés du même locuteur regroupés).")
    srows = []
    for spk, a in speakers.items():
        pr = a["proportions"]
        srows.append({"locuteur": spk, "verdict": LABEL_FR[a["tone"]],
                      "neg": pr["neg"], "neu": pr["neu"], "pos": pr["pos"],
                      "énoncés": a["n_clean"], "couverture": a["coverage"],
                      "confiance": a["confidence"], "à_revoir": a["flag_review"]})
    sdf = pd.DataFrame(srows)
    st.dataframe(sdf, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Télécharger le CSV (par locuteur)",
                       sdf.to_csv(index=False).encode("utf-8"),
                       file_name=f"tonalite_locuteurs_{up.name}.csv", mime="text/csv")
    for spk, a in speakers.items():
        st.markdown(
            f"**{spk}** — <span style='color:{COL[a['tone']]}'>{LABEL_FR[a['tone']]}</span> "
            f"<span style='color:#888;font-size:0.8em'>({a['n_clean']}/{a['n_utterances']} "
            f"énoncés intelligibles)</span>",
            unsafe_allow_html=True)
        if a["flag_review"]:
            st.caption(f"⚠ à revoir — {a['review_reason']}")
        st.pyplot(proportions_bar(a["proportions"]), use_container_width=True)
elif per_speaker:
    st.info("Aucune étiquette de locuteur « [SPEAKER_XX] » détectée — ce SRT n'est "
            "pas diarisé, donc l'analyse par locuteur n'est pas disponible.")
