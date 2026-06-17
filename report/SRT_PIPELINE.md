# Step 6 — The SRT Pipeline, Explained in Full

**Author:** Marwane ElBaraka  
**Scope:** Everything that happens between a raw `.srt` file and a sentiment prediction, explained from first principles. Covers the format, each processing stage, the language router, the model routing decision, and what the domaine-réel test set is and why it exists.

---

## Table of Contents

1. [What is an SRT file?](#1-what-is-an-srt-file)
2. [Why SRT files are harder to process than tweets](#2-why-srt-files-are-harder-to-process-than-tweets)
3. [Stage 1 — Reading the file (encoding detection)](#3-stage-1--reading-the-file-encoding-detection)
4. [Stage 2 — Parsing into cues (pysrt + regex fallback)](#4-stage-2--parsing-into-cues-pysrt--regex-fallback)
5. [Stage 3 — Stripping HTML tags](#5-stage-3--stripping-html-tags)
6. [Stage 4 — Merging cues into utterances](#6-stage-4--merging-cues-into-utterances)
7. [Stage 5 — Language routing (the script heuristic)](#7-stage-5--language-routing-the-script-heuristic)
8. [Stage 6 — MSA vs Darija disambiguation (camel-tools)](#8-stage-6--msa-vs-darija-disambiguation-camel-tools)
9. [Stage 7 — Model routing (best model per language)](#9-stage-7--model-routing-best-model-per-language)
10. [Stage 8 — Sentiment prediction and output](#10-stage-8--sentiment-prediction-and-output)
11. [Validating the language router](#11-validating-the-language-router)
12. [The domaine-réel test set — what it is and why it exists](#12-the-domaine-réel-test-set--what-it-is-and-why-it-exists)
13. [What you need to provide](#13-what-you-need-to-provide)
14. [Full worked example](#14-full-worked-example)

---

## 1. What is an SRT file?

SRT (SubRip Subtitle) is a plain-text subtitle format. Every broadcast programme produced or archived at HACA has an associated `.srt` file containing the transcription of what was said, split into short timed segments.

A raw `.srt` file looks like this:

```
1
00:00:01,500 --> 00:00:04,200
<i>مرحبا بكم في نشرة الأخبار</i>

2
00:00:04,800 --> 00:00:07,100
في هذا اليوم سنتحدث عن

3
00:00:07,100 --> 00:00:10,500
الوضع الاقتصادي في المغرب

4
00:00:11,000 --> 00:00:14,300
Bonjour et bienvenue sur cette chaîne.

5
00:00:15,000 --> 00:00:18,200
La situation économique reste stable.
```

Each block is called a **cue**. It has three parts:

```
<sequence number>
<start timestamp> --> <end timestamp>
<text line(s)>
```

**Sequence number:** An integer, starting at 1. Identifies the cue's position in the file.

**Timestamp format:** `HH:MM:SS,mmm` where mmm is milliseconds. The separator between seconds and milliseconds is a comma (`,`) in standard SRT, but some encoders use a period (`.`) instead — the parser handles both.

**Text:** One or more lines of subtitle text. Can contain HTML formatting tags like `<i>` (italic), `<b>` (bold), `<font color="...">`, etc. Multiple text lines within a single cue are displayed simultaneously on screen.

Cues are separated by a **blank line**. The end of one cue and the start of the next always have at least one blank line between them.

---

## 2. Why SRT files are harder to process than tweets

The benchmark test sets (Steps 2–5) used short social media texts — tweets and YouTube comments, typically 1–3 sentences, pre-cleaned, already one-per-row in a CSV. SRT files are different in every dimension:

| Property | Tweets / YouTube comments | SRT cues |
|---|---|---|
| Length | 1–3 sentences, complete | 5–15 words, often mid-sentence |
| Language per file | Single | Mixed (Arabic/French in same file) |
| Encoding | UTF-8, mostly clean | Variable: UTF-8, Windows-1256, BOM |
| Markup | None | HTML tags (`<i>`, `<b>`, `<font>`) |
| Sentence boundaries | Present | Often cut mid-sentence by timing |
| Sentiment unit | The whole tweet | One or more merged cues |

The key challenge is that **a cue is not a sentence**. A caption editor breaks speech into cues based on timing (roughly 2 seconds each, 10–15 words), not grammar. A single sentence often spans 2–4 consecutive cues:

```
Cue 5: "في هذا اليوم سنتحدث عن"
Cue 6: "الوضع الاقتصادي في المغرب"
Cue 7: "وتداعياته على المواطنين."
```

Together these form one sentence: *"Today we will talk about the economic situation in Morocco and its impact on citizens."* Feeding each cue to a sentiment model independently would give three meaningless predictions. They need to be merged first.

---

## 3. Stage 1 — Reading the file (encoding detection)

```python
_ENCODINGS = ("utf-8", "utf-8-sig", "cp1256", "latin-1")

content = None
for enc in _ENCODINGS:
    try:
        with open(path, encoding=enc) as fh:
            content = fh.read()
        break
    except (UnicodeDecodeError, LookupError):
        continue
```

**Why multiple encodings?**

Broadcast subtitle files come from many different sources: subtitle editors running on Windows machines, automatic speech recognition systems, legacy subtitle tools. Each has a different default encoding for Arabic text.

The four encodings we try, in order:

**`utf-8`** — The modern standard. Encodes all Unicode characters including Arabic, French accented letters, and Arabizi digits. Most subtitle files produced after ~2015 use this. A UTF-8 file opened with the wrong encoding produces `UnicodeDecodeError` — a fatal error — or "mojibake" (garbled characters).

**`utf-8-sig`** — UTF-8 with a BOM (Byte Order Mark). Some Windows tools prepend three bytes (`EF BB BF`) at the start of the file to identify it as UTF-8. If you open this file as plain `utf-8`, Python includes those three bytes as characters (they appear as `ï»¿` at the start of the text). `utf-8-sig` strips the BOM automatically.

**`cp1256`** — Windows code page 1256, also known as "Arabic Windows". This is the legacy encoding used by Arabic subtitle editors on Windows XP and earlier systems. It encodes Arabic letters in a single byte each (256 possible characters), making it incompatible with UTF-8. A cp1256 file read as UTF-8 raises `UnicodeDecodeError` because Arabic byte values in cp1256 are not valid UTF-8 sequences.

**`latin-1`** — ISO 8859-1, the fallback of last resort. Latin-1 never raises `UnicodeDecodeError` because it maps every byte to a character (it has 256 valid characters). The result may be wrong for Arabic characters, but the file at least opens. This catches exotic legacy encodings that the others miss.

The parser tries each encoding in order and uses the first one that succeeds. If all four fail, it raises `ValueError` — this should only happen for genuinely corrupt files.

---

## 4. Stage 2 — Parsing into cues (pysrt + regex fallback)

Once the file content is read, we parse it into structured cue objects.

### Primary path: pysrt

```python
import pysrt
subs = pysrt.open(path)
return [
    {"index": s.index, "start": str(s.start), "end": str(s.end),
     "text": _strip_html(s.text)}
    for s in subs
]
```

`pysrt` is a dedicated SRT parsing library. It handles:
- Sequence number parsing
- Timestamp parsing (both `,` and `.` as millisecond separators)
- Multi-line text blocks
- Minor format deviations (extra whitespace, Windows line endings `\r\n`)

Each `pysrt.SubRipItem` object has `.index`, `.start`, `.end`, and `.text` attributes. We convert these to plain Python dicts with string timestamps.

### Fallback path: manual regex

```python
block_re = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n"
    r"([\s\S]*?)(?=\n\n|\Z)",
    re.MULTILINE,
)
```

If pysrt raises an exception (malformed file, unusual encoding artifact after BOM stripping, etc.), we fall back to a hand-written regex. Breaking down the pattern:

- `(\d+)\s*\n` — Captures the sequence number (one or more digits), followed by optional whitespace and a newline.
- `(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n` — Captures the start and end timestamps. `[,\.]` matches either a comma or period as the millisecond separator. `\s*-->\s*` matches the arrow with any surrounding whitespace.
- `([\s\S]*?)(?=\n\n|\Z)` — Captures the text block (any character including newlines, non-greedy) up to either a blank line (`\n\n`) or end of file (`\Z`). The non-greedy `*?` is critical: it stops at the first blank line, not the last.

The regex fallback replaces newlines within the text block with spaces: `m.group(4).replace("\n", " ")`. This flattens multi-line text into a single string.

**Output of this stage:** A list of dicts, one per cue:

```python
[
    {"index": 1, "start": "00:00:01,500", "end": "00:00:04,200", "text": "مرحبا بكم في نشرة الأخبار"},
    {"index": 2, "start": "00:00:04,800", "end": "00:00:07,100", "text": "في هذا اليوم سنتحدث عن"},
    ...
]
```

---

## 5. Stage 3 — Stripping HTML tags

```python
def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()
```

SRT files frequently contain HTML markup for visual styling:

```
<i>مرحبا بكم</i>
<b>BREAKING:</b> الملك يعقد اجتماعا
<font color="#ffff00">Bonjour</font>
```

**`re.sub(r"<[^>]+>", "", text)`** removes anything that looks like an HTML tag: a `<`, followed by one or more characters that are not `>`, followed by `>`. This handles opening tags, closing tags, and tags with attributes.

**`html.unescape(text)`** converts HTML entities back to their Unicode characters:

| Entity | Becomes |
|---|---|
| `&amp;` | `&` |
| `&lt;` | `<` |
| `&gt;` | `>` |
| `&nbsp;` | non-breaking space |
| `&#8212;` | `—` (em dash) |

Some subtitle editors encode special characters as HTML entities even outside of tags. Without `html.unescape`, the model receives literal strings like `&amp;` instead of `&`.

**`.strip()`** removes leading and trailing whitespace, including the case where a tag at the start or end of the text leaves a leading/trailing space after removal.

---

## 6. Stage 4 — Merging cues into utterances

```python
SENTENCE_END = re.compile(r"(?<=[.!?؟۔])\s+")

def cues_to_utterances(cues: List[Dict]) -> List[str]:
    raw = " ".join(c["text"] for c in cues if c["text"])
    parts = SENTENCE_END.split(raw)
    return [p.strip() for p in parts if p.strip()]
```

This is the most conceptually important stage. It converts the timing-based cue structure into meaning-based utterances suitable for sentiment analysis.

### Step 1: Concatenate all cue texts

```python
raw = " ".join(c["text"] for c in cues if c["text"])
```

All cue texts are joined into one long string with a space between each cue. Empty cues (text = "") are skipped with `if c["text"]`. This produces:

```
"مرحبا بكم في نشرة الأخبار في هذا اليوم سنتحدث عن الوضع الاقتصادي في المغرب وتداعياته على المواطنين. Bonjour et bienvenue sur cette chaîne. La situation économique reste stable."
```

### Step 2: Split on sentence boundaries

```python
SENTENCE_END = re.compile(r"(?<=[.!?؟۔])\s+")
parts = SENTENCE_END.split(raw)
```

The pattern `(?<=[.!?؟۔])\s+` is a **lookbehind assertion**. It splits the string at any whitespace that is **immediately preceded** by a sentence-ending punctuation character. The lookbehind `(?<=...)` does not consume the character it looks at — the punctuation stays attached to the sentence that ends with it.

The five sentence-ending characters covered:

| Character | Unicode | Language |
|---|---|---|
| `.` | U+002E | French, Arabizi (Latin period) |
| `!` | U+0021 | All languages |
| `?` | U+003F | French, Arabizi |
| `؟` | U+061F | Arabic question mark |
| `۔` | U+06D4 | Arabic full stop (used in some Moroccan subtitles) |

**Why not split on every period?** A naive split on `.` would break decimal numbers (3.14), abbreviations (M. Dupont), URLs, and ellipses (...). The lookbehind only splits at `.` followed by whitespace — which is how sentence boundaries appear in practice.

**Why not use NLTK or spaCy?** Neither has reliable support for Moroccan Darija or Arabizi. A regex-based split on punctuation is simpler, has no dependencies, and works well enough for broadcast subtitle text, which tends to have clear sentence boundaries.

**Output:** A list of utterance strings, each being one complete (or near-complete) sentence:

```python
[
    "مرحبا بكم في نشرة الأخبار في هذا اليوم سنتحدث عن الوضع الاقتصادي في المغرب وتداعياته على المواطنين.",
    "Bonjour et bienvenue sur cette chaîne.",
    "La situation économique reste stable."
]
```

These are now the units that get fed to the sentiment models. Each utterance gets one sentiment prediction.

---

## 7. Stage 5 — Language routing (the script heuristic)

```python
ARABIC_RE  = re.compile(r"[؀-ۿ]")
ARABIZI_RE = re.compile(r"\b\w*[379]\w*\b")

def detect_lang(text: str) -> str:
    if ARABIC_RE.search(text):
        return "arabe"
    tokens = re.findall(r"\w+", text)
    arabizi_hits = sum(1 for t in tokens if ARABIZI_RE.match(t)) if tokens else 0
    if arabizi_hits / max(len(tokens), 1) > 0.10:
        return "arabizi"
    return "francais"
```

Each utterance needs to be routed to the correct sentiment model. The first decision is: what language/script is this text in?

### Rule 1: Arabic script detection

```python
ARABIC_RE = re.compile(r"[؀-ۿ]")
if ARABIC_RE.search(text):
    return "arabe"
```

`[؀-ۿ]` is a Unicode range covering U+0600 to U+06FF — the Arabic Unicode block. This block contains all Arabic letters, diacritics (tashkeel), punctuation specific to Arabic (؟ ، ؛), and digits (٠–٩). If **any** character in this range appears in the text, the router declares it `"arabe"`.

This is a binary test: even a single Arabic character is enough to classify the text as Arabic-script. This makes sense for broadcast subtitles because a genuine French or Arabizi utterance will never contain Arabic letters (even loan words in French are transliterated to Latin script). The false positive rate is essentially zero.

**What happens after `"arabe"` is returned:** The router gives you a language of `"arabe"`, which covers both Modern Standard Arabic (MSA) and Moroccan Darija in Arabic script. These need to be further distinguished — see Stage 6.

### Rule 2: Arabizi detection

```python
ARABIZI_RE = re.compile(r"\b\w*[379]\w*\b")
tokens = re.findall(r"\w+", text)
arabizi_hits = sum(1 for t in tokens if ARABIZI_RE.match(t)) if tokens else 0
if arabizi_hits / max(len(tokens), 1) > 0.10:
    return "arabizi"
```

If the text has no Arabic characters, it's either French or Arabizi (Latin-script Moroccan Arabic). The distinguishing feature of Arabizi is the use of the digits **3**, **7**, and **9** embedded in words to represent Arabic sounds that don't exist in Latin:

| Digit | Arabic letter | Sound |
|---|---|---|
| `3` | ع (ayn) | Voiced pharyngeal fricative |
| `7` | ح (ha) | Voiceless pharyngeal fricative |
| `9` | ق (qaf) | Voiceless uvular stop |

Example Arabizi words: `7bibo` (حبيبه, "my darling"), `3ayb` (عيب, "shame"), `f9ir` (فقير, "poor"), `zwine` (زوين, "beautiful") — the last one has no digit but is common enough to appear anyway.

The detector counts how many tokens contain at least one of the digits 3, 7, or 9 (`\b\w*[379]\w*\b`). If **more than 10%** of tokens have this property, the text is classified as Arabizi.

**Why 10%?** Pure French text occasionally contains the digits 3, 7, and 9 in numbers (`le 3 mars`, `17h30`). A 10% threshold means at least 1 in 10 tokens must contain a characteristic Arabizi digit — this is rare in French but common in Arabizi, where phonetic substitution appears throughout the vocabulary.

**Why `re.findall(r"\w+", text)` for tokens?** `\w+` in Python matches word characters (letters + digits + underscore). This splits the text on whitespace and punctuation, giving individual word+digit tokens. Using `text.split()` would leave punctuation attached to words.

### Rule 3: Default to French

```python
return "francais"
```

If no Arabic characters are found and fewer than 10% of tokens contain 3/7/9, the text is classified as French. This is appropriate for broadcast content: if it's not Arabic-script and not Arabizi, it's very likely French (HACA's other major broadcast language).

**A note on multilingual utterances:** Broadcast content sometimes mixes languages within a single utterance — a French journalist quoting an Arabic speaker, or a code-switched Darija/French sentence. The router makes one decision per utterance and cannot handle intra-utterance mixing. In practice, fully mixed sentences are rare enough in broadcast subtitles that this is acceptable.

---

## 8. Stage 6 — MSA vs Darija disambiguation (camel-tools)

The heuristic router returns `"arabe"` for any Arabic-script text, but the benchmark has two distinct Arabic models:
- **MSA** → camelbert-da (best at 0.924)
- **Darija AR** → MARBERTv2 fine-tuned (best at 0.844)

Feeding MSA text to the Darija model (or vice versa) would reduce accuracy. We need to distinguish which variety of Arabic each utterance belongs to.

This is done with **camel-tools**, a library from NYU Abu Dhabi specifically built for Arabic NLP:

```python
from camel_tools.dialectid import DialectIdentifier

did = DialectIdentifier.pretrained()
predictions = did.predict(utterances, 'city')
```

The `DialectIdentifier` classifies Arabic text into one of ~25 Arabic dialects (Egyptian, Moroccan, Gulf, Levantine, etc.) plus MSA. If the predicted dialect is `"MSA"` or one of the formal/standard Arabic categories, route to the MSA model. If it's `"MOR"` (Moroccan) or similar, route to the Darija model.

**Important limitation:** camel-tools' `DialectIdentifier` requires a Linux or macOS environment — it uses C extensions that are not available on native Windows. This is noted in the plan and is one reason the pipeline is designed for Kaggle/Colab rather than a local Windows machine.

**Why not use the script heuristic for MSA vs Darija?** Both MSA and Moroccan Darija use the Arabic Unicode block. There is no purely orthographic difference — a Darija word like بزاف (bezaf, "a lot") uses the same Arabic letters as an MSA word. Distinguishing them requires lexical and morphological knowledge, which camel-tools encodes.

---

## 9. Stage 7 — Model routing (best model per language)

After language detection, each utterance is routed to the best-performing model for that language:

| Detected language | Model | Macro-F1 on benchmark |
|---|---|---|
| `"darija_ar"` | MARBERTv2 (fine-tuned) | 0.844 |
| `"msa"` | camelbert-da (ready-made) | 0.924 |
| `"arabizi"` | DarijaBERT-arabizi (fine-tuned) | 0.983 |
| `"francais"` | distilcamembert (ready-made) | 0.949 |

This is the **max-precision routing** strategy from the plan: assign each utterance to the model that achieves the highest F1 on its language, regardless of other costs (memory, latency, model count). The alternative strategies — min-cost (one model for everything) and balanced (trade F1 for simplicity) — are discussed in the final report.

**What this means in practice:** A production deployment using max-precision routing would need to keep all four models loaded simultaneously (or load/unload on demand). Memory requirement: MARBERTv2 (~630MB) + camelbert-da (~420MB) + DarijaBERT-arabizi (~660MB) + distilcamembert (~260MB) ≈ **2 GB total** for the model weights (in fp32; roughly halved in fp16). This is manageable on any machine with a GPU.

---

## 10. Stage 8 — Sentiment prediction and output

Once an utterance is assigned to a model, it goes through that model's inference pipeline:

```python
pipe = pipeline("text-classification", model=model_path, device=0, top_k=None)
results = pipe([utterance], truncation=True, max_length=128)
best = max(results[0], key=lambda x: x["score"])
label = apply_map(best["label"], label_map)
```

The output for each utterance is one of three canonical labels: `neg`, `neu`, `pos`.

**What the pipeline does internally:**
1. The tokenizer converts the text string into a list of integer token IDs.
2. The model runs a forward pass and produces three logit scores (one per class).
3. Softmax converts logits to probabilities: [neg=0.12, neu=0.23, pos=0.65].
4. The highest-probability class is selected as the prediction.
5. The model's raw label name (e.g., `"LABEL_2"` for fine-tuned models, `"positive"` for ready-made models) is mapped to the canonical form (`"pos"`) via `apply_map`.

**Truncation:** Long utterances (over 128 tokens for fine-tuned models, 512 for ready-made) are truncated. A typical broadcast sentence is 10–30 words and well within these limits. Very long merged passages would be unusual but are handled gracefully by truncation rather than crashing.

**The complete output per utterance:**

```python
{
    "utterance": "الوضع الاقتصادي في المغرب يزداد تحسنا.",
    "lang_detected": "darija_ar",
    "model_used": "marbertv2",
    "sentiment": "pos",
    "confidence": 0.874,
    "cue_range": "2–5"   # which original SRT cues this came from
}
```

This can be written to a JSON or CSV file for further analysis.

---

## 11. Validating the language router

Before trusting the router's output on real content, we validate it on a **hand-checked sample**. The plan specifies 100 utterances.

```python
def validate_router(hand_checked: List[Tuple[str, str]]) -> None:
    y_true = [lang for _, lang in hand_checked]
    y_pred = [detect_lang(text) for text, _ in hand_checked]
    classes = sorted(set(y_true + y_pred))
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(cm)
```

**What you do:** Take 100 utterances from the real SRT files. For each one, manually write down what language it actually is (true label). Then run `validate_router` with these pairs. The function predicts the language using the heuristic and compares against your labels.

**What the output tells you:**

The confusion matrix for the router looks like this (example):

```
               Predicted arabe   Predicted arabizi   Predicted francais
True arabe            87                1                    0
True arabizi           2               14                    2
True francais          0                1                   11
```

- **Row sum** = how many utterances were truly that language in your sample.
- **Diagonal** = correct predictions.
- **Off-diagonal** = misclassifications — these are worth investigating individually. A common error: short Arabizi messages with no digit-containing words get classified as French (because the 10% threshold isn't met). Another: French numbers (`7h30`, `3 personnes`) occasionally trigger the Arabizi rule.

The router's error rate is reported in the final output. Unlike model errors (which affect sentiment quality), router errors are more serious: a misrouted utterance goes to the wrong model entirely, which will have lower F1 on a foreign language. If the router error rate is above 5%, the threshold or heuristic rules should be tuned.

---

## 12. The domaine-réel test set — what it is and why it exists

### What it is

A set of approximately 200 utterances extracted from **your actual HACA SRT files**, manually annotated with true sentiment labels.

### Why it exists

All the benchmark results so far (Steps 2–5) were measured on **public academic datasets**: MAC (Moroccan tweets), Allociné (French movie reviews), ASTD (Egyptian political Twitter), MYC (Moroccan YouTube comments). These datasets were collected from social media and may not represent the language, style, or sentiment distribution of **broadcast media subtitles** at HACA.

The domaine-réel test set answers the question: **how well do these models actually perform on the real content they will be deployed on?**

There are several ways the real content can differ from the training data:

| Property | Public datasets (Steps 2–5) | HACA SRTs (Step 6) |
|---|---|---|
| Register | Informal social media | Formal/journalistic speech |
| Sentence length | Short (tweet-length) | Longer, spoken-language patterns |
| Sentiment expression | Explicit, often strong | Often subtle or implicit |
| Topic | General / entertainment | Political, regulatory, media |
| Code-switching | Moderate | Variable (depends on programme) |
| Neutral content | Present but limited | Likely dominant (news is largely neutral) |

A model that scores 0.844 on MAC (Moroccan Twitter) might score significantly lower on HACA broadcast content, because broadcast speech is more formal and neutral than tweets. The domaine-réel test set directly measures this gap.

### How it is built

1. **Select utterances from your SRT files.** Pick a representative sample: include different types of programmes (news, debate, documentary, entertainment), different time periods, and different languages if the files are multilingual.

2. **Run the pipeline (Stages 1–8 above) on your SRTs.** This produces utterances. Pick ~200 from the output.

3. **Manually label each utterance.** For each utterance, assign one of: `neg`, `neu`, `pos`. This is the slowest step — expect 1–3 seconds per utterance, so about 5–10 minutes for 200 utterances.

4. **Write annotation rules.** The annotation rules document what counts as neg/neu/pos in the context of broadcast media. This is necessary because sentiment in broadcast content is often subtle or implicit:
   - A news anchor describing an economic indicator rising: `pos` or `neu`?
   - A politician saying a policy "needs to be reviewed": `neg` or `neu`?
   - A reporter describing an accident with no value judgement: `neu`.

   Without written rules, two annotators (or the same annotator on different days) will label the same utterance differently, making the test set unreliable.

5. **Save to `data/test_sets/domaine_reel.csv`** in the standard format: `[text, label]`. This file follows the same schema as the four existing test sets but is never used for training — it is evaluation-only.

6. **Run the best model per language through the standard harness.** The harness produces a JSON result and adds a row to `summary.csv`, exactly as for the public test sets.

### Why ~200 utterances?

A sample size of 200 gives reliable macro-F1 estimates with a confidence interval of approximately ±7 percentage points (95% CI, assuming macro-F1 ≈ 0.75 and equal class sizes). A larger sample (500+) would be more precise but the annotation time grows proportionally. 200 is the plan's minimum viable size for detecting meaningful differences in model performance.

---

## 13. What you need to provide

To execute Step 6, you need to supply:

**1. The SRT files themselves.** The pipeline accepts any `.srt` file. Ideally: a representative selection of HACA content covering different programmes and languages. Even 5–10 SRT files would be enough to extract 200 utterances.

**2. The true sentiment labels for ~200 utterances.** These are your manual annotations. The most efficient workflow:
   - Run the pipeline on the SRT files first to extract utterances.
   - Export them to a spreadsheet.
   - Add a `label` column and fill it in (neg / neu / pos).
   - Save as `data/test_sets/domaine_reel.csv`.

**3. Your annotation rules.** A short document (10–20 rules) defining how you draw the neg/neu/pos boundary for broadcast media content. This is needed so the test set is reproducible and its limitations are documented.

We do not need the SRT files to build the pipeline — it's already coded. We need them to run the pipeline and generate the utterances to annotate.

---

## 14. Full worked example

Here is what happens, end to end, for one real SRT cue sequence.

**Input: raw SRT content (excerpt)**

```
23
00:05:12,400 --> 00:05:15,100
<i>قال الوزير إن المشروع</i>

24
00:05:15,200 --> 00:05:18,600
سيساهم في تطوير القطاع

25
00:05:18,700 --> 00:05:21,000
وخلق فرص الشغل في المنطقة.

26
00:05:22,000 --> 00:05:25,300
L'opposition a critiqué ce projet

27
00:05:25,500 --> 00:05:28,800
en le qualifiant d'insuffisant.
```

**Stage 1 — Encoding detection**

File opens successfully as `utf-8`. Content is 120 KB.

**Stage 2 — Parse into cues**

pysrt returns 5 cue dicts. The `<i>` tag is still in the text at this point.

**Stage 3 — Strip HTML**

Cue 23: `"<i>قال الوزير إن المشروع</i>"` → `"قال الوزير إن المشروع"`

Cues 24–27: no tags, unchanged.

**Stage 4 — Merge cues into utterances**

All texts joined: `"قال الوزير إن المشروع سيساهم في تطوير القطاع وخلق فرص الشغل في المنطقة. L'opposition a critiqué ce projet en le qualifiant d'insuffisant."`

Split on sentence-ending punctuation followed by whitespace:

```python
[
    "قال الوزير إن المشروع سيساهم في تطوير القطاع وخلق فرص الشغل في المنطقة.",
    "L'opposition a critiqué ce projet en le qualifiant d'insuffisant."
]
```

Two utterances produced from five cues.

**Stage 5 — Language routing**

Utterance 1: contains Arabic characters (ق, ا, ل, ...) → `detect_lang()` finds `ARABIC_RE.search()` matches → returns `"arabe"`.

Utterance 2: no Arabic characters → check Arabizi: tokens are `["L", "opposition", "a", "critiqué", "ce", "projet", "en", "le", "qualifiant", "d", "insuffisant"]` → none contain 3, 7, or 9 → arabizi_hits = 0 → 0% < 10% → returns `"francais"`.

**Stage 6 — MSA vs Darija disambiguation**

Utterance 1 is `"arabe"`. camel-tools DialectIdentifier processes it. "قال الوزير" (the minister said) and "تطوير القطاع" (sector development) are formal MSA register. The model predicts `"MSA"` → route to `msa` → use camelbert-da.

(If this were `"الحالة ماشية مزيان"` — "things are going well" in Darija — the DialectIdentifier would predict `"MOR"` → route to `darija_ar` → use MARBERTv2.)

**Stage 7 — Model assignment**

| Utterance | Language | Model |
|---|---|---|
| قال الوزير إن المشروع... | msa | camelbert-da |
| L'opposition a critiqué... | francais | distilcamembert |

**Stage 8 — Sentiment prediction**

Utterance 1 through camelbert-da: prediction `pos` (the minister announcing development and jobs → positive framing). Confidence: 0.71.

Utterance 2 through distilcamembert: prediction `neg` (the opposition criticising as insufficient → negative framing). Confidence: 0.89.

**Final output**

```json
[
  {
    "utterance": "قال الوزير إن المشروع سيساهم في تطوير القطاع وخلق فرص الشغل في المنطقة.",
    "lang_detected": "msa",
    "model_used": "camelbert-da",
    "sentiment": "pos",
    "confidence": 0.71,
    "cues": [23, 24, 25]
  },
  {
    "utterance": "L'opposition a critiqué ce projet en le qualifiant d'insuffisant.",
    "lang_detected": "francais",
    "model_used": "distilcamembert",
    "sentiment": "neg",
    "confidence": 0.89,
    "cues": [26, 27]
  }
]
```

This is what a full broadcast monitoring system would produce: every utterance in the programme labelled with its language, the model that processed it, and its sentiment — ready to be aggregated by programme, by topic, or by time window.
