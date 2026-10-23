# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Build Vocabulary
# MAGIC
# MAGIC Parses the EVA transcription (`voynich-ms.txt`, IVTFF format) and writes:
# MAGIC - `voynichtokencounts.csv` — every VMS word with its count, used by the encoder
# MAGIC   to pick anchor and base words
# MAGIC - `vms_baseline.json` — the VMS statistics the analysis compares cipher output against
# MAGIC
# MAGIC Run once after uploading `voynich-ms.txt`, and again whenever you switch to a
# MAGIC different transcription. **Rebuilding changes every seed's results** — the encoder
# MAGIC samples words from this list — so results record a `vocab_id` (hash of the CSV)
# MAGIC and should only be compared within the same vocabulary.
# MAGIC
# MAGIC **Parsing rules**
# MAGIC - Every text line is used: normal pages, foldout panels (`f67r1`, `f85r2`, …),
# MAGIC   the Rosettes page (`fRos`), and all unit types (paragraphs, labels, circles, radii)
# MAGIC - `.` is a word break; `,` (uncertain break) is a word break too (`COMMA_IS_BREAK`)
# MAGIC - Inline tags (`<->` drawing gaps, `<@H=2>` hand changes, `<!…>` comments) break words
# MAGIC   and are otherwise ignored
# MAGIC - `[a:b]` alternate readings → first reading; `{…}` braces removed; `!` `%` fillers removed
# MAGIC - Words with unreadable glyphs (`?`, `*`) or any other non-EVA character are dropped
# MAGIC   and reported
# MAGIC - Ordering is count (descending), then alphabetical — the same input always gives
# MAGIC   a byte-identical CSV

# COMMAND ----------

# MAGIC %pip install scipy scikit-learn

# COMMAND ----------

# ── Configuration ────────────────────────────────────────────────────────────
CATALOG = "workspace"     # ← update if your catalog is not "workspace"
SCHEMA  = "voynich"
VOLUME  = "raw_files"

VOLUME_PATH        = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
TRANSCRIPTION_PATH = f"{VOLUME_PATH}/voynich-ms.txt"
VOCAB_OUT_PATH     = f"{VOLUME_PATH}/voynichtokencounts.csv"
BASELINE_OUT_PATH  = f"{VOLUME_PATH}/vms_baseline.json"

COMMA_IS_BREAK = True     # False = join the two halves of an uncertain break
TRANSCRIBER    = None     # e.g. "H" for multi-transcriber IVTFF files; None = single-transcriber file
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md ### Parser

# COMMAND ----------

import re
from collections import Counter

PAGE_RE    = re.compile(r"^<([^.>\s]+)>")
# <page.line,unit;transcriber>  e.g. <f1r.1,@P0>  <f67r1.12,&Lz>  <fRos.3,@L0>  <f1r.1,@P0;H>
LOCUS_RE   = re.compile(r"^<([^.>\s]+)\.([^,;>\s]+)(?:,([^;>]*))?(?:;([^>]*))?>\s*(.*)$")
INLINE_TAG = re.compile(r"<[^>]*>")
ALT_RE     = re.compile(r"\[([^:\]]*)(?::[^\]]*)*\]")
EVA_WORD   = re.compile(r"^[a-z]+$")
UNREADABLE = set("?*")


def split_words(text: str, comma_is_break: bool = True) -> list:
    """Split one line of IVTFF text into raw words (not yet validated)."""
    text = INLINE_TAG.sub(".", text)
    text = ALT_RE.sub(r"\1", text)
    text = re.sub(r"[{}!%]", "", text)
    if not comma_is_break:
        text = text.replace(",", "")
    return [w for w in re.split(r"[.,\s]+", text) if w]


def parse_ivtff(path: str, comma_is_break: bool = True, transcriber: str = None) -> dict:
    """
    Parse an IVTFF transcription.
    Returns {"words": [...in manuscript order], "dropped": Counter, "pages": set,
             "lines": int, "units": Counter}.
    """
    words, pages, units = [], set(), Counter()
    dropped = {"unreadable": Counter(), "invalid": Counter()}
    transcribers = set()
    lines = 0

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            m = LOCUS_RE.match(line)
            if not m:
                if PAGE_RE.match(line):
                    pages.add(PAGE_RE.match(line).group(1))
                    continue
                raise ValueError(f"Unrecognised line: {line[:80]!r}")

            page, _, unit, who, text = m.groups()
            transcribers.add(who)
            if transcriber is not None and who != transcriber:
                continue

            lines += 1
            pages.add(page)
            kind = re.search(r"[A-Z]", unit or "")
            units[kind.group(0) if kind else "?"] += 1   # P, L, C, R …

            for w in split_words(text, comma_is_break):
                if UNREADABLE & set(w):
                    dropped["unreadable"][w] += 1
                elif not EVA_WORD.match(w):
                    dropped["invalid"][w] += 1
                else:
                    words.append(w)

    if transcriber is None and len(transcribers) > 1:
        raise ValueError(f"File has several transcribers {sorted(t for t in transcribers if t)}; "
                         "set TRANSCRIBER to pick one")

    return {"words": words, "dropped": dropped, "pages": pages, "lines": lines, "units": units}


def vocab_csv(words: list) -> str:
    """Token counts as CSV text: count descending, then alphabetical."""
    counts = Counter(words)
    rows   = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return "token,count\n" + "".join(f"{t},{c}\n" for t, c in rows)

# COMMAND ----------

# MAGIC %md ### Parse and write the vocabulary

# COMMAND ----------

parsed = parse_ivtff(TRANSCRIPTION_PATH, COMMA_IS_BREAK, TRANSCRIBER)
VMS_WORDS = parsed["words"]

print(f"Pages:        {len(parsed['pages'])}")
print(f"Text lines:   {parsed['lines']}")
print(f"Unit types:   {dict(parsed['units'])}   (P=paragraph, L=label, C=circle, R=radius)")
print(f"Words kept:   {len(VMS_WORDS):,}")
print(f"Vocabulary:   {len(set(VMS_WORDS)):,} distinct words")
for reason, c in parsed["dropped"].items():
    print(f"Dropped ({reason}): {sum(c.values())} words, e.g. {list(c)[:8]}")

with open(VOCAB_OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(vocab_csv(VMS_WORDS))
print(f"\nWrote {VOCAB_OUT_PATH}")

# COMMAND ----------

# MAGIC %md ### VMS baseline statistics
# MAGIC Computed with the same stat tests the sweep runs on cipher output.

# COMMAND ----------

# MAGIC %run ./02_encoder_lib

# COMMAND ----------

import json

stats    = run_all_stats(VMS_WORDS, "vms")
baseline = {
    "zipf_slope":          stats["vms_zipf_slope"],
    "zipf_r2":             stats["vms_zipf_r_squared"],
    "hapax_ratio":         stats["vms_hapax_ratio"],
    "unigram_entropy":     stats["vms_unigram_entropy"],
    "letter_unigram_pred": stats["vms_letter_unigram_predictability"],
    "letter_bigram_pred":  stats["vms_letter_bigram_predictability"],
    "avg_length":          stats["vms_avg_length"],
    "vocab_size":          stats["vms_vocab_size"],
    "token_count":         stats["vms_token_count"],
    "vocab_id":            VOCAB_ID,
}

with open(BASELINE_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2)

print(json.dumps(baseline, indent=2))
print(f"\nWrote {BASELINE_OUT_PATH}")
