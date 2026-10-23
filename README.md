# Voynich Cipher Sweep on Databricks

An experiment that asks a narrow question about the Voynich Manuscript (VMS):

> Can a simple, fully decodable cipher turn ordinary English text into something
> with the same statistical fingerprint as the Voynich Manuscript?

The pipeline encodes known plaintexts (the King James Bible and a Newton text) into
Voynich-style words using a randomly generated rule set. It checks that each ciphertext
decodes back to the original exactly, then measures how closely the ciphertext's
statistics (Zipf's law, entropy, hapax ratio, word length, letter predictability)
match the real manuscript. Running this across many random seeds shows which rule sets,
if any, get close.

Everything runs as Databricks notebooks. Results go to Unity Catalog Delta tables
and are tracked in MLflow.

---

## How the cipher works

Ciphertext words are written in [EVA](https://www.voynich.nu/extra/eva.html), the
standard transcription alphabet for the Voynich script. **Each plaintext letter
becomes one EVA word**, and the word encodes how far the alphabet moved since the
previous letter (`distance = (current − previous) mod 26`).

A rule set, generated from a **rule seed**, has three kinds of words:

| Rule type     | What it encodes | How the word is chosen |
|---------------|-----------------|------------------------|
| **Anchor**    | An absolute letter (a–z) | One real VMS word per letter, sampled by frequency from the VMS vocabulary |
| **Base word** | A fixed distance (0–25) | One more real VMS word per distance |
| **Mutation**  | A distance | Changes an earlier word (1–3 words back) by swapping, adding or removing a prefix, suffix, or core fragment (e.g. `qo-`, `ch-`, `-edy`, `-aiin`) |

The many mutation rules are what make the output look like the manuscript: they create
families of similar words (`qokeedy`, `okeedy`, `qokedy`, …) the way the real text does.

During encoding, several words are often valid for the same letter. The **encode seed**
decides which one is used. A mutation is only allowed when the result can't be confused
with an anchor, a base word, or another rule's output, so decoding is always deterministic
and never needs the encode seed.

- **Rule seed** → which rule set (the key)
- **Encode seed** → which of that key's many valid ciphertexts you get
- Same rule seed + same encode seed → the same ciphertext every time

Other details: the text is lower-cased and stripped to `a–z`. The ciphertext starts
with a gallows character (`k`, `t`, `f`, `p`) followed by an anchor word. The text is
encoded in 100-character chunks, which become lines.

### Encoders

The cipher above is one of several encoders registered in `ENCODERS` in
`02_encoder_lib.py`. A sweep picks one by name with its `encoder` widget.

| Encoder | Description |
|---------|-------------|
| `anchor_mutation` (default) | The cipher described above. |
| `substitution` | Control baseline. Each letter always becomes the same VMS word, so the output has only 26 word types. It shows what a naive cipher's statistics look like next to the manuscript. The encode seed has no effect. |

Every encoder implements the same three methods: `generate_config(seed)`,
`encode(text, config, encode_seed)` and `decode(tokens, config)`. The sweep only
calls those, so adding an encoder means writing a subclass of `Encoder` and adding it
to `ENCODERS`.

---

## Repository layout

| Notebook | Purpose |
|----------|---------|
| [`00_pipeline.py`](00_pipeline.py) | Entry point. Runs `03_sweep` through `dbutils.notebook.run()`, then shows the analysis for that run inline. |
| [`01_setup.py`](01_setup.py) | Run once. Creates the schema, the volume, and the `results`, `ciphertext`, and `failed_runs` Delta tables. |
| [`02_encoder_lib.py`](02_encoder_lib.py) | Library, loaded with `%run`. EVA parser, rule-set generation, encoder, decoder, the encoder registry (`ENCODERS`), and the statistical tests registry (`STAT_TESTS`). |
| [`03_sweep.py`](03_sweep.py) | Encodes every text with each seed in parallel, checks the decode, runs the stats, and writes to Delta and MLflow. |
| [`04_analysis.py`](04_analysis.py) | Standalone analysis of one run: best-seed table, Zipf plots, raw results, failures, and pass rate over time. |
| [`05_build_vocab.py`](05_build_vocab.py) | Run once after uploading the transcription. Parses `voynich-ms.txt` into the VMS vocabulary (`voynichtokencounts.csv`) and computes the VMS baseline statistics (`vms_baseline.json`). |

All files are Databricks notebooks in source format (`# Databricks notebook source`).
Import the repo into your workspace with **Repos / Git folders**.

---

## Setup

### 1. Set the catalog name

Each notebook has a `CATALOG` setting near the top, currently `"workspace"`. If your
Unity Catalog catalog has a different name, change it in **all six notebooks**. Also
change the hard-coded `VOLUME_PATH` in `03_sweep.py`
(`/Volumes/workspace/voynich/raw_files`).

### 2. Create the schema, volume and tables

Run `01_setup.py` once. It creates:

- Schema `<CATALOG>.voynich`
- Volume `<CATALOG>.voynich.raw_files`
- Tables `results`, `ciphertext`, `failed_runs`

### 3. Upload the source files

Go to **Catalog → voynich → Volumes → raw_files → Upload** and add:

| File | Used for |
|------|----------|
| `voynich-ms.txt` | EVA transcription of the manuscript in IVTFF format. The VMS vocabulary and baseline are built from it. |
| `akjv.txt` | Plaintext corpus `bible` (Authorized King James Version) |
| `newton.txt` | Plaintext corpus `newton` |

### 4. Build the VMS vocabulary

Run `05_build_vocab.py`. It parses the transcription and writes two files to the volume:

- `voynichtokencounts.csv`: every VMS word and its count. The encoder samples anchor
  and base words from this list.
- `vms_baseline.json`: the manuscript's own statistics, computed with the same stat
  tests that score the cipher output. The analysis compares results against these.

How the transcription is parsed:

- Every text line is used: normal pages, foldout panels (`f67r1`, `f85r2`, …), the
  Rosettes page (`fRos`), and labels, circles and radial lines as well as paragraphs.
- `.` and `,` (an uncertain break) both separate words. Inline tags such as `<->`
  (a gap where a drawing interrupts the line) also separate words and are otherwise ignored.
- Words containing an unreadable glyph (`?`) are dropped. The notebook reports how many.
- Words are sorted by count and then alphabetically, so the same transcription always
  produces exactly the same file.

To switch to a different transcription, upload it as `voynich-ms.txt` and run this
notebook again. **This changes every seed's results**, because the encoder picks words
by their position in the list. Each result row records a `vocab_id` (a hash of the
vocabulary file), so only compare runs that share one.

`03_sweep` stops with a clear error if a source file is missing or the vocabulary hasn't
been built.

### 5. Dependencies

The notebooks install what they need with `%pip install scipy scikit-learn matplotlib`.
Pandas, NumPy, PySpark, and MLflow come with the Databricks runtime. The notebooks work
on serverless compute (MLflow is set up for the Unity Catalog registry).

---

## Running a sweep

**Option A: whole pipeline.** Open `00_pipeline.py`, change the arguments passed to
`03_sweep` in Step 1 if you need to, and click Run all.

**Option B: run each step yourself.** Use this if `dbutils.notebook.run()` isn't
available in your environment. Run `03_sweep.py` with its widgets set, then run
`04_analysis.py` with the same `run_id`.

### Parameters (`03_sweep` widgets)

| Widget | Default | Meaning |
|--------|---------|---------|
| `num_seeds` | `1` | Number of rule seeds to sweep. Seeds `0 … num_seeds-1` are used. |
| `text_size` | `40000` | Characters read from each source text before cleaning |
| `encode_seed` | `0` | Seed for choosing between valid spellings (see above) |
| `run_id` | timestamp | Label for the run. Leave blank to use `YYYYMMDD_HHMMSS`. |
| `encoder` | `anchor_mutation` | Which registered encoder to use (dropdown) |

Each seed is processed for every text in a thread pool of 4 workers. Any exception
becomes a row in `failed_runs`, so one bad seed doesn't stop the sweep.

---

## Outputs

### Delta tables (`<CATALOG>.voynich.*`)

| Table | Grain | Contents |
|-------|-------|----------|
| `results` | one row per run × seed × text | `encoder`, `vocab_id`, `decode_ok`, and every metric below |
| `ciphertext` | one row per token | Full encoded sequence, with `is_gallows` and `is_anchor` flags |
| `failed_runs` | one row per failure | Error type and message for decode errors or exceptions |

### MLflow

Experiment **`/voynich-cipher-sweep`**. Each sweep gets a parent run with its
parameters, `pass_rate`, `failed_seeds`, and `mean_zipf_slope`. Each seed × text gets a
nested child run with all of its metrics.

---

## Interpreting results

**Pass / fail.** A seed *passes* for a text when `decode(encode(text)) == text`.
Only passing rows are compared against the Voynich statistics.

**Metrics.** Gallows, line breaks, and separators are removed before the stats are
computed. The VMS baselines come from `vms_baseline.json`, which `05_build_vocab`
computes with the same tests. The values below are for the current transcription.

| Metric | What it measures | VMS baseline |
|--------|------------------|-------------:|
| `zipf_slope` | Slope of log(frequency) vs. log(rank). Natural language is about −1. | −0.891 |
| `zipf_r2` | How well the word frequencies fit a power law | 0.910 |
| `hapax_ratio` | Share of distinct words that appear exactly once | 0.700 |
| `entropy` | Shannon entropy of the word distribution, in bits | 10.49 |
| `avg_length` | Mean word length in characters | 5.06 |
| `vocab_size` | Number of distinct words | 8418 |
| `letter_unigram_predictability` | % of characters matching the most common character, with 10-fold CV over word types | 12.97 % |
| `letter_bigram_predictability` | % of next characters correctly predicted from the previous one, with 10-fold CV | 38.32 % |

**Best seed.** For each text, the analysis picks the passing seed whose Zipf slope is
closest to the VMS value. It shows that seed's metrics next to the baseline, along with
the spread of Zipf slopes across seeds, a log-log rank/frequency plot, and the pass rate
across all past runs.

Some useful queries:

```sql
-- One run
SELECT * FROM workspace.voynich.results WHERE run_id = '<run_id>';

-- Closest Zipf slopes across all runs
SELECT run_id, encoder, seed, encode_seed, text_name, zipf_slope, hapax_ratio, entropy
FROM workspace.voynich.results
WHERE decode_ok AND vocab_id = '<vocab_id>'
ORDER BY abs(zipf_slope - (-0.891))
LIMIT 20;
```

---

## Roadmap

See [`TODO.md`](TODO.md). Planned work includes more encoders, more
statistical tests, and a UI for configuring, launching, and reviewing runs.
