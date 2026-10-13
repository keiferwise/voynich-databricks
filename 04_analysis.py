# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Analysis
# MAGIC
# MAGIC Queries the Delta results table and produces:
# MAGIC - Best-seed comparison table against VMS baseline
# MAGIC - Zipf slope distribution plots
# MAGIC - Zipf log-log plot for the best seed per text
# MAGIC - Raw results table for the selected run
# MAGIC - Historical pass rate across all runs
# MAGIC
# MAGIC **Before running:** update `CATALOG` in `02_encoder_lib` if your default
# MAGIC catalog is not `main`.

# COMMAND ----------

# MAGIC %run ./02_encoder_lib

# COMMAND ----------

# MAGIC %pip install scipy scikit-learn matplotlib

# COMMAND ----------

# MAGIC %md ### Parameters

# COMMAND ----------

dbutils.widgets.text("run_id", "", "Run ID (blank = latest)")

run_id = dbutils.widgets.get("run_id").strip()
CATALOG = "workspace"     # ← update if your default catalog is not "main"
SCHEMA  = "voynich"
VOLUME  = "raw_files"

# COMMAND ----------

# MAGIC %md ### Load Results

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

df_all = spark.table(f"{CATALOG}.{SCHEMA}.results").toPandas()

if run_id:
    df = df_all[df_all["run_id"] == run_id].copy()
    print(f"Loaded run: {run_id}")
else:
    latest = df_all["run_id"].max()
    df     = df_all[df_all["run_id"] == latest].copy()
    print(f"Loaded latest run: {latest}")

df_pass = df[df["decode_ok"] == True].copy()

total   = len(df)
passing = len(df_pass)
print(f"Seeds: {passing}/{total} passed decode ({passing/total:.1%})")

# COMMAND ----------

# MAGIC %md ### Best-Seed Comparison Table

# COMMAND ----------

# Real VMS baseline values
VOYNICH = {
    "zipf_slope":          -0.894,
    "zipf_r2":              0.911,
    "hapax_ratio":          0.698,
    "unigram_entropy":      9.666,
    "letter_unigram_pred": 14.65,
    "letter_bigram_pred":  40.02,
    "avg_length":           6.2,
    "vocab_size":          7091,
}

TEXT_NAMES = sorted(df_pass["text_name"].unique().tolist())


def show_best_seeds(df_pass: pd.DataFrame, text_names: list):
    for text_name in text_names:
        sub = df_pass[df_pass["text_name"] == text_name].copy()
        if sub.empty:
            print(f"\nNo passing seeds for {text_name}")
            continue

        slope_col = f"{text_name}_zipf_slope"
        r2_col    = f"{text_name}_zipf_r_squared"

        sub["_zipf_dist"] = (sub[slope_col] - VOYNICH["zipf_slope"]).abs()
        best = sub.loc[sub["_zipf_dist"].idxmin()]

        print(f"\n{'='*55}")
        print(f"  {text_name.upper()} — Best seed: {int(best['seed'])}")
        print(f"{'='*55}")
        print(f"  {'Metric':<38} {'Cipher':>9} {'Voynich':>9}")
        print(f"  {'-'*55}")
        print(f"  {'Zipf slope':<38} {best[slope_col]:>9.4f} {VOYNICH['zipf_slope']:>9.4f}")
        print(f"  {'Zipf R²':<38} {best[r2_col]:>9.4f} {VOYNICH['zipf_r2']:>9.4f}")

        for col, label, key in [
            (f"{text_name}_hapax_ratio",                      "Hapax ratio",                   "hapax_ratio"),
            (f"{text_name}_unigram_entropy",                  "Unigram entropy",               "unigram_entropy"),
            (f"{text_name}_letter_unigram_predictability",    "Letter unigram predictability", "letter_unigram_pred"),
            (f"{text_name}_letter_bigram_predictability",     "Letter bigram predictability",  "letter_bigram_pred"),
            (f"{text_name}_avg_length",                       "Avg token length",              "avg_length"),
        ]:
            if col in best and pd.notna(best[col]):
                pct = "%" if "predictability" in col else ""
                print(f"  {label:<38} {best[col]:>8.4f}{pct} {VOYNICH[key]:>8.4f}{pct}")

        if f"{text_name}_unique_words" in best:
            print(f"  {'Vocab size':<38} {int(best[f'{text_name}_unique_words']):>9} {VOYNICH['vocab_size']:>9}")


show_best_seeds(df_pass, TEXT_NAMES)

# COMMAND ----------

# MAGIC %md ### Zipf Slope Distribution

# COMMAND ----------

fig, axes = plt.subplots(1, len(TEXT_NAMES), figsize=(7 * len(TEXT_NAMES), 5))
if len(TEXT_NAMES) == 1:
    axes = [axes]

for ax, text_name in zip(axes, TEXT_NAMES):
    sub       = df_pass[df_pass["text_name"] == text_name]
    slope_col = f"{text_name}_zipf_slope"
    if slope_col not in sub.columns:
        continue

    values = sub[slope_col].dropna()
    ax.hist(values, bins=20, edgecolor="black", alpha=0.8)
    ax.axvline(VOYNICH["zipf_slope"], color="red",  linestyle="--",
               label=f"VMS ({VOYNICH['zipf_slope']})")
    ax.axvline(values.mean(), color="blue", linestyle=":",
               label=f"mean ({values.mean():.3f})")
    ax.set_xlabel("Zipf slope")
    ax.set_ylabel("Seed count")
    ax.set_title(f"{text_name.capitalize()} — Zipf slope distribution")
    ax.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ### Zipf Log-Log Plot (Best Seed per Text)

# COMMAND ----------

fig, axes = plt.subplots(1, len(TEXT_NAMES), figsize=(8 * len(TEXT_NAMES), 5))
if len(TEXT_NAMES) == 1:
    axes = [axes]

TEXT_FILE_MAP = {
    "bible":  f"{VOLUME_PATH}/akjv.txt",
    "newton": f"{VOLUME_PATH}/newton.txt",
}

for ax, text_name in zip(axes, TEXT_NAMES):
    sub       = df_pass[df_pass["text_name"] == text_name].copy()
    slope_col = f"{text_name}_zipf_slope"
    if slope_col not in sub.columns or sub.empty:
        continue

    sub["_zipf_dist"] = (sub[slope_col] - VOYNICH["zipf_slope"]).abs()
    best_row  = sub.loc[sub["_zipf_dist"].idxmin()]
    best_seed = int(best_row["seed"])
    # Re-encode with the same encode seed so the plot shows the scored ciphertext
    best_encode_seed = int(best_row["encode_seed"]) if pd.notna(best_row.get("encode_seed")) else 0

    text_file = TEXT_FILE_MAP.get(text_name)
    if not text_file:
        continue

    with open(text_file, encoding="utf-8") as f:
        raw = f.read()

    text    = clean_text(raw[:TEXT_SIZE])
    config  = generate_random_config(seed=best_seed)
    encoded = encode_text(text, config, encode_seed=best_encode_seed)
    tokens  = [t for t in encoded if t not in GALLOWS and t != "." and t != "\n"]

    from collections import Counter
    freq   = Counter(tokens)
    ranks  = np.arange(1, len(freq) + 1)
    counts = np.array([c for _, c in freq.most_common()])

    slope, intercept, *_ = scipy_stats.linregress(np.log(ranks), np.log(counts))

    ax.scatter(np.log(ranks), np.log(counts), s=4, alpha=0.7, label="Cipher output")
    ax.plot(np.log(ranks), slope * np.log(ranks) + intercept,
            color="red",   label=f"Fit (slope={slope:.3f})")
    ax.plot(np.log(ranks), VOYNICH["zipf_slope"] * np.log(ranks) + intercept,
            color="green", linestyle="--", label=f"VMS slope ({VOYNICH['zipf_slope']})")
    ax.set_xlabel("log(rank)")
    ax.set_ylabel("log(frequency)")
    ax.set_title(f"{text_name.capitalize()} — Zipf plot (seed {best_seed})")
    ax.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ### Raw Results Table

# COMMAND ----------

display(
    spark.table(f"{CATALOG}.{SCHEMA}.results")
         .filter(f"run_id = '{df['run_id'].iloc[0]}'")
         .filter("decode_ok = true")
         .orderBy("text_name", "seed")
)

# COMMAND ----------

# MAGIC %md ### Failed Runs (if any)

# COMMAND ----------

failed = spark.table(f"{CATALOG}.{SCHEMA}.failed_runs") \
              .filter(f"run_id = '{df['run_id'].iloc[0]}'")

count = failed.count()
if count > 0:
    print(f"{count} failure(s) recorded:")
    display(failed.orderBy("seed", "text_name"))
else:
    print("No failures recorded for this run.")

# COMMAND ----------

# MAGIC %md ### Historical Pass Rate (All Runs)

# COMMAND ----------

spark.sql(f"""
  SELECT
    run_id,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN decode_ok THEN 1 ELSE 0 END)                  AS passed,
    ROUND(AVG(CASE WHEN decode_ok THEN 1.0 ELSE 0 END) * 100, 1) AS pass_pct,
    ROUND(AVG(CASE WHEN decode_ok THEN bible_zipf_slope END), 4)  AS mean_zipf_slope
  FROM {CATALOG}.{SCHEMA}.results
  GROUP BY run_id
  ORDER BY run_id DESC
""").show(truncate=False)