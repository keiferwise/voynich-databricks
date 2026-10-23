# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 00 — Pipeline Entry Point
# MAGIC
# MAGIC Runs the full pipeline end to end:
# MAGIC 1. **Seed Sweep** — Calls `03_sweep` to generate and evaluate cipher variants
# MAGIC 2. **Analysis** — Queries results and produces comparison tables, plots, and metrics
# MAGIC
# MAGIC **Before first run:**
# MAGIC 1. Run `01_setup.py` once to create the schema, volume, and tables
# MAGIC 2. Upload source files to the volume via Catalog > voynich > Volumes > raw_files > Upload
# MAGIC 3. Update `CATALOG` in `02_encoder_lib.py` if your default catalog is not `main`
# MAGIC
# MAGIC **Note:** The seed sweep uses `dbutils.notebook.run()` to call `03_sweep`. If this
# MAGIC errors with a "not supported" message, just run `03_sweep` manually with the same
# MAGIC parameters — the analysis cells below will still work.

# COMMAND ----------

from datetime import datetime
CATALOG = "workspace"
SCHEMA  = "voynich"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 50)
print(f"  Voynich Pipeline — Run {RUN_ID}")
print("=" * 50)

# COMMAND ----------

# MAGIC %md ## Step 1 — Seed Sweep

# COMMAND ----------

# DBTITLE 1,Cell 4
print("Step 1/2: Running seed sweep...")

try:
    sweep_result = dbutils.notebook.run(
        "./03_sweep",
        timeout_seconds=7000,
        arguments={
            "num_seeds": "2",
            "text_size": "40000",
            "run_id":    RUN_ID,
            "encode_seed": "0",
            "encoder":     "anchor_mutation",  # or any name in ENCODERS (02_encoder_lib)
        },
    )
    print(f"Sweep complete. Run ID: {sweep_result}")

except Exception as e:
    error_msg = str(e)
    
    # Check for serverless compatibility issues
    if "not supported" in error_msg.lower():
        print("dbutils.notebook.run() is not available in this environment.")
        print("Please run 03_sweep and 04_analysis manually in order.")
        print(f"Set the run_id widget to: {RUN_ID}")
        dbutils.notebook.exit(f"Manual run required. Use run_id={RUN_ID}")
    
    # Check for workload failure (notebook ran but errored)
    elif "workload failed" in error_msg.lower():
        print("\n" + "="*50)
        print("  03_sweep failed during execution")
        print("="*50)
        print("\nThe notebook ran but encountered an error.")
        print("Run 03_sweep directly to see the actual error message.")
        print(f"\nSteps to debug:")
        print("  1. Open the 03_sweep notebook")
        print(f"  2. Set the run_id widget to: {RUN_ID}")
        print("  3. Run all cells to see where it fails")
        print("\nCommon issues:")
        print("  • Run 01_setup first to create schema/volume/tables")
        print("  • Upload source files to Catalog > voynich > Volumes > raw_files")
        print("  • Check CATALOG setting in 02_encoder_lib")
        dbutils.notebook.exit(f"Sweep failed. Debug via 03_sweep with run_id={RUN_ID}")
    
    else:
        raise

# COMMAND ----------

# MAGIC %md ## Step 2 — Analysis

# COMMAND ----------

# DBTITLE 1,Load encoder library
# MAGIC %run ./02_encoder_lib

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install scipy scikit-learn matplotlib

# COMMAND ----------

# DBTITLE 1,Load results
print("Step 2/2: Running analysis...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

df_all = spark.table(f"{CATALOG}.{SCHEMA}.results").toPandas()

df = df_all[df_all["run_id"] == RUN_ID].copy()
print(f"Loaded run: {RUN_ID}")

df_pass = df[df["decode_ok"] == True].copy()

total   = len(df)
passing = len(df_pass)
print(f"Seeds: {passing}/{total} passed decode ({passing/total:.1%})")

# COMMAND ----------

# DBTITLE 1,Best-seed comparison
# MAGIC %md ### Best-Seed Comparison Table

# COMMAND ----------

# DBTITLE 1,Show best seeds
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

        slope_col = "zipf_slope"
        r2_col    = "zipf_r2"

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
            ("hapax_ratio",                      "Hapax ratio",                   "hapax_ratio"),
            ("entropy",                          "Unigram entropy",               "unigram_entropy"),
            ("letter_unigram_predictability",   "Letter unigram predictability", "letter_unigram_pred"),
            ("letter_bigram_predictability",    "Letter bigram predictability",  "letter_bigram_pred"),
            ("avg_length",                      "Avg token length",              "avg_length"),
        ]:
            if col in best and pd.notna(best[col]):
                pct = "%" if "predictability" in col else ""
                print(f"  {label:<38} {best[col]:>8.4f}{pct} {VOYNICH[key]:>8.4f}{pct}")

        if "vocab_size" in best:
            print(f"  {'Vocab size':<38} {int(best['vocab_size']):>9} {VOYNICH['vocab_size']:>9}")


show_best_seeds(df_pass, TEXT_NAMES)

# COMMAND ----------

# DBTITLE 1,Zipf slope distribution header
# MAGIC %md ### Zipf Slope Distribution

# COMMAND ----------

# DBTITLE 1,Plot Zipf distribution
fig, axes = plt.subplots(1, len(TEXT_NAMES), figsize=(7 * len(TEXT_NAMES), 5))
if len(TEXT_NAMES) == 1:
    axes = [axes]

for ax, text_name in zip(axes, TEXT_NAMES):
    sub       = df_pass[df_pass["text_name"] == text_name]
    slope_col = "zipf_slope"
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

# DBTITLE 1,Zipf log-log header
# MAGIC %md ### Zipf Log-Log Plot (Best Seed per Text)

# COMMAND ----------

# DBTITLE 1,Plot Zipf log-log
fig, axes = plt.subplots(1, len(TEXT_NAMES), figsize=(8 * len(TEXT_NAMES), 5))
if len(TEXT_NAMES) == 1:
    axes = [axes]

TEXT_FILE_MAP = {
    "bible":  f"{VOLUME_PATH}/akjv.txt",
    "newton": f"{VOLUME_PATH}/newton.txt",
}

for ax, text_name in zip(axes, TEXT_NAMES):
    sub       = df_pass[df_pass["text_name"] == text_name].copy()
    slope_col = "zipf_slope"
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

    text    = clean_text(raw[:40000])  # TEXT_SIZE hardcoded since variable doesn't exist here
    # Rows written before the encoder column existed all used anchor_mutation
    encoder = get_encoder(best_row["encoder"] if pd.notna(best_row.get("encoder")) else DEFAULT_ENCODER)
    config  = encoder.generate_config(best_seed)
    encoded = encoder.encode(text, config, best_encode_seed)
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
    ax.set_title(f"{text_name.capitalize()} — Zipf plot ({encoder.name}, seed {best_seed})")
    ax.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Raw results header
# MAGIC %md ### Raw Results Table

# COMMAND ----------

# DBTITLE 1,Show raw results
display(
    spark.table(f"{CATALOG}.{SCHEMA}.results")
         .filter(f"run_id = '{RUN_ID}'")
         .filter("decode_ok = true")
         .orderBy("text_name", "seed")
)

# COMMAND ----------

# DBTITLE 1,Failed runs header
# MAGIC %md ### Failed Runs (if any)

# COMMAND ----------

# DBTITLE 1,Show failures
failed = spark.table(f"{CATALOG}.{SCHEMA}.failed_runs") \
              .filter(f"run_id = '{RUN_ID}'")

count = failed.count()
if count > 0:
    print(f"{count} failure(s) recorded:")
    display(failed.orderBy("seed", "text_name"))
else:
    print("No failures recorded for this run.")

# COMMAND ----------

# DBTITLE 1,Historical pass rate header
# MAGIC %md ### Historical Pass Rate (All Runs)

# COMMAND ----------

# DBTITLE 1,Show historical pass rate
spark.sql(f"""
  SELECT
    run_id,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN decode_ok THEN 1 ELSE 0 END)                  AS passed,
    ROUND(AVG(CASE WHEN decode_ok THEN 1.0 ELSE 0 END) * 100, 1) AS pass_pct,
    ROUND(AVG(CASE WHEN decode_ok THEN zipf_slope END), 4)      AS mean_zipf_slope
  FROM {CATALOG}.{SCHEMA}.results
  GROUP BY run_id
  ORDER BY run_id DESC
""").show(truncate=False)

# COMMAND ----------

print("\n" + "=" * 50)
print(f"  Pipeline complete — Run ID: {RUN_ID}")
print("=" * 50)
print()
print("View experiment results:")
print("  MLflow > Experiments > voynich-cipher-sweep")
print()
print("Query results directly:")
print(f"  SELECT * FROM main.voynich.results WHERE run_id = '{RUN_ID}'")
print()
print("Review any decode failures:")
print(f"  SELECT * FROM main.voynich.failed_runs WHERE run_id = '{RUN_ID}'")

# COMMAND ----------


