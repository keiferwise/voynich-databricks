# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Seed Sweep
# MAGIC
# MAGIC Encodes two plaintext corpora across N random seeds, verifies decode,
# MAGIC computes all registered statistical tests, and writes results to Delta + MLflow.
# MAGIC
# MAGIC **Before running:** update `CATALOG` in `02_encoder_lib` if your default
# MAGIC catalog is not `main`.
# MAGIC
# MAGIC **Parameters** (set via widgets or passed from `00_pipeline.py`):
# MAGIC - `num_seeds` — number of random seeds to sweep (default 100)
# MAGIC - `text_size` — characters of plaintext to use per corpus (default 40000)
# MAGIC - `run_id`    — unique run identifier; auto-generated if blank
# MAGIC - `encode_seed` — seeds the encoder's spelling choices (default 0). The rule
# MAGIC   seed picks the rule set; the encode seed picks which of its many valid
# MAGIC   ciphertexts you get. Same pair = same ciphertext.
# MAGIC - `encoder`   — which registered encoder to use (default `anchor_mutation`);
# MAGIC   see `ENCODERS` in `02_encoder_lib`

# COMMAND ----------

CATALOG = "workspace"

# COMMAND ----------

# MAGIC %run ./02_encoder_lib

# COMMAND ----------

# MAGIC %pip install scipy scikit-learn

# COMMAND ----------

# MAGIC %md ### Parameters

# COMMAND ----------

dbutils.widgets.text("num_seeds", "1",   "Number of seeds")
dbutils.widgets.text("text_size", "40000", "Characters per text")
dbutils.widgets.text("run_id",    "",      "Run ID (blank = auto)")
dbutils.widgets.text("encode_seed", "0",   "Encode seed")
dbutils.widgets.dropdown("encoder", DEFAULT_ENCODER, list(ENCODERS.keys()), "Encoder")

from datetime import datetime

NUM_SEEDS = int(dbutils.widgets.get("num_seeds"))
TEXT_SIZE = int(dbutils.widgets.get("text_size"))
RUN_ID    = dbutils.widgets.get("run_id").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
ENCODE_SEED = int(dbutils.widgets.get("encode_seed"))
ENCODER   = get_encoder(dbutils.widgets.get("encoder"))

print(f"Run ID:    {RUN_ID}")
print(f"Seeds:     {NUM_SEEDS}")
print(f"Text size: {TEXT_SIZE} chars per corpus")
print(f"Encode seed: {ENCODE_SEED}")
print(f"Encoder:   {ENCODER.name}")

# COMMAND ----------

# MAGIC %md ### Pre-flight Checks

# COMMAND ----------

import os
VOLUME_PATH = "/Volumes/workspace/voynich/raw_files"
REQUIRED_FILES = [
    f"{VOLUME_PATH}/akjv.txt",
    f"{VOLUME_PATH}/newton.txt",
    f"{VOLUME_PATH}/voynichtokencounts.csv",
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(
        "Pipeline aborted — missing source files:\n" + "\n".join(missing) +
        "\n\nUpload them via Catalog > voynich > Volumes > raw_files > Upload"
    )
print("Pre-flight checks passed.")

# COMMAND ----------

# MAGIC %md ### Load Texts

# COMMAND ----------

with open(f"{VOLUME_PATH}/akjv.txt",   encoding="utf-8") as f:
    bible_raw = f.read()
with open(f"{VOLUME_PATH}/newton.txt", encoding="utf-8") as f:
    newton_raw = f.read()

TEXTS = {
    "bible":  clean_text(bible_raw[:TEXT_SIZE]),
    "newton": clean_text(newton_raw[:TEXT_SIZE]),
}

print("Texts loaded:")
for name, text in TEXTS.items():
    print(f"  {name}: {len(text):,} characters")

# COMMAND ----------

# MAGIC %md ### Sweep

# COMMAND ----------

# DBTITLE 1,Cell 11
import mlflow
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# Configure MLflow for serverless (Unity Catalog registry)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/voynich-cipher-sweep")


def run_seed(seed: int) -> tuple:
    """
    Encode + decode + stats for one seed across all texts.
    Returns (result_rows, ciphertext_rows, failure_rows).
    Never raises — all exceptions are captured as failure rows.
    """
    config        = ENCODER.generate_config(seed)
    anchor_tokens = ENCODER.anchor_tokens(config)
    result_rows, cipher_rows, failure_rows = [], [], []
    ts = datetime.now(timezone.utc)

    for text_name, text in TEXTS.items():
        try:
            encoded   = ENCODER.encode(text, config, ENCODE_SEED)
            decoded   = ENCODER.decode(encoded, config)
            decode_ok = (decoded == text)

            row = {
                "run_id":     RUN_ID,
                "seed":       seed,
                "encode_seed": ENCODE_SEED,
                "encoder":    ENCODER.name,
                "text_name":  text_name,
                "decode_ok":  decode_ok,
                "created_at": ts,
            }

            if decode_ok:
                row.update(run_all_stats(encoded, text_name))
            else:
                failure_rows.append({
                    "run_id":        RUN_ID,
                    "seed":          seed,
                    "encode_seed":   ENCODE_SEED,
                    "encoder":       ENCODER.name,
                    "text_name":     text_name,
                    "error_type":    "decode_mismatch",
                    "error_message": f"decoded length={len(decoded)}, expected={len(text)}",
                    "created_at":    ts,
                })

            result_rows.append(row)

            for i, token in enumerate(encoded):
                cipher_rows.append({
                    "run_id":     RUN_ID,
                    "seed":       seed,
                    "encode_seed": ENCODE_SEED,
                    "encoder":    ENCODER.name,
                    "text_name":  text_name,
                    "position":   i,
                    "token":      token,
                    "is_gallows": token in GALLOWS,
                    "is_anchor":  token in anchor_tokens,
                    "created_at": ts,
                })

        except Exception as e:
            failure_rows.append({
                "run_id":        RUN_ID,
                "seed":          seed,
                "encode_seed":   ENCODE_SEED,
                "encoder":       ENCODER.name,
                "text_name":     text_name,
                "error_type":    type(e).__name__,
                "error_message": str(e),
                "created_at":    ts,
            })

    return result_rows, cipher_rows, failure_rows


# --- Parent MLflow run (sweep-level) ---
with mlflow.start_run(run_name=RUN_ID) as parent_run:

    mlflow.log_params({
        "num_seeds":  NUM_SEEDS,
        "encode_seed": ENCODE_SEED,
        "encoder":    ENCODER.name,
        "text_size":  TEXT_SIZE,
        **({"anchor_prob": 0.15} if ENCODER.name == "anchor_mutation" else {}),
        "stat_tests": str(list(STAT_TESTS.keys())),
    })
    mlflow.set_tag("status", "running")

    all_results, all_cipher, all_failures = [], [], []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_seed, s): s for s in range(NUM_SEEDS)}

        for future in futures:
            seed = futures[future]
            res, ciph, fails = future.result()
            all_results.extend(res)
            all_cipher.extend(ciph)
            all_failures.extend(fails)

            # Child MLflow run — one per seed x text
            for row in res:
                with mlflow.start_run(
                    run_name=f"seed_{seed}_{row['text_name']}",
                    nested=True
                ):
                    mlflow.log_param("seed",      seed)
                    mlflow.log_param("encode_seed", ENCODE_SEED)
                    mlflow.log_param("encoder",   ENCODER.name)
                    mlflow.log_param("text_name", row["text_name"])
                    mlflow.set_tag("decode_ok",   str(row["decode_ok"]))
                    if row["decode_ok"]:
                        numeric_metrics = {
                            k: v for k, v in row.items()
                            if isinstance(v, (int, float)) and k not in ("seed", "encode_seed")
                        }
                        mlflow.log_metrics(numeric_metrics)

    # Sweep-level summary metrics
    passing   = [r for r in all_results if r.get("decode_ok")]
    pass_rate = len(passing) / len(all_results) if all_results else 0
    zipf_vals = [r.get("bible_zipf_slope") for r in passing
                 if r.get("bible_zipf_slope") is not None]
    mean_zipf = round(sum(zipf_vals) / len(zipf_vals), 4) if zipf_vals else None

    mlflow.log_metrics({
        "pass_rate":    round(pass_rate, 4),
        "failed_seeds": len(all_failures),
        **({"mean_zipf_slope": mean_zipf} if mean_zipf is not None else {}),
    })
    mlflow.set_tag("status", "completed")

# COMMAND ----------

all_results_df = spark.createDataFrame(pd.DataFrame(all_results))

all_results_df.display()

# COMMAND ----------

# MAGIC %md ### Write to Delta

# COMMAND ----------

# DBTITLE 1,Cell 14
import pandas as pd
from pyspark.sql.functions import col

# Reshape: extract columns matching each row's text_name and remove prefix
def reshape_results(rows):
    reshaped = []
    for row in rows:
        text_name = row["text_name"]
        prefix = f"{text_name}_"
        
        # Start with core columns
        new_row = {
            "run_id": row["run_id"],
            "seed": row["seed"],
            "encode_seed": row["encode_seed"],
            "encoder": row["encoder"],
            "text_name": text_name,
            "decode_ok": row["decode_ok"],
            "created_at": row["created_at"],
        }
        
        # Extract prefixed columns and rename
        for key, val in row.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]  # Remove prefix
                new_row[new_key] = val
        
        reshaped.append(new_row)
    return reshaped

reshaped_results = reshape_results(all_results)

# Rename columns and select only those in table schema
from pyspark.sql.functions import col

df = spark.createDataFrame(pd.DataFrame(reshaped_results)) \
     .withColumn("seed", col("seed").cast("int")) \
     .withColumn("encode_seed", col("encode_seed").cast("int")) \
     .withColumn("vocab_size", col("vocab_size").cast("int")) \
     .withColumn("token_count", col("token_count").cast("int")) \
     .withColumnRenamed("zipf_r_squared", "zipf_r2") \
     .withColumnRenamed("unigram_entropy", "entropy") \
     .select(
         "run_id", "seed", "encode_seed", "encoder", "text_name", "decode_ok",
         "zipf_slope", "zipf_r2", "hapax_ratio", "entropy",
         "vocab_size", "token_count", "avg_length",
         "letter_unigram_predictability", "letter_bigram_predictability",
         "created_at"
     )

# mergeSchema adds the encode_seed column to tables created before it existed
df.write.mode("append").option("mergeSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.results")
print(f"Wrote {len(reshaped_results)} rows to {CATALOG}.{SCHEMA}.results")

spark.createDataFrame(pd.DataFrame(all_cipher)) \
     .withColumn("seed", col("seed").cast("int")) \
     .withColumn("encode_seed", col("encode_seed").cast("int")) \
     .withColumn("position", col("position").cast("int")) \
     .write.mode("append").option("mergeSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.ciphertext")
print(f"Wrote {len(all_cipher)} rows to {CATALOG}.{SCHEMA}.ciphertext")

if all_failures:
    spark.createDataFrame(pd.DataFrame(all_failures)) \
         .withColumn("seed", col("seed").cast("int")) \
         .withColumn("encode_seed", col("encode_seed").cast("int")) \
         .write.mode("append").option("mergeSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.failed_runs")
    print(f"Wrote {len(all_failures)} rows to {CATALOG}.{SCHEMA}.failed_runs")

# COMMAND ----------

spark.sql("SHOW TABLES IN voynich").show()

# COMMAND ----------

spark.sql("SHOW CATALOGS").show()


# COMMAND ----------

dbutils.fs.ls("dbfs:/")

# COMMAND ----------

# MAGIC %md ### Summary

# COMMAND ----------

print(f"\n{'='*45}")
print(f"  Run:        {RUN_ID}")
print(f"  Seeds:      {NUM_SEEDS}")
print(f"  Pass rate:  {pass_rate:.1%}  ({len(passing)}/{len(all_results)} text encodings)")
print(f"  Failures:   {len(all_failures)}")
if mean_zipf:
    print(f"  Mean Zipf:  {mean_zipf:.4f}  (VMS baseline: -0.894)")
print(f"{'='*45}")
print()
print("View in MLflow: Experiments > voynich-cipher-sweep")
print(f"Query results:  SELECT * FROM {CATALOG}.{SCHEMA}.results WHERE run_id = '{RUN_ID}'")

dbutils.notebook.exit(RUN_ID)