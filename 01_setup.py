# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Setup
# MAGIC Run this notebook **once** to create the Unity Catalog schema, volume, and
# MAGIC Delta tables.
# MAGIC
# MAGIC **Before running:** check your catalog name in the Catalog Explorer on the left
# MAGIC sidebar and update `CATALOG` below if it is not `main`.
# MAGIC
# MAGIC After running, upload the four source files manually:
# MAGIC **Catalog > voynich > Volumes > raw_files > Upload**

# COMMAND ----------

# ── Configuration ────────────────────────────────────────────────────────────
CATALOG = "workspace"     # ← update if your default catalog is not "main"
SCHEMA  = "voynich"
VOLUME  = "raw_files"
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %pip install scipy scikit-learn

# COMMAND ----------

# MAGIC %md ### Schema and Volume

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema {CATALOG}.{SCHEMA} ready.")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Volume  {CATALOG}.{SCHEMA}.{VOLUME} ready.")

print()
print("Next: upload these four files via Catalog > voynich > Volumes > raw_files > Upload:")
print("  - voynich-ms.txt")
print("  - akjv.txt")
print("  - newton.txt")
print("  - voynichtokencounts.csv")

# COMMAND ----------

# MAGIC %md ### Delta Tables

# COMMAND ----------

# Results table — one row per seed x text
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.results (
    run_id        STRING    NOT NULL,
    seed          INT       NOT NULL,
    encode_seed   INT,
    encoder       STRING,
    text_name     STRING    NOT NULL,
    decode_ok     BOOLEAN   NOT NULL,
    zipf_slope    DOUBLE,
    zipf_r2       DOUBLE,
    hapax_ratio   DOUBLE,
    entropy       DOUBLE,
    vocab_size    INT,
    token_count   INT,
    avg_length    DOUBLE,
    letter_unigram_predictability DOUBLE,
    letter_bigram_predictability  DOUBLE,
    created_at    TIMESTAMP NOT NULL
  ) USING DELTA
  PARTITIONED BY (text_name)
  TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print(f"Table {CATALOG}.{SCHEMA}.results created.")

# COMMAND ----------

# Ciphertext table — full encoded token sequence per seed
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.ciphertext (
    run_id        STRING    NOT NULL,
    seed          INT       NOT NULL,
    encode_seed   INT,
    encoder       STRING,
    text_name     STRING    NOT NULL,
    position      INT       NOT NULL,
    token         STRING    NOT NULL,
    is_gallows    BOOLEAN   NOT NULL,
    is_anchor     BOOLEAN   NOT NULL,
    created_at    TIMESTAMP NOT NULL
  ) USING DELTA
  PARTITIONED BY (seed, text_name)
""")
print(f"Table {CATALOG}.{SCHEMA}.ciphertext created.")

# COMMAND ----------

# Failed runs table — decode failures and exceptions with error details
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.failed_runs (
    run_id        STRING    NOT NULL,
    seed          INT       NOT NULL,
    encode_seed   INT,
    encoder       STRING,
    text_name     STRING    NOT NULL,
    error_type    STRING,
    error_message STRING,
    created_at    TIMESTAMP NOT NULL
  ) USING DELTA
""")
print(f"Table {CATALOG}.{SCHEMA}.failed_runs created.")

# COMMAND ----------

# Delta constraints — reject obviously bad rows at write time
spark.sql(f"""
  ALTER TABLE {CATALOG}.{SCHEMA}.results
  ADD CONSTRAINT valid_seed CHECK (seed >= 0)
""")
spark.sql(f"""
  ALTER TABLE {CATALOG}.{SCHEMA}.results
  ADD CONSTRAINT valid_hapax CHECK (hapax_ratio IS NULL OR (hapax_ratio >= 0 AND hapax_ratio <= 1))
""")
spark.sql(f"""
  ALTER TABLE {CATALOG}.{SCHEMA}.results
  ADD CONSTRAINT valid_zipf_r2 CHECK (zipf_r2 IS NULL OR (zipf_r2 >= 0 AND zipf_r2 <= 1))
""")
print("Constraints added.")

# COMMAND ----------

print("\n=== Setup complete ===")
spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").show()