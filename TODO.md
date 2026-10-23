# TODO

Suggested order: README → pluggable encoders → pluggable stats tests → UI.
The UI's "pick an encoder / pick tests" options depend on the two registries existing first.

## 1. README for GitHub
- [x] Project overview: what the Voynich hypothesis is and what this pipeline tests
- [x] How it works: encoder → decoder round-trip → statistical comparison against the Voynich (EVA) text
- [x] Notebook map (`00_pipeline` … `04_analysis`) and what each one does
- [x] Setup on Databricks: catalog/schema/volume (`01_setup`), uploading source files, `CATALOG` setting
- [x] Running a sweep: parameters (`num_seeds`, `text_size`, `encode_seed`, `run_id`) and what rule seed vs. encode seed control
- [x] Interpreting results: Zipf slope, entropy, token lengths, letter predictability, pass/fail criteria
- [ ] Sample output (table / Zipf plot screenshot)
- [ ] Data sources and licensing/attribution for the input texts
- [ ] ~~Contributing section~~ (dropped for now)

## 2. Pluggable encoders / decoders
- [x] Define an encoder interface (e.g. `name`, `generate_config(seed)`, `encode(text, config, encode_seed)`, `decode(tokens, config)`)
- [x] Wrap the current anchor/mutation encoder in `02_encoder_lib.py` as the first implementation
- [x] Encoder registry (name → implementation) so a run can select one by name
- [x] Add `encoder` parameter to `03_sweep` and record it in the results tables
- [x] Update Delta table schemas in `01_setup` (encoder name, encoder version/params)
- [ ] Round-trip test harness that any new encoder must pass (encode → decode == cleaned plaintext)
- [x] Add at least one second encoder (e.g. a simple baseline/control cipher) to prove the abstraction

## 3. Pluggable statistical tests
- [ ] Define a test interface (e.g. `name`, `compute(tokens) -> dict`, optional pass/fail threshold)
- [ ] Move existing tests (Zipf, entropy, basic stats, token lengths, letter predictability) into a registry
- [ ] Make `run_all_stats` iterate over a selected list of tests instead of a hard-coded set
- [ ] Add `tests` parameter to `03_sweep`; store results in a long/narrow format so new metrics don't require schema changes
- [ ] Make `04_analysis` / `00_pipeline` render whatever metrics are present rather than hard-coded columns
- [ ] Candidate new tests:
  - [ ] Heaps' law (vocabulary growth)
  - [ ] Conditional / second-order entropy (h1, h2)
  - [ ] Word-position effects (line-initial / line-final token distributions)
  - [ ] Character n-gram distributions vs. Voynich
  - [ ] Repetition / reduplication rate (consecutive identical or near-identical words)
  - [ ] Distance metric between encoded output and Voynich across all metrics (single comparable score)

## 4. UI for business users / admins
- [ ] Decide on platform: Databricks App (Streamlit/Dash/Gradio) vs. notebook widgets/dashboard vs. external web app
- [ ] **Configure runs**
  - [ ] Seeds: number of rule seeds, explicit seed list/range, encode seed
  - [ ] Text inputs and text size
  - [ ] Encoder / decoder selection (from registry in section 2)
  - [ ] Statistical tests / analytics selection (from registry in section 3)
  - [ ] Save / load named run configurations (presets)
- [ ] **Launch runs**
  - [ ] Trigger via Databricks Jobs API (parameterized job wrapping `03_sweep`)
  - [ ] Show run status / progress, link to job run, surface errors
- [ ] **View results / analytics**
  - [ ] Run history list with config summary
  - [ ] Best-seed comparison table, Zipf plots, metric distributions
  - [ ] Compare two or more runs side by side
  - [ ] Export results (CSV)
- [ ] Access control: admin vs. read-only users (Unity Catalog permissions / app auth)

## Housekeeping (noticed along the way)
- [ ] Move `CATALOG` to a single config location instead of editing it in multiple notebooks
- [ ] Replace `dbutils.notebook.run()` fallback note in `00_pipeline` with a Databricks Job definition
- [ ] Pin dependency versions (`scipy`, `scikit-learn`, `matplotlib`)
- [x] Generate the VMS vocabulary from the transcription (`05_build_vocab`): fixes dropped foldout/`fRos` pages, `?` words and markup leaking into words
- [x] Compute VMS baselines with the same stat tests as the cipher output (`vms_baseline.json`) instead of hard-coding them
- [x] Record `vocab_id` on every result so runs from different vocabularies aren't mixed
- [x] Make letter predictability deterministic (sort word types before the CV split)
- [ ] Note which transcription `voynich-ms.txt` is (file header says `IVTFF Eva- 2.0`) in the README data sources
