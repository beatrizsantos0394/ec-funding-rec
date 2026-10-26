# Strategic Diversification Paths — code
```
company description (LLM + web search)           modules/hs22_enrichment.py
        ↓
HS22 product codes (LLM, 3 runs, ≥2 votes)       modules/hs22_classification.py
        ↓
product space, relatedness, complexity           modules/complexity_module.py   (trade.csv)
        ↓  top-10 target products                modules/diversification_context.py
HS22 → HS17 → CPC → ISIC → NACE                  modules/policy_translation.py  (UNSD tables)
        ↓
EU funding recommender, WITH and WITHOUT         modules/funding_recommender.py (web search)
        ↓
real topic pages fetched (Edge via Playwright)   modules/evaluation/topic_html.py
        ↓
metrics, tests, results workbook                 modules/evaluation/metrics.py, scripts/evaluate_results.py,
                                                 scripts/analyze_diversification*.py
        ↓
paired WITH/WITHOUT table                        scripts/paired_table.py
```

Commands are for **Windows PowerShell**, run from the project folder (the one containing `config.py`).

---

## Part A — Set up (once)

### Step 1. Install
You need Python 3.10 or newer and Microsoft Edge (already on Windows 10/11).
Edge is used to read the EU portal's topic pages; if it is missing, run `python -m playwright install msedge`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2. Configure `.env`
```powershell
Copy-Item .env.example .env
```
Edit `.env`:

| Setting | What to put |
|---|---|
| `THESIS_PROJECT_DIR` | full path of this folder (the one with `config.py`). Every script reads `data/` and writes `results/` here |
| `OPEN_AI_API_KEY` | your Azure OpenAI key, spelled with the underscore. Leave empty if you want to use a Windows system variable and using the same naming convention |
| `AZURE_ENDPOINT` | `https://<your-resource>.openai.azure.com/`, **ending with `/`** |
| `AZURE_CHAT_DEPLOYMENT` | name of your chat deployment; it must allow the `web_search` tool |

All three of the last settings are required: `config.py` has empty defaults. Never commit `.env`.

Check that the project folder is read correctly:
```powershell
python -c "import config; print(config.BASE_DIR); print(config.AZURE_ENDPOINT); print(bool(config.OPENAI_API_KEY))"
```
The first line must be this folder, the second your endpoint, the third `True`.

---

## Part B — Get the data (once)

Nothing in `data/` is in git. Create it in these steps.

### Step 3. CORDIS projects
1. Download the **HORIZON projects (xlsx)** export from the CORDIS open-data portal (https://data.europa.eu/data/datasets/cordis-eu-research-projects-under-horizon-europe-2021-2027?locale=en): `cordis-HORIZONprojects-xlsx.zip`. 
2. Extract the files to the folder data/raw/cordis_horizon
### Step 4. Build `data/processed/projects_clean.csv` (by hand, in Excel / Power Query)
The only step done outside Python. Join `project`, `organization` and `topics`, one row per project:

| Rule | Detail |
|---|---|
| Organisation | `role` = coordinator **and** `activityType` = PRC (private company) |
| Date | `ecSignatureDate` on or after 2022-01-01 |
| DOI | `grantDoi` starts with `10.` |
| Topic join | match `{topics, id}` of the project to `{topic, projectID}` of `topics` |
| Import | read the CORDIS sheets with `QuoteStyle.Csv`, or semicolons inside free text break the columns |
| Export | **CSV UTF-8**, delimiter **`;`**. Plain "CSV (comma delimited)" corrupts accents and quotes |

Required columns, in this order:
`id; acronym; status; title; startDate; endDate; topics; ecSignatureDate; masterCall; subCall; fundingScheme; objective; grantDoi; keywords; Human-validated; legalBasis; organizationID; org_VAT; org_name; org_SME; org_activitytype; org_country; org_url; org_role; topic_title; topic_url`

Dates are `M/D/YYYY`. The thesis file has **2,302 projects**.

### Step 5. Draw the sample of 500 companies
```powershell
python scripts/build_pilot_sample_500.py
```
Writes `data/processed/pilot_sample_500.csv`: 500 companies, one per organisation, sampled across call clusters with seed 42, so it is identical every time (it reproduces the thesis sample). The "WARNING … taking all available" lines are normal.

### Step 6. Trade data for the product space
Download the **country × product × year trade file, HS 2022, 6-digit** from the Harvard Growth Lab Atlas (https://atlas.hks.harvard.edu/data-downloads) and save it as `data/raw/complexity/trade.csv`.
Required columns: `year`, `country_iso3_code`, `product_hs22_code`, `export_value`.

### Step 7. Mapping tables (HS22 → NACE)
Download four UNSD correspondence tables from https://unstats.un.org/unsd/classifications/Econ into `data/raw/mapping/`. **Keep the words in the file names**, the code finds each file by name:

| Table | File name must contain | Key columns |
|---|---|---|
| HS 2022 → HS 2017 (xlsx) | `hs2022` | `From HS2022`, `To HS2017` |
| HS 2017 → CPC 2.1 (csv) | `cpc` and `hs2017` | `HS 2017` (dotted, e.g. `101.21`), `CPC Ver. 2.1` |
| CPC 2.1 → ISIC 4 (txt) | `cpc` and `isic4` | `CPC21code`, `ISIC4code` |
| ISIC 4 → NACE 2 (txt) | `isic4` and `nace2` | `ISIC4code`, `NACE2code` |

Typical names: `HS2022toHS2017ConversionAndCorrelationTables.xlsx`, `CPC21-HS2017.csv`, `cpc21-isic4.txt`, `ISIC4_NACE2.txt`.

### Step 8. HS22 labels
Used by the alignment analysis and to name the target products in the prompt.
1. Download `HSCodeandDescription.xlsx` from the UN Statistics Division:
   https://unstats.un.org/unsd/classifications/Econ/download/In%20Text/HSCodeandDescription.xlsx and save it on `data/raw/HS_codes`
2. Build `data/processed/hs22_labels.csv` (columns `hs22_code`, `description`) from the `HS22` sheet, keeping only the 6-digit products:
```powershell
python scripts/build_hs22_labels.py
```
The result has 5,613 rows.
### Step 9. Check that everything is in place
```powershell
python -c "import config as c; [print(p.exists(), p) for p in (c.CORDIS_CANDIDATE_POOL_CSV, c.PROCESSED_DIR/'pilot_sample_500.csv', c.COMPLEXITY_INPUT, c.MAPPING_CSV.parent, c.PROCESSED_DIR/'hs22_labels.csv')]"
```
Every line must print `True`.
```powershell
python -c "import modules.policy_translation as p; print(len(p.load_hs22_to_nace_mapping()))"
```
This must print a number (the rows of the HS22 → NACE mapping, about 14,700 with the full UNSD tables) and not an error.
---

## Part C — Run the pipeline

### Step 10. Try 3 companies first
A full run makes many LLM and web-search calls per company, and the workbook is written only at the very end (a crash loses the run). Test with 3 companies:
```powershell
Copy-Item data\processed\pilot_sample_500.csv data\processed\pilot_sample_500.full.csv
python -c "import pandas as pd; p='data/processed/pilot_sample_500.csv'; pd.read_csv(p,sep=';',encoding='utf-8-sig').head(3).to_csv(p,sep=';',index=False,encoding='utf-8-sig')"
python scripts/pipeline.py
```
Then restore the full sample and move the test workbook out of `results\` (the analysis scripts use the newest file there):
```powershell
Move-Item -Force data\processed\pilot_sample_500.full.csv data\processed\pilot_sample_500.csv
Move-Item results\Recommender_results_N.xlsx $env:TEMP\     # N = the number just written
```
**A good test run** prints for each company a `description:`, a `company_hs22:` list and the complexity step, then fetches the topic pages and ends with
`Saved … recommendation rows, … case results to results\Recommender_results_N.xlsx`. Open it: the `Recommendations` sheet must have WITH, WITHOUT and GROUND_TRUTH rows for each company, and `real_scope` should be filled for most rows (empty cells with `fetch_status = error` mean Edge or the portal failed).
A company can legitimately be skipped ("no description" / "no hs22 codes"), for example a service firm.

### Step 11. Run all 500
```powershell
python scripts/pipeline.py
```
For every company: (1) pre-project description (LLM + web search); (2) HS22 classification, 3 runs, ≥2 votes; (3) top-10 target products from the product space, translated to NACE; (4) the funding recommender WITH the targets (`ec_feasible` variant) and WITHOUT; (5) the real funded topic as `GROUND_TRUTH`.
Then for all companies: fetch the real topic pages (cached in `data/processed/HTML_Topics/`, so reruns are cheap), evaluate against the ground truth, run both diversification analyses and write the workbook.
The first run also computes the complexity metrics and proximity matrix (a few minutes) and caches them as `data/processed/complexity_results.csv` and `proximity_results.csv`; delete those two files if you change `trade.csv`.
Each run writes a new `results\Recommender_results_N.xlsx`; nothing is overwritten.

---

## Part D — Read the results

### Step 12. The workbook
| Sheet | Contents |
|---|---|
| `Recommendations` | one row per recommended topic: `company`, `condition` (WITH / WITHOUT / GROUND_TRUTH), `call_id`, `topic_id`, `match_reason`, the real topic text (`real_scope`, `real_expected_outcome`, `real_trl` …), `target_description` |
| `Case Results`, `Summary` | accuracy vs the real topic: composite score, family hit, exact hit, reciprocal rank |
| `Diversification Cases`, `Diversification Summary` | distinct families / calls / action types, pairwise dissimilarity, share of justifications using diversification language |
| `Alignment Cases`, `Alignment Summary` | text similarity between each recommendation and the targets shown to the model |

### Step 13. The paired table (thesis Table 5.3)
```powershell
python scripts/paired_table.py results\Recommender_results_6.xlsx
```
Prints the table and writes `paired_table.csv` (n, means, difference, 95% bootstrap CI with 2,000 resamples and seed 42, p-value, significance at 0.05/7 = 0.0071). Expected output for Run 6:

| Metric | n | Difference | 95% CI | p |
|---|---|---|---|---|
| Best composite score | 197 | +0.0066 | [−0.0181, 0.0324] | 0.4094 |
| Family hit | 197 | −0.0051 | [−0.0558, 0.0457] | 1.0000 |
| Distinct families | 197 | +0.1218 | [−0.0406, 0.2741] | 0.0724 |
| Distinct calls | 197 | +0.3553 | [0.1066, 0.6142] | **0.0066** |
| Distinct action types | 197 | +0.1066 | [−0.1066, 0.3147] | 0.2479 |
| Pairwise dissimilarity | 143 | +0.0138 | [−0.0053, 0.0357] | 0.3003 |
| Alignment to targets | 167 | +0.0025 | [0.0012, 0.0039] | **0.0008** |

If you have the Run 6 workbook, this is a no-cost test of the analysis: it needs no API and no internet.

---

## Every file

| File | What it is | Run it? |
|---|---|---|
| `config.py` | all paths, settings and constants; reads `.env` | imported, never run |
| `.env.example`, `.gitignore`, `requirements.txt` | settings template, git exclusions, packages | — |
| `scripts/setup_data_folder.py` | creates `data/` folders and unpacks the CORDIS zip | step 3, once |
| `scripts/build_pilot_sample_500.py` | draws the 500-company sample | step 5, once |
| `scripts/pipeline.py` | **the whole experiment**, ends with the results workbook | steps 10–11 |
| `scripts/paired_table.py` | paired WITH/WITHOUT table from a workbook | step 13 |
| `scripts/enrich_topic_csv.py` | adds real topic-page columns to a CSV of recommendations; the pipeline already calls it | only for old CSV files |
| `scripts/evaluate_results.py` | accuracy vs ground truth; called by the pipeline | standalone only on an old `.csv` run |
| `scripts/analyze_diversification.py`, `analyze_diversification_alignment.py` | the two diversification analyses; called by the pipeline | optional, see below |
| `modules/hs22_enrichment.py` | pre-project company description (and project HS22 label, keyword-leakage check) | called by pipeline |
| `modules/hs22_classification.py` | HS22 classification with consensus voting | called by pipeline |
| `modules/complexity_module.py` | ecomplexity metrics and proximity matrix, cached | called by pipeline |
| `modules/diversification_context.py` | relatedness density, feasibility, top-10 targets, prompt text | called by pipeline |
| `modules/policy_translation.py` | HS22→NACE chain and HS22 labels | called by pipeline |
| `modules/funding_recommender.py` | the funding recommender (Azure Responses API + web search) | called by pipeline |
| `modules/evaluation/topic_html.py` | fetches and parses real topic pages with Edge | called by pipeline |
| `modules/evaluation/topic_details.py` | topic family / action-type / TRL helpers (its API-based functions are unused) | called by evaluation |
| `modules/evaluation/metrics.py` | bootstrap CIs, paired tests, composite score, diversification measures | called by evaluation |


## Settings (`config.py`)
| Setting | Value | Meaning |
|---|---|---|
| `CONSENSUS_N_RUNS` / `CONSENSUS_MIN_VOTES` | 3 / 2 | classification and description runs, and votes needed |
| `TOP_N_DIVERSIFICATION_PATHS` | 10 | targets shown to the model |
| `LOGSUPERMODULARITY_SAMPLES` | 85000 | diagnostic only; tuned for the real 231 × 5,612 matrix. On a tiny test matrix it can use far too much memory, so lower it only for toy data |













