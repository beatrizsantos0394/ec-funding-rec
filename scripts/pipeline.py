import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
from datetime import date
import pandas as pd

import config
import modules.hs22_enrichment as hs22_enrichment
import modules.hs22_classification as hs22_classification
import modules.complexity_module as complexity_module
import modules.diversification_context as diversification_context
import modules.policy_translation as policy_translation
import modules.funding_recommender as funding_recommender
from enrich_topic_csv import enrich_df
from evaluate_results import evaluate
from analyze_diversification import analyze as analyze_diversification
from analyze_diversification_alignment import analyze as analyze_alignment

cases = pd.read_csv(config.PROCESSED_DIR / "pilot_sample_500.csv", sep=";", encoding="utf-8-sig")

client = hs22_classification.get_client()

print("Loading complexity data...")
cdata, prox_df = complexity_module.run_complexity_analysis()
mapping_df = policy_translation.load_hs22_to_nace_mapping()
hs22_labels = policy_translation.load_hs22_labels()

rows = []

for _, row in cases.iterrows():
    company = row["org_name"]
    country = row["org_country"]
    print(f"\n=== {company} ===")

    # enrich: pre-project description (the system's input)
    enrichment = hs22_enrichment.enrich_case(
        objective=row["objective"],
        keywords=row["keywords"],
        title=row["title"],
        company_name=company,
        country=country,
        as_of_date=row["ecSignatureDate"],
        client=client,
    )
    description = enrichment["company_products_description"]
    print("description:", description)

    if not description:
        print("no description, skipping")
        continue

    # classification
    classification = hs22_classification.run_hs22_classification(description, client)
    company_hs22 = [c["hs_code"] for c in classification["company_hs22"]]
    print("company_hs22:", company_hs22)

    if not company_hs22:
        print("no hs22 codes, skipping")
        continue

    # complexity -> nearest/diversification candidates
    year = date.today().year
    div_table = diversification_context.compute_diversification_table(company_hs22, prox_df, cdata, year=year)
    div_table = policy_translation.translate_to_nace(div_table, mapping_df)
    context_text = diversification_context.build_diversification_context(div_table, variant="ec_feasible")

    # for scripts/analyze_diversification_alignment.py: pull the exact
    # target_hs22 codes that were actually shown to the model in
    # context_text (not re-derived from div_table, so this can't drift
    # from what the LLM actually saw), and turn them into readable text
    # via the HS22 label lookup so they're comparable to real topic text
    target_hs22_codes = re.findall(r"target_hs22=(\d+)", context_text)
    target_description = "; ".join(
        hs22_labels[c] for c in target_hs22_codes if c in hs22_labels
    )

    company_profile = {"name": company, "country": country, "sme": True, "description": description}

    # recommender: with and without diversification context
    for condition, div_context in [("WITH", context_text), ("WITHOUT", None)]:
        result = funding_recommender.recommend_eu_funds(
            company_profile, diversification_context=div_context, client=client,
        )
        for m in result["funding_matches"]:
            topic_id = m.get("topic_id", "")
            rows.append({
                "company": company,
                "condition": condition,
                "call_id": m.get("call_id", ""),
                "call_title": m.get("call_title", ""),
                "topic_id": topic_id,
                "match_reason": m.get("match_reason", ""),
                "topic_url": f"https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/{topic_id}",
                "target_description": target_description,
            })

    # ground truth: the topic this company's real project was actually funded under
    true_topic_id = row["topics"]
    rows.append({
        "company": company,
        "condition": "GROUND_TRUTH",
        "call_id": "",
        "call_title": "",
        "topic_id": true_topic_id,
        "topic_url": f"https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/{true_topic_id}",
    })

results_df = pd.DataFrame(rows)

# last step: same enrichment as scripts/enrich_topic_csv.py, called directly
# instead of run separately on the terminal
print("\nFetching real topic pages...")
cache_dir = config.PROCESSED_DIR / "HTML_Topics"
results_df = enrich_df(results_df, topic_col="topic_id", cache_dir=cache_dir)

# evaluate WITH vs WITHOUT against ground truth (predictability of the framework)
print("\nEvaluating...")
case_results, summary = evaluate(results_df)

# separate analysis: WITH vs WITHOUT diversification of the recommendations
# themselves, independent of ground truth
print("\nAnalyzing diversification...")
div_case_results, div_summary = analyze_diversification(results_df)

# does WITH move recommendations toward the EC module's own flagged
# targets (target_description), more directly than the family/call/type
# counts above -- see scripts/analyze_diversification_alignment.py
print("\nAnalyzing diversification alignment...")
align_case_results, align_summary = analyze_alignment(results_df)

existing = list(config.RESULTS_DIR.glob("Recommender_results_*.xlsx"))
existing_nums = [int(p.stem.rsplit("_", 1)[-1]) for p in existing if p.stem.rsplit("_", 1)[-1].isdigit()]
iteration = max(existing_nums, default=0) + 1
out_path = config.RESULTS_DIR / f"Recommender_results_{iteration}.xlsx"

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Recommendations", index=False)
    case_results.to_excel(writer, sheet_name="Case Results", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)
    div_case_results.to_excel(writer, sheet_name="Diversification Cases", index=False)
    div_summary.to_excel(writer, sheet_name="Diversification Summary", index=False)
    align_case_results.to_excel(writer, sheet_name="Alignment Cases", index=False)
    align_summary.to_excel(writer, sheet_name="Alignment Summary", index=False)

print(f"\nSaved {len(results_df)} recommendation rows, {len(case_results)} case results to {out_path}")