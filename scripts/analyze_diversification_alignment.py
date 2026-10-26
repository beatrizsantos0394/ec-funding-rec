"""
scripts/analyze_diversification_alignment.py

The most direct test of the hypothesis: does giving the model economic
complexity diversification context (WITH) actually move its recommendations
TOWARD the specific product/NACE targets the EC module flagged for that
company, more than WITHOUT does? The other diversification metrics
(analyze_diversification.py) only measure the SHAPE of the recommendation
set (how spread out it is across families/calls/types) -- none of them
check whether it points toward the EC-computed targets specifically.

Requires pipeline.py to have saved a "target_description" column per
company (the human-readable HS22 descriptions of the exact target_hs22
codes shown to the model in the WITH prompt, via modules/policy_translation.py's
load_hs22_labels()). Only runs on results that have that column -- older
result files (e.g. Recommender_results_4.xlsx and earlier) don't have it
and can't be re-analyzed this way retroactively, since the target codes
were never saved.

Per company x condition (WITH/WITHOUT):
  alignment_score -- average TF-IDF similarity between the company's
                     target_description text and each recommended topic's
                     own real_scope/real_expected_outcome/real_expected_impact
                     text. Higher = recommendations textually resemble the
                     EC-flagged targets more.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
import modules.evaluation.metrics as metrics
from evaluate_results import s, tfidf_similarity


def analyze(df: pd.DataFrame):
    rows = []

    for company, g in df.groupby("company"):
        target_description = s(g["target_description"].dropna().iloc[0]) if "target_description" in g.columns and g["target_description"].notna().any() else ""
        if not target_description:
            print(f"{company}: no target_description, skipping (rerun pipeline.py to capture it)")
            continue

        for condition in ["WITH", "WITHOUT"]:
            candidates = g[g["condition"] == condition]
            if candidates.empty:
                continue

            sims = []
            for _, cand in candidates.iterrows():
                cand_text = f"{s(cand.get('real_scope', ''))} {s(cand.get('real_expected_outcome', ''))} {s(cand.get('real_expected_impact', ''))}".strip()
                sim = tfidf_similarity(target_description, cand_text)
                if sim is not None:
                    sims.append(sim)

            alignment_score = sum(sims) / len(sims) if sims else None

            rows.append({
                "case_id": company,
                "prompt_variant": condition,
                "n_recommendations": len(candidates),
                "alignment_score": alignment_score,
            })
            print(f"{company} [{condition}]: n={len(candidates)} alignment_score={alignment_score}")

    case_results = pd.DataFrame(rows)

    if case_results.empty:
        print("\nNo cases with target_description -- nothing to summarize.")
        return case_results, pd.DataFrame()

    summary = metrics.summarize_by_variant(case_results, metric_cols=["alignment_score"])
    print("\n=== Alignment summary by condition (mean + 95% CI) ===")
    print(summary.to_string(index=False))

    print("\n=== WITH vs WITHOUT paired test (diversification alignment) ===")
    print("alignment_score:", metrics.paired_test(case_results, "alignment_score", "WITHOUT", "WITH"))

    return case_results, summary


if __name__ == "__main__":
    existing = list(config.RESULTS_DIR.glob("Recommender_results_*.xlsx"))
    latest = max(existing, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing {latest}")
    df = pd.read_excel(latest, sheet_name="Recommendations")
    case_results, summary = analyze(df)

    out_path = config.RESULTS_DIR / f"{latest.stem}_alignment.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        case_results.to_excel(writer, sheet_name="Case Results", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"\nSaved to {out_path}")