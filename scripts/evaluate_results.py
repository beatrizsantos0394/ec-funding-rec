import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
import modules.evaluation.topic_details as topic_details
import modules.evaluation.metrics as metrics


def s(x):
    # pandas reads empty CSV cells as NaN (float), not "" -- normalize to string everywhere
    return "" if pd.isna(x) else str(x)


def tfidf_similarity(text_a, text_b):
    # local, no API -- lexical-overlap proxy for scope/expected_outcome similarity
    if not text_a or not text_b:
        return None
    vecs = TfidfVectorizer(stop_words="english").fit_transform([text_a, text_b])
    return float(cosine_similarity(vecs[0], vecs[1])[0][0])


def evaluate(df: pd.DataFrame):
    """df is a results_df with company/condition/topic_id/real_* columns
    (WITH/WITHOUT/GROUND_TRUTH per company, same shape pipeline.py builds).
    Returns (case_results, summary) DataFrames."""
    rows = []

    for company, g in df.groupby("company"):
        truth = g[g["condition"] == "GROUND_TRUTH"]
        if truth.empty:
            print(f"{company}: no ground truth row, skipping")
            continue
        truth = truth.iloc[0]
        true_topic_id = truth["topic_id"]
        true_family = topic_details.extract_topic_family(true_topic_id)
        true_type = s(truth.get("real_type_of_action", ""))
        true_scope = s(truth.get("real_scope", ""))
        true_trl = s(truth.get("real_trl", ""))
        # some calls (EIC/Mission topics especially) describe results under
        # "Expected Impact" instead of or alongside "Expected Outcome" --
        # fold both into one field so neither gets silently ignored
        true_outcome = f"{s(truth.get('real_expected_outcome', ''))} {s(truth.get('real_expected_impact', ''))}".strip()
        true_text = f"{true_scope} {true_outcome}".strip()

        for condition in ["WITH", "WITHOUT"]:
            candidates = g[g["condition"] == condition]
            if candidates.empty:
                continue

            best_score = None
            exact_hit = 0
            family_hit = 0
            families = set()

            for _, cand in candidates.iterrows():
                cand_topic_id = cand["topic_id"]
                cand_family = topic_details.extract_topic_family(cand_topic_id)
                families.add(cand_family)

                if str(cand_topic_id) == str(true_topic_id):
                    exact_hit = 1
                if cand_family and cand_family == true_family:
                    family_hit = 1

                cand_scope = s(cand.get("real_scope", ""))
                cand_outcome = f"{s(cand.get('real_expected_outcome', ''))} {s(cand.get('real_expected_impact', ''))}".strip()
                cand_type = s(cand.get("real_type_of_action", ""))
                cand_trl = s(cand.get("real_trl", ""))
                cand_text = f"{cand_scope} {cand_outcome}".strip()
                similarity = tfidf_similarity(true_text, cand_text)

                score = metrics.weighted_topic_score(
                    true_topic_id=true_topic_id, true_type_of_action=true_type, true_trl=true_trl,
                    true_scope=true_scope, true_expected_outcome=true_outcome,
                    candidate_topic_id=cand_topic_id, candidate_type_of_action=cand_type,
                    candidate_trl=cand_trl, candidate_scope=cand_scope, candidate_expected_outcome=cand_outcome,
                    embedding_similarity=similarity,
                )
                if best_score is None or score["composite_score"] > best_score["composite_score"]:
                    best_score = score

            rr = metrics.reciprocal_rank(candidates, true_topic_id)

            rows.append({
                "case_id": company,
                "prompt_variant": condition,  # reusing metrics.py's summarize_by_variant/paired_test as-is
                "best_composite_score": best_score["composite_score"] if best_score else 0.0,
                "exact_hit": exact_hit,
                "family_hit": family_hit,
                "n_distinct_families": len(families),
                "reciprocal_rank": rr,
            })
            print(f"{company} [{condition}]: best_score={rows[-1]['best_composite_score']:.3f} "
                  f"exact_hit={exact_hit} family_hit={family_hit} "
                  f"n_distinct_families={len(families)} rr={rr:.3f}")

    case_results = pd.DataFrame(rows)

    summary = metrics.summarize_by_variant(
        case_results,
        metric_cols=["best_composite_score", "exact_hit", "family_hit", "n_distinct_families", "reciprocal_rank"],
    )

    print("\n=== Summary by condition (mean + 95% CI) ===")
    print(summary.to_string(index=False))

    print("\n=== WITH vs WITHOUT paired tests ===")
    print("composite score (Wilcoxon):", metrics.paired_test(case_results, "best_composite_score", "WITHOUT", "WITH"))
    print("exact_hit (McNemar):", metrics.paired_test(case_results, "exact_hit", "WITHOUT", "WITH", binary=True))
    print("family_hit (McNemar):", metrics.paired_test(case_results, "family_hit", "WITHOUT", "WITH", binary=True))
    print("n_distinct_families (Wilcoxon, diversification proof):",
          metrics.paired_test(case_results, "n_distinct_families", "WITHOUT", "WITH"))

    return case_results, summary


if __name__ == "__main__":
    # pick the latest Recommender_results_*.csv/.xlsx in results/, evaluate it standalone
    existing = list(config.RESULTS_DIR.glob("Recommender_results_*.csv"))
    latest = max(existing, key=lambda p: p.stat().st_mtime)
    print(f"Evaluating {latest}")
    df = pd.read_csv(latest, sep=";", encoding="utf-8-sig")
    case_results, summary = evaluate(df)
    case_results.to_csv(config.RESULTS_DIR / "evaluation_case_results.csv", sep=";", index=False, encoding="utf-8-sig")
    summary.to_csv(config.RESULTS_DIR / "evaluation_summary.csv", sep=";", index=False, encoding="utf-8-sig")