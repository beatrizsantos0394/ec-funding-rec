"""
scripts/analyze_diversification.py

Separate from evaluate_results.py on purpose: that script asks "does the
system predict the real funded topic" (accuracy vs ground truth).
This script asks a different question -- "does giving the model economic
complexity diversification context change the SHAPE of what it recommends,
independent of whether any of it matches the real outcome." No ground
truth is used here at all.

Per company x condition (WITH/WITHOUT), computed purely from the
recommended set itself:
  n_distinct_families     -- how many different call families recommended
                              (coarsest level, e.g. "HLTH" -- ignores year
                              and call round, can span several calls)
  n_distinct_calls        -- how many different specific call rounds
                              recommended (e.g. "HORIZON-HLTH-2026-01" --
                              in between family and exact topic_id: two
                              topics can share a family but belong to
                              different calls, or belong to the very same
                              call)
  n_distinct_type         -- how many different type-of-action categories
  avg_pairwise_dissimilarity -- how different the recommended topics' own
                                 scope/outcome text are from EACH OTHER
                                 (1 - TF-IDF cosine similarity, averaged
                                 over all pairs). Higher = more spread out.
  frac_diversification_language -- MANIPULATION CHECK, not a diversification
                                 outcome measure: the fraction of this
                                 condition's match_reason justifications
                                 that explicitly invoke diversification /
                                 capability-extension vocabulary. Confirms
                                 whether the EC context is actually shaping
                                 the model's stated reasoning, independent
                                 of whether that reasoning changes which
                                 topics get recommended (the other metrics
                                 above test that outcome).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from itertools import combinations
import pandas as pd

import config
import modules.evaluation.topic_details as topic_details
import modules.evaluation.metrics as metrics
from evaluate_results import s, tfidf_similarity

# manipulation-check vocabulary -- direct synonyms of the construct itself
# (diversification / capability extension), fixed before inspecting which
# condition they'd favor, not tuned post-hoc to maximize a gap
DIVERSIFICATION_LANGUAGE = [
    "diversif", "new capabilit", "adjacent", "expand", "extend its",
    "move into", "moving into", "branch into",
]


def mentions_diversification_language(text: str) -> bool:
    t = s(text).lower()
    return any(k in t for k in DIVERSIFICATION_LANGUAGE)


def analyze(df: pd.DataFrame):
    rows = []

    for company, g in df.groupby("company"):
        for condition in ["WITH", "WITHOUT"]:
            candidates = g[g["condition"] == condition]
            if candidates.empty:
                continue

            families = set()
            calls = set()
            types = set()
            texts = []
            div_language_flags = []

            for _, cand in candidates.iterrows():
                topic_id = cand["topic_id"]
                families.add(topic_details.extract_topic_family(topic_id))
                call_id = s(cand.get("call_id", "")).strip()
                if call_id:
                    calls.add(call_id)
                types.add(topic_details.normalize_action_type(s(cand.get("real_type_of_action", ""))))
                div_language_flags.append(mentions_diversification_language(cand.get("match_reason", "")))

                text = f"{s(cand.get('real_scope', ''))} {s(cand.get('real_expected_outcome', ''))} {s(cand.get('real_expected_impact', ''))}".strip()
                if text:
                    texts.append(text)

            frac_div_language = (
                sum(div_language_flags) / len(div_language_flags) if div_language_flags else None
            )

            # average pairwise dissimilarity among this condition's own
            # recommended topics -- not compared to any ground truth
            pairs = list(combinations(texts, 2))
            dissimilarities = []
            for text_a, text_b in pairs:
                sim = tfidf_similarity(text_a, text_b)
                if sim is not None:
                    dissimilarities.append(1 - sim)
            avg_dissimilarity = sum(dissimilarities) / len(dissimilarities) if dissimilarities else None

            rows.append({
                "case_id": company,
                "prompt_variant": condition,
                "n_recommendations": len(candidates),
                "n_distinct_families": len(families),
                "n_distinct_calls": len(calls),
                "n_distinct_type": len(types - {""}),
                "avg_pairwise_dissimilarity": avg_dissimilarity,
                "frac_diversification_language": frac_div_language,
            })
            print(f"{company} [{condition}]: n={len(candidates)} "
                  f"n_distinct_families={len(families)} n_distinct_calls={len(calls)} "
                  f"n_distinct_type={len(types - {''})} "
                  f"avg_pairwise_dissimilarity={avg_dissimilarity} "
                  f"frac_diversification_language={frac_div_language}")

    case_results = pd.DataFrame(rows)

    summary = metrics.summarize_by_variant(
        case_results,
        metric_cols=[
            "n_distinct_families", "n_distinct_calls", "n_distinct_type",
            "avg_pairwise_dissimilarity", "frac_diversification_language",
        ],
    )
    print("\n=== Diversification summary by condition (mean + 95% CI) ===")
    print(summary.to_string(index=False))

    print("\n=== WITH vs WITHOUT paired tests (diversification only) ===")
    print("n_distinct_families:", metrics.paired_test(case_results, "n_distinct_families", "WITHOUT", "WITH"))
    print("n_distinct_calls:", metrics.paired_test(case_results, "n_distinct_calls", "WITHOUT", "WITH"))
    print("n_distinct_type:", metrics.paired_test(case_results, "n_distinct_type", "WITHOUT", "WITH"))
    print("avg_pairwise_dissimilarity:", metrics.paired_test(case_results, "avg_pairwise_dissimilarity", "WITHOUT", "WITH"))
    print("frac_diversification_language (manipulation check):",
          metrics.paired_test(case_results, "frac_diversification_language", "WITHOUT", "WITH"))

    return case_results, summary


if __name__ == "__main__":
    existing = list(config.RESULTS_DIR.glob("Recommender_results_*.xlsx"))
    latest = max(existing, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing {latest}")
    df = pd.read_excel(latest, sheet_name="Recommendations")
    case_results, summary = analyze(df)

    out_path = config.RESULTS_DIR / f"{latest.stem}_diversification.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        case_results.to_excel(writer, sheet_name="Case Results", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"\nSaved to {out_path}")