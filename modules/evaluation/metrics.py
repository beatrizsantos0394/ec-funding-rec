import re
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------
# Call-level relevance (D1): exact / family / content / miss
# ---------------------------------------------------------------

def call_relevance_tier(
    funding_matches: pd.DataFrame,
    true_topic_id: str,
    true_programme: str,
    true_scheme: str,
    reference_text: str = None,
    embed_fn=None,
    similarity_threshold: float = None,
) -> dict:
    """Returns {"tier": 0-3, "best_match_topic_id": str or None,
    "best_match_similarity": float or nan} per D1's graded scale."""
    if funding_matches is None or funding_matches.empty:
        return {"tier": 0, "best_match_topic_id": None, "best_match_similarity": np.nan}

    # Level 3: exact topic match
    if "topic_id" in funding_matches.columns and true_topic_id:
        exact = funding_matches[funding_matches["topic_id"].astype(str) == str(true_topic_id)]
        if not exact.empty:
            return {"tier": 3, "best_match_topic_id": true_topic_id, "best_match_similarity": np.nan}

    # Level 2: call-family match (programme + funding_scheme)
    if all(c in funding_matches.columns for c in ["programme", "funding_scheme"]):
        family = funding_matches[
            (funding_matches["programme"].astype(str) == str(true_programme)) &
            (funding_matches["funding_scheme"].astype(str) == str(true_scheme))
        ]
        if not family.empty:
            return {"tier": 2, "best_match_topic_id": family.iloc[0].get("topic_id"),
                    "best_match_similarity": np.nan}

    # Level 1: content match via calibrated similarity threshold (D2)
    if embed_fn is not None and reference_text and "scope" in funding_matches.columns:
        texts = [t for t in funding_matches["scope"].dropna().tolist() if t]
        if texts:
            from sklearn.metrics.pairwise import cosine_similarity
            vecs = embed_fn(texts + [reference_text])
            ref_vec = np.array(vecs[-1]).reshape(1, -1)
            cand_vecs = np.array(vecs[:-1])
            sims = cosine_similarity(cand_vecs, ref_vec).flatten()
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if similarity_threshold is not None and best_sim >= similarity_threshold:
                return {"tier": 1, "best_match_topic_id": funding_matches.iloc[best_idx].get("topic_id"),
                        "best_match_similarity": best_sim}
            return {"tier": 0, "best_match_topic_id": None, "best_match_similarity": best_sim}

    return {"tier": 0, "best_match_topic_id": None, "best_match_similarity": np.nan}


def reciprocal_rank(funding_matches: pd.DataFrame, true_topic_id: str) -> float:
    if funding_matches is None or funding_matches.empty or not true_topic_id:
        return 0.0
    if "topic_id" not in funding_matches.columns:
        return 0.0
    ids = funding_matches["topic_id"].astype(str).tolist()
    if str(true_topic_id) in ids:
        return 1.0 / (ids.index(str(true_topic_id)) + 1)
    return 0.0


def ndcg_relevance(tier: int) -> float:
    """Graded relevance weight for aggregation, bounded [0,1], exact
    match = 1.0."""
    return tier / 3.0


# ---------------------------------------------------------------
# Product-level ground truth (D26): does the system recommend the
# project's own new-output code, independent of call matching?
# ---------------------------------------------------------------

def product_hit(funding_matches: pd.DataFrame, project_hs22: str) -> int:
    """1 if any recommended call's matched_hs22 list CONTAINS project_hs22
    -- tests economic-complexity theory's core claim directly: does the
    system suggest the right PRODUCT, independent of whether it also
    found the right funding CALL.

    matched_hs22 is a LIST per row (e.g. ["284290"], sometimes several
    codes for one call -- see funding_recommender.py's real schema, drawn
    from the author's original draft), not a single scalar string. An
    earlier version of this function compared the list object directly
    to a string, which would silently always evaluate to 0 -- fixed here
    to check membership within each row's list."""
    if funding_matches is None or funding_matches.empty or not project_hs22 or pd.isna(project_hs22):
        return 0
    if "matched_hs22" not in funding_matches.columns:
        return 0
    target = str(project_hs22).strip()
    target = re.sub(r"\.0$", "", target)  # pandas float-read artifact, e.g. "680911.0" -> "680911"
    try:
        target_padded = target.zfill(6)
    except Exception:
        target_padded = target

    for codes in funding_matches["matched_hs22"]:
        code_list = codes if isinstance(codes, list) else []
        normalized = {str(c).strip().zfill(6) for c in code_list if str(c).strip()}
        if target_padded in normalized or target in normalized:
            return 1
    return 0


# ---------------------------------------------------------------
# Threshold calibration (D2)
# ---------------------------------------------------------------

def calibrate_content_threshold(
    positive_pairs: list[tuple[str, str]],
    negative_pairs: list[tuple[str, str]],
    embed_fn,
) -> dict:
    """positive_pairs/negative_pairs: list of (text_a, text_b) tuples.
    Simple midpoint-of-means threshold; can be swapped for a full
    ROC-based choice later without changing the interface."""
    from sklearn.metrics.pairwise import cosine_similarity

    def sims_for(pairs):
        sims = []
        for a, b in pairs:
            vecs = embed_fn([a, b])
            s = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
            sims.append(float(s))
        return sims

    pos_sims = sims_for(positive_pairs)
    neg_sims = sims_for(negative_pairs)
    threshold = (np.mean(pos_sims) + np.mean(neg_sims)) / 2 if pos_sims and neg_sims else 0.7

    return {
        "threshold": float(threshold),
        "positive_similarities": pos_sims,
        "negative_similarities": neg_sims,
        "positive_pairs_n": len(positive_pairs),
        "negative_pairs_n": len(negative_pairs),
    }


# ---------------------------------------------------------------
# Aggregation + paired significance tests
# ---------------------------------------------------------------

def summarize_by_variant(case_results: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for variant, g in case_results.groupby("prompt_variant"):
        row = {"prompt_variant": variant, "n_cases": len(g)}
        for col in metric_cols:
            vals = g[col].dropna().values
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_ci_lo"] = np.nan
                row[f"{col}_ci_hi"] = np.nan
                continue
            boot_means = [np.mean(np.random.choice(vals, size=len(vals), replace=True))
                          for _ in range(2000)]
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_ci_lo"] = float(np.percentile(boot_means, 2.5))
            row[f"{col}_ci_hi"] = float(np.percentile(boot_means, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(
    case_results: pd.DataFrame, metric_col: str, variant_a: str, variant_b: str, binary: bool = False,
) -> dict:
    a = case_results[case_results["prompt_variant"] == variant_a].set_index("case_id")[metric_col]
    b = case_results[case_results["prompt_variant"] == variant_b].set_index("case_id")[metric_col]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].dropna(), b.loc[common].dropna()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    if len(a) < 2:
        return {"n_paired": len(a), "test": None, "statistic": np.nan, "p_value": np.nan, "mean_delta": np.nan}

    mean_delta = float((b - a).mean())

    if binary:
        n01 = int(((a == 0) & (b == 1)).sum())
        n10 = int(((a == 1) & (b == 0)).sum())
        if n01 + n10 == 0:
            stat, p = 0.0, 1.0
        else:
            # McNemar's test with continuity correction -- NOT
            # chi2_contingency (which fails whenever the 2x2 discordant
            # table has a zero row/column, a common case at small sample
            # sizes; confirmed by reproducing it during testing).
            stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p = float(stats.chi2.sf(stat, df=1))
        test_name = "mcnemar_chi2"
    else:
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            stat, p = np.nan, np.nan
        test_name = "wilcoxon"

    return {
        "n_paired": len(a), "test": test_name,
        "statistic": float(stat) if stat == stat else np.nan,
        "p_value": float(p) if p == p else np.nan,
        "mean_delta": mean_delta,
    }


# ---------------------------------------------------------------
# Weighted topic-match score: family + type-of-action + TRL + embedding
# similarity, each reported separately AND combined, per the author's
# explicit design (D33). If scope/expected_outcome are empty on either
# side (the topic-details page couldn't be read -- see the JS-rendering
# limitation found for the EU Portal), similarity contributes 0 to the
# weighted score rather than being excluded -- per the author's explicit
# instruction that an empty/unverifiable result should show up as a LOW
# score, not be skipped from evaluation. This is a deliberate departure
# from topic_details.py's "None means unscoreable, don't treat as a
# miss" rule for its own raw comparisons -- here, at the scoring-summary
# level, "we don't know" and "it doesn't match" are intentionally both
# penalized, since an unverifiable recommendation is not a usable one for
# the person relying on this system.
# ---------------------------------------------------------------

DEFAULT_TOPIC_SCORE_WEIGHTS = {
    "similarity": 0.5,
    "family_match": 0.2,
    "type_match": 0.2,
    "trl_overlap": 0.1,
}


def weighted_topic_score(
    true_topic_id: str, true_type_of_action: str, true_trl: str,
    true_scope: str, true_expected_outcome: str,
    candidate_topic_id: str, candidate_type_of_action: str, candidate_trl: str,
    candidate_scope: str, candidate_expected_outcome: str,
    embedding_similarity,  # float 0-1, or None if not computed/available
    weights: dict = None,
) -> dict:
    """
    Combines family match (ignoring year), type-of-action match, TRL
    overlap, and embedding similarity into one weighted score, with every
    component reported individually alongside the composite -- never just
    the single number.

    embedding_similarity should come from topic_details.py's own
    comparison (real fetched scope+expected_outcome on both sides). If
    None or either scope/expected_outcome is empty, similarity
    contributes 0 -- see module-level note above on why this differs from
    topic_details.py's own None-means-unscoreable convention.
    """
    import modules.evaluation.topic_details as topic_details
    weights = weights or DEFAULT_TOPIC_SCORE_WEIGHTS

    family_match = (
        topic_details.extract_topic_family(true_topic_id) != ""
        and topic_details.extract_topic_family(true_topic_id)
        == topic_details.extract_topic_family(candidate_topic_id)
    )

    type_match = (
        topic_details.normalize_action_type(true_type_of_action) != ""
        and topic_details.normalize_action_type(true_type_of_action)
        == topic_details.normalize_action_type(candidate_type_of_action)
    )

    trl_overlap_result = topic_details.trl_overlap(true_trl, candidate_trl)
    # None (unparseable on either side) treated as no contribution, not
    # a bonus -- consistent with the "unverifiable = penalized" rule above.
    trl_component = 1.0 if trl_overlap_result else 0.0

    has_real_content = bool(
        (true_scope or true_expected_outcome) and (candidate_scope or candidate_expected_outcome)
    )
    similarity_component = (
        float(embedding_similarity)
        if (has_real_content and embedding_similarity is not None)
        else 0.0
    )

    composite = (
        weights["similarity"] * similarity_component
        + weights["family_match"] * (1.0 if family_match else 0.0)
        + weights["type_match"] * (1.0 if type_match else 0.0)
        + weights["trl_overlap"] * trl_component
    )

    return {
        "composite_score": composite,
        "similarity_component": similarity_component,
        "family_match": family_match,
        "type_match": type_match,
        "trl_overlap": trl_overlap_result,  # True/False/None (None = unparseable, shown honestly)
        "has_real_content": has_real_content,
        "weights_used": weights,
    }


# ---------------------------------------------------------------
# Diversification-quality metric: does EC information actually produce
# recommendations pointing toward NEW, RELATED products, not just
# recommendations for the company's existing business? This is a
# separate, distinct test from topic-match accuracy above -- it tests
# the core economic-complexity claim directly (D33), independent of
# whether the specific historical call was found.
# ---------------------------------------------------------------

def diversification_quality(
    funding_matches, company_hs22: list, diversification_table,
) -> dict:
    """
    funding_matches: DataFrame with a 'matched_hs22' column (list per row).
    company_hs22: list of HS22 codes the company ALREADY has (its
        existing basket, from hs22_classification.py).
    diversification_table: the computed candidate table (from
        diversification_context.compute_diversification_table) -- the
        EC/complexity model's own top candidates for this company, with
        'product_hs22_code_2' and 'feasibility_score' columns.

    Returns:
        new_product_rate: fraction of all matched_hs22 codes (deduped)
            that are NOT already in the company's existing basket --
            "does it recommend something actually new, not just more of
            the same."
        diversification_hit_rate: of the NEW codes, what fraction also
            appear in the complexity model's own computed candidate
            list -- "when it recommends something new, is it a well-
            reasoned, CLOSE diversification per relatedness theory, not
            just an arbitrary new product." This is the direct test of
            whether EC information is actually being used, not just
            present in the prompt.
        avg_feasibility_of_hits: mean feasibility_score (from the
            complexity model) among the new codes that landed inside the
            candidate list -- a continuous "how good/close" measure, not
            just a hit-rate.
        n_new_codes, n_diversification_hits: raw counts, for transparency.
    """
    company_set = {str(c).strip().zfill(6) for c in (company_hs22 or [])}

    all_matched = []
    if funding_matches is not None and not funding_matches.empty and "matched_hs22" in funding_matches.columns:
        for codes in funding_matches["matched_hs22"]:
            code_list = codes if isinstance(codes, list) else []
            for c in code_list:
                c = str(c).strip().zfill(6)
                if c:
                    all_matched.append(c)
    unique_matched = set(all_matched)

    if not unique_matched:
        return {
            "new_product_rate": None, "diversification_hit_rate": None,
            "avg_feasibility_of_hits": None, "n_matched_total": 0,
            "n_new_codes": 0, "n_diversification_hits": 0,
        }

    new_codes = unique_matched - company_set
    new_product_rate = len(new_codes) / len(unique_matched)

    candidate_lookup = {}
    if diversification_table is not None and not diversification_table.empty:
        for _, row in diversification_table.iterrows():
            code = str(row.get("product_hs22_code_2", "")).strip().zfill(6)
            if code:
                candidate_lookup[code] = row.get("feasibility_score")

    diversification_hits = new_codes & set(candidate_lookup.keys())
    diversification_hit_rate = (
        len(diversification_hits) / len(new_codes) if new_codes else None
    )
    feasibility_values = [
        float(candidate_lookup[c]) for c in diversification_hits
        if candidate_lookup.get(c) is not None
    ]
    avg_feasibility = float(np.mean(feasibility_values)) if feasibility_values else None

    return {
        "new_product_rate": new_product_rate,
        "diversification_hit_rate": diversification_hit_rate,
        "avg_feasibility_of_hits": avg_feasibility,
        "n_matched_total": len(unique_matched),
        "n_new_codes": len(new_codes),
        "n_diversification_hits": len(diversification_hits),
    }