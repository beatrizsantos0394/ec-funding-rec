import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import pandas as pd
import config

MERGE_THRESHOLD = 10
N_TOTAL = 500
RANDOM_SEED = 42  # fixed and documented -- see DECISIONS.md D20


def extract_cluster(master_call) -> str:
    if pd.isna(master_call):
        return "UNKNOWN"
    m = re.match(r"HORIZON-([A-Z]+\d*)-", str(master_call))
    return m.group(1) if m else "OTHER"


def allocate_slots(strata_counts: pd.Series, n_total: int) -> dict:
    alloc = {s: 1 for s in strata_counts.index}
    remaining = n_total - len(strata_counts)
    sorted_strata = strata_counts.sort_values(ascending=False).index.tolist()
    i = 0
    while remaining > 0:
        s = sorted_strata[i % len(sorted_strata)]
        if alloc[s] < strata_counts[s]:
            alloc[s] += 1
            remaining -= 1
        i += 1
    return alloc


def alloc_pool_size(df: pd.DataFrame, stratum: str) -> int:
    return (df["stratum"] == stratum).sum()


def sample_without_company_duplicates(
    df: pd.DataFrame, alloc: dict, id_col: str = "organizationID", seed: int = RANDOM_SEED
) -> pd.DataFrame:
    used_org_ids = set()
    sampled_parts = []
    for stratum in sorted(alloc.keys(), key=lambda s: -alloc_pool_size(df, s)):
        n = alloc[stratum]
        pool = df[df["stratum"] == stratum]
        # a company can have more than one project in the same stratum --
        # drop_duplicates so a single .sample() draw can't pick the same
        # organizationID twice within this stratum
        eligible = pool[~pool[id_col].isin(used_org_ids)].drop_duplicates(subset=id_col)
        if len(eligible) < n:
            print(f"  [WARNING] Stratum '{stratum}' has only {len(eligible)} companies "
                  f"left after excluding duplicates, needed {n}. Taking all available.")
            n = len(eligible)
        sampled = eligible.sample(n=n, random_state=seed)
        used_org_ids.update(sampled[id_col].tolist())
        sampled_parts.append(sampled)

    pilot = pd.concat(sampled_parts)

    # a small stratum can run out of unique companies, leaving the total
    # short of n_total -- top up from the biggest remaining pool
    deficit = sum(alloc.values()) - len(pilot)
    if deficit > 0:
        print(f"  Topping up {deficit} slot(s) short after dedup, drawing from the largest pool...")
        eligible = df[~df[id_col].isin(used_org_ids)].drop_duplicates(subset=id_col)
        topup = eligible.sample(n=deficit, random_state=seed)
        pilot = pd.concat([pilot, topup])

    return pilot


df = pd.read_csv(config.CORDIS_CANDIDATE_POOL_CSV, sep=";", encoding="utf-8-sig")
print(f"Loaded candidate pool: {len(df):,} projects")

df["cluster_proxy"] = df["masterCall"].apply(extract_cluster)
counts = df["cluster_proxy"].value_counts()
small_labels = counts[counts < MERGE_THRESHOLD].index.tolist()
df["stratum"] = df["cluster_proxy"].apply(lambda x: "OTHER" if x in small_labels else x)

strata_counts = df["stratum"].value_counts()
print(f"\n{len(strata_counts)} strata after merging labels with <{MERGE_THRESHOLD} projects:")
print(strata_counts.to_string())

alloc = allocate_slots(strata_counts, N_TOTAL)
print(f"\nAllocation (seed={RANDOM_SEED}):")
for s, n in sorted(alloc.items(), key=lambda x: -x[1]):
    print(f"  {s}: {n} of {strata_counts[s]}")

pilot = sample_without_company_duplicates(df, alloc).drop(columns=["cluster_proxy", "stratum"])
assert len(pilot) == N_TOTAL, f"Expected {N_TOTAL} rows, got {len(pilot)}"
assert pilot["organizationID"].nunique() == N_TOTAL, "Duplicate companies -- deduplication failed"

out_path = config.PROCESSED_DIR / "pilot_sample_500.csv"
config.ensure_dirs()
pilot.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
print(f"\nSaved {len(pilot)} cases to {out_path}")