from pathlib import Path
import pandas as pd
from ecomplexity import ecomplexity, proximity as _proximity

import config


def run_complexity_analysis(
    file_path=None,
    output_path=None,
    proximity_output_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_path = Path(file_path or config.COMPLEXITY_INPUT)
    output_path = Path(output_path or config.COMPLEXITY_RESULTS_CSV)
    proximity_output_path = Path(proximity_output_path or config.PROXIMITY_RESULTS_CSV)
    config.ensure_dirs()

    trade_cols = {
        "time": "year",
        "loc": "country_iso3_code",
        "prod": "product_hs22_code",
        "val": "export_value",
    }

    print(f"Loading raw trade data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    df["product_hs22_code"] = df["product_hs22_code"].astype(str).str.zfill(6)

    if output_path.exists():
        print(f"Loading cached complexity metrics from {output_path}...")
        cdata = pd.read_csv(output_path)
    else:
        print("Computing complexity metrics (ecomplexity)... this may take a few minutes.")
        cdata = ecomplexity(
            df, trade_cols,
            check_logsupermodularity=config.LOGSUPERMODULARITY_SAMPLES,
        )
        cdata.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    if proximity_output_path.exists():
        print(f"Loading cached proximity matrix from {proximity_output_path}...")
        prox_df = pd.read_csv(
            proximity_output_path,
            dtype={"product_hs22_code_1": str, "product_hs22_code_2": str},
        )
    else:
        print("Computing proximity matrix... this may take a few minutes.")
        prox_df = _proximity(df, trade_cols)
        prox_df.to_csv(proximity_output_path, index=False)
        print(f"Saved to {proximity_output_path}")

    return cdata, prox_df


def recommend_products(
    company_df: pd.DataFrame,
    prox_df: pd.DataFrame,
    company_code_col: str,
    year: int,
    top_n: int = 20,
) -> pd.DataFrame | None:
    company_codes = (
        company_df[company_code_col].astype(str).str.zfill(6).dropna().unique()
    )

    work_df = prox_df.copy()
    if year is not None and "year" in work_df.columns:
        work_df = work_df[work_df["year"] == year]

    work_df["product_hs22_code_1"] = work_df["product_hs22_code_1"].astype(str).str.zfill(6)
    work_df["product_hs22_code_2"] = work_df["product_hs22_code_2"].astype(str).str.zfill(6)

    reachable = work_df[work_df["product_hs22_code_1"].isin(company_codes)]
    candidates = reachable[~reachable["product_hs22_code_2"].isin(company_codes)]

    if candidates.empty:
        return None

    recs = (
        candidates.groupby("product_hs22_code_2", as_index=False)["proximity"]
        .max()
        .rename(columns={"proximity": "feasibility_score"})
    )
    support_counts = (
        candidates.groupby("product_hs22_code_2")["product_hs22_code_1"]
        .nunique()
        .reset_index(name="supporting_products")
    )
    recs = recs.merge(support_counts, on="product_hs22_code_2", how="left")
    recs = recs.sort_values(
        by=["feasibility_score", "supporting_products"], ascending=[False, False]
    ).reset_index(drop=True)

    return recs.head(top_n)