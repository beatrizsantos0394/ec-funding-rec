import pandas as pd
import config
import modules.complexity_module as complexity_module


def compute_diversification_table(
    company_hs_codes: list[str],
    prox_df: pd.DataFrame,
    cdata: pd.DataFrame,
    year: int,
    top_n: int = None,
) -> pd.DataFrame:
    """
    Computes feasibility_score, relatedness_density (omega, Eq. 4), and
    pci for every candidate product NOT already in company_hs_codes.
    ALWAYS computes all fields regardless of prompt variant.

    Returns columns: product_hs22_code_2, feasibility_score,
    supporting_products, relatedness_density, pci.
    Empty DataFrame if no candidates found (e.g. empty company basket).
    """
    top_n = top_n or config.TOP_N_DIVERSIFICATION_PATHS

    if not company_hs_codes:
        return pd.DataFrame(columns=[
            "product_hs22_code_2", "feasibility_score", "supporting_products",
            "relatedness_density", "pci",
        ])

    # Trade data (Harvard Atlas HS2022) only covers a fixed range of
    # years (2022-2024 at time of writing, per D12). A case whose
    # as_of_date falls after the latest available trade year (e.g. a
    # HORIZON grant signed in 2026, still within HORIZON's active
    # 2021-2027 window) would otherwise silently produce an EMPTY
    # diversification table -- not because the company has no products,
    # but because there's no trade-year data for that year at all. This
    # was found by inspecting a real run where company_hs22 was
    # correctly populated but diversification_context came back empty
    # for every variant. Fall back to the nearest available past year,
    # consistent with D8's own reasoning that product-relatedness
    # structure doesn't shift much year-to-year -- the alternative
    # (returning nothing) would silently exclude every case signed after
    # the trade data's latest year from the EC comparison entirely.
    available_years = sorted(prox_df["year"].unique()) if "year" in prox_df.columns else []
    effective_year = year
    if available_years and year not in available_years:
        past_years = [y for y in available_years if y <= year]
        effective_year = max(past_years) if past_years else max(available_years)
        print(f"  [NOTE] No trade data for year={year}; using year={effective_year} instead "
              f"(nearest available <= requested, or latest available if none). See D29.")

    company_df = pd.DataFrame({"product_hs22_code": company_hs_codes})
    recs = complexity_module.recommend_products(
        company_df=company_df, prox_df=prox_df,
        company_code_col="product_hs22_code", year=effective_year, top_n=top_n,
    )
    if recs is None or recs.empty:
        return pd.DataFrame(columns=[
            "product_hs22_code_2", "feasibility_score", "supporting_products",
            "relatedness_density", "pci",
        ])

    # Relatedness density omega_{f,p} = sum(phi to owned products) / sum(phi to all products)
    work = prox_df.copy()
    if effective_year is not None and "year" in work.columns:
        work = work[work["year"] == effective_year]
    work["product_hs22_code_1"] = work["product_hs22_code_1"].astype(str).str.zfill(6)
    work["product_hs22_code_2"] = work["product_hs22_code_2"].astype(str).str.zfill(6)

    total_phi = work.groupby("product_hs22_code_2")["proximity"].sum().rename("sum_phi_all")
    owned_phi = (
        work[work["product_hs22_code_1"].isin(company_hs_codes)]
        .groupby("product_hs22_code_2")["proximity"].sum()
        .rename("sum_phi_owned")
    )
    density = pd.concat([owned_phi, total_phi], axis=1).fillna(0.0)
    density["relatedness_density"] = density["sum_phi_owned"] / density["sum_phi_all"].replace(0, pd.NA)
    density = density.reset_index()

    recs["product_hs22_code_2"] = recs["product_hs22_code_2"].astype(str).str.zfill(6)
    recs = recs.merge(density[["product_hs22_code_2", "relatedness_density"]],
                       on="product_hs22_code_2", how="left")

    # Product complexity index, if available for that year
    if cdata is not None and "pci" in cdata.columns:
        pci_col = cdata[cdata["year"] == effective_year][["product_hs22_code", "pci"]].drop_duplicates()
        pci_col["product_hs22_code"] = pci_col["product_hs22_code"].astype(str).str.zfill(6)
        recs = recs.merge(
            pci_col.rename(columns={"product_hs22_code": "product_hs22_code_2"}),
            on="product_hs22_code_2", how="left",
        )
    else:
        recs["pci"] = pd.NA

    return recs


def build_diversification_context(
    diversification_table: pd.DataFrame,
    variant: str,
    top_n_paths: int = None,
) -> str:
    """
    Renders diversification_table as prompt text. `variant` controls
    which columns are shown -- THIS is the experimental treatment
    (methodology 3.8) -- never recomputes anything, only selects/formats
    columns already present.

    If diversification_table has a 'NACE2code' column (i.e. it's already
    been through policy_translation.translate_to_nace()), that's shown
    for all variants; 'policy_tag' is required for the "policy" variant
    specifically.
    """
    top_n_paths = top_n_paths or config.TOP_N_DIVERSIFICATION_PATHS

    if diversification_table is None or diversification_table.empty:
        return "No diversification candidates available (empty company basket)."

    work = diversification_table.copy()
    if "feasibility_score" in work.columns:
        work = work.sort_values("feasibility_score", ascending=False)
    work = work.head(top_n_paths).reset_index(drop=True)

    lines = []
    for i, row in work.iterrows():
        hs22 = str(row.get("product_hs22_code_2", "NA")).zfill(6)
        nace = row.get("NACE2code", "NA")
        supporting = row.get("supporting_products", "NA")
        base = f"{i+1}. target_hs22={hs22} | NACE2={nace} | supporting_products={supporting}"

        if variant == "content":
            lines.append(base)

        elif variant == "ec_feasible":
            omega = row.get("relatedness_density", "NA")
            pci = row.get("pci", "NA")
            omega = round(omega, 3) if pd.notna(omega) else "NA"
            pci = round(pci, 3) if pd.notna(pci) else "NA"
            lines.append(base + f" | relatedness_density={omega} | product_complexity_index={pci}")

        elif variant == "capability":
            omega = row.get("relatedness_density", None)
            closeness = ("close to existing capabilities"
                         if (omega is not None and pd.notna(omega) and omega > 0.5)
                         else "requires new capabilities")
            lines.append(base + f" | capability_fit={closeness}")

        elif variant == "policy":
            omega = row.get("relatedness_density", "NA")
            pci = row.get("pci", "NA")
            omega = round(omega, 3) if pd.notna(omega) else "NA"
            pci = round(pci, 3) if pd.notna(pci) else "NA"
            policy_tag = row.get("policy_tag", "NA")
            lines.append(
                base + f" | relatedness_density={omega} | product_complexity_index={pci}"
                       f" | policy_alignment={policy_tag}"
            )
        else:
            raise ValueError(f"Unknown variant: {variant}. Expected one of {config.PROMPT_VARIANTS}")

    return "\n".join(lines)