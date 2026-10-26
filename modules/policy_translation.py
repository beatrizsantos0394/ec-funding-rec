import pandas as pd
import config


def load_hs22_labels(path=None) -> dict:
    """
    Loads the HS22 6-digit code -> description lookup (config.PROCESSED_DIR
    / "hs22_labels.csv", sourced from the public-domain UN Comtrade-derived
    "datasets/harmonized-system" package -- not part of the UNSD
    correspondence chain above, only used to turn target_hs22 codes from
    diversification_context.py into human-readable text for the
    diversification-alignment analysis (scripts/analyze_diversification_alignment.py).
    """
    path = path or (config.PROCESSED_DIR / "hs22_labels.csv")
    labels = pd.read_csv(path, dtype=str)
    labels["hs22_code"] = labels["hs22_code"].str.zfill(6)
    return dict(zip(labels["hs22_code"], labels["description"]))


def _clean_code(series: pd.Series, pad_to: int | None = None) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    if pad_to is not None:
        s = s.str.zfill(pad_to)
    return s


def _clean_hs_dotted_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    parts = s.str.split(".", n=1, expand=True)
    if parts.shape[1] < 2:
        raise ValueError("Expected dotted 'chapter.subheading' HS notation (e.g. '101.21').")
    left, right = parts[0], parts[1]
    bad_right = right[right.str.len() != 2]
    if not bad_right.empty:
        raise ValueError(
            f"{len(bad_right)} HS codes have a subheading part that isn't exactly "
            f"2 digits after the dot (e.g. {bad_right.iloc[0]!r}). This reliably "
            "indicates the source file lost a trailing zero -- see DECISIONS.md D15."
        )
    return left.str.zfill(4) + right.str.zfill(2)


def _read_delimited_flexible(path, required_cols: list[str]) -> pd.DataFrame:
    last_error = None
    for sep in ["\t", ",", ";", r"\s{2,}"]:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, engine="python")
            if all(c in df.columns for c in required_cols):
                return df
        except Exception as e:
            last_error = e
    raise ValueError(
        f"Could not parse {path} with any known delimiter. "
        f"Expected columns {required_cols}. Last error: {last_error}"
    )


def load_hs22_to_nace_mapping(mapping_dir=None) -> pd.DataFrame:
    mapping_dir = mapping_dir or config.MAPPING_RAW_DIR
    files = {f.name.lower(): f for f in mapping_dir.glob("*")}

    def find(*keywords):
        for name, path in files.items():
            if all(k in name for k in keywords):
                return path
        raise FileNotFoundError(f"No file matching {keywords} found in {mapping_dir}")

    hs_path = find("hs2022")
    cpc_hs_path = find("cpc", "hs2017")
    cpc_isic_path = find("cpc", "isic4")
    isic_nace_path = find("isic4", "nace2")

    hs_df = pd.read_excel(hs_path, dtype=str)
    hs_df["From HS2022"] = _clean_code(hs_df["From HS2022"], pad_to=6)
    hs_df["To HS2017"] = _clean_code(hs_df["To HS2017"], pad_to=6)

    cpc_hs_df = pd.read_csv(cpc_hs_path, dtype=str)
    cpc_hs_df["HS 2017"] = _clean_hs_dotted_code(cpc_hs_df["HS 2017"])
    cpc_hs_df["CPC Ver. 2.1"] = _clean_code(cpc_hs_df["CPC Ver. 2.1"])
    cpc_hs_df = cpc_hs_df.rename(columns={"HS partial": "hs_partial", "CPC partial": "cpc_partial"})

    step1 = hs_df.merge(
        cpc_hs_df[["HS 2017", "CPC Ver. 2.1", "hs_partial", "cpc_partial"]],
        left_on="To HS2017", right_on="HS 2017", how="left",
    ).drop(columns=["HS 2017"])

    cpc_isic_df = _read_delimited_flexible(cpc_isic_path, ["CPC21code", "ISIC4code"])
    cpc_isic_df["CPC21code"] = _clean_code(cpc_isic_df["CPC21code"])
    cpc_isic_df["ISIC4code"] = _clean_code(cpc_isic_df["ISIC4code"])
    cpc_isic_df = cpc_isic_df.rename(columns={"ISIC4partial": "isic4_partial"})

    step2 = step1.merge(
        cpc_isic_df[["CPC21code", "ISIC4code", "isic4_partial"]],
        left_on="CPC Ver. 2.1", right_on="CPC21code", how="left",
    ).drop(columns=["CPC21code"])

    isic_nace_df = _read_delimited_flexible(isic_nace_path, ["ISIC4code", "NACE2code"])
    isic_nace_df["ISIC4code"] = _clean_code(isic_nace_df["ISIC4code"])
    isic_nace_df["NACE2code"] = isic_nace_df["NACE2code"].astype(str).str.strip()
    isic_nace_df = isic_nace_df.rename(columns={"NACE2part": "nace2_partial"})

    step3 = step2.merge(
        isic_nace_df[["ISIC4code", "NACE2code", "nace2_partial"]], on="ISIC4code", how="left",
    )
    return step3


def translate_to_nace(diversification_table: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    work = diversification_table.copy()
    work["product_hs22_code_2"] = work["product_hs22_code_2"].astype(str).str.zfill(6)

    def _join_unique(s):
        vals = sorted(set(x for x in s.dropna().astype(str) if x and x != "nan"))
        return ";".join(vals) if vals else None

    valid = mapping_df.dropna(subset=["From HS2022"]).copy()
    valid = valid.drop_duplicates(subset=["From HS2022", "CPC Ver. 2.1", "ISIC4code", "NACE2code"])

    def _summarize(g: pd.DataFrame) -> pd.Series:
        triples = g.dropna(subset=["CPC Ver. 2.1", "ISIC4code", "NACE2code"])
        paths = sorted(set(
            f"{c}\u2192{i}\u2192{n}"
            for c, i, n in zip(triples["CPC Ver. 2.1"], triples["ISIC4code"], triples["NACE2code"])
        ))
        return pd.Series({
            "nace_mapping_paths": "; ".join(paths) if paths else None,
            "NACE2code": _join_unique(g["NACE2code"]),
            "ISIC4code": _join_unique(g["ISIC4code"]),
            "CPC Ver. 2.1": _join_unique(g["CPC Ver. 2.1"]),
        })

    collapsed = valid.groupby("From HS2022").apply(_summarize, include_groups=False).reset_index()

    return work.merge(collapsed, left_on="product_hs22_code_2", right_on="From HS2022", how="left")


def add_policy_tag(diversification_table_nace, policy_priorities_path=None, project_id_col="projectID"):
    path = policy_priorities_path or config.CORDIS_POLICY_FILE
    pp = pd.read_excel(path) if str(path).endswith(".xlsx") else pd.read_csv(path, sep=";")
    flag_cols = [c for c in pp.columns if c != project_id_col]
    pp = pp.copy()
    pp[flag_cols] = pp[flag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(bool)

    def _tags(row):
        active = [c for c in flag_cols if row[c]]
        return ";".join(active) if active else "none"

    pp["policy_tag"] = pp.apply(_tags, axis=1)
    return diversification_table_nace.merge(pp[[project_id_col, "policy_tag"]], on=project_id_col, how="left")