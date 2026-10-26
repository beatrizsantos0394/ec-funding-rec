import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BASE_DIR  # noqa: E402
from modules.evaluation.topic_html import fetch_and_extract  # noqa: E402

DEFAULT_CACHE_DIR = BASE_DIR / "data" / "processed" / "HTML_Topics"

NEW_COLUMNS = [
    "real_status", "real_opening_date", "real_deadline_date",
    "real_type_of_action", "real_scope", "real_expected_outcome", "real_trl",
    "fetch_status", "fetch_error",
]


def enrich_df(df: pd.DataFrame, topic_col: str = "topic_id",
              cache_dir: Path = DEFAULT_CACHE_DIR, refresh: bool = False,
              headless: bool = True) -> pd.DataFrame:
    """Add the real_* / fetch_status / fetch_error columns onto df, matched
    by topic_col. Same logic used by the CLI below and by pipeline.py."""
    if topic_col not in df.columns:
        raise SystemExit(f"Column '{topic_col}' not found. Available columns: {list(df.columns)}")

    topic_ids = sorted({
        str(t).strip() for t in df[topic_col].dropna().tolist() if str(t).strip()
    })
    print(f"{len(df)} rows, {len(topic_ids)} distinct topic_id(s) to fetch.\n")

    # Fetch each distinct topic_id exactly once, even if it repeats across
    # rows (e.g. the same topic recommended under both WITH and WITHOUT
    # conditions).
    fetched = {}
    for i, topic_id in enumerate(topic_ids, 1):
        print(f"[{i}/{len(topic_ids)}] {topic_id} ...", end=" ")
        data = fetch_and_extract(
            topic_id, cache_dir,
            force_refetch=refresh,
            headless=headless,
        )
        fetched[topic_id] = data
        print(data["fetch_status"] if not data["error"] else f"error: {data['error']}")

    def col(topic_id, field, default=""):
        t = str(topic_id).strip()
        return fetched.get(t, {}).get(field, default)

    df["real_status"] = df[topic_col].apply(lambda t: col(t, "status"))
    df["real_opening_date"] = df[topic_col].apply(lambda t: col(t, "opening_date"))
    df["real_deadline_date"] = df[topic_col].apply(lambda t: col(t, "due_date"))
    df["real_type_of_action"] = df[topic_col].apply(lambda t: col(t, "type_of_action"))
    df["real_scope"] = df[topic_col].apply(lambda t: col(t, "scope"))
    df["real_expected_outcome"] = df[topic_col].apply(lambda t: col(t, "expected_outcome"))
    df["real_expected_impact"] = df[topic_col].apply(lambda t: col(t, "expected_impact"))
    df["real_trl"] = df[topic_col].apply(lambda t: col(t, "trl"))
    df["fetch_status"] = df[topic_col].apply(lambda t: col(t, "fetch_status", "no_topic_id"))
    df["fetch_error"] = df[topic_col].apply(lambda t: col(t, "error"))

    n_ok = (df["fetch_status"] == "fetched").sum() + (df["fetch_status"] == "cached").sum()
    n_err = (df["fetch_status"] == "error").sum()
    print(f"\nDone. {n_ok} row(s) enriched with real content, {n_err} error(s).")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add real (fetched) topic-page columns onto the "
                    "recommender's own output CSV, matched by topic_id."
    )
    parser.add_argument("--csv", required=True,
                        help="The CSV written by test_funding_recommender_minimal.py.")
    parser.add_argument("--out", default=None,
                        help="Where to write the result. Default: overwrite --csv in place.")
    parser.add_argument("--sep", default=";",
                        help="CSV delimiter (default ';', matching the project convention).")
    parser.add_argument("--topic-col", default="topic_id",
                        help="Column holding the topic ID (default: topic_id).")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                        help=f"HTML cache directory (default: {DEFAULT_CACHE_DIR}).")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch even if a cached HTML file already exists.")
    parser.add_argument("--show-browser", action="store_true",
                        help="Run the browser visibly (useful for debugging).")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path

    df = pd.read_csv(csv_path, sep=args.sep, encoding="utf-8-sig")
    df = enrich_df(
        df, topic_col=args.topic_col, cache_dir=Path(args.cache_dir),
        refresh=args.refresh, headless=not args.show_browser,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep=args.sep, index=False, encoding="utf-8-sig")
    print(f"Written to {out_path.resolve()}")


if __name__ == "__main__":
    main()