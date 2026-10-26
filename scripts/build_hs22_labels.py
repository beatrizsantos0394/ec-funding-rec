from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "data" / "raw" / "HS_codes" / "HSCodeandDescription.xlsx"
OUT = BASE / "data" / "processed" / "hs22_labels.csv"

if not SRC.exists():
    raise SystemExit(f"File not found: {SRC}\nDownload HSCodeandDescription.xlsx into that folder.")

sheets = pd.ExcelFile(SRC).sheet_names
if "HS22" not in sheets:
    raise SystemExit(f"No 'HS22' sheet in {SRC.name}. Sheets found: {sheets}")

df = pd.read_excel(SRC, sheet_name="HS22", dtype=str)
missing = {"Code", "Description", "Level"} - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns {sorted(missing)}. Columns found: {list(df.columns)}")

df = df[df["Level"].str.strip() == "6"].rename(columns={"Code": "hs22_code", "Description": "description"})
df["hs22_code"] = df["hs22_code"].str.strip().str.zfill(6)
labels = df[["hs22_code", "description"]].drop_duplicates("hs22_code")

OUT.parent.mkdir(parents=True, exist_ok=True)
labels.to_csv(OUT, index=False)
print(f"Wrote {len(labels)} rows to {OUT}")