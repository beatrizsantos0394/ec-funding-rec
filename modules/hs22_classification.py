import json
import re
from collections import Counter, defaultdict

from openai import OpenAI
import config


SYSTEM_PROMPT = """
You are an expert in international trade classification (HS 2022).

Task: given a company's business description, identify UP TO 5 candidate
6-digit HS 2022 codes for tangible goods the company produces, manufactures,
or sells. Classify only tangible goods clearly described -- ignore
services, consulting, software, research, or administrative activities
even if mentioned alongside real products.

A company with a multi-faceted product line can reasonably map to several
HS headings -- list plausible candidates rather than forcing a single
answer, ordered by how well each fits the description.

Rules:
- If the company is mainly services/software/finance/consulting with no
  clear tangible goods, return an empty candidates list.
- HS code must be exactly 6 digits.
- Omit candidates with confidence below 60. At most 5 candidates.
- Order from highest to lowest confidence.

Return ONLY this JSON object, no text before or after:
{
  "candidates": [
    {"code": "######", "confidence": 0, "notes": "short reason"}
  ]
}
"""


def get_client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        raise ValueError(f"{config.API_KEY_ENV} is not set.")
    return OpenAI(api_key=config.OPENAI_API_KEY, base_url=f"{config.AZURE_ENDPOINT}openai/v1/")


def classify_company_hs_once(description: str, client: OpenAI) -> list[dict]:
    """One LLM call, no web search. Returns a shortlist of candidate
    codes (possibly empty)."""
    try:
        response = client.chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Company description:\n{description}"},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=600,
        )
        data = json.loads(response.choices[0].message.content.strip())
    except Exception:
        return []

    candidates = []
    for item in data.get("candidates", []):
        code = re.sub(r"\D", "", str(item.get("code", "")))
        conf = item.get("confidence", 0)
        try:
            conf = int(conf)
        except Exception:
            conf = 0
        if len(code) == 6 and conf >= 60:
            candidates.append({"code": code, "confidence": conf})
    candidates.sort(key=lambda x: -x["confidence"])
    return candidates[:5]


def classify_with_consensus(
    description: str, client: OpenAI, n_runs: int = None, min_votes: int = None,
) -> list[dict]:
    """Pools candidate shortlists across n_runs (D28 pattern -- fixes the
    false-negative failure mode where each run's single top pick differs
    but a shared code still appears across shortlists). Returns
    [{"hs_code": str, "votes": int, "confidence": int}, ...] for codes
    reaching min_votes agreement, sorted by votes then confidence."""
    n_runs = n_runs or config.CONSENSUS_N_RUNS
    min_votes = min_votes or config.CONSENSUS_MIN_VOTES

    all_codes = []
    code_confidences = defaultdict(list)
    for _ in range(n_runs):
        for item in classify_company_hs_once(description, client):
            all_codes.append(item["code"])
            code_confidences[item["code"]].append(item["confidence"])

    counts = Counter(all_codes)
    consensus = [
        {"hs_code": code, "votes": votes, "confidence": max(code_confidences[code])}
        for code, votes in counts.items() if votes >= min_votes
    ]
    consensus.sort(key=lambda x: (-x["votes"], -x["confidence"]))
    return consensus


def run_hs22_classification(description: str, client: OpenAI = None) -> dict:
    """
    Top-level entry point for this module (DESIGN.md 2.2 contract,
    adapted: input is a description string, not a company_profile
    DataFrame, since that's what hs22_enrichment.py now produces).

    Returns: {"company_hs22": [...], "reliable": bool} -- reliable=False
    (empty basket) is a valid, expected outcome for a services/consulting
    company, not an error (see D26/D28 discussion of the pilot's ~45%
    non-product-company rate) -- callers must NOT treat this as a failure
    to retry or paper over.
    """
    client = client or get_client()
    consensus = classify_with_consensus(description, client)
    return {"company_hs22": consensus, "reliable": len(consensus) > 0}