import json
import re
from collections import Counter

from openai import OpenAI
import config


PROJECT_SYSTEM_PROMPT = """
You are an expert in international trade classification (HS 2022).

Task: given a funded R&I project's own description, identify UP TO 3
candidate 6-digit HS 2022 codes that could represent the project's main
NEW codable product, good, or technology output -- not the organisations
involved, not the research process itself, only the resulting tangible
good/product if the project is aimed at producing or enabling one.

A complex project can reasonably map to more than one plausible HS
heading (e.g. a medical device platform could be classified under
medical instruments, under monitors, or under a related equipment
heading) -- list the plausible candidates rather than forcing a single
answer, ordered by how well each fits.

Rules:
- Only tangible, HS22-codable goods. If the project's output is a
  service, software, pure research findings, or something not codable as
  a physical good, return an empty candidates list.
- Return ONLY this JSON object, no text before or after:
{
  "candidates": [
    {"code": "######", "confidence": 0, "notes": "short reason"}
  ],
  "codable": true
}
- Omit candidates with confidence below 60. At most 3 candidates, ordered
  from highest to lowest confidence.
- "codable" is false (and candidates []) if the project's output isn't a
  tangible HS22-codable good at all.
"""

DESCRIPTION_SYSTEM_PROMPT = """
You are a business research assistant with web search access.

Task: write a comprehensive business profile describing what a company
did BEFORE a specific date -- its PRE-EXISTING activity, in full.

Describe the company's activity broadly and naturally, the way you would
describe any real business: its products, systems, equipment, and
technologies where relevant, AND its services, technical activities, and
the sectors/clients it serves. Do NOT filter this down to "tangible goods
only" -- a real business description naturally mixes products and
services together, and a separate classification step downstream will
extract whatever is codable from your full description. An overly
narrow, goods-only description is LESS useful here, not more.

Write 2-4 sentences, in the style of a company "About us" profile -- not
a single short fragment. Write in PRESENT TENSE, as if as_of_date were
today (e.g. "the company develops and sells...", "its offering
combines..." -- NOT "developed", "sold", "combined"). The goal is to
simulate what the company's own current profile would say at that point
in time, not a historical retrospective written from today's vantage
point. Do NOT state or reference the as_of_date explicitly in the text
itself (no "Before [date], ..." phrasing) -- just describe the business
as if it is the company's present.

CRITICAL temporal rule (governs what you research, not how you phrase it):
- Only use information describing the company's activity BEFORE
  as_of_date.
- Prefer dated historical sources: older press coverage, company
  registry records, an archived version of the company's website from
  before as_of_date (check web.archive.org / the Wayback Machine for a
  snapshot from before that date if the live site doesn't make the
  history clear).
- If you cannot find reliable historical information (only present-day
  information is available), say so explicitly rather than describing
  the company's current business as if it were historical.

CRITICAL exclusion rule:
- This company later received funding for a new project, described
  below. Your description must describe ONLY the company's PRE-EXISTING
  business -- do NOT mention, describe, or hint at anything in that new
  project's product/technology area, even if you find evidence the
  company was already moving in that direction before the official
  project start date. This description is used to test whether a
  separate system can correctly recommend that exact new direction as a
  good diversification move -- if your description already gives it
  away, the test becomes meaningless.

Return ONLY this JSON object, no text before or after:
{
  "description": "a full 2-4 sentence business profile, plain language, no date mentioned",
  "historical_data_found": true,
  "search_note": "one sentence on what kind of source you relied on"
}

Rules:
- "historical_data_found" is false if you had to fall back to
  present-day-only information -- in that case "description" should say
  so plainly (e.g. "No reliable pre-project information found.") rather
  than describing current business.
- Do not use HS codes or classification jargon in the description --
  plain business language only, as if describing the company to a
  colleague.
"""


def get_client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        raise ValueError(f"{config.API_KEY_ENV} is not set.")
    return OpenAI(api_key=config.OPENAI_API_KEY, base_url=f"{config.AZURE_ENDPOINT}openai/v1/")


# ---------------------------------------------------------------
# Ground truth label: project's own new-output code
# ---------------------------------------------------------------

def classify_project_hs22_once(objective: str, keywords: str, title: str, client: OpenAI) -> dict:
    user_prompt = f"""
Project title: {title}
Project keywords: {keywords}

Project objective:
{objective}
""".strip()
    try:
        response = client.chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": PROJECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=500,
        )
        data = json.loads(response.choices[0].message.content.strip())
    except Exception:
        return {"candidates": [], "codable": False}

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

    return {"candidates": candidates[:3], "codable": bool(candidates) and bool(data.get("codable", False))}


def classify_project_hs22_consensus(
    objective: str, keywords: str, title: str, client: OpenAI,
    n_runs: int = None, min_votes: int = None,
) -> dict:
    """Pools candidate codes across n_runs shortlists (matching the
    company_hs22 approach) and picks the code with the most votes, rather
    than requiring a single per-run answer to match exactly -- a
    complex/multi-faceted project can plausibly map to several reasonable
    HS headings, and requiring literal single-answer agreement across
    runs was found to produce false negatives (empty project_hs22) for
    companies with clearly codable products, e.g. a VR-based medical
    device platform could reasonably land under medical instruments,
    monitors, or related equipment headings across different runs."""
    n_runs = n_runs or config.CONSENSUS_N_RUNS
    min_votes = min_votes or config.CONSENSUS_MIN_VOTES

    all_codes = []
    code_confidences = {}
    for _ in range(n_runs):
        result = classify_project_hs22_once(objective, keywords, title, client)
        for c in result["candidates"]:
            all_codes.append(c["code"])
            code_confidences.setdefault(c["code"], []).append(c["confidence"])

    if not all_codes:
        return {"project_hs22": "", "votes": 0, "reliable": False}

    counts = Counter(all_codes)
    # Rank by (votes, max confidence) so ties favor the more confident code
    ranked = sorted(
        counts.items(),
        key=lambda kv: (kv[1], max(code_confidences[kv[0]])),
        reverse=True,
    )
    top_code, top_votes = ranked[0]
    return {
        "project_hs22": top_code if top_votes >= min_votes else "",
        "votes": top_votes,
        "reliable": top_votes >= min_votes,
    }


# ---------------------------------------------------------------
# Company's pre-project description -- web search, historical-only,
# explicitly excluding the winning project's product area
# ---------------------------------------------------------------

def build_description_user_prompt(
    company_name: str, country: str, as_of_date: str,
    project_title: str, project_objective: str, project_keywords: str,
) -> str:
    return f"""
Company: {company_name}
Country: {country}
as_of_date: {as_of_date}

This company later received funding for this NEW project (do NOT
describe this or anything in its product/technology area):
Project title: {project_title}
Project keywords: {project_keywords}
Project objective: {project_objective}

Describe what tangible goods/products this company made or sold BEFORE
{as_of_date}, excluding anything related to the new project above.
""".strip()


def get_company_description_once(
    company_name: str, country: str, as_of_date: str,
    project_title: str, project_objective: str, project_keywords: str,
    client: OpenAI,
) -> dict:
    try:
        response = client.responses.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            tools=[{"type": "web_search"}],
            tool_choice="required",
            input=[
                {"role": "system", "content": DESCRIPTION_SYSTEM_PROMPT},
                {"role": "user", "content": build_description_user_prompt(
                    company_name, country, as_of_date,
                    project_title, project_objective, project_keywords)},
            ],
            max_output_tokens=800,
        )
        data = json.loads((response.output_text or "").strip())
    except Exception as e:
        return {"description": "", "historical_data_found": False, "search_note": f"error: {e}"}

    return {
        "description": str(data.get("description", ""))[:800],
        "historical_data_found": bool(data.get("historical_data_found", False)),
        "search_note": str(data.get("search_note", "")),
    }


def check_keyword_leakage(description: str, project_keywords: str) -> list[str]:
    """Flags (doesn't drop) any distinctive project keyword appearing
    verbatim in the generated description -- a strong signal the
    exclusion instruction didn't work for this run. Ignores very short/
    generic keywords (<=4 chars) to avoid noisy false positives."""
    if not description or not project_keywords:
        return []
    desc_lower = description.lower()
    hits = []
    for kw in re.split(r"[;,]", str(project_keywords)):
        kw = kw.strip().lower()
        if len(kw) > 4 and kw in desc_lower:
            hits.append(kw)
    return hits


def get_company_description_consensus(
    company_name: str, country: str, as_of_date: str,
    project_title: str, project_objective: str, project_keywords: str,
    client: OpenAI, n_runs: int = None, min_votes: int = None,
) -> dict:
    """Runs get_company_description_once n_runs times. A company is
    'reliable' only if at least min_votes runs report historical data
    found; the returned description is the first one from such a run
    (never from a run that admitted falling back to present-day info --
    D25's lesson applied here from the start). Every run's description is
    checked for keyword leakage regardless of historical_data_found,
    since leakage could happen even when the model claims success."""
    n_runs = n_runs or config.CONSENSUS_N_RUNS
    min_votes = min_votes or config.CONSENSUS_MIN_VOTES

    historical_found_count = 0
    chosen_description = ""
    search_notes = []
    leakage_flags = []

    for _ in range(n_runs):
        result = get_company_description_once(
            company_name, country, as_of_date,
            project_title, project_objective, project_keywords, client,
        )
        search_notes.append(result["search_note"])

        hits = check_keyword_leakage(result["description"], project_keywords)
        if hits:
            leakage_flags.append(hits)

        if result["historical_data_found"]:
            historical_found_count += 1
            if not chosen_description and result["description"]:
                chosen_description = result["description"]

    return {
        "company_products_description": chosen_description,
        "historical_data_found_in_n_of_runs": f"{historical_found_count}/{n_runs}",
        "search_notes": " | ".join(search_notes),
        "reliable": historical_found_count >= min_votes and bool(chosen_description),
        "possible_leakage": leakage_flags,  # non-empty = manual review needed
    }


def enrich_case(
    objective: str, keywords: str, title: str,
    company_name: str, country: str, as_of_date: str, client: OpenAI,
) -> dict:
    """Full per-case enrichment for the benchmark: ground-truth label
    (project_hs22) plus the system's actual input (a clean pre-project
    description). Does NOT classify company_hs22 -- see module docstring."""
    project_result = classify_project_hs22_consensus(objective, keywords, title, client)
    description_result = get_company_description_consensus(
        company_name, country, as_of_date, title, objective, keywords, client,
    )
    return {
        "project_hs22": project_result["project_hs22"],
        "project_hs22_votes": project_result["votes"],
        "project_hs22_reliable": project_result["reliable"],
        "company_products_description": description_result["company_products_description"],
        "historical_data_found_in_n_of_runs": description_result["historical_data_found_in_n_of_runs"],
        "search_notes": description_result["search_notes"],
        "description_reliable": description_result["reliable"],
        "possible_leakage": description_result["possible_leakage"],
    }