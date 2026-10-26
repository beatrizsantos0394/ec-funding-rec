import json
import os
from datetime import date, datetime

from openai import OpenAI

DEFAULT_ENDPOINT = "https://ai-iscott.openai.azure.com/"
DEFAULT_DEPLOYMENT = "gpt-5.4"
API_KEY_ENV = "OPEN_AI_API_KEY"  # non-standard name, underscored -- matches this project's .env


def get_client() -> OpenAI:
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise ValueError(
            f"{API_KEY_ENV} is not set. Check your .env file (project root, "
            f"next to config.py) or the system environment."
        )
    endpoint = os.getenv("AZURE_ENDPOINT", DEFAULT_ENDPOINT)
    return OpenAI(api_key=api_key, base_url=f"{endpoint}openai/v1/")


def format_company_profile(company_profile: dict) -> str:
    """company_profile keys: name, country, sme (bool), description."""
    return (
        f"name: {company_profile.get('name', '')}\n"
        f"country: {company_profile.get('country', '')}\n"
        f"SME: {'YES' if company_profile.get('sme') else 'NO'}\n"
        f"description: {company_profile.get('description', '')}"
    )


def build_user_message(
    company_profile: dict,
    diversification_context: str = None,
    as_of_date: str = None,
    limit_date: str = None,
) -> str:
    """
    diversification_context: pre-formatted text block (the
    "N. target_hs22=... | feasibility_score=... | ..." lines) for
    ec_flag=1. Pass None (or "") for ec_flag=0 -- gated here, at this one
    place, exactly like the earlier fix that prevented diversification
    data from leaking into the ec_flag=0 condition.
    """
    as_of_date = as_of_date or date.today().isoformat()
    limit_date = limit_date or date.today().isoformat()

    profile_text = format_company_profile(company_profile)

    if diversification_context:
        intention_block = (
            "\nThe company is seeking funding to diversify into the following related\n"
            f"directions (specific intention):\n{diversification_context}\n"
        )
    else:
        intention_block = (
            "\nThe company is seeking relevant EU funding opportunities (no specific\n"
            "diversification direction provided).\n"
        )

    schema_and_rules = f"""
Task: Access https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/calls-for-proposals

and identify and rank real EU funding calls/topics that would support this company, following this rule:
Closed calls are acceptable: a call may be identified even if it is now
closed, as long as it was OPEN at some point during the period {as_of_date} to {limit_date}.
Do not identify a call whose entire opening_date-to-deadline_date
window falls entirely outside that period (i.e. it closed before {as_of_date},
or it only opens after {limit_date}).

Do NOT attempt to read or report the topic's Scope, Expected Outcome, TRL,
status or dates -- the topic-details page cannot be read this way and that
content is retrieved separately, afterwards, by a different process. Your
only job here is to identify and rank which real calls/topics are relevant.

VERIFICATION: only include a topic_id you have actually confirmed exists via
web search on a real EU Funding & Tenders Portal page. Do not guess or
reconstruct a topic_id from memory or pattern-matching. If you cannot verify
a candidate, leave it out rather than include an unconfirmed one.

TOPIC ID FORMAT: topic_id must be the real portal topic code (e.g.
"HORIZON-CL5-2026-09-D4-02"), never a call's plain-language name. Some calls
(e.g. EIC Accelerator, EIC Pathfinder) are continuously open with a single
standing topic code rather than a dated one -- use that standing topic code
exactly as published on the portal, not a description like "EIC Accelerator
Open".

RANKING: order funding_matches from most to least relevant to this company.

Return ONLY valid JSON with this exact schema:

{{
    "funding_matches": [
        {{
            "call_id": "",
            "call_title": "",
            "topic_id": "",
            "topic_title": "",
            "match_reason": "one sentence: why this specific call/topic fits this company's profile or diversification target"
        }}
    ]
}}

Example (illustrative only, not a real recommendation for this company):
{{
    "funding_matches": [
        {{
            "call_id": "HORIZON-CL5-2026-02",
            "call_title": "Cluster 5 Call 02-2026 (WP 2025)",
            "topic_id": "HORIZON-CL5-2026-02-D4-05",
            "topic_title": "Buildings and renovation for a decarbonised built environment",
            "match_reason": "Company manufactures bio-based construction materials, which directly matches this topic's focus on low-carbon building products."
        }},
        {{
            "call_id": "HORIZON-EIC-2026-ACCELERATOR",
            "call_title": "EIC Accelerator",
            "topic_id": "HORIZON-EIC-2026-ACCELERATOR-01-OPEN",
            "topic_title": "EIC Accelerator Open",
            "match_reason": "Continuously open call suited to an SME scaling a validated deep-tech product, matching the company's stage and product profile."
        }}
    ]
}}
"""

    return f"""Today is {as_of_date}.

Company profile:
{profile_text}
{intention_block}
{schema_and_rules}
""".strip()


def recommend_eu_funds(
    company_profile: dict,
    diversification_context: str = None,
    as_of_date: str = None,
    limit_date: str = None,
    client: OpenAI = None,
    deployment: str = None,
    max_output_tokens: int = 6000,
) -> dict:
    """
    The one call this module exists for. Same web_search-grounded LLM
    call as test_funding_recommender_minimal.py, just parameterized.

    Returns:
        {"funding_matches": [...], "n_sources": int, "raw_text": str,
         "parse_error": str or None}
    n_sources / parse_error let a caller tell "genuinely empty result"
    apart from "broken call that looks empty" -- same check as
    test_api_connection.py, kept here so every caller gets it for free.
    """
    client = client or get_client()
    deployment = deployment or os.getenv("AZURE_CHAT_DEPLOYMENT", DEFAULT_DEPLOYMENT)
    as_of_date = as_of_date or date.today().isoformat()
    limit_date = limit_date or date.today().isoformat()

    user_message = build_user_message(company_profile, diversification_context, as_of_date, limit_date)

    response = client.responses.create(
        model=deployment,
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        include=["web_search_call.action.sources"],
        input=[
            {
                "role": "system",
                "content": (
                    "You are a funding recommendation assistant. "
                    "You will be given a company profile describing what the company makes or does. "
                    "Your task is to IDENTIFY and RANK real, currently relevant EU funding calls or "
                    "topics that this company could realistically apply to, based on that profile. "
                    "You MUST use web search to inspect official EU funding sources, especially the "
                    "Funding and Tenders Portal. "
                    f"Only identify calls that were open (opening_date to deadline_date overlapping) "
                    f"at some point during {as_of_date} to {limit_date} -- a call that is now closed is "
                    f"still acceptable as long as it was open during that period. "
                    "Do NOT try to read or report a topic's Scope, Expected Outcome, TRL, status, or "
                    "dates -- only its call_id, call_title, topic_id, and topic_title. That page's "
                    "content is fetched and parsed separately, by a different process, precisely "
                    "because it cannot be reliably read this way. "
                    "Only include a topic_id you have actually confirmed via web search on a real "
                    "portal page -- never guess or reconstruct one from memory. topic_id must be the "
                    "real portal topic code, never a call's plain-language name; for continuously open "
                    "calls (e.g. EIC Accelerator, EIC Pathfinder) use their real standing topic code, "
                    "not a description. For each match, include one short sentence in match_reason "
                    "explaining why it fits this specific company. Order funding_matches from most to "
                    "least relevant. "
                    "When you answer, you MUST output ONLY a single valid JSON object that conforms "
                    "exactly to the schema specified by the user. "
                    "Do not include any explanation, comments, or text outside the JSON."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        max_output_tokens=max_output_tokens,
    )

    raw_text = response.output_text or ""
    funding_matches, parse_error = [], None
    if raw_text.strip():
        try:
            data = json.loads(raw_text)
            funding_matches = data.get("funding_matches", [])
        except json.JSONDecodeError as e:
            parse_error = f"{type(e).__name__}: {e}"
    else:
        parse_error = "empty_output"

    n_sources = 0
    try:
        for item in getattr(response, "output", None) or []:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else item
            if item_dict.get("type") == "web_search_call":
                n_sources += len((item_dict.get("action", {}) or {}).get("sources", []) or [])
    except Exception:
        pass

    return {
        "funding_matches": funding_matches,
        "n_sources": n_sources,
        "raw_text": raw_text,
        "parse_error": parse_error,
    }


# ---------------------------------------------------------------------------
# Date-window filter (ported as-is from test_funding_recommender_minimal.py,
# direction-fixed version: an open call's deadline is, by definition, in the
# future, so the filter keeps deadline_date >= today, not <= today).
# ---------------------------------------------------------------------------


def _parse_date(s: str):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def apply_date_window(funding_matches: list, today: date = None) -> list:
    today = today or date.today()
    kept = []
    for m in funding_matches:
        deadline_dt = _parse_date(m.get("deadline_date", ""))
        if deadline_dt is None:
            kept.append(m)  # can't disprove, keep it
            continue
        if deadline_dt >= today:
            kept.append(m)
    return kept