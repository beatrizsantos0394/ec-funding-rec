import json
import re
from typing import Optional
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

import config


def extract_topic_family(topic_id: str) -> str:
    """
    Coarse family extraction, ignoring year -- confirmed necessary since
    exact ID structure varies across years/tracks (CL-track destination
    vs call-number order flipped 2021->2025+, EIC's own format changed
    three times). Extracts just the STABLE family token: the segment
    after "HORIZON-" (or the whole first segment for non-HORIZON
    prefixes like ERC-/EURATOM-), with JU/JTI sub-programmes joined with
    their next segment since they cover very different themes (e.g.
    JU-CBE vs JU-IHI vs JU-SNS are NOT comparable families despite
    sharing "JU").

    KNOWN LIMITATION: does not catch a track renaming itself over time
    (e.g. HORIZON-CL4-... became HORIZON-CL4-INDUSTRY-... in the
    2026-2027 work programme) -- these are treated as different families
    even though they're the same cluster. Not fixed here; flagging
    honestly rather than silently mapping renamed tracks, since a wrong
    guess at equivalence is worse than an honest non-match.
    """
    if not topic_id or not str(topic_id).strip():
        return ""
    parts = str(topic_id).strip().split("-")
    if not parts:
        return ""
    if parts[0] == "HORIZON" and len(parts) > 1:
        if parts[1] in ("JU", "JTI") and len(parts) > 2:
            return f"{parts[1]}-{parts[2]}"
        return parts[1]
    return parts[0]


_ACTION_TYPE_PATTERNS = [
    ("EIC-ACC", ["eic accelerator", "eic-accelerator", "accelerator", "eic-acc", "eic acc"]),
    ("EIC-PATHFINDER", ["eic pathfinder", "pathfinder"]),
    ("EIC-TRANSITION", ["eic transition", "transition"]),
    ("MSCA", ["msca", "marie"]),
    ("COFUND", ["cofund"]),
    ("RIA", ["research and innovation action", " ria", "ria)", "ria "]),
    ("IA", ["innovation action", " ia", "ia)", "ia "]),
    ("CSA", ["coordination and support", " csa", "csa)", "csa "]),
    ("ERC", ["erc"]),
]


def normalize_action_type(raw: str) -> str:
    """Maps free-text type_of_action / funding_scheme strings to a small
    set of canonical categories -- the model and CORDIS format these very
    differently (model: "HORIZON EIC Accelerator"; CORDIS:
    "HORIZON-EIC-ACC"), so exact string equality would almost never match
    even when they mean the same thing. Order matters: specific patterns
    (EIC-ACC) checked before generic ones (IA) so "accelerator" isn't
    missed by a plain "ia" substring check."""
    text = (raw or "").strip().lower()
    if not text:
        return ""
    for canonical, patterns in _ACTION_TYPE_PATTERNS:
        if any(p in text for p in patterns):
            return canonical
    return text.upper()[:20]  # fallback: best-effort raw token, truncated


def parse_trl_range(trl_text: str):
    """Extracts a (min, max) integer TRL range from free text like
    'TRL 6-8', 'TRL 6+', 'TRL 5', 'End TRL 6-8; ...'. None if no TRL
    number is found at all."""
    text = (trl_text or "")
    numbers = []
    for match in re.finditer(r"TRL\s*(\d)(?:\s*[-–to]+\s*(\d))?", text, flags=re.IGNORECASE):
        numbers.append(int(match.group(1)))
        if match.group(2):
            numbers.append(int(match.group(2)))
    if not numbers:
        numbers = [int(n) for n in re.findall(r"\b([1-9])\b", text)]
    if not numbers:
        return None
    return (min(numbers), max(numbers))


def trl_overlap(true_trl: str, candidate_trl: str):
    """True if the two TRL ranges overlap at all (not exact match --
    'TRL 6-8' and 'TRL 6' should count as consistent). None if either
    side has no parseable TRL (can't disprove, not scored as a miss)."""
    true_range = parse_trl_range(true_trl)
    cand_range = parse_trl_range(candidate_trl)
    if true_range is None or cand_range is None:
        return None
    return not (true_range[1] < cand_range[0] or cand_range[1] < true_range[0])



TOPIC_DETAILS_SYSTEM_PROMPT = """
You are a research funding analyst with web search access to the EU
Funding & Tenders Portal.

CRITICAL: you must open the SPECIFIC topic page, not rely on search
result snippets. The topic's page URL follows this exact pattern:

https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/{TOPIC_ID}

Construct this URL by substituting {TOPIC_ID} with the exact topic ID you
were given (e.g. for topic ID "HORIZON-EIC-2021-TRANSITIONCHALLENGES-01-01"
the URL is
.../topic-details/HORIZON-EIC-2021-TRANSITIONCHALLENGES-01-01).
Open that exact URL and read its actual page content -- do not answer
from a search snippet, a summary page, or general knowledge.

Task: extract the topic's REAL published content from that page. Do not
guess or paraphrase from general knowledge -- only report what you
actually read on the page. If the page doesn't load or the topic isn't
found at that URL, say so explicitly (found: false) rather than
inventing content or answering from a search snippet instead.

The page has distinct "Scope" and "Expected Outcome" sections -- you MUST
extract the actual text under each of these headings as they appear on
the page. If you successfully opened the page but genuinely cannot find
either section's text, set found to false rather than returning found:
true with empty fields -- a partial result without the actual scope and
expected outcome text is not useful and must not be reported as success.

Return ONLY this JSON object, no text before or after:
{
  "found": true,
  "call_title": "",
  "scope": "the topic's official Scope section text, verbatim or close paraphrase from the page",
  "expected_outcome": "the topic's official Expected Outcome section text, verbatim or close paraphrase from the page",
  "trl": "",
  "type_of_action": ""
}
"""


def get_client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        raise ValueError(f"{config.API_KEY_ENV} is not set.")
    return OpenAI(api_key=config.OPENAI_API_KEY, base_url=f"{config.AZURE_ENDPOINT}openai/v1/")


def fetch_topic_details(topic_id: str, client: OpenAI, max_attempts: int = 3) -> dict:
    """One LLM call with web search per attempt, retried if the attempt
    doesn't produce a genuinely usable result -- live web search is
    genuinely variable (same lesson as D23/D28's consensus voting), and a
    single failed attempt on the ground-truth topic is a costly single
    point of failure since it's cached and reused across all 4 prompt
    variants for a case.

    CRITICAL: does not trust the model's self-reported found=true alone.
    A real test run showed found=true with call_title populated but
    scope/expected_outcome BOTH empty -- the model found the topic's
    title via a search snippet but never actually opened the full page
    (despite the prompt asking it to). A result is only treated as
    genuinely successful if found=true AND both scope and
    expected_outcome contain real text; otherwise it's retried like any
    other failure. Prompting alone was not sufficient here -- this is a
    code-level check, not just an instruction.
    """
    if not topic_id or not str(topic_id).strip():
        return {"found": False, "topic_id": topic_id, "reason": "empty topic_id"}

    last_result = None
    for attempt in range(1, max_attempts + 1):
        last_result = _fetch_topic_details_once(topic_id, client)
        has_real_content = (
            last_result.get("found")
            and str(last_result.get("scope", "")).strip()
            and str(last_result.get("expected_outcome", "")).strip()
        )
        if has_real_content:
            if attempt > 1:
                last_result["found_on_attempt"] = attempt
            return last_result
        # Downgrade a hollow found=true (title only, no real content) to
        # found=false so it's treated consistently as a failed attempt.
        if last_result.get("found") and not has_real_content:
            last_result["found"] = False
            last_result["reason"] = "found=true but scope/expected_outcome were empty (hollow result)"
    last_result["attempts_exhausted"] = max_attempts
    return last_result


def _fetch_topic_details_once(topic_id: str, client: OpenAI) -> dict:
    try:
        response = client.responses.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            tools=[{"type": "web_search"}],
            tool_choice="required",
            input=[
                {"role": "system", "content": TOPIC_DETAILS_SYSTEM_PROMPT},
                {"role": "user", "content": f"Topic ID: {topic_id}"},
            ],
            max_output_tokens=1500,
        )
        data = json.loads((response.output_text or "").strip())
        data["topic_id"] = topic_id
        return data
    except Exception as e:
        return {"found": False, "topic_id": topic_id, "reason": f"error: {e}"}


def get_embedding(text: str, client: OpenAI):
    text = (text or "").strip()
    if not text:
        return None
    response = client.embeddings.create(model=config.AZURE_EMBEDDING_DEPLOYMENT, input=text)
    return response.data[0].embedding


def _combined_text(details: dict) -> str:
    return f"{details.get('scope', '')} {details.get('expected_outcome', '')}".strip()


def compare_topic_pair(true_details: dict, candidate_details: dict, client: OpenAI) -> dict:
    """Compares one true topic vs one candidate. Similarity is the
    primary continuous score; TRL and type_of_action are reported as
    SEPARATE corroborating checks, never folded into one opaque number."""
    if not true_details.get("found") or not candidate_details.get("found"):
        return {"similarity": None, "trl_match": None, "type_of_action_match": None,
                "reason": "one or both topics not found/fetchable"}

    true_text = _combined_text(true_details)
    cand_text = _combined_text(candidate_details)
    if not true_text or not cand_text:
        return {"similarity": None, "trl_match": None, "type_of_action_match": None,
                "reason": "missing scope/expected_outcome text"}

    true_vec = get_embedding(true_text, client)
    cand_vec = get_embedding(cand_text, client)
    if true_vec is None or cand_vec is None:
        return {"similarity": None, "trl_match": None, "type_of_action_match": None,
                "reason": "embedding failed"}

    similarity = float(cosine_similarity([true_vec], [cand_vec])[0][0])

    trl_match = None
    if true_details.get("trl") and candidate_details.get("trl"):
        trl_match = str(true_details["trl"]).strip() == str(candidate_details["trl"]).strip()

    type_match = None
    if true_details.get("type_of_action") and candidate_details.get("type_of_action"):
        type_match = (str(true_details["type_of_action"]).strip().lower()
                      == str(candidate_details["type_of_action"]).strip().lower())

    return {"similarity": similarity, "trl_match": trl_match, "type_of_action_match": type_match}


def compare_recommended_to_ground_truth(
    true_topic_details: dict,
    recommended_topic_ids: list[str],
    client: OpenAI,
    top_k: int = 3,
) -> dict:
    """
    Compares up to top_k recommended topics (in ranked order -- the first
    top_k of the already fit_score-sorted funding_matches) against the
    pre-fetched ground-truth topic details (fetch true_topic_details ONCE
    per case, not per variant, and pass it in -- it doesn't change across
    prompt variants). Returns the BEST match found among the checked
    candidates, plus per-candidate detail for audit.
    """
    true_id = str(true_topic_details.get("topic_id", ""))
    exact_match = true_id != "" and true_id in [str(t) for t in recommended_topic_ids]

    candidates_to_check = recommended_topic_ids[:top_k]
    results = []
    for topic_id in candidates_to_check:
        cand_details = fetch_topic_details(topic_id, client)
        comparison = compare_topic_pair(true_topic_details, cand_details, client)
        results.append({"topic_id": topic_id, **comparison})

    valid = [(r["topic_id"], r["similarity"]) for r in results if r["similarity"] is not None]
    if valid:
        best_topic_id, best_similarity = max(valid, key=lambda x: x[1])
        best_result = next(r for r in results if r["topic_id"] == best_topic_id)
    else:
        best_topic_id, best_similarity, best_result = None, None, None

    return {
        "exact_match": exact_match,
        "best_match_topic_id": best_topic_id,
        "best_similarity": best_similarity,
        "best_trl_match": best_result["trl_match"] if best_result else None,
        "best_type_of_action_match": best_result["type_of_action_match"] if best_result else None,
        "n_candidates_checked": len(candidates_to_check),
        "n_candidates_found": sum(1 for r in results if r.get("similarity") is not None),
        "per_candidate": results,
    }