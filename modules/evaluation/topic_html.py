import html as html_mod
import re
import sys
from pathlib import Path

BASE_URL = (
    "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/"
    "opportunities/topic-details/{topic}"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

CSV_FIELDS = [
    "topic_id", "title", "status", "call", "type_of_action",
    "opening_date", "due_date", "expected_outcome", "expected_impact", "scope", "trl",
    "fetch_status", "error",
]

# ---------------------------------------------------------------------------
# Fetching (from fetch_htlm_script.py, unchanged logic)
# ---------------------------------------------------------------------------


def fetch_rendered_html(url: str, timeout_ms: int = 60_000, headless: bool = True) -> str:
    """Load the page in headless Chromium and return the rendered HTML."""
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, channel="msedge")
        context = browser.new_context(user_agent=USER_AGENT, locale="en-GB")
        page = context.new_page()

        print(f"Loading {url} ...")
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        try:
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except PWTimeout:
            print("Warning: network did not go idle; continuing anyway.", file=sys.stderr)

        topic_id = url.rstrip("/").split("/")[-1]
        try:
            page.wait_for_function(
                "id => document.body && document.body.innerText.includes(id)",
                arg=topic_id,
                timeout=timeout_ms,
            )
        except PWTimeout:
            print("Warning: topic content not detected; HTML may be incomplete.",
                  file=sys.stderr)

        rendered_html = page.content()
        browser.close()
        return rendered_html


# ---------------------------------------------------------------------------
# HTML -> text helpers (from extract_topic_info.py, unchanged logic)
# ---------------------------------------------------------------------------


def html_to_text(fragment: str) -> str:
    text = fragment
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "\n- ", text)
    text = re.sub(r"(?i)</(p|div|li|ul|ol|tr|h[1-6])>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_mod.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?im)^\s*Show (more|less)\s*$\n?", "", text)
    return text.strip()


def clean_inline(fragment: str) -> str:
    return re.sub(r"\s+", " ", html_to_text(fragment)).strip()


# ---------------------------------------------------------------------------
# Field extractors (from extract_topic_info.py, unchanged logic)
# ---------------------------------------------------------------------------


def extract_label_values(page: str) -> dict:
    pattern = re.compile(
        r'<div[^>]*class="[^"]*eui-u-f-bold[^"]*"[^>]*>(.*?)</div>\s*'
        r'<div[^>]*>(.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    )
    values = {}
    for label_html, value_html in pattern.findall(page):
        label = clean_inline(label_html)
        value = clean_inline(value_html)
        if label and value and value not in values.get(label, []):
            values.setdefault(label, []).append(value)
    return {k: " | ".join(v) for k, v in values.items()}


def extract_description_sections(page: str) -> dict:
    heading = re.compile(
        r'<p[^>]*class="[^"]*topicdescriptionkind[^"]*"[^>]*>(.*?)</p>',
        re.IGNORECASE | re.DOTALL,
    )
    marks = list(heading.finditer(page))
    sections = {}
    for i, m in enumerate(marks):
        name = clean_inline(m.group(1)).rstrip(":").strip()
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(page)
        body = page[start:end]
        if i + 1 == len(marks):
            cut = re.search(r"</eui-card-content>", body, re.IGNORECASE)
            if cut:
                body = body[: cut.start()]
        sections[name.lower()] = body
    return sections


def first_paragraph(section_html: str) -> str:
    for m in re.finditer(r"<p[^>]*>(.*?)</p>", section_html,
                         re.IGNORECASE | re.DOTALL):
        text = html_to_text(m.group(1))
        if text:
            return text
    cut = re.search(r"<(ul|ol)\b", section_html, re.IGNORECASE)
    head = section_html[: cut.start()] if cut else section_html
    return html_to_text(head)


def extract_topic_id_field(page: str, fallback: str) -> str:
    m = re.search(
        r'class="[^"]*eui-common-header__sub-label-text[^"]*"[^>]*>(.*?)</div>',
        page, re.IGNORECASE | re.DOTALL,
    )
    if m:
        value = clean_inline(m.group(1))
        if value:
            return value
    m = re.search(r"(HORIZON-[A-Z0-9\-]+)", page)
    return m.group(1) if m else fallback


def extract_title(page: str) -> str:
    m = re.search(
        r'class="[^"]*eui-common-header__label-text[^"]*"[^>]*>(.*?)</div>',
        page, re.IGNORECASE | re.DOTALL,
    )
    if m:
        title = clean_inline(m.group(1))
        if title:
            return title
    m = re.search(r"<title[^>]*>(.*?)</title>", page, re.IGNORECASE | re.DOTALL)
    return clean_inline(m.group(1)) if m else ""


def extract_status(page: str) -> str:
    m = re.search(
        r"<sedia-project-status[^>]*>(.*?)</sedia-project-status>",
        page, re.IGNORECASE | re.DOTALL,
    )
    return clean_inline(m.group(1)) if m else ""


def find_section(sections: dict, contains: str) -> str:
    """Some topics don't use the exact heading text we look up directly
    (e.g. the TRL heading is 'Technology Readiness Level - Technology
    Readiness Level expected from completed projects', not just 'TRL') --
    match on a substring of the heading instead of an exact key."""
    for name, body in sections.items():
        if contains in name:
            return body
    return ""


def extract_from_html(page: str, fallback_id: str) -> dict:
    """Parse an already-in-memory HTML string (no disk re-read needed)."""
    labels = extract_label_values(page)
    sections = extract_description_sections(page)

    # "scope" is the primary field, but ~2% of topics (confirmed across an
    # 881-topic sample) use "Objective" instead -- fall back to it rather
    # than silently scoring those topics as having no scope at all.
    scope = first_paragraph(sections.get("scope", "")) or first_paragraph(sections.get("objective", ""))

    return {
        "topic_id": extract_topic_id_field(page, fallback_id),
        "title": extract_title(page),
        "status": extract_status(page),
        "programme": labels.get("Programme", ""),
        "call": labels.get("Call", ""),
        "type_of_action": labels.get("Type of action", ""),
        "opening_date": labels.get("Opening date", ""),
        "due_date": labels.get("Deadline date", "") or labels.get("Deadline", ""),
        "expected_outcome": html_to_text(sections.get("expected outcome", "")),
        "expected_impact": html_to_text(sections.get("expected impact", "")),
        "scope": scope,
        "trl": html_to_text(find_section(sections, "technology readiness level")),
    }


# ---------------------------------------------------------------------------
# Combined fetch + extract, with a cache keyed by topic_id
# ---------------------------------------------------------------------------


def fetch_and_extract(
    topic_id: str,
    cache_dir: Path,
    force_refetch: bool = False,
    headless: bool = True,
    timeout_ms: int = 60_000,
) -> dict:
    """
    Returns extract_from_html()'s field dict plus 'fetch_status'
    ('cached' | 'fetched') and 'error' (empty string on success).

    Cache hit: skips Playwright entirely (no network, no browser launch) --
    this is what makes repeat runs over the same topic_id list cheap.
    Cache miss: fetches once, writes the HTML to cache_dir/{topic_id}.html
    for next time, and parses the string already in hand rather than
    reading the file back.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{topic_id}.html"

    error = ""
    try:
        if cache_path.exists() and not force_refetch:
            page_html = cache_path.read_text(encoding="utf-8", errors="replace")
            fetch_status = "cached"
        else:
            url = BASE_URL.format(topic=topic_id)
            page_html = fetch_rendered_html(url, timeout_ms=timeout_ms, headless=headless)
            cache_path.write_text(page_html, encoding="utf-8")
            fetch_status = "fetched"

        data = extract_from_html(page_html, fallback_id=topic_id)
    except Exception as exc:  # keep batch runs going past one bad topic_id
        data = {k: "" for k in CSV_FIELDS if k not in ("fetch_status", "error")}
        data["topic_id"] = topic_id
        fetch_status = "error"
        error = f"{type(exc).__name__}: {exc}"

    data["fetch_status"] = fetch_status
    data["error"] = error
    return data