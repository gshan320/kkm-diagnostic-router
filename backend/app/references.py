"""Verified external references for what the indexed corpus cannot answer.

Why this exists
---------------
Some authoritative KKM guidance is not a PDF the index can hold. The National
Antimicrobial Guideline 2024 (4th edition) is published ONLY as a website, and
it is the guidance that decides antibiotic choice. When a report reaches a
question like that, "no data found" wastes the clinician's time and a guessed
answer is worse. The report instead shows a link to the exact page.

The one rule that makes this safe: **a URL is never written by the model.**
Language models produce plausible, non-existent addresses. Every link comes from
the registry in data/reference_links.json, which is built by crawling the real
site (so every address was served when it was recorded) and re-checked by a
link checker; a link whose last check failed is not shown.

    python -m app.references --build         # crawl NAG, rewrite the registry
    python -m app.references --check         # re-check every link, record status + date
    python -m app.references --build-cited   # the DOIs / URLs KKM documents cite
    python -m app.references --check-cited   # confirm each (DOIs via the doi.org handle API)

Which link a report gets is decided by code, from the working diagnosis
matched against each page's own section headings (NAG "A11: Respiratory
infections" lists "Community acquired pneumonia", "Infective exacerbation of
COPD", ...). Links are labelled as references to consult, not sources the
report was generated from.
"""

from __future__ import annotations

import html
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import date
from functools import lru_cache

from . import config, indications

REGISTRY = config.DATA_DIR / "reference_links.json"
# The references KKM documents themselves cite (their reference lists), with a
# DOI or web address - built by build_cited() from the indexed PDFs.
CITED = config.DATA_DIR / "cited_references.json"
NAG_ROOT = "https://sites.google.com/moh.gov.my/nag"
NAG_CONTENTS = f"{NAG_ROOT}/contents"
UA = {"User-Agent": "Mozilla/5.0 (KKM Diagnostic Router link checker)"}

# Always-available fallbacks, shown only when nothing more specific applies.
STATIC = [
    {
        "id": "mahtas-cpg-list",
        "title": "MaHTAS Clinical Practice Guidelines - full list",
        "publisher": "Ministry of Health Malaysia",
        "url": "https://mymahtas.moh.gov.my/index.php/docman-list/publications/cpg-list",
        "population": "all",
        "topics": [],
        "role": "fallback_guideline",
    },
    {
        "id": "amm-cpg-list",
        "title": "Academy of Medicine of Malaysia - Clinical Practice Guidelines",
        "publisher": "Academy of Medicine of Malaysia",
        "url": "https://www.acadmed.org.my/index.cfm?&menuid=67",
        "population": "all",
        "topics": [],
        "role": "fallback_guideline",
    },
]


def _get(url: str, timeout: float = 30) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def build() -> list[dict]:
    status, page = _get(NAG_CONTENTS)
    if status != 200:
        raise SystemExit(f"NAG contents page returned {status}")
    paths = sorted(set(re.findall(r'href="(/moh\.gov\.my/nag/contents/section-[^"#?]+)"', page)))
    entries: list[dict] = []
    today = date.today().isoformat()
    for path in paths:
        if path.rstrip("/").count("/") < 5:  # a section index page, not a chapter
            continue
        url = "https://sites.google.com" + path
        code, body = _get(url)
        if code != 200:
            continue
        title = _text(re.search(r"<title>(.*?)</title>", body, re.S).group(1)) if "<title>" in body else path
        title = title.replace("National Antimicrobial Guideline (NAG), Ministry of Health Malaysia - ", "")
        heads = [_text(h) for _, h in re.findall(r"<h([1-3])[^>]*>(.*?)</h\1>", body, re.S)]
        topics = [title] + [re.sub(r"^\d+\.\s*", "", h) for h in heads if 3 < len(h) < 140]
        section = "paediatric" if "section-b-paediatrics" in path else (
            "primary_care" if "section-c-clinical-pathways" in path else "adult")
        entries.append({
            "id": "nag-" + path.rstrip("/").rsplit("/", 1)[-1],
            "title": f"National Antimicrobial Guideline 2024 - {title.title() if title.isupper() else title}",
            "publisher": "Ministry of Health Malaysia, Pharmaceutical Services Programme",
            "url": url,
            "population": {"paediatric": "paediatric", "adult": "adult"}.get(section, "all"),
            "topics": topics,
            "role": "antimicrobial",
            "checked": today,
            "status": 200,
        })
    registry = entries + [dict(s, checked=None, status=None) for s in STATIC]
    REGISTRY.write_text(json.dumps(registry, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return registry


_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_URL = re.compile(r"https?://[^\s\"<>]+", re.I)
_REF_START = re.compile(r"(?:^|\n)\s*\[?\d{1,3}[.)\]]\s")
_GUIDELINE_WORDS = re.compile(r"guideline|consensus|recommendation|position statement|practice parameter|"
                              r"management of|scientific statement|task force", re.I)
# A site's front page is not a reference to read.
_BARE_SITE = re.compile(r"^https?://[^/]+/?$", re.I)


def _entry_text(page: str, pos: int) -> str:
    """The reference entry ending at `pos`: back to the list number that
    starts it, else the preceding 250 characters."""
    head = page[max(0, pos - 400):pos]
    starts = list(_REF_START.finditer(head))
    if starts:
        head = head[starts[-1].end():]
    else:
        head = head[-250:]
    return re.sub(r"\s+", " ", head).strip(" .;,")


def build_cited() -> list[dict]:
    """Every DOI or web address in the reference lists of the indexed KKM and
    Malaysian documents, with the entry text around it and where it is cited.
    These are the sources KKM itself relies on - the only outside reading this
    project points to (source_policy.py)."""
    import pymupdf  # noqa: PLC0415

    from . import population, source_policy  # noqa: PLC0415
    from .ingest import parse_title  # noqa: PLC0415
    out: dict[str, dict] = {}
    for path in sorted(config.RAW_PDF_DIR.glob("*.pdf")):
        basis, _ = source_policy.basis(path.name)
        if basis not in (source_policy.KKM, source_policy.MALAYSIAN_BODY):
            continue
        title = parse_title(path.name)
        try:
            doc = pymupdf.open(path)
        except Exception:
            continue
        for pno, page in enumerate(doc, start=1):
            text = page.get_text()
            found = [(m.start(), "https://doi.org/" + m.group(1).rstrip(".,;)]")) for m in _DOI.finditer(text)]
            found += [(m.start(), m.group(0).rstrip(".,;)]")) for m in _URL.finditer(text)
                      if "doi.org" not in m.group(0).lower()]
            for pos, url in found:
                if _BARE_SITE.match(url) or url in out:
                    continue
                entry = _entry_text(text, pos)
                if len(entry) < 25:
                    continue
                out[url] = {
                    "id": "cited-" + re.sub(r"[^a-z0-9]+", "-", url.lower())[-60:].strip("-"),
                    "title": entry[:220],
                    "publisher": f"Cited by {title} p{pno}",
                    "url": url,
                    "population": population.title_population(title),
                    "topics": [entry],
                    "role": "cited_reference",
                    "guideline": bool(_GUIDELINE_WORDS.search(entry)),
                    "cited_by": title,
                    "page": pno,
                    "checked": None,
                    "status": None,
                }
    rows = sorted(out.values(), key=lambda r: (r["cited_by"], r["page"]))
    CITED.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


def _doi_exists(url: str) -> int:
    """200 when doi.org knows the DOI (its handle API, one small JSON reply -
    publishers often refuse scripted requests to the article itself)."""
    doi = url.split("doi.org/", 1)[1]
    code, body = _get("https://doi.org/api/handles/" + urllib.parse.quote(doi, safe="/"), timeout=15)
    return 200 if code == 200 and '"responseCode":1' in body.replace(" ", "") else (code or 0)


def check(cited: bool = False) -> list[dict]:
    path = CITED if cited else REGISTRY
    rows = json.loads(path.read_text(encoding="utf-8"))
    today = date.today().isoformat()
    for e in rows:
        code = _doi_exists(e["url"]) if "doi.org/" in e["url"] else _get(e["url"])[0]
        e["status"], e["checked"] = code, today
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


@lru_cache(maxsize=1)
def registry() -> list[dict]:
    if not REGISTRY.exists():
        return []
    rows = json.loads(REGISTRY.read_text(encoding="utf-8"))
    # A link whose last check failed is never shown.
    return [r for r in rows if r.get("status") in (None, 200)]


@lru_cache(maxsize=1)
def cited() -> list[dict]:
    """The KKM-cited references - only those a link check has confirmed."""
    if not CITED.exists():
        return []
    return [r for r in json.loads(CITED.read_text(encoding="utf-8")) if r.get("status") == 200]


def select_cited(diagnosis: str, age: float, limit: int = 2) -> list[dict]:
    """References KKM documents cite that are about this diagnosis - the
    diagnosis's own specific words in the entry - guidelines first, then the
    most recent citing document's order."""
    words = {c for c in indications.concepts(diagnosis or "", expand=False) if " " in c or len(c) >= 6}
    if not words:
        return []
    out = []
    for e in cited():
        if not _population_ok(e, age):
            continue
        hit = indications.mentions(words, e["title"])
        if hit:
            out.append(((not e.get("guideline"), -len(hit)), {
                **e, "reason": f"cited by {e['cited_by']} p{e['page']}; about \"{hit}\""}))
    out.sort(key=lambda x: x[0])
    return [e for _, e in out[:limit]]


def _population_ok(entry: dict, age: float) -> bool:
    pop = entry.get("population", "all")
    if pop == "paediatric":
        return age < 18
    if pop == "adult":
        return age >= 12
    return True


def select(diagnosis: str, age: float, need_guideline: bool, limit: int = 2) -> list[dict]:
    """Registry entries for this diagnosis, most specific first.

    `need_guideline` adds the MOH CPG list as a fallback when the report has no
    indexed guideline governing its diagnosis."""
    def strong(cs: set[str]) -> set[str]:
        return {c for c in cs if " " in c or len(c) >= 6}

    # The diagnosis's own words outrank synonym expansions: "sepsis" must land
    # on "A12: Sepsis" before its expansion "bacterial infection" finds a page
    # that merely mentions bacterial infections.
    direct = strong(indications.concepts(diagnosis or "", expand=False))
    expanded = strong(indications.concepts(diagnosis or "")) - direct
    out = []
    for e in registry():
        if e.get("role") == "fallback_guideline" or not _population_ok(e, age):
            continue
        best = None
        for topic in e.get("topics", []):
            for tier, pool in ((0, direct), (1, expanded)):
                hit = indications.mentions(pool, topic)
                if hit:
                    key = (tier, -len(hit))
                    if best is None or key < best[0]:
                        best = (key, topic)
                    break
        if best:
            out.append((best[0], {**e, "reason": f"covers \"{best[1]}\""}))
    out.sort(key=lambda x: x[0])
    chosen = [e for _, e in out[:limit]]
    if need_guideline and not chosen:
        chosen += [dict(e, reason="no indexed guideline governs this diagnosis")
                   for e in registry() if e.get("role") == "fallback_guideline"]
    return chosen


if __name__ == "__main__":
    if "--build" in sys.argv:
        rows = build()
        print(f"{len(rows)} reference links -> {REGISTRY}")
    elif "--build-cited" in sys.argv:
        rows = build_cited()
        print(f"{len(rows)} cited references -> {CITED}")
    elif "--check-cited" in sys.argv:
        rows = check(cited=True)
        bad = [r for r in rows if r.get("status") != 200]
        print(f"{len(rows)} checked, {len(bad)} not OK")
    elif "--check" in sys.argv:
        rows = check()
        bad = [r for r in rows if r.get("status") != 200]
        print(f"{len(rows)} checked, {len(bad)} not OK: " + ", ".join(r["id"] for r in bad))
    else:
        print(__doc__)
