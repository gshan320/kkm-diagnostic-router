"""Fetch the National Antimicrobial Guideline 2024 (4th edition) text for indexing.

Why this exists
---------------
The NAG decides antibiotic choice across MOH facilities and is published ONLY
as a website (https://sites.google.com/moh.gov.my/nag) - there is no PDF to put
in raw_pdfs/. Without it, every infection presentation is answered from CPGs
that were not written to settle antibiotic choice. This fetches each chapter
page listed in the verified reference registry (references.py) and saves its
text; `python -m app.ingest --web-only` then indexes it, each chunk carrying
the page's URL so a citation opens the live NAG page.

    python -m app.scrape_nag          # -> data/raw_json/nag_pages.json
    python -m app.ingest --web-only   # add those pages to the index, no rebuild

The output is regenerable and git-ignored, like the FUKKM JSON. Re-run both
commands when the NAG is updated (its "What's new" page announces revisions).
"""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import date

from . import config, references

OUT = config.RAW_JSON_DIR / "nag_pages.json"


def page_text(body: str) -> str:
    """The readable content of a Google Sites page: scripts and chrome dropped,
    block elements turned into line breaks so table rows stay separate."""
    s = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", body)
    m = re.search(r'(?is)<div role="main".*', s)
    s = m.group(0) if m else s
    s = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|td|th)>", "\n", s)
    t = html.unescape(re.sub(r"<[^>]+>", " ", s))
    t = re.sub(r"[ \t ]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def build() -> list[dict]:
    rows = []
    for e in references.registry():
        if e.get("role") != "antimicrobial":
            continue
        code, body = references._get(e["url"])
        if code != 200 or not body:
            print(f"  skip {e['id']}: HTTP {code}")
            continue
        text = page_text(body)
        if len(text) < 200:
            continue
        rows.append({"id": e["id"], "url": e["url"], "title": e["title"],
                     "population": e.get("population", "all"),
                     "fetched": date.today().isoformat(), "text": text})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    return rows


if __name__ == "__main__":
    rows = build()
    chars = sum(len(r["text"]) for r in rows)
    print(f"{len(rows)} NAG pages, {chars:,} characters -> {OUT}")
    sys.exit(0 if rows else 1)
