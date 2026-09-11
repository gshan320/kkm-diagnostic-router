"""FUKKM (MOH Medicines Formulary / "Blue Book") scraper.

The first pass of this scraper read the listing table by fixed column index and
silently lost the generic drug name: `drug_name` ended up holding the row
number, and every later field was shifted one column left. This version maps
columns by their table *header text*, so a column re-order upstream can no
longer corrupt the output, and it validates the result before overwriting a
good file.

Usage
-----
    python -m app.scrape_fukkm                  # full scrape -> raw_json/
    python -m app.scrape_fukkm --max-pages 2    # sample first, inspect, then commit
    python -m app.scrape_fukkm --out /tmp/f.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from . import config

BASE_URL = "https://pharmacy.moh.gov.my/en/apps/fukkm"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_DELAY_S = 1.0
MAX_RETRIES = 3

# header keyword -> canonical field. Order matters: first match wins.
COLUMN_RULES: list[tuple[tuple[str, ...], str]] = [
    (("no.", "no", "bil", "#"), "fukkm_no"),
    (("generic", "drug", "name", "ubat"), "drug_name"),
    (("mdc", "code", "kod"), "mdc_code"),
    (("prescriber", "category", "kategori"), "prescriber_category"),
    (("indication", "indikasi"), "indication"),
    (("dose", "dosage", "strength", "presentation"), "dosage"),
]
FIELDS = ["fukkm_no", "drug_name", "mdc_code", "prescriber_category", "indication", "dosage"]
# Positional layout observed on the live listing, used only when the table has
# no usable header row.
FALLBACK_ORDER = ["fukkm_no", "drug_name", "mdc_code", "prescriber_category", "indication"]

CATEGORY_CODE_RE = re.compile(
    r"^\s*(?:A\*?|B|C\+?|KK|A/KK|A\*/KK)(?:\s*,\s*(?:A\*?|B|C\+?|KK|A/KK|A\*/KK))*\s*$"
)


def fetch(page: int) -> str:
    url = f"{BASE_URL}?generic=&page={page}"
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.text
            last = RuntimeError(f"HTTP {resp.status_code}")
        except requests.RequestException as exc:
            last = exc
        wait = attempt * 2
        print(f"   retry {attempt}/{MAX_RETRIES} for page {page} in {wait}s ({last})")
        time.sleep(wait)
    raise RuntimeError(f"Could not fetch page {page}: {last}")


def map_columns(table) -> dict[int, str] | None:
    """Build {column index -> canonical field} from the table header."""
    header_cells = []
    if (thead := table.find("thead")) and (row := thead.find("tr")):
        header_cells = row.find_all(["th", "td"])
    elif (row := table.find("tr")) and row.find("th"):
        header_cells = row.find_all(["th", "td"])
    if not header_cells:
        return None

    mapping: dict[int, str] = {}
    taken: set[str] = set()
    for idx, cell in enumerate(header_cells):
        label = " ".join(cell.get_text(" ", strip=True).lower().split())
        for keywords, field in COLUMN_RULES:
            if field in taken:
                continue
            if any(k in label for k in keywords):
                mapping[idx] = field
                taken.add(field)
                break
    return mapping or None


def parse_page(html: str) -> tuple[list[dict], bool]:
    """Return (records, has_next_page)."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return [], False

    mapping = map_columns(table)
    if mapping is None:
        mapping = dict(enumerate(FALLBACK_ORDER))
        print("   ! no usable header row; falling back to positional columns")

    body = table.find("tbody") or table
    rows = body.find_all("tr")
    records: list[dict] = []
    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue  # header row
        record = {field: "" for field in FIELDS}
        record["source"] = "FUKKM (MOH Medicines Formulary / Blue Book)"
        for idx, cell in enumerate(cells):
            field = mapping.get(idx)
            if field:
                record[field] = cell.get_text(" ", strip=True)
        if any(record[f] for f in ("drug_name", "mdc_code", "indication")):
            records.append(record)

    has_next = bool(
        soup.find("li", class_="pager-next")
        or soup.find("a", attrs={"rel": "next"})
        or soup.find("a", string=re.compile(r"next", re.IGNORECASE))
    )
    return records, has_next


def validate(records: list[dict]) -> list[str]:
    """Catch the exact failure that corrupted the first scrape."""
    problems: list[str] = []
    if not records:
        return ["no records scraped"]
    total = len(records)
    numeric_names = sum(1 for r in records if r["drug_name"].strip().isdigit())
    named = sum(1 for r in records if r["drug_name"].strip() and not r["drug_name"].strip().isdigit())
    cat_ok = sum(1 for r in records if CATEGORY_CODE_RE.match(r["prescriber_category"]))
    cat_misplaced = sum(1 for r in records if CATEGORY_CODE_RE.match(r["indication"]))

    if named < total * 0.5:
        problems.append(
            f"only {named}/{total} rows carry a real generic name "
            f"({numeric_names} hold a bare number) - the column mapping is wrong"
        )
    if cat_misplaced > total * 0.3:
        problems.append(
            f"{cat_misplaced}/{total} rows have a prescriber-category code sitting in "
            "the `indication` column - the columns are shifted"
        )
    if cat_ok < total * 0.5:
        problems.append(
            f"only {cat_ok}/{total} rows have a recognisable prescriber category "
            "(A, A*, A/KK, B, C, C+)"
        )
    return problems


def scrape(max_pages: int | None = None) -> list[dict]:
    all_records: list[dict] = []
    seen: set[str] = set()
    page = 0
    print(f"Scraping FUKKM from {BASE_URL}")
    while True:
        if max_pages is not None and page >= max_pages:
            print(f"Stopping at --max-pages={max_pages}.")
            break
        print(f"  page {page} ...", end=" ", flush=True)
        records, has_next = parse_page(fetch(page))
        fresh = [r for r in records if (key := json.dumps(r, sort_keys=True)) not in seen
                 and not seen.add(key)]
        print(f"{len(fresh)} new rows (total {len(all_records) + len(fresh)})")
        if not fresh:
            print("  no new rows - stopping.")
            break
        all_records.extend(fresh)
        if not has_next:
            print("  no next-page link - stopping.")
            break
        page += 1
        time.sleep(REQUEST_DELAY_S)
    return all_records


def main() -> int:
    ap = argparse.ArgumentParser(description="Scrape the FUKKM drug listing to JSON.")
    ap.add_argument("--out", type=Path, default=config.FUKKM_JSON)
    ap.add_argument("--max-pages", type=int, default=None, help="sample N pages then stop")
    ap.add_argument("--force", action="store_true", help="write even if validation fails")
    args = ap.parse_args()

    records = scrape(args.max_pages)
    problems = validate(records)

    if problems:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        if records:
            print("\nFirst scraped row, for inspection:", file=sys.stderr)
            print(json.dumps(records[0], indent=2, ensure_ascii=False), file=sys.stderr)
        if not args.force:
            print(
                "\nRefusing to overwrite "
                f"{args.out} with data that would silently corrupt drug lookup. "
                "Inspect the page structure, fix COLUMN_RULES, or pass --force.",
                file=sys.stderr,
            )
            return 1
        print("\n--force given: writing anyway.", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {len(records)} drug records to {args.out}")
    print("Next: python -m app.ingest --reset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
