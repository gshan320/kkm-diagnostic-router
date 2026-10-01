"""Table-aware page text for the MOH Standard Practice Guidelines for Assistant
Medical Officers in Emergency Medicine and Trauma Services (2023).

Why this exists
---------------
The document is a core ED source - 20 chapters, 9 procedures, 38 appendices,
published by the MOH Medical Practice Division - and a plain text extraction
reads it badly in two ways (measured 2026-09-30):

  1. Its clinical content is in 4-column tables (Activity | Work Process |
     Standard | Requirement) on 50 pages and 2-column tables (Component |
     Description) on 30. Extracted as a stream, the columns interleave: the
     equipment list ("PPE, ECG machine") runs into the clinical steps, and the
     standard a step rests on ("Management of Acute STEMI 4th Edition 2019")
     floats free of it.
  2. Its flow charts and many appendices are IMAGES - the asthma management
     table (p104), the bradycardia and tachycardia algorithms (p108, p111),
     the aggressive-patient flow chart (p116), burn resuscitation fluid rates
     (p141). A text extraction has only their captions.

So each page becomes:
  - its chapter / procedure / appendix heading, carried across pages;
  - every table row as one block - "Activity: ... / Work process: ... /
    Standard: ... / Requirement: ..." - so a chunk never splits a step from
    the standard it cites;
  - the remaining prose;
  - the text of the page's images, read by OCR (tools/ocr_pdf.swift), as lines
    NOT already in the text layer, each kept only if most of its words are
    real words (the flow-chart images were compressed twice and some OCR is
    noise: "Banned Par", "Perfomed intal"). Marked as OCR so a reader knows to
    check the page image.
"""

from __future__ import annotations

import re

import pymupdf

RUNNING_HEADERS = re.compile(
    r"^\s*(?:Clinical|Standard Practice) Guidelines For AMO in EMTS MOH\s*$|^\s*[ivxlc]+\s*$|^\s*\d{1,3}\s*$",
    re.IGNORECASE)
HEADING = re.compile(r"\b(CHAPTER\s+\d+\s*:\s*[^\n|]{3,80}|PROCEDURE\s+\d+\s*:\s*[^\n|]{3,80}|"
                     r"APPENDIX\s+\d+\s*:\s*[^\n|]{3,90})", re.IGNORECASE)
_COLUMN_LABELS = {
    ("activity", "work process", "standard", "requirement"):
        ("Activity", "Work process", "Standard (source)", "Requirement (equipment)"),
    ("component", "description"): ("Component", "Description"),
}


def _clean(cell: str | None) -> str:
    t = (cell or "").replace("’", "'").replace("‘", "'")
    t = re.sub(r"[-]", " ", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _norm(text: str) -> str:
    t = (text or "").lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9%/.' ]+", " ", re.sub(r"\s+", " ", t))


def _table_blocks(page: pymupdf.Page) -> tuple[list[str], list[pymupdf.Rect]]:
    blocks: list[str] = []
    rects: list[pymupdf.Rect] = []
    try:
        tables = page.find_tables().tables
    except Exception:  # noqa: BLE001 - a page that defeats the table finder reads as prose
        return blocks, rects
    for t in tables:
        rows = t.extract()
        if not rows:
            continue
        head = tuple(_clean(c).lower() for c in rows[0])
        labels = _COLUMN_LABELS.get(head)
        body = rows[1:] if labels else rows
        for row in body:
            cells = [_clean(c) for c in row]
            if not any(cells):
                continue
            if labels and len(labels) == len(cells):
                parts = [f"{lab}: {val}" for lab, val in zip(labels, cells) if val]
                # A Component/Description row reads as one line: "Objectives: ..."
                if len(labels) == 2 and cells[0]:
                    parts = [f"{cells[0]}: {cells[1]}"] if cells[1] else [cells[0]]
            else:
                parts = [c for c in cells if c]
            blocks.append("\n".join(parts))
        rects.append(pymupdf.Rect(t.bbox))
    return blocks, rects


def _prose_outside(page: pymupdf.Page, rects: list[pymupdf.Rect]) -> str:
    lines = []
    for b in page.get_text("blocks"):
        r = pymupdf.Rect(b[:4])
        if any(r.intersects(t) and (r & t).get_area() > 0.5 * r.get_area() for t in rects):
            continue
        for line in (b[4] or "").splitlines():
            if line.strip() and not RUNNING_HEADERS.match(line):
                lines.append(line.strip())
    return _clean("\n".join(lines)).replace(" ", " ")


_DOSE_LINE = re.compile(r"\d+(?:\.\d+)?\s*(?:-|\u2013|to)?\s*\d*\s*(?:ml|mg|mcg|g|kg|%|hr|h\b|units?|iu|l\b|min)", re.I)


def _image_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Figure images - not the full-page backgrounds this document prints on
    every page, not icons."""
    area = page.rect.get_area()
    out = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if 0.01 * area < r.get_area() < 0.7 * area:
            out.append(r)
    return out


def _reading_order(lines: list[dict], W: float) -> list[dict]:
    """Columns first. Then: short cells in three or more columns are a TABLE
    and read row by row (burn fluid rates: category | age | rate | urine
    output); longer lines are boxed PROSE and read column by column (the asthma
    severity columns)."""
    if not lines:
        return lines
    xs = sorted(lines, key=lambda x: x["x0"])
    cols: list[list[dict]] = [[xs[0]]]
    for ln in xs[1:]:
        if ln["x0"] - max(c["x0"] for c in cols[-1]) > 0.08 * W:
            cols.append([ln])
        else:
            cols[-1].append(ln)
    avg = sum(len(ln["t"]) for ln in lines) / len(lines)
    if len(cols) >= 3 and avg < 32:
        rows: list[list[dict]] = []
        for ln in sorted(lines, key=lambda x: x["cy"]):
            if rows and abs(ln["cy"] - rows[-1][0]["cy"]) < 9:
                rows[-1].append(ln)
            else:
                rows.append([ln])
        return [ln for r in rows for ln in sorted(r, key=lambda x: x["x0"])]
    return [ln for col in cols for ln in sorted(col, key=lambda x: x["cy"])]


# Misreadings seen in this document's figure OCR (2026-09-30), corrected only
# in OCR text - never in the text layer. Each is unambiguous in context.
_OCR_FIXES = (
    (re.compile(r"\b(SaO|SpO)[,.]?\s*2(\d{2})\s*%"), r"\g<1>2 ≥\2%"),   # "SaO, 290%" -> "SaO2 ≥90%"
    (re.compile(r"\b(SaO|SpO)[,.](?=\s)"), r"\g<1>2"),
    (re.compile(r"\bFEV[,.](?=\s)"), "FEV1"),
    (re.compile(r"\bO, saturation"), "O2 saturation"),
    (re.compile(r"\bEGG\b"), "ECG"),
    (re.compile(r"\bCritically III\b"), "Critically Ill"),
    (re.compile(r"\bkgx\b"), "kg x"),
)


def _fix_ocr(text: str) -> str:
    for rx, rep in _OCR_FIXES:
        text = rx.sub(rep, text)
    return text


def _ocr_supplement(page: pymupdf.Page, ocr_lines: list[dict], layer_text: str, known: set[str]) -> list[str]:
    """Text of the page's figure images: OCR lines inside an image, confident,
    and made of real words (a dose line needs only the confidence)."""
    W, H = page.rect.width, page.rect.height
    rects = _image_rects(page)
    if not rects or not ocr_lines:
        return []
    layer = _norm(layer_text)
    kept: list[str] = []
    used: set[int] = set()   # the same image is often drawn twice
    for rect in rects:
        inside = []
        for k, ln in enumerate(ocr_lines):
            if k in used:
                continue
            x0, y0 = ln["x"] * W, ln["y"] * H
            cx, cy = x0 + ln["w"] * W / 2, y0 + ln["h"] * H / 2
            if rect.contains(pymupdf.Point(cx, cy)):
                used.add(k)
                inside.append({**ln, "x0": x0, "cx": cx, "cy": cy})
        for ln in _reading_order(inside, W):
            text = _fix_ocr(ln["t"].strip())
            n = _norm(text).strip()
            if len(text) < 2 or not n or n in layer or ln.get("c", 1.0) < 0.5:
                continue
            words = re.findall(r"[a-z]{3,}", n)
            if _DOSE_LINE.search(text):
                kept.append(text)
            elif words and sum(1 for w in words if w in known) / len(words) >= (0.5 if len(words) <= 3 else 0.75):
                kept.append(text)
    return kept


def pages(doc: pymupdf.Document, ocr: list[str], known: set[str],
          ocr_lines: list[list[dict]] | None = None) -> list[str]:
    """One text per page, structured as the module docstring describes."""
    out: list[str] = []
    heading = ""
    for i, page in enumerate(doc):
        layer = page.get_text()
        # A heading STARTS a line, and a page naming three or more is a table
        # of contents or a list of appendices - it sets none (the contents page
        # had carried "Chapter 1" into the front matter).
        starts = [m.group(1) for m in re.finditer(r"(?im)^\s*(" + HEADING.pattern[3:-1] + r")", layer)]
        if not starts and i < len(ocr):
            starts = [m.group(1) for m in re.finditer(r"(?im)^\s*(" + HEADING.pattern[3:-1] + r")", ocr[i])]
        if len(HEADING.findall(layer)) >= 3:
            starts = []
        if starts:
            heading = re.sub(r"\s+", " ", starts[0]).strip().rstrip(":").title().replace("Amo", "AMO")
        blocks, rects = _table_blocks(page)
        prose = _prose_outside(page, rects)
        parts = []
        if heading:
            parts.append(f"Section: {heading}")
        if prose:
            parts.append(prose)
        parts.extend(blocks)
        # A chapter's flow chart repeats its Activity table and, in this twice-
        # compressed copy, reads as noise ("Assess branding" for "Assess
        # breathing"): it is named, not transcribed. Appendix and procedure
        # figures - the asthma table, burn fluid rates - are transcribed.
        chapter_flowchart = bool(re.search(r"^\s*Flow ?chart\s*$", layer, re.I | re.M)) and \
            heading.lower().startswith("chapter")
        extra = [] if chapter_flowchart else _ocr_supplement(
            page, (ocr_lines or [])[i] if ocr_lines and i < len(ocr_lines) else [], layer, known)
        if chapter_flowchart and _image_rects(page):
            parts.append(f"[Flow chart for {heading}: an image not machine-readable in this copy - see the page]")
        if extra:
            parts.append("[Text of the page image (flow chart / figure / table), read by OCR - check the page]\n"
                         + "\n".join(extra))
        out.append("\n\n".join(p for p in parts if p.strip()))
    return out
