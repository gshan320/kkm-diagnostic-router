"""Which documents may enter the index: KKM, Malaysian, or KKM-cited.

Why this exists
---------------
Run 11 of the rhabdomyolysis case (2026-09-30) scored 96.2, and 10 of its 21
sources were CHAMP 2025 - a US Department of Defense guideline that no KKM
document cites. The user's rule, stated that day: Malaysian patients get
Malaysian handling. A non-KKM document is allowed ONLY if KKM or Malaysian
hospitals recognise, approve or use it; where KKM has no source, the report
states the gap and links out (references.py) instead of importing one.

The objective test is a citation: an indexed KKM document names the non-KKM
one, and the record here carries that document's title, page and words. The
test suite checks each quote against the index, so a qualification cannot be
typed from memory. The one other basis is an explicit decision by the user,
recorded with its date.

Checked at ingest (a PDF that fails is skipped with a warning) and by
`python -m app.source_policy`, which lists every indexed document with the
basis on which it is allowed.
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache

from . import config

QUALIFICATIONS = config.DATA_DIR / "source_qualifications.json"

KKM = "kkm"                       # published by the Ministry of Health or its agencies
MALAYSIAN_BODY = "malaysian_body"  # Malaysian professional society / consensus, used in MOH hospitals
KKM_CITATION = "kkm_citation"      # cited by an indexed KKM document (title + page + quote)
USER_APPROVED = "user_approved"    # kept by an explicit, dated decision of the user

# Filename conventions of the corpus (raw_pdfs/README.md). "JKN" is a state
# health department of the Ministry; NAG is published on an MOH site.
_KKM_NAME = re.compile(r"^(?:CPG|QR|MOH|MTS|MaHTAS|JKN)\b|^Paediatric Protocols for Malaysian Hospitals"
                       r"|^fukkm_database\.json$|^nag-[a-z0-9-]+\.html$", re.I)
_MALAYSIAN_NAME = re.compile(r"^(?:MSN|MSIC|MSAI|Malaysian)\b", re.I)


@lru_cache(maxsize=1)
def qualifications() -> tuple[dict, ...]:
    if not QUALIFICATIONS.exists():
        return ()
    return tuple(json.loads(QUALIFICATIONS.read_text(encoding="utf-8")))


def basis(filename: str) -> tuple[str, dict | None]:
    """(basis, qualification record or None); basis "" means NOT allowed."""
    name = filename.strip()
    if _KKM_NAME.search(name):
        return KKM, None
    if _MALAYSIAN_NAME.search(name):
        return MALAYSIAN_BODY, None
    for q in qualifications():
        if name.lower().startswith(q["filename_prefix"].lower()):
            return q["basis"], q
    return "", None


def allowed(filename: str) -> bool:
    return bool(basis(filename)[0])


def main() -> None:
    from .cards_build import load_index  # noqa: PLC0415
    index = load_index()
    files = sorted({m.get("filename", "") for m in index["metas"]})
    bad = 0
    for f in files:
        b, q = basis(f)
        bad += not b
        cited = f" - cited by {q['cited_by']} p{q['page']}" if q and q.get("cited_by") else ""
        print(f"{b or 'NOT ALLOWED':<15} {f}{cited}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
