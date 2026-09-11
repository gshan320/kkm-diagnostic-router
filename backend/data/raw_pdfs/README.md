# `raw_pdfs/` — the source corpus (not in git)

This folder is intentionally empty in the repository. It holds the 53 KKM /
MOH PDFs the index is built from; they are inputs rather than source code, they
are large (254 MB), and their licence terms are not uniform — see the
`.gitignore` entry for the reasoning.

## How to populate it

Guidelines come from the Academy of Medicine Malaysia library:
<https://www.acadmed.org.my/index.cfm?&menuid=67>

Downloads are Cloudflare Turnstile CAPTCHA-gated and **cannot be scripted** —
fetch them by hand. Filenames are load-bearing: `ingest.py` derives
`cpg_title`, `edition_year`, `edition` and `doc_type` from them, so follow the
convention in README §Ingestion exactly. The year is mandatory and must be
explicit, and the `CPG ` / `QR ` prefixes decide document type.

    CPG <Portal Title> (<N>th Edition) <YEAR>[ v<YYYYMMDD>].pdf
    QR  <same title>   (<N>th Edition) <YEAR>[ v<YYYYMMDD>].pdf

Then rebuild:

    cd backend
    python -m app.scrape_fukkm      # regenerates data/raw_json/fukkm_database.json
    python -m app.ingest --dry-run  # check parsing and chunk counts first
    python -m app.ingest --reset

`python -m app.ingest --dry-run` reports per-document chunk counts and warns on
any file whose edition year it could not resolve. Read the ingest log after a
rebuild; a clean `/api/v1/stats` is not the same as a clean corpus.
