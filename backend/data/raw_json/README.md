# `raw_json/` — the FUKKM formulary snapshot (not in git)

Holds `fukkm_database.json`, ~1,686 drug records scraped from the MOH Medicines
Formulary. Not committed: it is regenerable, and it is a snapshot of someone
else's database rather than source code.

Regenerate with:

    cd backend
    python -m app.scrape_fukkm --max-pages 2   # sample first, inspect output
    python -m app.scrape_fukkm                 # full re-scrape
    python -m app.ingest --reset

`scrape_fukkm.py` maps columns by **header text**, not by index, and refuses to
overwrite a good file when validation fails — an earlier index-based version was
off by one and left no generic drug name in the file at all. `ingest.py` still
detects that old `column_shifted` shape and re-maps what it can, so a stale file
degrades rather than corrupts.
