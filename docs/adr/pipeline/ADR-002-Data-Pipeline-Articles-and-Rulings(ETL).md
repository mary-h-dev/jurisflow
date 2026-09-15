# ADR-002 — Data Pipeline: Articles and Rulings (ETL)

**Date:** 2026-07
**Status:** Accepted
**Component:** `pipeline/graph-rag/laws & cases`
---

## Context

The knowledge graph needs two structurally different types of legal data:
statutory articles (5 core codes) and court rulings that cite them. The
two live on different sites with different access constraints:
`qavanin.ir` (statutes) sits behind a JS anti-bot challenge; `ara.jri.ac.ir`
(rulings) is scriptable but must be rate-limited and is large (thousands
of rulings per statute).

## Decision

Use **two independent Scrape → Parse → Load pipelines** — `laws/` +
`main_laws.py` for statutes (Rule Graph), `cases/` + `main_cases.py` for
rulings (Fact Graph) — sharing only `common/fetcher.py` and
`database/connection.py`. The two graphs converge only at the Neo4j
level, via `Ruling -[:CITES]-> Article`.

### Scrape

- **Laws**: `qavanin.ir`'s anti-bot wall blocks automated fetching, so
  pages are saved manually from the browser (Ctrl+S → "Webpage,
  Complete") and parsed locally. Requires an Iran IP to load the pages
  in the first place.
- **Cases**: fetched automatically via `requests`, one ruling per JSON
  file. Resumable — already-downloaded files are skipped on re-run.

### Parse

- `laws/parser.py`: HTML → flat list of `Article` (with optional
  `Note`s), via regex anchored on "ماده"/"تبصره". Per-statute
  configuration (tag, class, patterns) lives in `laws/configs.py`, so a
  statute with a different HTML structure only needs a new config entry,
  not a parser change.
- `cases/parser.py`: HTML → structured `Ruling` (title, summary, verdict
  metadata, sections split by litigation tier, cited articles from both
  the official citation box and free-text extraction).

### Load

- `database/law_loader.py` / `database/case_loader.py` write into Neo4j.
  Both `main_laws.py load` and `main_cases.py load` are resumable and
  retry transient Neo4j/network errors with exponential backoff. Load
  must run from a non-Iran IP (Aura access constraint) — the opposite
  constraint from scrape.

## Alternatives Considered

| Option | Reason Rejected |
|---|---|
| One unified parser/loader for both laws and cases | Statutes are a flat, uniform structure (Law → Article → Note); rulings are multi-tier, free-text-heavy, and require separate handling for official vs. text-extracted citations. Forcing one parser to cover both would increase complexity without a real benefit, since the two are consumed independently downstream. |
| Fully automated scraping for statutes | `qavanin.ir`'s anti-bot challenge blocks automated requests; a headless-browser workaround (e.g. Playwright) was not pursued given the fixed, small set of 5 statutes. |

## Consequences

**Positive:**
- Clear separation of concerns — each pipeline can be debugged, extended,
  or re-run independently
- Resumable at every load stage, tolerant of the network instability
  implied by the Iran/non-Iran IP split
- Per-statute config isolates HTML-structure differences from parsing
  logic

**Negative:**
- The manual scrape step for statutes is a workaround, not something
  that runs unattended or scales past a small, fixed set of statutes
- No explicit logging of parse failures — text that fails to match any
  pattern before the first recognized article is silently dropped
  (typically just page headers, but unverified as such by the pipeline)
