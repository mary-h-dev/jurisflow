# ADR-003 — Auditor Design: Checklist-Based Verification

**Date:** 2026-08  
**Status:** Accepted  
**Component:** `backend/apps/auditor`

## 2. Auditor Layer

### 2.1 Purpose
The Auditor verifies each retrieved statutory article against a fixed, article-specific checklist of necessary legal conditions.  
It produces two independent labels:
- `topically_relevant`: whether the article’s subject matter matches the query
- `is_applicable`: whether the necessary conditions are satisfied given the case facts

A failed necessary condition is treated as informative evidence rather than noise.

### 2.2 Checklist Design
- Checklists are generated **once per article** (offline) and stored in `checklist_db.json`.
- Maximum 3 necessary conditions per article.
- Conditions are written so that `satisfied=True` always favors applicability (polarity rule).
- Exceptions in the article text are inverted into positive conditions.

### 2.3 Evidence Bundle
For each candidate article the Auditor receives:
- The full article text
- Supporting rulings that cite the article
- Known relevant factors from `GROUNDED_IN` edges (ranked by NPMI, min co-occurrence = 3)

### 2.4 Decision Logic (`decide`)
- If any condition is marked `necessary`:
  - Article is applicable **only if all necessary conditions are satisfied**.
- Otherwise:
  - Soft majority rule with threshold 0.6.
- `auditor_confidence` = fraction of satisfied conditions used in the decision.

### 2.5 MIN_ARTICLES Fallback
If fewer than 3 articles remain with `is_applicable=True`, the highest-confidence `topically_relevant` articles (even if `is_applicable=False`) are added back so that agents always receive a minimum of 3 candidates.

### 2.6 Query-level Confidence
- Mean of `auditor_confidence` over articles with `is_applicable=True`.
- Prune ratio = rejected / (kept + rejected) is reported as a noise signal for later fusion.


## Related
- `backend/apps/auditor/services.py`