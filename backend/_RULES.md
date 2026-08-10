# backend/_RULES.md

## Stack

- Python 3.12 + Django 5 + Django-Ninja
- Package manager: `uv` (never use pip directly)
- Database: SQLite for dev, PostgreSQL for prod
- Graph DB: Neo4j AuraDB
- Task queue: Celery + Redis

---

## Project Layout

```
backend/
├── config/
│   ├── settings/
│   │   ├── base.py          # shared settings — no secrets here
│   │   ├── development.py   # dev overrides
│   │   └── production.py    # prod overrides (uses os.environ[], not getenv)
│   ├── urls.py              # NinjaAPI routers only — no endpoint logic
│   └── celery.py
├── apps/
│   ├── users/               # auth, JWT, onboarding, plan management
│   ├── search/              # hybrid retrieval (vector + graph)
│   ├── agents/              # LangGraph multi-agent case analysis
│   └── documents/           # file upload, OCR, Celery tasks
├── core/
│   ├── auth.py              # JWT helpers + JWTAuth bearer
│   └── permissions.py       # plan-based access control
└── manage.py
```

---

## App Structure Rules

Every app must follow this exact structure:

```
apps/<name>/
├── __init__.py      # empty
├── models.py        # Django ORM models only
├── schemas.py       # Pydantic schemas (input/output) only
├── services.py      # business logic — no HTTP concerns
└── api.py           # endpoints only — delegate to services
```

Agents app additionally has:

```
apps/agents/
├── state.py         # LangGraph CaseState TypedDict
├── graph.py         # StateGraph definition + build_graph()
└── nodes/           # one file per agent node
```

---

## API Rules

- All routers use `django-ninja` — never `djangorestframework`
- Authentication: JWT Bearer via `core.auth.jwt_auth`
- Plan check: call `check_plan_limit(user)` before heavy operations
- Error messages must be in **Persian**
- HTTP status codes must follow REST conventions

---

## Model Rules

- Always define `Meta.verbose_name` and `Meta.verbose_name_plural` in Persian
- Use `TextChoices` for all enum-like fields
- Atomic operations for counter updates (use `select_for_update()`)
- Never use `save()` without `update_fields` for partial updates

---

## Settings Rules

- `base.py`: no secrets, no environment-specific config
- `development.py`: use `os.getenv()` with safe defaults
- `production.py`: use `os.environ[]` — crash if missing
- `config/settings/__init__.py` imports from `development` by default

---

## Neo4j Rules

- Use lazy connection via `@property` in service classes
- Always call `.close()` in `finally` blocks
- Vector search index name: `article_embedding`
- Avoid `db.index.vector.queryNodes` (deprecated) — migrate to `SEARCH` in v2

---

## Celery Rules

- Task names must follow: `<app>.<action>` (e.g., `documents.process_ocr`)
- Always use `bind=True, max_retries=3`
- Update model status at start (`PROCESSING`) and end (`COMPLETED`/`FAILED`)
- Never do heavy work inside Django request cycle — always delegate to Celery

---

## Confidence Scoring

```
Final = 0.5 * embedding_score + 0.3 * llm_confidence + 0.2 * graph_score

high   >= 0.8  → no warning
medium >= 0.6  → show review warning
low    <  0.6  → recommend consulting a lawyer
```

---

## Forbidden

- No `django.contrib.admin` URLs (not used)
- No raw SQL — use ORM or Neo4j Cypher
- No `print()` in production code — use `logging`
- No `os.getenv()` in `production.py`
- No direct calls between apps — go through services