# ADR-001 — Why Neo4j as the Primary Knowledge Store

**Date:** 2026-07  
**Status:** Accepted

---

## Context

Iranian legal documents have a deeply hierarchical and interconnected structure:

- Laws contain Volumes → Books → Chapters → Sections → Articles → Notes
- Articles frequently reference other articles
- A single legal question may require traversing multiple levels and relationships

We needed a database that could:
1. Store this hierarchy natively
2. Traverse relationships efficiently at query time
3. Support vector similarity search on the same nodes

---

## Decision

Use **Neo4j AuraDB** as the primary knowledge store for all legal content.

---

## Alternatives Considered

| Option | Reason Rejected |
|--------|----------------|
| PostgreSQL only | Recursive CTEs for hierarchy are slow and hard to maintain |
| Elasticsearch | No native graph traversal; relationships must be denormalized |
| Pinecone + PostgreSQL | Two databases to sync; no relationship traversal |
| Weaviate | Graph traversal limited; less mature Cypher-equivalent |

---

## Consequences

**Positive:**
- Single database for both graph traversal and vector search
- Cypher queries express legal hierarchy naturally
- `REFERENCES` and `HAS_TOPIC` edges enable multi-hop reasoning
- Native vector index (`article_embedding`) avoids a separate vector DB

**Negative:**
- Learning curve for Cypher
- Neo4j AuraDB free tier has node/relationship limits
- `db.index.vector.queryNodes` is deprecated → migration to `SEARCH` needed in v2

---

## Graph Schema Summary

```
Law → Section → Article → Note
Article -[:REFERENCES]→ Article
Article -[:HAS_TOPIC]→ Topic
```
