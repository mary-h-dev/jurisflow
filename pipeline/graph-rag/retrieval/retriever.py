"""
retrieval/retriever.py

معماری: Hybrid Parallel Retrieval
    سه کانال کاملاً مستقل (نه زنجیره‌ای) روی یک query اجرا می‌شن:
    ۱. Feature Search  — جست‌وجو در LegalConcept/Role/Action/Object/Fact
    ۲. Ruling Search   — جست‌وجو در RulingSection (متن کامل پرونده‌ها)
    ۳. Article Search  — جست‌وجو در Article (مواد قانونی)

    بعد نتایج هر کانال با Reciprocal Rank Fusion (RRF) ترکیب می‌شن.

چرا موازی نه زنجیره‌ای؟
    اگه زنجیره‌ای باشه، خطای کانال اول روی همه منتقل می‌شه. با موازی،
    هر کانال مستقل کار می‌کنه و RRF نتایج رو باهم ترکیب می‌کنه.

چرا RRF نه جمع ساده‌ی امتیازها؟
    امتیاز cosine similarity بین Feature node (چند کلمه) و RulingSection
    (چند صفحه) قابل مقایسه‌ی مستقیم نیست — طول متفاوت باعث تفاوت
    inherent در distribution می‌شه. RRF فقط از rank استفاده می‌کنه،
    نه مقدار خام امتیاز.

⚠️ تاریخچه‌ی رفع باگ (هر دو مهم برای uncertainty estimation بودن،
   چون بی‌صدا نتیجه‌ی خالی برمی‌گردوندن، نه خطای قابل‌مشاهده):

   ۱. Syntax نامعتبر Cypher: نسخه‌ی قبلی از یک syntax ساختگی استفاده
      می‌کرد («VECTOR SEARCH INDEX ... FOR ... TOP ... YIELD») که در
      Neo4j اصلاً وجود ندارد. syntax صحیح، فراخوانی procedure داخلی
      Neo4j است:
          CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
          YIELD node, score
      چون هر سه تابع _search_* یک except Exception: continue/return []
      داشتند، این خطای syntax همیشه رخ می‌داد ولی بی‌صدا بلعیده می‌شد —
      یعنی retrieval همیشه نتیجه‌ی خالی می‌داد، بدون هیچ خطای قابل‌دیدن.

   ۲. Property نادرست روی Feature node ها: کوئری دنبال feature.value
      می‌گشت، در حالی که database/feature_loader.py نودها را با
      property به اسم name می‌سازد (MERGE (n:{label} {name: $value})).
      همچنین evidence quote باید از خودِ رابطه (rel.evidence) خوانده
      شود، نه از نود — چون evidence مخصوص هر ذکرِ خاص در متن است، نه
      خاصیتِ ثابت خودِ مفهوم.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from common.embedder import embed_text
from database.connection import Neo4jConnection


# ──────────────────────────────────────────────
# ساختار داده‌ی خروجی
# ──────────────────────────────────────────────

@dataclass
class RetrievedEvidence:
    """یک قطعه‌ی شواهد از یکی از سه کانال retrieval"""
    source_type: str          # "feature" | "ruling" | "article"
    ruling_id: Optional[str]  # None برای article-only results
    text: str                 # متن اصلی (evidence quote یا section text یا article text)
    score: float              # امتیاز cosine similarity (0 تا 1)
    rrf_score: float = 0.0   # امتیاز نهایی بعد از RRF

    # فیلدهای اختیاری بسته به source_type
    feature_value: Optional[str] = None   # مثلاً «خواهان» برای feature
    feature_category: Optional[str] = None  # concept/role/action/object/fact
    article_number: Optional[int] = None
    law_name: Optional[str] = None
    confidence: Optional[float] = None    # از Feature extraction


@dataclass
class RetrievalResult:
    """نتیجه‌ی کامل retrieval برای یک query"""
    query: str
    evidences: list[RetrievedEvidence] = field(default_factory=list)

    # نتایج هر کانال جدا (برای uncertainty estimation و ارزیابی)
    feature_results: list[RetrievedEvidence] = field(default_factory=list)
    ruling_results: list[RetrievedEvidence] = field(default_factory=list)
    article_results: list[RetrievedEvidence] = field(default_factory=list)

    @property
    def ruling_ids(self) -> list[str]:
        """لیست ruling_id های یکتای پیداشده"""
        return list({e.ruling_id for e in self.evidences if e.ruling_id})

    # نکته: محاسبه‌ی امتیاز اطمینان (retrieval confidence) عمداً اینجا
    # نیست — توی retrieval/confidence.py جداست، چون محاسبه‌ی امتیاز و
    # اجرای جست‌وجو دو مسئولیت متفاوتن. برای گرفتن امتیاز:
    #     from retrieval.confidence import score_result
    #     score, vec = score_result(result)


# ──────────────────────────────────────────────
# RRF (Reciprocal Rank Fusion)
# ──────────────────────────────────────────────

_RRF_K = 60  # مقدار استاندارد در ادبیات


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    return 1.0 / (k + rank)


def _fuse_with_rrf(
    *ranked_lists: list[RetrievedEvidence],
    top_k: int = 10,
) -> list[RetrievedEvidence]:
    """
    Reciprocal Rank Fusion روی چند لیست رتبه‌بندی‌شده.
    برای هر آیتم، مجموع 1/(k+rank) از همه‌ی لیست‌هایی که توشه رو
    حساب می‌کنه.
    """
    scores: dict[str, float] = {}
    items: dict[str, RetrievedEvidence] = {}

    for ranked_list in ranked_lists:
        for rank, evidence in enumerate(ranked_list, start=1):
            # کلید یکتا برای dedup
            key = f"{evidence.source_type}:{evidence.ruling_id}:{evidence.text[:50]}"
            scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
            if key not in items:
                items[key] = evidence

    sorted_keys = sorted(scores.keys(), key=lambda k: -scores[k])[:top_k]
    results = []
    for key in sorted_keys:
        ev = items[key]
        ev.rrf_score = scores[key]
        results.append(ev)

    return results


# ──────────────────────────────────────────────
# کانال ۱: Feature Search
# ──────────────────────────────────────────────

_FEATURE_LABELS = [
    "LegalConcept", "LegalRole", "LegalAction", "LegalObject", "LegalFact"
]

_FEATURE_QUERY = """
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node AS feature, score
MATCH (r:Ruling)-[rel]->(feature)
WHERE type(rel) IN ['HAS_CONCEPT','HAS_ROLE','HAS_ACTION','HAS_OBJECT','HAS_FACT']
RETURN
    feature.name       AS value,
    labels(feature)[0] AS category,
    rel.evidence        AS quote,
    rel.confidence      AS confidence,
    rel.start_char      AS start_char,
    rel.end_char        AS end_char,
    r.ruling_id         AS ruling_id,
    score
ORDER BY score DESC
LIMIT $top_k
"""


def _search_features(
    session, query_embedding: list[float], top_k: int
) -> list[RetrievedEvidence]:
    results = []
    seen = set()

    for label in _FEATURE_LABELS:
        index_name = f"{label.lower()}_embedding"
        try:
            records = session.run(
                _FEATURE_QUERY,
                index_name=index_name,
                embedding=query_embedding,
                top_k=top_k,
            )
            for r in records:
                key = f"{r['ruling_id']}:{r['value']}"
                if key in seen:
                    continue
                seen.add(key)
                results.append(RetrievedEvidence(
                    source_type="feature",
                    ruling_id=str(r["ruling_id"]) if r["ruling_id"] else None,
                    text=r["quote"] or r["value"] or "",
                    score=float(r["score"]),
                    feature_value=r["value"],
                    feature_category=r["category"],
                    confidence=float(r["confidence"]) if r["confidence"] else None,
                ))
        except Exception as e:
            # اگه index هنوز ساخته نشده (قبل از embed_features.py)، skip می‌کنیم
            # ولی حداقل لاگ می‌کنیم — silent failure دیگه تکرار نشه
            print(f"⚠️ Feature search برای {label} شکست خورد: {type(e).__name__}: {e}")
            continue

    return sorted(results, key=lambda e: -e.score)[:top_k]


# ──────────────────────────────────────────────
# کانال ۲: Ruling Search (روی RulingSection)
# ──────────────────────────────────────────────

_RULING_QUERY = """
CALL db.index.vector.queryNodes('ruling_section_embedding', $top_k, $embedding)
YIELD node AS section, score
MATCH (r:Ruling)-[:HAS_SECTION]->(section)
RETURN
    r.ruling_id     AS ruling_id,
    section.text    AS text,
    r.domain        AS domain,
    r.case_type     AS case_type,
    score
ORDER BY score DESC
LIMIT $top_k
"""


def _search_rulings(
    session, query_embedding: list[float], top_k: int
) -> list[RetrievedEvidence]:
    try:
        records = session.run(
            _RULING_QUERY,
            embedding=query_embedding,
            top_k=top_k,
        )
        return [
            RetrievedEvidence(
                source_type="ruling",
                ruling_id=str(r["ruling_id"]) if r["ruling_id"] else None,
                text=r["text"] or "",
                score=float(r["score"]),
            )
            for r in records
        ]
    except Exception as e:
        print(f"⚠️ Ruling search شکست خورد: {type(e).__name__}: {e}")
        return []


# ──────────────────────────────────────────────
# کانال ۳: Article Search
# ──────────────────────────────────────────────

_ARTICLE_QUERY = """
CALL db.index.vector.queryNodes('article_embedding', $top_k, $embedding)
YIELD node AS article, score
MATCH (law:Law)-[:CONTAINS]->(article)
RETURN
    article.article_number AS article_number,
    article.content         AS text,
    law.name                AS law_name,
    score
ORDER BY score DESC
LIMIT $top_k
"""


def _search_articles(
    session, query_embedding: list[float], top_k: int
) -> list[RetrievedEvidence]:
    try:
        records = session.run(
            _ARTICLE_QUERY,
            embedding=query_embedding,
            top_k=top_k,
        )
        return [
            RetrievedEvidence(
                source_type="article",
                ruling_id=None,
                text=r["text"] or "",
                score=float(r["score"]),
                article_number=r["article_number"],
                law_name=r["law_name"],
            )
            for r in records
        ]
    except Exception as e:
        print(f"⚠️ Article search شکست خورد: {type(e).__name__}: {e}")
        return []


# ──────────────────────────────────────────────
# Retriever اصلی
# ──────────────────────────────────────────────

class LegalRetriever:
    """
    Hybrid Parallel Retriever برای سیستم GraphRAG حقوقی.

    نحوه‌ی استفاده:
        connection = Neo4jConnection(URI, USER, PASS)
        retriever = LegalRetriever(connection)
        result = retriever.retrieve("آیا مستأجر می‌تواند بدون اجازه موجر اجاره‌دهد؟")
        print(result.retrieval_score)  # ورودی uncertainty estimator
        for ev in result.evidences:
            print(ev.source_type, ev.text[:100])
    """

    def __init__(
        self,
        connection: Neo4jConnection,
        feature_top_k: int = 10,
        ruling_top_k: int = 5,
        article_top_k: int = 5,
        final_top_k: int = 10,
    ):
        self._connection = connection
        self._feature_top_k = feature_top_k
        self._ruling_top_k = ruling_top_k
        self._article_top_k = article_top_k
        self._final_top_k = final_top_k

    def retrieve(self, query: str) -> RetrievalResult:
        """
        ورودی: سؤال کاربر (متن آزاد فارسی)
        خروجی: RetrievalResult با شواهد از هر سه کانال + امتیاز کلی
        """
        query_embedding = embed_text(query)

        with self._connection.session() as session:
            feature_results = _search_features(
                session, query_embedding, self._feature_top_k
            )
            ruling_results = _search_rulings(
                session, query_embedding, self._ruling_top_k
            )
            article_results = _search_articles(
                session, query_embedding, self._article_top_k
            )

        fused = _fuse_with_rrf(
            feature_results,
            ruling_results,
            article_results,
            top_k=self._final_top_k,
        )

        return RetrievalResult(
            query=query,
            evidences=fused,
            feature_results=feature_results,
            ruling_results=ruling_results,
            article_results=article_results,
        )

    def retrieve_batch(self, queries: list[str]) -> list[RetrievalResult]:
        return [self.retrieve(q) for q in queries]