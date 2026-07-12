from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import requests
from django.conf import settings
from groq import Groq
from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ArticleResult:
    """نتیجه خام از Neo4j"""
    num:            int
    content:        str
    law:            str
    llm_confidence: float
    vector_score:   float = 0.0


@dataclass
class ClassificationResult:
    """نتیجه classification سوال"""
    topics:     list[str]
    confidence: float
    primary:    str | None  # مهم‌ترین topic


@dataclass
class ConfidenceResult:
    final:     float
    level:     str
    note:      str | None
    embedding: float
    llm:       float
    graph:     float


# ── Search Service ────────────────────────────────────────────────────────────

class SearchService:

    def __init__(self):
        self._driver: Driver | None = None
        self._groq:   Groq   | None = None

    # ── Lazy Connections ──────────────────────────────────────────────────────

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
            )
        return self._driver

    @property
    def groq(self) -> Groq:
        if self._groq is None:
            self._groq = Groq(api_key=settings.GROQ_API_KEY)
        return self._groq

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    # ── Classification ────────────────────────────────────────────────────────

    LEGAL_TOPICS = [
        "مالکیت و اموال",
        "قراردادها و معاملات",
        "اجاره و روابط موجر و مستاجر",
        "ازدواج و طلاق",
        "ارث و وصیت",
        "مسئولیت مدنی",
        "اهلیت و شخصیت حقوقی",
        "اثبات دعوا و ادله",
        "حقوق بین‌الملل خصوصی",
        "سایر",
    ]

    def _classify_query(self, query: str) -> ClassificationResult:
        """
        موضوع سوال رو با Groq تشخیص میده.
        اگه classification fail شد، با topics خالی برمیگرده
        تا fallback به جستجوی بدون فیلتر بشه.
        """
        prompt = f"""
سوال حقوقی زیر به کدام موضوع یا موضوعات مربوط می‌شود؟
فقط از لیست زیر انتخاب کن:

{chr(10).join(f"- {t}" for t in self.LEGAL_TOPICS)}

سوال: {query}

فقط JSON برگردون، هیچ توضیح اضافه‌ای نده:
{{"topics": ["موضوع اول"], "confidence": 0.9}}
"""
        try:
            response = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)

            topics = data.get("topics", [])
            confidence = float(data.get("confidence", 0.0))

            return ClassificationResult(
                topics=topics,
                confidence=confidence,
                primary=topics[0] if topics else None,
            )

        except Exception as e:
            logger.warning(f"Classification failed: {e} — falling back to no filter")
            return ClassificationResult(topics=[], confidence=0.0, primary=None)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """
        Development  → Ollama (local, بدون rate limit)
        Production   → Gemini (دقیق‌تر برای فارسی)
        """
        if settings.DEBUG:
            return self._embed_ollama(text)
        return self._embed_gemini(text)

    def _embed_ollama(self, text: str) -> list[float]:
        response = requests.post(
            f"{settings.OLLAMA_URL}/api/embeddings",
            json={"model": settings.OLLAMA_MODEL, "prompt": text[:2000]},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding") or data.get("embeddings", [[]])[0]

    def _embed_gemini(self, text: str) -> list[float]:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text[:2000],
        )
        return result.embeddings[0].values

    # ── Vector Search ─────────────────────────────────────────────────────────

    def _vector_search(
        self,
        embedding: list[float],
        law:       str,
        top_k:     int = 10,
        topic:     str | None = None,
    ) -> list[ArticleResult]:
        """
        جستجوی semantic در Neo4j.
        اگه topic داده شده، اول با فیلتر topic سرچ می‌کنه.
        """
        topic_filter = """
            AND EXISTS {
                MATCH (node)-[:HAS_TOPIC]->(t:Topic {name: $topic})
            }
        """ if topic else ""

        with self.driver.session() as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes(
                    'article_embedding', $top_k, $embedding
                ) YIELD node, score
                WHERE node.status <> 'abolished'
                  AND ($law = '' OR node.law = $law)
                  {topic_filter}
                RETURN
                    node.article_number            AS num,
                    node.content                   AS content,
                    node.law                       AS law,
                    coalesce(node.confidence, 0.7) AS llm_confidence,
                    score                          AS vector_score
                ORDER BY score DESC
                """,
                embedding=embedding,
                top_k=top_k,
                law=law,
                topic=topic or "",
            )
            return [
                ArticleResult(
                    num=r["num"],
                    content=r["content"],
                    law=r["law"],
                    llm_confidence=r["llm_confidence"],
                    vector_score=r["vector_score"],
                )
                for r in result
            ]

    def _vector_search_with_fallback(
        self,
        embedding: list[float],
        law:       str,
        topic:     str | None,
        top_k:     int = 10,
    ) -> list[ArticleResult]:
        """
        استراتژی Fallback:
        ۱. اگه topic داشتیم، اول با فیلتر topic سرچ کن
        ۲. اگه نتیجه کمتر از ۳ تا بود یا topic نداشتیم،
           بدون فیلتر دوباره سرچ کن
        """
        if topic:
            results = self._vector_search(embedding, law, top_k, topic=topic)
            if len(results) >= 3:
                logger.debug(f"Topic filter hit: {topic} → {len(results)} results")
                return results
            logger.debug(f"Topic filter miss: {topic} → fallback to no filter")

        return self._vector_search(embedding, law, top_k, topic=None)

    # ── Graph Traversal ───────────────────────────────────────────────────────

    def _graph_traversal(
        self,
        article_nums: list[int],
        law:          str,
    ) -> list[ArticleResult]:
        """
        مواد مرتبط رو از طریق REFERENCES و HAS_TOPIC پیدا می‌کنه.
        محدود به ۱۰ نتیجه برای جلوگیری از context overflow.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $nums AS num
                MATCH (a:Article {article_number: num, law: $law})

                OPTIONAL MATCH (a)-[:REFERENCES]->(ref:Article)

                OPTIONAL MATCH (a)-[:HAS_TOPIC]->(t:Topic)
                    <-[:HAS_TOPIC]-(related:Article)
                WHERE related.article_number <> num
                  AND related.status <> 'abolished'

                WITH
                    collect(DISTINCT ref)[..3]    AS refs,
                    collect(DISTINCT related)[..3] AS relateds

                WITH refs + relateds AS extras
                UNWIND extras AS extra
                WITH extra WHERE extra IS NOT NULL

                RETURN DISTINCT
                    extra.article_number            AS num,
                    extra.content                   AS content,
                    extra.law                       AS law,
                    coalesce(extra.confidence, 0.7) AS llm_confidence
                LIMIT 10
                """,
                nums=article_nums[:3],
                law=law,
            )
            return [
                ArticleResult(
                    num=r["num"],
                    content=r["content"],
                    law=r["law"],
                    llm_confidence=r["llm_confidence"],
                )
                for r in result
            ]

    # ── Confidence ────────────────────────────────────────────────────────────

    def _calculate_confidence(
        self,
        top_vector_score:  float,
        llm_confidence:    float,
        has_graph_results: bool,
    ) -> ConfidenceResult:
        graph_score = 1.0 if has_graph_results else 0.5

        final = round(
            0.5 * top_vector_score +
            0.3 * llm_confidence   +
            0.2 * graph_score,
            3,
        )

        if final >= 0.8:
            level, note = "high", None
        elif final >= 0.6:
            level, note = "medium", "پاسخ نیاز به بررسی بیشتر دارد"
        else:
            level, note = "low", "اطمینان کافی وجود ندارد. با وکیل مشورت کنید"

        return ConfidenceResult(
            final=final,
            level=level,
            note=note,
            embedding=round(top_vector_score, 3),
            llm=round(llm_confidence, 3),
            graph=round(graph_score, 3),
        )

    # ── Answer Generation ─────────────────────────────────────────────────────

    def _generate_answer(
        self,
        query:    str,
        articles: list[ArticleResult],
    ) -> str:
        context = "\n\n".join([
            f"ماده {a.num} — {a.law}:\n{a.content}"
            for a in articles[:7]
        ])

        prompt = f"""تو یک دستیار حقوقی متخصص در قوانین ایران هستی.

سوال کاربر: {query}

مواد قانونی مرتبط:
{context}

قوانین پاسخ‌دهی:
- فقط از مواد ارائه‌شده استفاده کن
- شماره ماده و نام قانون رو در جواب ذکر کن
- اگه جواب در مواد نیست، صادقانه بگو
- جواب رو به فارسی روان بنویس

جواب:"""

        response = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    # ── Main Search ───────────────────────────────────────────────────────────

    def search(self, query: str, law: str = "قانون مدنی") -> dict | None:
        try:
            # ۱. classification — موضوع سوال رو پیدا کن
            classification = self._classify_query(query)
            topic = (
                classification.primary
                if classification.confidence >= 0.7
                else None
            )
            logger.debug(f"Classification: {classification}")

            # ۲. embed سوال
            embedding = self._embed(query)

            # ۳. vector search با fallback strategy
            vector_results = self._vector_search_with_fallback(
                embedding=embedding,
                law=law,
                topic=topic,
                top_k=10,
            )

            if not vector_results:
                logger.warning(f"No results for query: {query}")
                return None

            # ۴. graph traversal
            nums          = [r.num for r in vector_results]
            graph_results = self._graph_traversal(nums, law)

            # ۵. ترکیب و dedup
            seen:         set[int]          = set()
            all_articles: list[ArticleResult] = []
            for article in vector_results + graph_results:
                if article.num not in seen:
                    seen.add(article.num)
                    all_articles.append(article)

            # ۶. confidence
            top        = vector_results[0]
            confidence = self._calculate_confidence(
                top_vector_score=top.vector_score,
                llm_confidence=top.llm_confidence,
                has_graph_results=bool(graph_results),
            )

            # ۷. جواب نهایی
            answer = self._generate_answer(query, all_articles)

            return {
                "answer":     answer,
                "confidence": confidence,
                "sources":    all_articles,
            }

        except Exception as e:
            logger.error(f"SearchService.search error: {e}", exc_info=True)
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────
search_service = SearchService()