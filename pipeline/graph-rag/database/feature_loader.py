"""
features/database/feature_loader.py — بارگذاری گراف سوم (Feature Graph) در Neo4j

از همون Neo4jConnection مشترک استفاده می‌کنه (database/connection.py)،
دقیقاً مثل database/case_loader.py و database/law_loader.py.

ساختار گراف:
    (:Ruling)-[:HAS_CONCEPT   {evidence, start_char, end_char, confidence}]->(:LegalConcept   {name})
    (:Ruling)-[:HAS_ACTION    {...}]->(:LegalAction    {name})
    (:Ruling)-[:HAS_ROLE      {...}]->(:LegalRole      {name})
    (:Ruling)-[:HAS_OBJECT    {...}]->(:LegalObject    {name})
    (:Ruling)-[:HAS_FACT      {...}]->(:LegalFact      {text})

چرا closed-vocab node ها MERGE می‌شوند ولی LegalFact نه؟
    چون Concept/Action/Role/Object از یک لیست بسته و
    یکتاشده (بعد از canonicalization در vocabulary_categorizer.py)
    می‌آیند — یک node واحد برای «بیع» در کل گراف کافی است، و MERGE
    باعث می‌شود همه‌ی رأی‌های مرتبط به یک node وصل شوند (این دقیقاً
    همون چیزیه که query زدن رو قدرتمند می‌کنه).
    Factها اما آزاد و به‌شدت متنوعند؛ اگر آن‌ها را هم MERGE کنیم،
    Factهای مشابه ولی نه‌کاملاً یکسان (که در متن حقوقی خیلی رایج است)
    به‌اشتباه در یک node ادغام می‌شوند و اطلاعات خاص هر پرونده گم
    می‌شود. برای همین هر Fact یک node مستقل با CREATE می‌گیرد، حتی اگر
    شبیه یک Fact در پرونده‌ی دیگر باشد.

چرا خودِ رابطه‌ها (نه فقط node ها) با CREATE ساخته می‌شوند، نه MERGE؟
    قبلاً `MERGE (r)-[rel:...]->(n)` بود که یک باگ واقعی داشت: اگر یک
    مفهوم (مثلاً «فسخ») دو بار در بخش‌های مختلف یک رأی ذکر شده باشد،
    MERGE دومین نوشتن evidence را جای اولی می‌گذاشت و شاهدِ اول برای
    همیشه گم می‌شد. با CREATE، هر بار که یک Feature در یک رأی دیده
    می‌شود یک رابطه‌ی جدا و evidence خودش را می‌گیرد؛ Neo4j اجازه‌ی
    چند رابطه‌ی هم‌نوع بین دو node را می‌دهد، پس این مشکلی ایجاد
    نمی‌کند و query زدن هم تغییری نمی‌کند (فقط شاهدها کامل‌تر می‌مانند).

    این تغییر یک اثر جانبی دارد: اگر load_result() روی یک ruling دو
    بار صدا زده شود (مثلاً اجرای مجدد main_features.py load)، رابطه‌ها
    تکراری ساخته می‌شوند. برای همین is_ruling_loaded/mark_ruling_loaded
    اضافه شده — قبل از بارگذاری هر ruling چک می‌شود که قبلاً بارگذاری
    نشده باشد (idempotency در سطح ruling، نه در سطح رابطه‌ی تکی).

چرا retry دور نوشتن‌های Neo4j؟
    چون main_features.py قرار است روی صدها/هزاران رأی پشت سر هم
    اجرا شود؛ یک قطعی لحظه‌ای شبکه یا Neo4j Aura نباید کل اجرا را
    crash کند و کاری که تا الان انجام شده را از دست بدهد.
"""


import time

from database.connection import Neo4jConnection
from features.configs import FACT_CATEGORY, VOCAB_CATEGORIES
from features.schemas import ExtractedFeature, FeatureExtractionResult

# نگاشت کلید دسته → نام فیلد لیست در FeatureExtractionResult
_FIELD_NAMES = {
    "concept": "concepts",
    "action": "actions",
    "role": "roles",
    "object": "objects",
}

_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 5


def _with_retry(fn, *args, **kwargs):
    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — قطعی شبکه/Neo4j هم باید اینجا گرفته بشه
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                print(f"   ⏳ خطای موقت در Neo4j، تلاش دوباره "
                      f"({attempt + 1}/{_MAX_RETRIES}): {e}")
                time.sleep(_RETRY_DELAY_SECONDS)
    raise RuntimeError(f"❌ نوشتن در Neo4j بعد از {_MAX_RETRIES} تلاش شکست خورد: {last_error}")


class FeatureGraphLoader:
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection

    def create_indexes(self):
        queries = [
            f"CREATE INDEX IF NOT EXISTS FOR (n:{cat.node_label}) ON (n.name)"
            for cat in VOCAB_CATEGORIES.values()
        ]
        with self.connection.session() as session:
            for q in queries:
                _with_retry(session.run, q)
        print("✅ Index های Feature Graph آماده‌اند.")

    def ruling_exists(self, ruling_id: str) -> bool:
        """
        بررسی وجود نود :Ruling در دیتابیس برای جلوگیری از Silent Failure روی پرونده‌های یتیم.
        """
        ruling_id_str = str(ruling_id)
        with self.connection.session() as session:
            result = _with_retry(
                session.run,
                "MATCH (r:Ruling {ruling_id: $ruling_id}) RETURN count(r) AS cnt",
                ruling_id=ruling_id_str,
            )
            record = result.single()
            return bool(record and record["cnt"] > 0)

    def is_ruling_loaded(self, ruling_id: str) -> bool:
        """
        چک idempotency: آیا این ruling قبلاً کامل بارگذاری شده؟
        """
        ruling_id_str = str(ruling_id)
        with self.connection.session() as session:
            result = _with_retry(
                session.run,
                "MATCH (r:Ruling {ruling_id: $ruling_id}) RETURN r.features_loaded AS loaded",
                ruling_id=ruling_id_str,
            )
            record = result.single()
            return bool(record and record["loaded"])

    def load_result(self, result: FeatureExtractionResult):
        ruling_id_str = str(result.ruling_id)

        # ۱. بررسی وجود نود Ruling قبل از بارگذاری (جلوگیری از Silent Failure)
        if not self.ruling_exists(ruling_id_str):
            raise ValueError(
                f"❌ نود :Ruling برای ruling_id='{ruling_id_str}' در Neo4j وجود ندارد (پرونده یتیم/Orphan)."
            )

        # ۲. بارگذاری ویژگی‌ها و ثبت علامت موفقیت
        with self.connection.session() as session:
            for category_key, field_name in _FIELD_NAMES.items():
                cat = VOCAB_CATEGORIES[category_key]
                for item in getattr(result, field_name):
                    _with_retry(session.execute_write, self._link_closed_vocab, ruling_id_str, item, cat)
            for fact in result.facts:
                _with_retry(session.execute_write, self._link_fact, ruling_id_str, fact)
            _with_retry(
                session.run,
                "MATCH (r:Ruling {ruling_id: $ruling_id}) SET r.features_loaded = true",
                ruling_id=ruling_id_str,
            )

    @staticmethod
    def _link_closed_vocab(tx, ruling_id: str, item: ExtractedFeature, cat):
        tx.run(
            f"""
            MATCH (r:Ruling {{ruling_id: $ruling_id}})
            MERGE (n:{cat.node_label} {{name: $value}})
            CREATE (r)-[rel:{cat.relation_type}]->(n)
            SET rel.evidence = $quote, rel.start_char = $start, rel.end_char = $end,
                rel.confidence = $confidence
            """,
            ruling_id=ruling_id,
            value=item.value,
            quote=item.evidence.quote,
            start=item.evidence.start_char,
            end=item.evidence.end_char,
            confidence=item.evidence.confidence,
        )

    @staticmethod
    def _link_fact(tx, ruling_id: str, item: ExtractedFeature):
        tx.run(
            f"""
            MATCH (r:Ruling {{ruling_id: $ruling_id}})
            CREATE (f:{FACT_CATEGORY.node_label} {{text: $value}})
            CREATE (r)-[rel:{FACT_CATEGORY.relation_type}]->(f)
            SET rel.evidence = $quote, rel.start_char = $start, rel.end_char = $end,
                rel.confidence = $confidence
            """,
            ruling_id=ruling_id,
            value=item.value,
            quote=item.evidence.quote,
            start=item.evidence.start_char,
            end=item.evidence.end_char,
            confidence=item.evidence.confidence,
        )



