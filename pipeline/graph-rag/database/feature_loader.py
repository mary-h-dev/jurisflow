"""
features/database/feature_loader.py — بارگذاری گراف سوم (Feature Graph) در Neo4j

از همون Neo4jConnection مشترک استفاده می‌کنه (database/connection.py)،
دقیقاً مثل database/case_loader.py و database/law_loader.py.

ساختار گراف:
    (:Ruling)-[:HAS_CONCEPT   {evidence, start_char, end_char, confidence}]->(:LegalConcept   {name})
    (:Ruling)-[:HAS_ACTION    {...}]->(:LegalAction    {name})
    (:Ruling)-[:HAS_ROLE      {...}]->(:LegalRole      {name})
    (:Ruling)-[:HAS_OBJECT    {...}]->(:LegalObject    {name})
    (:Ruling)-[:HAS_PRINCIPLE {...}]->(:LegalPrinciple {name})
    (:Ruling)-[:HAS_FACT      {...}]->(:LegalFact      {text})

چرا closed-vocab node ها MERGE می‌شوند ولی LegalFact نه؟
    چون Concept/Action/Role/Object/Principle از یک لیست بسته و
    یکتاشده (بعد از canonicalization در vocabulary_categorizer.py)
    می‌آیند — یک node واحد برای «بیع» در کل گراف کافی است، و MERGE
    باعث می‌شود همه‌ی رأی‌های مرتبط به یک node وصل شوند (این دقیقاً
    همون چیزیه که query زدن رو قدرتمند می‌کنه).
    Factها اما آزاد و به‌شدت متنوعند؛ اگر آن‌ها را هم MERGE کنیم،
    Factهای مشابه ولی نه‌کاملاً یکسان (که در متن حقوقی خیلی رایج است)
    به‌اشتباه در یک node ادغام می‌شوند و اطلاعات خاص هر پرونده گم
    می‌شود. برای همین هر Fact یک node مستقل با CREATE می‌گیرد، حتی اگر
    شبیه یک Fact در پرونده‌ی دیگر باشد.
"""


from database.connection import Neo4jConnection
from features.configs import FACT_CATEGORY, VOCAB_CATEGORIES
from features.schemas import ExtractedFeature, FeatureExtractionResult

# نگاشت کلید دسته → نام فیلد لیست در FeatureExtractionResult
_FIELD_NAMES = {
    "concept": "concepts",
    "action": "actions",
    "role": "roles",
    "object": "objects",
    "principle": "principles",
}


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
                session.run(q)
        print("✅ Index های Feature Graph آماده‌اند.")

    def load_result(self, result: FeatureExtractionResult):
        with self.connection.session() as session:
            for category_key, field_name in _FIELD_NAMES.items():
                cat = VOCAB_CATEGORIES[category_key]
                for item in getattr(result, field_name):
                    session.execute_write(self._link_closed_vocab, result.ruling_id, item, cat)
            for fact in result.facts:
                session.execute_write(self._link_fact, result.ruling_id, fact)

    @staticmethod
    def _link_closed_vocab(tx, ruling_id: str, item: ExtractedFeature, cat):
        tx.run(
            f"""
            MATCH (r:Ruling {{ruling_id: $ruling_id}})
            MERGE (n:{cat.node_label} {{name: $value}})
            MERGE (r)-[rel:{cat.relation_type}]->(n)
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