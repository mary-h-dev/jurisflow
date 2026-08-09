# test_retrieval.py
import os
from dotenv import load_dotenv
from database.connection import Neo4jConnection
from retrieval.retriever import LegalRetriever
from retrieval.confidence import score_result

load_dotenv()

connection = Neo4jConnection(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USERNAME"),
    os.getenv("NEO4J_PASSWORD"),
)

retriever = LegalRetriever(connection)

queries = [
    "آیا مستأجر می‌تواند بدون اجازه موجر اجاره‌دهد؟",
    "شرایط فسخ قرارداد در حقوق ایران چیست؟",
    "حضانت فرزند بعد از طلاق با کیست؟",
]

for q in queries:
    result = retriever.retrieve(q)
    score, vec = score_result(result)

    print(f"\nQuery: {q}")
    print(f"Retrieval score: {score:.3f}")
    print(f"  کیفیت feature: {vec.feature_quality:.3f}")
    print(f"  کیفیت ruling:  {vec.ruling_quality:.3f}")
    print(f"  کیفیت article: {vec.article_quality:.3f}")
    if vec.missing_channels:
        print(f"  ⚠️ کانال‌های خالی: {vec.missing_channels}")

    print(f"Evidence count (بعد از RRF): {len(result.evidences)}")
    print(f"Rulings found: {result.ruling_ids[:10]}")
    for ev in result.evidences[:5]:
        print(f"  [{ev.source_type}] {ev.text[:80]}...")

connection.close()