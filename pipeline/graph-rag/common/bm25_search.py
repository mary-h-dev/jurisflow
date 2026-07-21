"""
common/bm25_search.py — جستجوی BM25 سبک، کاملاً محلی (بدون سرور جدا)

نصب: pip install rank-bm25

چون پیکره‌مون کوچیکه (~۲۴۰۰ پرونده + ~۲۰۰۰ ماده)، کل ایندکس توی حافظه
جا می‌شه — نیازی به Elasticsearch یا موتور جدا نیست. این عمداً مستقل از
embedding نگه داشته شده: هدف اینه که وقتی retrieval رو نوشتیم، بشه امتیاز
BM25 و امتیاز embedding رو جدا سنجید (دقیقاً همون چیزی که قبلاً توافق
کردیم — هر کانال باید قابل ارزیابیِ جدا باشه).
"""

import re
from rank_bm25 import BM25Okapi



def simple_tokenize(text: str) -> list[str]:
    """
    توکنایزر ساده برای فارسی: نیم‌فاصله رو با فاصله یکی می‌کنه، بعد فقط
    دنباله‌های حروف فارسی/عربی و اعداد رو به‌عنوان توکن می‌گیره.
    برای دقت بیشتر می‌شه بعداً با یک lemmatizer فارسی (مثل Hazm)
    جایگزینش کرد؛ برای شروع و محک اولیه همین کافیه.
    """
    text = text.replace("\u200c", " ")
    return re.findall(r"[\u0600-\u06FF]+|\d+", text)



class BM25Search:
    def __init__(self, documents: list[dict], text_key: str = "text", id_key: str = "id"):
        """
        documents: لیستی از dict — هرکدوم حداقل یک فیلد متن (text_key)
        و یک شناسه‌ی یکتا (id_key، مثلاً article_number یا ruling_id).
        """
        self.ids = [d[id_key] for d in documents]
        tokenized_corpus = [simple_tokenize(d[text_key]) for d in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)



    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """خروجی: لیست (شناسه، امتیاز) مرتب‌شده بر اساس بیشترین امتیاز"""
        query_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]