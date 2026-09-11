"""
common/bm25_search.py — lightweight, fully local BM25 search (no separate server)

Install: pip install rank-bm25

Since our corpus is small (~2400 rulings + ~2000 articles), the whole
index fits in memory — no need for Elasticsearch or a separate engine.
This is deliberately kept independent of embeddings: the goal is that
once retrieval is written, the BM25 score and the embedding score can be
evaluated separately (exactly what we agreed on before — each channel
must be independently evaluable).
"""

import re
from rank_bm25 import BM25Okapi



def simple_tokenize(text: str) -> list[str]:
    """
    A simple tokenizer for Persian: merges ZWNJ with a regular space, then
    only takes sequences of Persian/Arabic letters and digits as tokens.
    For better accuracy this can later be replaced with a Persian
    lemmatizer (like Hazm); for a first pass and initial benchmark this is
    enough.
    """
    text = text.replace("\u200c", " ")
    return re.findall(r"[\u0600-\u06FF]+|\d+", text)



class BM25Search:
    def __init__(self, documents: list[dict], text_key: str = "text", id_key: str = "id"):
        """
        documents: a list of dicts — each with at least one text field
        (text_key) and one unique identifier (id_key, e.g. article_number
        or ruling_id).
        """
        self.ids = [d[id_key] for d in documents]
        tokenized_corpus = [simple_tokenize(d[text_key]) for d in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)



    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Output: a list of (id, score), sorted by highest score first"""
        query_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]