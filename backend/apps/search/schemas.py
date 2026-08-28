from pydantic import BaseModel
from typing import Optional


class SearchIn(BaseModel):
    query: str


class EvidenceOut(BaseModel):
    source_type:      str
    text:             str
    score:            float
    rrf_score:        float
    ruling_id:        Optional[str] = None
    feature_value:    Optional[str] = None
    feature_category: Optional[str] = None
    article_number:   Optional[int] = None
    law_name:         Optional[str] = None
    cited_articles:   list[str] = []   



class ChannelQualityOut(BaseModel):
    feature: float
    ruling:  float
    article: float
    missing: list[str]



class RoutingOut(BaseModel):
    rewritten_query:    str
    channels:           list[str]
    routing_confidence: float
    intent:             str
    case_type_hint:     Optional[str] = None
    ambiguity_flag:     bool
    raw_ok:             bool




class ConfidenceOut(BaseModel):
    score:    float
    level:    str
    note:     Optional[str] = None
    channels: ChannelQualityOut


class SearchOut(BaseModel):
    query:        str
    evidences:    list[EvidenceOut]
    confidence:   ConfidenceOut
    routing:      RoutingOut
    ruling_ids:   list[str]
    article_refs: list[str]   