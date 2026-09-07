from dataclasses import dataclass, field


@dataclass
class Note:
    """A note (تبصره) attached to an article"""
    note_number: int
    content: str
    status: str = "active"        # active | amended | abolished | interpreted


@dataclass
class Article:
    """A single statutory article — the core unit of the Rule Graph"""
    article_number: int
    content: str
    law_name: str                 # e.g. "Civil Code"
    domain: str                   # Civil | Criminal | Commercial | Civil Procedure | Criminal Procedure
    case_type: str                # civil | criminal — always derived from domain, never entered manually
    notes: list[Note] = field(default_factory=list)
    status: str = "active"
    references: list[int] = field(default_factory=list)


@dataclass
class Law:
    """A complete statute, represented as a flat list of articles"""
    name: str
    domain: str
    case_type: str                # civil | criminal — always derived from domain
    url: str
    articles: list[Article] = field(default_factory=list)