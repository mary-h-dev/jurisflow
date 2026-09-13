"""
features/schemas.py — output data structures for Feature Extraction
                       (third graph layer)

Why is Evidence a separate dataclass, not just a string?
    Per the original evidence-based design, every feature must carry the
    evidence text (a quote from the ruling), its exact location
    (start_char/end_char), and the model's confidence score — so it can
    later be wired directly into the Uncertainty module and highlighted
    precisely in the UI.

Why doesn't the LLM itself produce start_char/end_char?
    Because language models are unreliable at counting exact character
    positions in long text — this is a known LLM limitation, not
    something a better prompt fixes. Instead, the LLM is only asked for
    "the exact evidence text" (a verbatim quote from the ruling);
    extractor.py locates its position in the source text with str.find.
    If not found (the LLM slightly altered the text), start_char/end_char
    stay None, but the evidence text itself is kept — more honest than
    guessing a wrong number.

Why do Concept/Action/Role/Object and Fact share one class
(ExtractedFeature) instead of separate classes for each?
    Because their structure (category, value, evidence) is identical —
    the only difference is which ones must come from the closed
    vocabulary and which are free (see features/configs.py →
    from_closed_vocabulary).
"""


from dataclasses import dataclass, field


@dataclass
class Evidence:
    quote: str                        # exact evidence text, verbatim from the ruling
    start_char: int | None = None     # start position in the source text (if found)
    end_char: int | None = None
    confidence: float = 0.0           # model confidence, between 0 and 1


@dataclass
class ExtractedFeature:
    category: str    # one of the VOCAB_CATEGORIES keys ("concept"|"action"|...) or "fact"
    value: str        # for closed-vocab: must exactly match the list; for fact: free
    evidence: Evidence


@dataclass
class FeatureExtractionResult:
    """Full feature-extraction result for one Ruling"""
    ruling_id: str
    concepts: list[ExtractedFeature] = field(default_factory=list)
    actions: list[ExtractedFeature] = field(default_factory=list)
    roles: list[ExtractedFeature] = field(default_factory=list)
    objects: list[ExtractedFeature] = field(default_factory=list)
    facts: list[ExtractedFeature] = field(default_factory=list)

    def all_features(self) -> list[ExtractedFeature]:
        return (
            self.concepts + self.actions + self.roles
            + self.objects + self.facts
        )