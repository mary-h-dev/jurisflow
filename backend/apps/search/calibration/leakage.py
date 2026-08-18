from __future__ import annotations

import dataclasses

from apps.search.services import SearchResult


def exclude_self_ruling(search_result: SearchResult, ruling_id: str) -> SearchResult:
    """
    Returns a copy of `search_result` with every piece of evidence that
    came from `ruling_id` removed -- from ruling_results, from
    feature_results (features are extracted per-ruling, so they carry
    ruling_id too), and from the fused evidences list. article_results
    are left untouched: they come directly from the article_channel's
    embedding search over statute text, not from any specific ruling, so
    they carry no leakage risk here.

    Use this ONLY in eval/calibration scripts where the query being
    tested was paraphrased FROM a specific known ruling_id (see
    case_grounded_100.json). If that same ruling is still retrievable
    from the graph, the ruling_channel can trivially return the source
    ruling's own reasoning and verdict as "evidence" -- which is not
    independent precedent, it's the answer key. Production queries (a
    real user's fresh question) have no such source ruling_id and don't
    need this filter.
    """
    filtered_ruling_results = [
        e for e in search_result.ruling_results if e.ruling_id != ruling_id
    ]
    filtered_feature_results = [
        e for e in search_result.feature_results if e.ruling_id != ruling_id
    ]
    filtered_evidences = [
        e for e in search_result.evidences if e.ruling_id != ruling_id
    ]

    return dataclasses.replace(
        search_result,
        evidences=filtered_evidences,
        feature_results=filtered_feature_results,
        ruling_results=filtered_ruling_results,
    )