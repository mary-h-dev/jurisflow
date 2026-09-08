"""
cases/configs.py — maps each core statute to its "Laws" ID on the National
                    Rulings System (ara.jri.ac.ir)

This file is completely independent of laws/configs.py:
    laws/configs.py   → for scraping the statute's own text from qavanin.ir
    cases/configs.py  → only the ID of that same statute on the rulings
                         system (ara.jri.ac.ir) — a different site with a
                         completely different ID space.

Dynamic by design:
    The keys (civil, penal, ...) must exactly match the keys in
    laws.configs.LAW_CONFIGS. The cases pipeline loops directly over
    LAW_CONFIGS (not a separate manual list) — meaning that if a new
    statute is later added to laws/configs.py, it will automatically be
    included in the case-scraping step too, as long as its ID is also
    registered here.

    If a statute exists in LAW_CONFIGS but its ID is missing here, the
    pipeline stops with a clear error (rather than silently skipping it) —
    so that whoever adds a new statute doesn't forget to also add its
    rulings-system ID.

    How to find a new statute's ID: type the statute's name into the
    statute search box on the National Rulings System (ara.jri.ac.ir);
    in the HTML, the <option value="..."> is the ID you need.
"""

CASE_SEARCH_LAW_IDS: dict[str, int] = {
    "civil":               2,
    "civil_procedure":     4671,
    "commercial":          2970,
    "penal":               867,
    "criminal_procedure":  2446,
}


def get_case_search_id(law_key: str) -> int:
    """
    Returns the rulings-system ID for a given statute key.
    Raises a clear error if it isn't registered (instead of silently
    ignoring it).
    """
    if law_key not in CASE_SEARCH_LAW_IDS:
        raise KeyError(
            f"No rulings-system ID registered for statute '{law_key}'. "
            f"Search for the statute's name on ara.jri.ac.ir, find its "
            f"<option> id, and add it to CASE_SEARCH_LAW_IDS in cases/configs.py."
        )
    return CASE_SEARCH_LAW_IDS[law_key]