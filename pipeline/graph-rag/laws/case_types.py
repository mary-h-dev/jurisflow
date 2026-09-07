DOMAIN_TO_CASE_TYPE = {
    "مدنی":          "حقوقی",
    "تجاری":         "حقوقی",
    "دادرسی مدنی":   "حقوقی",
    "کیفری":         "کیفری",
    "دادرسی کیفری":  "کیفری",
}


def case_type_of(domain: str) -> str:
    """Returns the overall case type (civil | criminal) based on the domain."""
    if domain not in DOMAIN_TO_CASE_TYPE:
        raise ValueError(
            f"Unknown domain: '{domain}' — add it to DOMAIN_TO_CASE_TYPE."
        )
    return DOMAIN_TO_CASE_TYPE[domain]