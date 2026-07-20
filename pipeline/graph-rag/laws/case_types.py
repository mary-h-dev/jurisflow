

DOMAIN_TO_CASE_TYPE = {
    "مدنی":          "حقوقی",
    "تجاری":         "حقوقی",
    "دادرسی مدنی":   "حقوقی",
    "کیفری":         "کیفری",
    "دادرسی کیفری":  "کیفری",
}


def case_type_of(domain: str) -> str:
    """نوع کلی پرونده (حقوقی|کیفری) را از روی domain برمی‌گرداند"""
    if domain not in DOMAIN_TO_CASE_TYPE:
        raise ValueError(
            f"domain ناشناخته: '{domain}' — آن را در DOMAIN_TO_CASE_TYPE اضافه کنید."
        )
    return DOMAIN_TO_CASE_TYPE[domain]