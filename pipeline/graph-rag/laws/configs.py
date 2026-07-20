

from laws.parser import LawParseConfig

LAW_CONFIGS = {
    "civil": LawParseConfig(
        law_name="قانون مدنی",
        domain="مدنی",
    ),
    "penal": LawParseConfig(
        law_name="قانون مجازات اسلامی",
        domain="کیفری",
    ),
    "commercial": LawParseConfig(
        law_name="قانون تجارت",
        domain="تجاری",
    ),
    "criminal_procedure": LawParseConfig(
        law_name="قانون آیین دادرسی کیفری",
        domain="دادرسی کیفری",
    ),
    "civil_procedure": LawParseConfig(
        law_name="قانون آیین دادرسی مدنی",
        domain="دادرسی مدنی",
    ),
}

