"""
features/schemas.py — ساختار داده‌ی خروجیِ Feature Extraction (لایه‌ی سوم گراف)

چرا Evidence یک dataclass جداست، نه فقط یک رشته؟
    چون طبق تصمیم اصلی (evidence-based)، هر Feature باید هم متن مدرک
    (quote از رأی)، هم موقعیت دقیقش (start_char/end_char)، و هم عدد
    اطمینان مدل را داشته باشد — تا بعداً بشه مستقیم به ماژول
    Uncertainty وصل کرد و در UI محل دقیقش را هایلایت کرد.

چرا start_char/end_char را خودِ LLM تولید نمی‌کند؟
    چون مدل‌های زبانی در شمردن دقیق موقعیت کاراکتر در متن‌های طولانی
    غیرقابل‌اعتمادند — این یک محدودیت شناخته‌شده‌ی LLM هاست، نه چیزی که
    با پرامپت بهتر حل بشه. به‌جایش از LLM فقط «متن دقیق مدرک» (quote
    عیناً از رأی) خواسته می‌شود؛ extractor.py با str.find روی متن اصلی
    موقعیتش را پیدا می‌کند. اگر پیدا نشد (LLM کمی متن را تغییر داده)،
    start_char/end_char برابر None می‌مانند، ولی خودِ متنِ مدرک نگه
    داشته می‌شود — صادقانه‌تر از این‌که یک عدد غلط حدس بزنیم.

چرا Concept/Action/Role/Object و Fact یک کلاس مشترک
(ExtractedFeature) دارند، نه کلاس‌های جدا برای هرکدام؟
    چون ساختارشان (دسته، مقدار، مدرک) کاملاً یکسان است — تنها فرقشان
    این است که مقدارِ کدام‌ها باید از closed vocabulary باشد و کدام
    آزادند (نگاه کن به features/configs.py → from_closed_vocabulary).
"""




from dataclasses import dataclass, field


@dataclass
class Evidence:
    quote: str                        # متن دقیق مدرک، عیناً از رأی
    start_char: int | None = None     # موقعیت شروع در متن اصلی (اگر پیدا شد)
    end_char: int | None = None
    confidence: float = 0.0           # اطمینان مدل، بین 0 و 1


@dataclass
class ExtractedFeature:
    category: str    # یکی از کلیدهای VOCAB_CATEGORIES ("concept"|"action"|...) یا "fact"
    value: str        # برای closed-vocab: باید دقیقاً از لیست باشد؛ برای fact: آزاد
    evidence: Evidence


@dataclass
class FeatureExtractionResult:
    """نتیجه‌ی کامل استخراج Feature برای یک Ruling"""
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