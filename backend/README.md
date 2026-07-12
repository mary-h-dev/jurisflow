START
  ↓
analyzer (استخراج اطلاعات پرونده)
  ↓
searcher (جستجوی قوانین از Neo4j)
  ↓
Lead Agent
  ↓ تسک میده به سه teammate به صورت parallel
┌─────────────────────────────────┐
Teammate 1      Teammate 2        Teammate 3
وکیل مدافع    دادستان/مخالف     قاضی بی‌طرف
  ↓                ↓                 ↓
└─────────────────────────────────┘
  ↓ نظرات به Lead برمیگرده
Lead Agent → گزارش نهایی
  ↓
END




pipeline/
├── data/          ← همون کدهای قبلی (scraper, embedder, classifier)
└── ocr/           ← جدید
    ├── __init__.py
    ├── base.py
    ├── gemini_ocr.py
    └── glm_ocr.py
