"""
features/vocab_audit.py — چک‌های کیفیت بعد از دسته‌بندی واژه‌نامه

چرا این اسکریپت جداست، نه فقط چند خط دستی توی ترمینال؟
    چون این خطاها تکرارشونده‌اند: هر بار vocabulary_categorizer.py از
    نو اجرا بشه (مدل عوض بشه، batch size عوض بشه، ...)، همین دسته
    مشکلات دوباره ممکنه پیش بیان. این اسکریپت همون بررسی‌هایی رو که
    دستی روی خروجی انجام دادیم خودکار می‌کنه، تا بعد از هر اجرا فقط
    یک دستور بزنی، نه اینکه دوباره صدها واژه رو چشمی مرور کنی.

پنج چک انجام می‌شود:
    ۱. واژه‌های تکراری بین چند دسته (concept/action/role/object/principle)
    ۲. رکوردهای خراب (حرف لاتین وسط واژه‌ی فارسی)
    ۳. مترادف‌های تایپی که ادغام نشده‌اند (فقط تفاوت املایی/همزه)
    ۴. تناقض بین skipped.json و یک دسته‌ی طبقه‌بندی‌شده (نشونه‌ی
       checkpoint آلوده/قدیمی — دقیقاً همون باگی که با «خلاف» دیدیم)
    ۵. افتادن کلمه‌ی نقش‌ساز («طرف»، «له»، «عليه») هنگام ریشه‌سازی در
       دسته‌ی role — الگویی که کشف شد: مثلاً «طرف دعوی» به ریشه‌ی
       «دعوی» کوتاه شده، که با مفهوم مستقل «دعوی» در concepts.json
       تصادم می‌کند. این الگو مخصوص role است چون کلمه‌ی حذف‌شده دقیقاً
       همان چیزیه که تعیین می‌کند این یک «نقش» است نه یک مفهوم/شیء.

نحوه‌ی اجرا:
    uv run -m features.vocab_audit
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from features.configs import VOCAB_CATEGORIES

_HERE = Path(__file__).resolve().parent.parent  # graph-rag/
VOCAB_DIR = _HERE / "data" / "legal-vocabulary" / "categorized"

CATEGORY_KEYS = list(VOCAB_CATEGORIES.keys())

# کلمه‌های نقش‌ساز که اگر هنگام ریشه‌سازی حذف بشن، معنی/دسته عوض می‌شه
ROLE_MARKERS_PREFIX = ["طرف "]
ROLE_MARKERS_SUFFIX = [" له", " عليه", " علیه"]


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all():
    canon, aliases = {}, {}
    for key in CATEGORY_KEYS:
        c = _load_json(VOCAB_DIR / f"{key}s.json")
        if c is None:
            print(f"⚠️ {key}s.json پیدا نشد — رد شد.")
            continue
        canon[key] = c
        aliases[key] = _load_json(VOCAB_DIR / f"{key}s_aliases.json") or {}
    skipped = _load_json(VOCAB_DIR / "skipped.json") or []
    return canon, aliases, skipped


def check_cross_category(canon: dict) -> dict:
    where = defaultdict(list)
    for cat, words in canon.items():
        for w in words:
            where[w].append(cat)
    return {w: cats for w, cats in where.items() if len(cats) > 1}


def check_corrupted(canon: dict) -> list[tuple[str, str]]:
    return [(cat, w) for cat, words in canon.items() for w in words if re.search(r"[a-zA-Z]", w)]


def _normalize(s: str) -> str:
    return (
        s.replace("ء", "").replace("أ", "ا").replace("إ", "ا").replace("ة", "ه")
        .replace("ي", "ی").replace("ك", "ک").replace(" ", "").replace("‌", "")
    )


def check_near_duplicates(canon: dict) -> dict:
    result = {}
    for cat, words in canon.items():
        buckets = defaultdict(list)
        for w in words:
            buckets[_normalize(w)].append(w)
        dups = [group for group in buckets.values() if len(group) > 1]
        if dups:
            result[cat] = dups
    return result


def check_skip_overlap(canon: dict, skipped: list) -> dict:
    skipped_set = set(skipped)
    return {cat: sorted(skipped_set & set(words)) for cat, words in canon.items() if skipped_set & set(words)}


def check_role_marker_truncation(aliases: dict) -> list[dict]:
    findings = []
    for root, raws in aliases.get("role", {}).items():
        for raw in raws:
            for m in ROLE_MARKERS_PREFIX:
                if raw.startswith(m) and raw[len(m):] == root:
                    findings.append({"root": root, "raw": raw, "marker": m.strip(), "position": "پیشوند"})
            for m in ROLE_MARKERS_SUFFIX:
                if raw.endswith(m) and raw[: -len(m)] == root:
                    findings.append({"root": root, "raw": raw, "marker": m.strip(), "position": "پسوند"})
    return findings


def _section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run():
    canon, aliases, skipped = load_all()

    _section("۱. واژه‌های تکراری بین چند دسته")
    cross = check_cross_category(canon)
    if not cross:
        print("✅ چیزی پیدا نشد.")
    for w, cats in sorted(cross.items()):
        print(f"  ⚠️ «{w}»: {cats}")

    _section("۲. رکوردهای خراب (حرف لاتین وسط واژه‌ی فارسی)")
    corrupted = check_corrupted(canon)
    if not corrupted:
        print("✅ چیزی پیدا نشد.")
    for cat, w in corrupted:
        print(f"  ⚠️ [{cat}] «{w}»")

    _section("۳. مترادف‌های تایپی ادغام‌نشده (فقط تفاوت املایی)")
    near_dup = check_near_duplicates(canon)
    if not near_dup:
        print("✅ چیزی پیدا نشد.")
    for cat, groups in near_dup.items():
        for g in groups:
            print(f"  ⚠️ [{cat}] {g}")

    _section("۴. تناقض skip در برابر دسته‌بندی‌شده (checkpoint آلوده)")
    skip_conflict = check_skip_overlap(canon, skipped)
    if not skip_conflict:
        print("✅ چیزی پیدا نشد.")
    for cat, words in skip_conflict.items():
        print(f"  ⚠️ [{cat}] {words}")

    _section("۵. افتادن کلمه‌ی نقش‌ساز هنگام ریشه‌سازی (دسته‌ی role)")
    role_trunc = check_role_marker_truncation(aliases)
    if not role_trunc:
        print("✅ چیزی پیدا نشد.")
    for f in role_trunc:
        print(f"  ⚠️ ریشه=«{f['root']}» <- خام=«{f['raw']}» (کلمه‌ی «{f['marker']}» به‌صورت {f['position']} افتاده)")

    total = (
        len(cross) + len(corrupted)
        + sum(len(g) for g in near_dup.values())
        + sum(len(w) for w in skip_conflict.values())
        + len(role_trunc)
    )
    _section(f"🎯 جمع کل موارد یافت‌شده: {total}")
    print("این‌ها را دستی در _checkpoint.jsonl یا فایل‌های categorized/*.json اصلاح کن.")


if __name__ == "__main__":
    run()