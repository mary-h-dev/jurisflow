"""
features/vocab_audit.py — quality checks after vocabulary categorization

Why is this script separate, not just a few manual lines in a terminal?
    Because these errors recur: every time vocabulary_categorizer.py is
    re-run (model changed, batch size changed, ...), the same class of
    problems can reappear. This script automates the checks that were
    previously done manually on the output, so after each run you just
    run one command instead of eyeballing hundreds of terms again.

Five checks are performed:
    1. Terms duplicated across multiple categories (concept/action/role/object/principle)
    2. Corrupted records (Latin letters inside a Persian term)
    3. Unmerged typo-synonyms (spelling/hamza differences only)
    4. Conflict between skipped.json and a categorized category (a sign
       of a stale/contaminated checkpoint — exactly the bug seen with "خلاف")
    5. Loss of the role-forming word ("طرف", "له", "عليه") during root
       resolution in the role category — a discovered pattern: e.g.
       "طرف دعوی" got shortened to root "دعوی", which collides with the
       independent concept "دعوی" in concepts.json. This pattern is
       specific to role because the dropped word is exactly what
       determines that this is a "role" rather than a concept/object.

Usage:
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

# Role-forming words that, if dropped during root resolution, change the meaning/category
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
            print(f"⚠️ {key}s.json not found — skipped.")
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
                    findings.append({"root": root, "raw": raw, "marker": m.strip(), "position": "prefix"})
            for m in ROLE_MARKERS_SUFFIX:
                if raw.endswith(m) and raw[: -len(m)] == root:
                    findings.append({"root": root, "raw": raw, "marker": m.strip(), "position": "suffix"})
    return findings


def _section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run():
    canon, aliases, skipped = load_all()

    _section("1. Terms duplicated across multiple categories")
    cross = check_cross_category(canon)
    if not cross:
        print("✅ Nothing found.")
    for w, cats in sorted(cross.items()):
        print(f"  ⚠️ «{w}»: {cats}")

    _section("2. Corrupted records (Latin letter inside a Persian term)")
    corrupted = check_corrupted(canon)
    if not corrupted:
        print("✅ Nothing found.")
    for cat, w in corrupted:
        print(f"  ⚠️ [{cat}] «{w}»")

    _section("3. Unmerged typo synonyms (spelling difference only)")
    near_dup = check_near_duplicates(canon)
    if not near_dup:
        print("✅ Nothing found.")
    for cat, groups in near_dup.items():
        for g in groups:
            print(f"  ⚠️ [{cat}] {g}")

    _section("4. Skip conflict vs. categorized (contaminated checkpoint)")
    skip_conflict = check_skip_overlap(canon, skipped)
    if not skip_conflict:
        print("✅ Nothing found.")
    for cat, words in skip_conflict.items():
        print(f"  ⚠️ [{cat}] {words}")

    _section("5. Dropped role-forming word during root resolution (role category)")
    role_trunc = check_role_marker_truncation(aliases)
    if not role_trunc:
        print("✅ Nothing found.")
    for f in role_trunc:
        print(f"  ⚠️ root=«{f['root']}» <- raw=«{f['raw']}» (word «{f['marker']}» dropped as a {f['position']})")

    total = (
        len(cross) + len(corrupted)
        + sum(len(g) for g in near_dup.values())
        + sum(len(w) for w in skip_conflict.values())
        + len(role_trunc)
    )
    _section(f"🎯 Total findings: {total}")
    print("Fix these manually in _checkpoint.jsonl or the categorized/*.json files.")


if __name__ == "__main__":
    run()