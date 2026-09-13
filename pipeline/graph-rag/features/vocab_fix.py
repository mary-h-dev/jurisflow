"""
features/vocab_fix.py — automatically applies the deterministic (not
                          judgment-based) fixes found by vocab_audit.py

Why are only some of the 52 findings auto-fixed here, not all?
    Because of vocab_audit.py's 5 checks, only 4 (corrupted record, typo
    synonym, skip conflict, dropped role-forming word) have an
    unambiguously "correct" answer. Check #1 (a term duplicated across
    categories) is not like that — most of those 34 cases (e.g. "تصرف",
    which is both a verb and a concept) are a natural phenomenon of
    legal Persian (an Arabic-derived noun/verb that refers to both an
    act and a state) and are left untouched. Only the subset directly
    caused by the "dropped role-forming word" bug (دعوی، حواله، شکایت،
    قرارداد، مشروط) are also fixed here — because their root cause is
    known.

    The rest of check #1's cases (e.g. whether "تصویر"/"جواز"/"غرامت"
    should stay object-only or also be concept) were proposed separately
    in a text response, not here — because they require domain judgment
    that's better made by looking at your actual case data, not a fixed
    rule in code.

Before any change, a backup of the entire categorized/ folder is taken.

Usage:
    uv run -m features.vocab_fix
    uv run -m features.vocab_audit   # to confirm the fixed items are gone
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from features.configs import VOCAB_CATEGORIES

_HERE = Path(__file__).resolve().parent.parent  # graph-rag/
VOCAB_DIR = _HERE / "data" / "legal-vocabulary" / "categorized"

# --- 1. corrupted record (Latin letter inside a Persian term) ---
CORRUPTED_TO_REMOVE = {"action": ["جبران خسarat"]}

# --- 2. merging typo synonyms: {category: {drop -> keep}} ---
TYPO_MERGES = {
    "concept": {"هم‌زمانی": "همزمانی"},
    "action": {"استعفاء": "استعفا", "پیش‌گیری": "پیشگیری"},
    "object": {"رای": "رأی"},
}

# --- 3. skip/category conflict: these go back to skip (because "ذاتی"/
#         "ثابت" are generic adjectives, not a specific, citable
#         principle/concept) ---
SKIP_WINS = {"concept": ["ثابت"], "principle": ["ذاتی"]}

# --- 4. removal from role: either directly caused by the "طرف"/"له"/
#         "عليه" dropped-word bug (collision with an unrelated root), or
#         fully redundant with a more precise, already-existing role
#         (سوم ≈ ثالث/شخص ثالث which is in roles.json) ---
ROLE_REMOVE = ["دعوی", "حواله", "شکایت", "قرارداد", "مقابل", "مشروط", "سوم"]

# --- 5. concept+object terms decided to keep as object only — all of
#         them are more of a "specific object/document" than an
#         "abstract concept" (e.g. "ضمانت‌نامه" itself has the "-نامه"
#         suffix meaning it's a document) ---
CONCEPT_REMOVE_KEEP_AS_OBJECT = [
    "تصویر", "جواز", "حصه", "ضمانت‌نامه", "طلب", "غرامت", "مالیات",
]

# --- 6. concept+action terms narrowed to a single category (domain
#         judgment; if your experience with real data differs, change
#         it here)
#
#     Deliberately NOT included here, staying dual, because both usages
#     are genuinely common in ruling text, not a bug:
#       - "تصرف" (both the verb "took possession" and the abstract
#         property-law concept)
#       - the six action+object terms (حکم/دادخواست/شکایت/مجوز/کمک/گواهی):
#         both the act of issuing/filing matters and the document itself
#
#     Selection logic: if the term denotes a stable state/institution/
#     relationship -> concept. If it denotes a one-time act with a
#     specific actor -> action.
REMOVE_FROM = {
    # remove these from action, keep only as concept
    "action": ["افلاس", "بخشودگی", "عطف", "لطمه", "وکالت"],
    # remove these from concept (10 stay action-only, the last 2 —
    # بی‌طرفی/ثبات — stay principle-only)
    "concept": [
        "اقاله", "تأسیس", "تعلیق", "جعل", "دفاع", "لغو", "مذاکره",
        "نقض", "نقل و انتقال", "پرداخت", "بی‌طرفی", "ثبات",
    ],
}


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def backup():
    backup_dir = VOCAB_DIR.parent / "categorized_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(VOCAB_DIR, backup_dir)
    print(f"🗄️ Backup saved to {backup_dir}.")


def run():
    backup()

    # Load all canon lists up front — step 6 needs to check that when we
    # remove a term from category X, it genuinely still exists in
    # another category (otherwise its meaning would be lost entirely)
    all_canon = {}
    for key in VOCAB_CATEGORIES:
        p = VOCAB_DIR / f"{key}s.json"
        if p.exists():
            all_canon[key] = set(_load(p))

    for key in VOCAB_CATEGORIES:
        canon_path = VOCAB_DIR / f"{key}s.json"
        alias_path = VOCAB_DIR / f"{key}s_aliases.json"
        if not canon_path.exists():
            print(f"⚠️ {canon_path.name} not found — skipped.")
            continue

        canon = _load(canon_path)
        aliases = _load(alias_path) if alias_path.exists() else {}
        changed = False

        # 1. corrupted record
        for bad in CORRUPTED_TO_REMOVE.get(key, []):
            if bad in canon:
                canon.remove(bad)
                changed = True
                print(f"  🗑️ [{key}] removed corrupted record: «{bad}»")
            for raws in aliases.values():
                if bad in raws:
                    raws.remove(bad)

        # 2. merge typo synonym
        for drop, keep in TYPO_MERGES.get(key, {}).items():
            if drop in canon:
                canon.remove(drop)
                changed = True
                print(f"  🔗 [{key}] merged: «{drop}» -> «{keep}»")
                if drop in aliases:
                    aliases.setdefault(keep, [])
                    aliases[keep].extend(aliases.pop(drop))

        # 3. skip conflict -> revert to skip
        for w in SKIP_WINS.get(key, []):
            if w in canon:
                canon.remove(w)
                changed = True
                print(f"  ↩️ [{key}] reverted to skip: «{w}»")
                aliases.pop(w, None)

        # 6. dual concept/action terms narrowed to one category (safety
        #    check: only if it genuinely exists in at least one other category)
        for w in REMOVE_FROM.get(key, []):
            if w not in canon:
                continue
            exists_elsewhere = any(w in words for k2, words in all_canon.items() if k2 != key)
            if exists_elsewhere:
                canon.remove(w)
                changed = True
                print(f"  🗑️ [{key}] removed (kept in the other category): «{w}»")
                aliases.pop(w, None)
            else:
                print(f"  ⚠️ [{key}] «{w}» not found elsewhere — not removed "
                      f"(to avoid losing this term entirely)")

        # 4. remove collision/redundant role nodes
        if key == "role":
            for w in ROLE_REMOVE:
                if w in canon:
                    canon.remove(w)
                    changed = True
                    print(f"  🗑️ [role] removed (colliding or redundant): «{w}»")
                    aliases.pop(w, None)

        # 5. concept+object -> object only (with the safety check that it
        #    actually exists in objects.json too, otherwise it's not removed
        #    since it would lose its meaning entirely)
        if key == "concept":
            objects_canon = _load(VOCAB_DIR / "objects.json") if (VOCAB_DIR / "objects.json").exists() else []
            for w in CONCEPT_REMOVE_KEEP_AS_OBJECT:
                if w in canon:
                    if w in objects_canon:
                        canon.remove(w)
                        changed = True
                        print(f"  🗑️ [concept] removed (stays object-only): «{w}»")
                        aliases.pop(w, None)
                    else:
                        print(f"  ⚠️ [concept] «{w}» not found in objects.json — not removed "
                              f"(to avoid losing this concept entirely)")

        if changed:
            _save(canon_path, sorted(set(canon)))
            _save(alias_path, aliases)

    # update skipped.json with terms that were reverted
    skipped_path = VOCAB_DIR / "skipped.json"
    skipped = _load(skipped_path) if skipped_path.exists() else []
    for words in SKIP_WINS.values():
        for w in words:
            if w not in skipped:
                skipped.append(w)
    _save(skipped_path, sorted(set(skipped)))

    print("\n✅ Deterministic fixes applied.")
    print("   Re-run: uv run -m features.vocab_audit")
    print("   (Check #1 will still show some items -- these are intentionally left untouched, see the note at the top of this file)")


if __name__ == "__main__":
    run()