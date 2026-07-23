"""
features/vocab_fix.py — اعمال خودکارِ اصلاحاتِ قطعی (نه قضاوتی) که با
                          vocab_audit.py پیدا شدند

چرا فقط بعضی از ۵۲ مورد اینجا خودکار اصلاح می‌شوند، نه همه؟
    چون از ۵ چکِ vocab_audit.py، فقط ۴ تاشون (رکورد خراب، مترادف
    تایپی، تناقض skip، افتادنِ کلمه‌ی نقش‌ساز) پاسخ «درست» بدون ابهام
    دارند. چک شماره‌ی ۱ (واژه‌ی تکراری بین چند دسته) این‌طور نیست —
    اکثر آن ۳۴ مورد (مثل «تصرف» که هم فعل است هم مفهوم) پدیده‌ی طبیعی
    زبان حقوقی فارسی‌اند (مصدر عربی که هم به عمل اشاره می‌کند هم به
    حالت) و دست‌نخورده رها می‌شوند. فقط زیرمجموعه‌ای از آن‌ها که مستقیم
    ناشی از باگ «افتادن کلمه‌ی نقش‌ساز» بودند (دعوی، حواله، شکایت،
    قرارداد، مشروط) اینجا هم اصلاح می‌شوند — چون ریشه‌شان مشخص است.

    بقیه‌ی موارد چک ۱ (مثلاً آیا «تصویر»/«جواز»/«غرامت» باید فقط object
    بمانند یا هم concept) در پاسخ متنی جداگانه پیشنهاد داده شده‌اند، نه
    اینجا — چون قضاوت دامنه‌ای (domain judgment) لازم دارند که بهتره
    خودت با نگاه به داده‌ی واقعی پرونده‌هات تصمیم بگیری، نه یک قانون
    ثابت در کد.

قبل از هر تغییری، از کل پوشه‌ی categorized/ یک نسخه‌ی پشتیبان می‌گیرد.

نحوه‌ی اجرا:
    uv run -m features.vocab_fix
    uv run -m features.vocab_audit   # برای تأیید اینکه موارد رفع‌شده دیگه نیستن
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from features.configs import VOCAB_CATEGORIES

_HERE = Path(__file__).resolve().parent.parent  # graph-rag/
VOCAB_DIR = _HERE / "data" / "legal-vocabulary" / "categorized"

# --- ۱. رکورد خراب (حرف لاتین وسط واژه‌ی فارسی) ---
CORRUPTED_TO_REMOVE = {"action": ["جبران خسarat"]}

# --- ۲. ادغام مترادف‌های تایپی: {دسته: {حذف‌شونده -> نگه‌داشته‌شونده}} ---
TYPO_MERGES = {
    "concept": {"هم‌زمانی": "همزمانی"},
    "action": {"استعفاء": "استعفا", "پیش‌گیری": "پیشگیری"},
    "object": {"رای": "رأی"},
}

# --- ۳. تناقض skip/دسته: این‌ها دوباره skip می‌شوند (چون «ذاتی»/«ثابت»
#         صفت عمومی‌اند، نه یک principle/concept مشخص و قابل‌استناد) ---
SKIP_WINS = {"concept": ["ثابت"], "principle": ["ذاتی"]}

# --- ۴. حذف از role: یا مستقیماً ناشیِ باگِ افتادنِ «طرف»/«له»/«عليه»
#         هستند (تصادم با یک ریشه‌ی نامرتبط)، یا با یک role دقیق‌تر و
#         از قبل موجود کاملاً زائدند (سوم ≈ ثالث/شخص ثالث که در roles.json هست) ---
ROLE_REMOVE = ["دعوی", "حواله", "شکایت", "قرارداد", "مقابل", "مشروط", "سوم"]

# --- ۵. concept+object که تصمیم گرفتیم فقط object بمانند — همه‌شون
#         بیشتر «شیء/سند مشخص» هستن تا «مفهوم انتزاعی» (مثلاً
#         «ضمانت‌نامه» خودش پسوند «-نامه» یعنی سند دارد) ---
CONCEPT_REMOVE_KEEP_AS_OBJECT = [
    "تصویر", "جواز", "حصه", "ضمانت‌نامه", "طلب", "غرامت", "مالیات",
]

# --- ۶. concept+action که به یک دسته محدود می‌شوند (قضاوت دامنه‌ای؛
#         اگر با تجربه‌ی خودت روی داده‌ی واقعی فرق داشت، همینجا عوضش کن)
#
#     عمداً این‌ها اینجا نیستند و دوگانه می‌مانند، چون واقعاً هر دو
#     کاربرد در متن رأی رایج است، نه یک باگ:
#       - «تصرف» (هم فعل «تصرف کرد» هم مفهوم انتزاعی حقوق اموال)
#       - شش‌تای action+object (حکم/دادخواست/شکایت/مجوز/کمک/گواهی):
#         هم فعل صدور/ارائه مهم است هم خودِ سند
#
#     منطق انتخاب: اگر واژه یک وضعیت/نهاد/رابطه‌ی پایدار را نشان
#     می‌دهد -> concept. اگر یک عمل یک‌باره با یک انجام‌دهنده‌ی مشخص
#     را نشان می‌دهد -> action.
REMOVE_FROM = {
    # این‌ها را از action حذف می‌کنیم، فقط concept می‌مانند
    "action": ["افلاس", "بخشودگی", "عطف", "لطمه", "وکالت"],
    # این‌ها را از concept حذف می‌کنیم (۱۰ تا فقط action می‌مانند،
    # ۲ تا آخر یعنی بی‌طرفی/ثبات فقط principle می‌مانند)
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
    print(f"🗄️ نسخه‌ی پشتیبان در {backup_dir} ذخیره شد.")


def run():
    backup()

    # همه‌ی canon ها رو از قبل می‌خونیم -- چون قدم ۶ باید بتونه چک کنه
    # که وقتی یه واژه رو از دسته‌ی X حذف می‌کنیم، واقعاً توی یه دسته‌ی
    # دیگه هم هست (وگرنه معنیش کامل گم می‌شه)
    all_canon = {}
    for key in VOCAB_CATEGORIES:
        p = VOCAB_DIR / f"{key}s.json"
        if p.exists():
            all_canon[key] = set(_load(p))

    for key in VOCAB_CATEGORIES:
        canon_path = VOCAB_DIR / f"{key}s.json"
        alias_path = VOCAB_DIR / f"{key}s_aliases.json"
        if not canon_path.exists():
            print(f"⚠️ {canon_path.name} پیدا نشد — رد شد.")
            continue

        canon = _load(canon_path)
        aliases = _load(alias_path) if alias_path.exists() else {}
        changed = False

        # ۱. رکورد خراب
        for bad in CORRUPTED_TO_REMOVE.get(key, []):
            if bad in canon:
                canon.remove(bad)
                changed = True
                print(f"  🗑️ [{key}] رکورد خراب حذف شد: «{bad}»")
            for raws in aliases.values():
                if bad in raws:
                    raws.remove(bad)

        # ۲. ادغام مترادف تایپی
        for drop, keep in TYPO_MERGES.get(key, {}).items():
            if drop in canon:
                canon.remove(drop)
                changed = True
                print(f"  🔗 [{key}] ادغام شد: «{drop}» -> «{keep}»")
                if drop in aliases:
                    aliases.setdefault(keep, [])
                    aliases[keep].extend(aliases.pop(drop))

        # ۳. تناقض skip -> برگشت به skip
        for w in SKIP_WINS.get(key, []):
            if w in canon:
                canon.remove(w)
                changed = True
                print(f"  ↩️ [{key}] به skip برگردانده شد: «{w}»")
                aliases.pop(w, None)

        # ۶. concept/action دوگانه که به یک دسته محدود می‌شن (چک ایمنی:
        #    فقط اگه توی حداقل یه دسته‌ی دیگه هم واقعاً موجود باشه)
        for w in REMOVE_FROM.get(key, []):
            if w not in canon:
                continue
            exists_elsewhere = any(w in words for k2, words in all_canon.items() if k2 != key)
            if exists_elsewhere:
                canon.remove(w)
                changed = True
                print(f"  🗑️ [{key}] حذف شد (دسته‌ی دیگه نگه‌داشته می‌شود): «{w}»")
                aliases.pop(w, None)
            else:
                print(f"  ⚠️ [{key}] «{w}» جای دیگه‌ای پیدا نشد — حذف نشد "
                      f"(برای جلوگیری از گم‌شدن کامل این واژه)")

        # ۴. حذف نودهای role تصادمی/زائد
        if key == "role":
            for w in ROLE_REMOVE:
                if w in canon:
                    canon.remove(w)
                    changed = True
                    print(f"  🗑️ [role] حذف شد (تصادمی یا زائد): «{w}»")
                    aliases.pop(w, None)

        # ۵. concept+object -> فقط object (با چک ایمنی که واقعاً توی
        #    objects.json هم هست، وگرنه اصلاً حذفش نمی‌کنیم چون معنیش
        #    کامل از دست می‌ره)
        if key == "concept":
            objects_canon = _load(VOCAB_DIR / "objects.json") if (VOCAB_DIR / "objects.json").exists() else []
            for w in CONCEPT_REMOVE_KEEP_AS_OBJECT:
                if w in canon:
                    if w in objects_canon:
                        canon.remove(w)
                        changed = True
                        print(f"  🗑️ [concept] حذف شد (فقط object می‌ماند): «{w}»")
                        aliases.pop(w, None)
                    else:
                        print(f"  ⚠️ [concept] «{w}» در objects.json پیدا نشد — حذف نشد "
                              f"(برای جلوگیری از گم‌شدن کامل این مفهوم)")

        if changed:
            _save(canon_path, sorted(set(canon)))
            _save(alias_path, aliases)

    # به‌روزرسانی skipped.json با واژه‌هایی که برگشتند
    skipped_path = VOCAB_DIR / "skipped.json"
    skipped = _load(skipped_path) if skipped_path.exists() else []
    for words in SKIP_WINS.values():
        for w in words:
            if w not in skipped:
                skipped.append(w)
    _save(skipped_path, sorted(set(skipped)))

    print("\n✅ اصلاحات قطعی اعمال شد.")
    print("   دوباره اجرا کن: uv run -m features.vocab_audit")
    print("   (چک ۱ هنوز چند مورد نشون می‌ده -- اون‌ها عمداً دست‌نخورده موندن، نگاه کن به توضیح بالای فایل)")


if __name__ == "__main__":
    run()