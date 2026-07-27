"""
features/vocabulary_categorizer.py — دسته‌بندیِ یک‌بارِ واژه‌نامه‌ی حقوقی
                                      به closed vocabulary برای هر Feature

چرا این کار لازم است؟
    data/legal-vocabulary/legal_vocabulary.json یک دیکشنری عمومی حقوقی
    است (۲۴۱۶ رکورد {واژه, معنی})، نه یک لیست دسته‌بندی‌شده. طرح اصلی
    می‌گفت «LLM فقط از بین Conceptهای موجود انتخاب کند» — این فقط وقتی
    درست کار می‌کند که از قبل بدانیم کدام واژه Concept است، کدام Action،
    کدام Role، کدام Object. این اسکریپت دقیقاً همان کار را می‌کند —
    **یک‌بار**، نه به‌ازای هر پرونده.

چرا هر واژه یک «ریشه» (canonical form) هم می‌گیرد؟
    چون واژه‌نامه پر از مترادف/زیرشاخه است: «بیع»، «عقد بیع»،
    «قرارداد بیع» نباید سه node جدا در گراف شوند. به‌جای یک pass دوم
    جداگانه برای ادغام (که یعنی دوباره هزینه‌ی LLM روی کل واژه‌نامه)،
    همین‌جا از LLM خواسته می‌شود همزمان با دسته‌بندی، ریشه‌ی هر واژه را
    هم مشخص کند. اگر واژه‌ای خودش ریشه است، «ریشه» برابر خودِ واژه
    برمی‌گردد. خروجی نهایی (export_categorized) بر اساس این ریشه‌ها
    دسته‌بندی و یکتاسازی می‌شود؛ نگاشت alias→ریشه هم جداگانه نگه
    داشته می‌شود تا در extractor.py بعداً بشه واژه‌ی خام متن را به
    node درست در گراف وصل کرد.

چرا batch (نه یکی‌یکی و نه همه‌باهم)؟
    یکی‌یکی = ۲۴۱۶ فراخوانی LLM، گران و کند.
    همه‌باهم = یک پرامپت غول‌آسا با ریسک بالای خطای JSON و از دست رفتن
               کل نتیجه با یک خطا.
    پس در دسته‌های VOCAB_BATCH_SIZE‌تایی (پیش‌فرض ۶۰) کار می‌کنیم —
    حدود ۴۰ فراخوانی برای کل واژه‌نامه.

چرا resumable با checkpoint (نه فقط یک خروجی نهایی)؟
    اگر روی batch شماره‌ی ۳۰ از ۴۰ خطا بگیریم (rate limit، قطعی شبکه...)،
    نمی‌خواهیم ۲۹ batch قبلی که هزینه‌ش پرداخت شده از دست برود. هر batch
    که کامل شد، بلافاصله در یک فایل jsonl append می‌شود؛ اجرای مجدد
    اسکریپت واژه‌هایی که قبلاً دسته‌بندی شده‌اند را رد می‌کند.

خروجی نهایی:
    data/legal-vocabulary/categorized/{concepts,actions,roles,objects}.json
        هرکدوم یک لیست ساده از رشته‌های *ریشه* (canonical، نه همه‌ی
        alias های خام) — همون چیزی که extractor.py بعداً به‌عنوان
        closed vocabulary در پرامپت LLM استفاده می‌کند.
    data/legal-vocabulary/categorized/{concepts,actions,roles,objects}_aliases.json
        نگاشت ریشه → لیست alias های خام (برای هر دسته جدا)، مثلاً
        {"بیع": ["بیع", "عقد بیع", "قرارداد بیع"]}. این برای
        قابل‌ردیابی بودن نگه داشته می‌شود، نه برای استفاده‌ی مستقیم در
        پرامپت.
    data/legal-vocabulary/categorized/skipped.json
        واژه‌هایی که «مفید نیستند» تشخیص داده شدند — نگه داشته می‌شود
        فقط برای مرور دستی و اطمینان از درستی تصمیم LLM، نه برای استفاده
        در pipeline.

نحوه‌ی اجرا:
    uv run -m features.vocabulary_categorizer
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv

from features.configs import (
    ALL_VOCAB_LABELS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SKIP_LABEL,
    VOCAB_BATCH_SIZE,
    VOCAB_CATEGORIES,
    VOCAB_MEANING_CHAR_CAP,
    get_llm_client,
)

load_dotenv()

_HERE = Path(__file__).resolve().parent.parent  # graph-rag/
VOCAB_SOURCE = _HERE / "data" / "legal-vocabulary" / "legal_vocabulary.json"
OUTPUT_DIR = _HERE / "data" / "legal-vocabulary" / "categorized"
CHECKPOINT_PATH = OUTPUT_DIR / "_checkpoint.jsonl"

_client = get_llm_client()


PROMPT_TEMPLATE = """
تو یک دستیار حقوقی متخصص در قوانین ایران هستی. وظیفه‌ات دسته‌بندی
واژه‌های زیر (از یک دیکشنری حقوقی) به یکی از دسته‌های مشخص‌شده است.

دسته‌ها:
{categories_desc}

- "{skip_label}": اگر واژه یک اصطلاح فنی حقوقیِ مفید برای دسته‌بندی
  پرونده‌ها نیست (مثلاً یک واژه‌ی صرفاً زبانی/عمومی است، یا خیلی کلی/
  نامشخص است که در هیچ‌کدام از دسته‌های بالا جا نمی‌شود).

علاوه بر دسته، برای هر واژه یک «ریشه» (canonical form) هم مشخص کن:
- اگر چند واژه در همین لیست مترادف یا زیرشاخه‌ی همدیگرند (مثلاً «بیع»،
  «عقد بیع»، «قرارداد بیع»)، همه باید یک «ریشه» یکسان بگیرند — کوتاه‌ترین
  و رایج‌ترین شکل اصطلاح را به‌عنوان ریشه انتخاب کن (مثلاً «بیع»).
- اگر واژه‌ای مترادف/زیرشاخه‌ی هیچ واژه‌ی دیگری در همین لیست نیست،
  «ریشه» همان خودِ واژه است.
- برای واژه‌های "{skip_label}"، «ریشه» را برابر خودِ واژه بگذار
  (استفاده نمی‌شود، ولی فیلد باید پر باشد).

قوانین مهم:
- هر واژه دقیقاً یک دسته و یک ریشه می‌گیرد.
- خروجی باید شامل *دقیقاً* همان تعداد و همان ترتیب واژه‌های ورودی باشد.
- فقط از برچسب‌های داده‌شده استفاده کن: {all_labels}

واژه‌ها (به همراه معنی برای رفع ابهام):
{words_block}

فقط JSON برگردون — بدون هیچ توضیح اضافه، به این فرم:
[
  {{"واژه": "...", "دسته": "...", "ریشه": "..."}},
  ...
]
"""


def _build_categories_desc() -> str:
    lines = []
    for cat in VOCAB_CATEGORIES.values():
        examples = "، ".join(cat.examples)
        lines.append(f'- "{cat.key}" ({cat.label_fa}): {cat.description} مثال: {examples}.')
    return "\n".join(lines)


_CATEGORIES_DESC = _build_categories_desc()


def load_vocabulary() -> list[dict]:
    """
    واژه‌نامه‌ی خام را می‌خواند و واژه‌های تکراری را ادغام می‌کند.

    چرا این ادغام لازم است؟
        در legal_vocabulary.json حدود ۹۰ واژه (مثل «خلاف»، «تصرف») بیش
        از یک‌بار با تعریف‌های متفاوت آمده‌اند. اگر این رکوردهای تکراری
        مستقل به LLM فرستاده شوند، ممکن است برچسب‌های متفاوتی بگیرند
        (چون هرکدام تعریف متفاوتی می‌بینند) — و چون checkpoint بر اساس
        «واژه» dict می‌سازد، یکی از این نتایج بی‌صدا توسط دیگری overwrite
        می‌شود (یک واژه، دو تصمیمِ متناقض، فقط آخری باقی می‌ماند). برای
        جلوگیری از این از دست‌رفتنِ بی‌صدا، همه‌ی تعریف‌های یک واژه قبل
        از ارسال به LLM با «؛» ادغام می‌شوند تا مدل یک تصمیم واحد و
        آگاه از هر دو معنی بگیرد.
    """
    with open(VOCAB_SOURCE, encoding="utf-8") as f:
        raw = json.load(f)

    merged: dict[str, list[str]] = {}
    for row in raw:
        merged.setdefault(row["واژه"], [])
        meaning = row["معنی"].strip()
        if meaning and meaning not in merged[row["واژه"]]:
            merged[row["واژه"]].append(meaning)

    duplicates = {w: ms for w, ms in merged.items() if len(ms) > 1}
    if duplicates:
        print(f"ℹ️ {len(duplicates)} واژه‌ی تکراری در منبع پیدا و ادغام شد "
              f"(مثال: {list(duplicates.keys())[:3]})")

    return [
        {"واژه": word, "معنی": " ؛ ".join(meanings)}
        for word, meanings in merged.items()
    ]


def load_checkpoint() -> dict[str, dict]:
    """واژه‌هایی که قبلاً دسته‌بندی شده‌اند را برمی‌گرداند: {واژه: {"دسته", "ریشه"}}"""
    if not CHECKPOINT_PATH.exists():
        return {}
    done: dict[str, dict] = {}
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[row["واژه"]] = {"دسته": row["دسته"], "ریشه": row["ریشه"]}
    return done


def append_checkpoint(rows: list[dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _batch(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def classify_batch(batch: list[dict], retries: int = 2) -> list[dict]:
    """
    یک batch از واژه‌ها را به LLM می‌دهد و لیست {"واژه", "دسته"} برمی‌گرداند.
    اگر تعداد یا محتوای خروجی با ورودی نخواند، batch را نامعتبر می‌داند
    و به‌جای حدس زدن، آن را (پس از retry) با خطا گزارش می‌کند — تا داده‌ی
    نادرست بی‌صدا وارد closed vocabulary نشود.
    """
    words_block = "\n".join(
        f'{idx+1}. {row["واژه"]}: {row["معنی"][:VOCAB_MEANING_CHAR_CAP]}'
        for idx, row in enumerate(batch)
    )
    prompt = PROMPT_TEMPLATE.format(
        categories_desc=_CATEGORIES_DESC,
        skip_label=SKIP_LABEL,
        all_labels="، ".join(ALL_VOCAB_LABELS),
        words_block=words_block,
    )

    expected_words = [row["واژه"] for row in batch]
    last_error = None

    for attempt in range(retries + 1):
        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)

            got_words = [row["واژه"] for row in parsed]
            if got_words != expected_words:
                raise ValueError(
                    f"عدم تطابق ترتیب/تعداد واژه‌ها "
                    f"(انتظار {len(expected_words)}, دریافت {len(got_words)})"
                )
            for row in parsed:
                if row["دسته"] not in ALL_VOCAB_LABELS:
                    raise ValueError(f"برچسب نامعتبر: {row['دسته']}")
                if not row.get("ریشه", "").strip():
                    raise ValueError(f"فیلد «ریشه» خالی برای واژه‌ی «{row['واژه']}»")

            return parsed

        except Exception as e:  # noqa: BLE001 — می‌خوایم خطای JSON/schema هم بگیریم
            last_error = e
            if attempt < retries:
                print(f"  ⏳ خطا در batch، تلاش دوباره ({attempt + 1}/{retries}): {e}")
                time.sleep(3)

    raise RuntimeError(f"❌ batch شکست خورد بعد از {retries + 1} تلاش: {last_error}")


def audit_checkpoint_conflicts() -> dict[str, list[dict]]:
    """
    checkpoint خام (jsonl، پیش از dict-collapse) را می‌خواند و واژه‌هایی
    را که بیش از یک بار با «دسته» یا «ریشه» متفاوت ثبت شده‌اند گزارش
    می‌کند. مخصوصاً برای checkpointهایی که با نسخه‌ی قبل از رفع باگِ
    ادغام واژه‌های تکراری (load_vocabulary) ساخته شده‌اند — تا قبل از
    export_categorized() بشود دستی تصمیم گرفت کدام برچسب درست است.
    """
    if not CHECKPOINT_PATH.exists():
        print("⚠️ فایل checkpoint پیدا نشد.")
        return {}

    seen: dict[str, list[dict]] = {}
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seen.setdefault(row["واژه"], []).append(row)

    conflicts = {
        w: rows for w, rows in seen.items()
        if len({(r["دسته"], r["ریشه"]) for r in rows}) > 1
    }

    if not conflicts:
        print("✅ هیچ تناقضی در checkpoint پیدا نشد.")
    else:
        print(f"⚠️ {len(conflicts)} واژه با برچسب‌های متناقض پیدا شد "
              f"(در export_categorized فعلی فقط آخرین رکورد باقی می‌ماند):")
        for w, rows in list(conflicts.items())[:20]:
            labels = " | ".join(f'{r["دسته"]}/{r["ریشه"]}' for r in rows)
            print(f"  - {w}: {labels}")
        if len(conflicts) > 20:
            print(f"  ... و {len(conflicts) - 20} مورد دیگر")

    return conflicts


def run(limit: int | None = None):
    vocabulary = load_vocabulary()
    if limit:
        vocabulary = vocabulary[:limit]

    already_done = load_checkpoint()
    remaining = [row for row in vocabulary if row["واژه"] not in already_done]

    print(f"📚 کل واژه‌نامه: {len(vocabulary)} | قبلاً دسته‌بندی‌شده: {len(already_done)} "
          f"| باقی‌مانده: {len(remaining)}")

    if not remaining:
        print("✅ همه‌ی واژه‌ها قبلاً دسته‌بندی شده‌اند. برو سراغ export_categorized().")
        return

    total_batches = (len(remaining) + VOCAB_BATCH_SIZE - 1) // VOCAB_BATCH_SIZE
    for i, batch in enumerate(_batch(remaining, VOCAB_BATCH_SIZE), start=1):
        print(f"  🔎 batch {i}/{total_batches} ({len(batch)} واژه)...")
        try:
            results = classify_batch(batch)
        except RuntimeError as e:
            print(f"  ⚠️ رد شد و در اجرای بعدی دوباره تلاش می‌شود: {e}")
            continue
        append_checkpoint(results)
        time.sleep(0.5)  # فاصله‌ی کوچک برای رعایت rate limit

    print("✅ دسته‌بندی تمام batchها انجام شد (یا برای اجرای بعدی صف شد).")


def export_categorized():
    """
    checkpoint خام (jsonl، یک ردیف به‌ازای هر واژه‌ی خام) را می‌خواند و
    بر اساس «ریشه» یکتاسازی می‌کند — یعنی خروجی نهایی لیستِ *مفاهیمِ
    ریشه* است (مثلاً «بیع»)، نه همه‌ی ۲۴۱۶ واژه‌ی خام (مثلاً «بیع»،
    «عقد بیع»، «قرارداد بیع» هرسه). به‌ازای هر دسته دو فایل تولید
    می‌شود: لیست ریشه‌ها (برای استفاده در پرامپت extractor.py) و
    نگاشت ریشه→alias ها (برای ردیابی/دیباگ).
    """
    done = load_checkpoint()
    if not done:
        print("⚠️ هنوز چیزی دسته‌بندی نشده. اول run() را اجرا کن.")
        return

    # buckets[category][ریشه] = [alias1, alias2, ...]
    buckets: dict[str, dict[str, list[str]]] = {key: {} for key in VOCAB_CATEGORIES}
    skipped: list[str] = []

    for word, info in done.items():
        label = info["دسته"]
        root = info["ریشه"]
        if label == SKIP_LABEL:
            skipped.append(word)
        elif label in buckets:
            buckets[label].setdefault(root, []).append(word)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n📊 گزارش نهایی (بعد از ادغام مترادف‌ها/aliasها):")
    total_roots = 0
    for key, root_map in buckets.items():
        canonical_list = sorted(root_map.keys())
        total_roots += len(canonical_list)

        canon_path = OUTPUT_DIR / f"{key}s.json"
        with open(canon_path, "w", encoding="utf-8") as f:
            json.dump(canonical_list, f, ensure_ascii=False, indent=2)

        alias_path = OUTPUT_DIR / f"{key}s_aliases.json"
        with open(alias_path, "w", encoding="utf-8") as f:
            json.dump({r: sorted(a) for r, a in sorted(root_map.items())}, f,
                       ensure_ascii=False, indent=2)

        raw_count = sum(len(a) for a in root_map.values())
        print(f"  💾 {canon_path.name}: {len(canonical_list)} مفهوم ریشه "
              f"(از {raw_count} واژه‌ی خام)")

    with open(OUTPUT_DIR / "skipped.json", "w", encoding="utf-8") as f:
        json.dump(sorted(skipped), f, ensure_ascii=False, indent=2)
    print(f"  💾 skipped.json: {len(skipped)} واژه (نادیده گرفته‌شده)")

    print(f"\n🎯 جمع کل مفاهیم ریشه در همه‌ی دسته‌ها: {total_roots} "
          f"(هدف تقریبی پیشنهادشده: ۴۰۰-۶۰۰ — این فقط یک راهنماست، "
          f"نه یک قانون؛ خودت با نگاه به aliasها قضاوت کن)")


if __name__ == "__main__":
    run()
    export_categorized()