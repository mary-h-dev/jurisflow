"""
features/vocabulary_categorizer.py — one-time categorization of the legal
                                      vocabulary into a closed vocabulary
                                      per Feature category

Why is this needed?
    data/legal-vocabulary/legal_vocabulary.json is a general legal
    dictionary (2416 {term, meaning} records), not a categorized list.
    The original design said "the LLM must only choose among existing
    concepts" — this only works if we already know which term is a
    Concept, which is an Action, which is a Role, which is an Object.
    This script does exactly that — **once**, not per case.

Why does every term also get a "root" (canonical form)?
    Because the dictionary is full of synonyms/variants: "بیع", "عقد
    بیع", "قرارداد بیع" should not become three separate graph nodes.
    Instead of a separate second pass just for merging (which means
    paying the LLM cost over the whole vocabulary again), the LLM is
    asked to determine each term's root at the same time as
    categorizing it. If a term is itself the root, "root" equals the
    term itself. The final output (export_categorized) is categorized
    and deduplicated based on these roots; the alias→root mapping is
    also kept separately so that extractor.py can later connect a raw
    term from the text to the correct graph node.

Why batching (neither one-by-one nor all at once)?
    One-by-one = 2416 LLM calls, expensive and slow.
    All at once = one giant prompt with a high risk of a JSON error and
                  losing the entire result to a single mistake.
    So we work in batches of VOCAB_BATCH_SIZE (default 60) —
    about 40 calls for the whole vocabulary.

Why resumable with a checkpoint (not just one final output)?
    If we hit an error on batch 30 of 40 (rate limit, network outage,
    ...), we don't want to lose the 29 previous batches we already paid
    for. As soon as a batch completes, it's immediately appended to a
    jsonl file; re-running the script skips terms already categorized.

Note on JSON keys: the checkpoint/output records use the Persian keys
"واژه" (term), "دسته" (category), "ریشه" (root) throughout this file —
kept as-is rather than translated to English, since a real checkpoint
file on disk already uses this schema; renaming would break resuming
from it without a migration step.

Final output:
    data/legal-vocabulary/categorized/{concepts,actions,roles,objects}.json
        each a plain list of *root* strings (canonical, not all raw
        aliases) — exactly what extractor.py later uses as the closed
        vocabulary in the LLM prompt.
    data/legal-vocabulary/categorized/{concepts,actions,roles,objects}_aliases.json
        root → list of raw aliases mapping (per category), e.g.
        {"بیع": ["بیع", "عقد بیع", "قرارداد بیع"]}. Kept for
        traceability, not for direct use in the prompt.
    data/legal-vocabulary/categorized/skipped.json
        terms judged "not useful" — kept only for manual review and to
        confirm the LLM's decision was correct, not for use in the
        pipeline.

Usage:
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


# NOTE: instructional text translated to English below; the category
# descriptions/examples embedded via {categories_desc} come from
# configs.py and stay in Persian (they're the actual Persian legal
# vocabulary content being categorized). The requested output keys
# ("واژه"/"دسته"/"ریشه") are also left as-is — see module docstring.
PROMPT_TEMPLATE = """
You are a legal assistant specialized in Iranian law. Your task is to
categorize the following terms (from a legal dictionary) into one of the
categories specified below.

Categories:
{categories_desc}

- "{skip_label}": if the term is not a useful legal/technical term for
  categorizing cases (e.g. it is a purely linguistic/general word, or is
  too vague/unclear to fit any of the categories above).

In addition to the category, also determine a "root" (canonical form)
for each term:
- If several terms in this list are synonyms or variants of one another
  (e.g. "بیع", "عقد بیع", "قرارداد بیع"), they must all get the same
  root — choose the shortest, most common form of the term as the root
  (e.g. "بیع").
- If a term is not a synonym/variant of any other term in this list,
  its root is the term itself.
- For "{skip_label}" terms, set the root equal to the term itself
  (unused, but the field must still be filled).

Important rules:
- Every term gets exactly one category and one root.
- The output must contain *exactly* the same number and order of terms
  as the input.
- Only use the given labels: {all_labels}

Terms (with their meaning, to resolve ambiguity):
{words_block}

Return only JSON — no extra explanation — in exactly this form:
[
  {{"واژه": "...", "دسته": "...", "ریشه": "..."}},
  ...
]
"""


def _build_categories_desc() -> str:
    lines = []
    for cat in VOCAB_CATEGORIES.values():
        examples = "، ".join(cat.examples)
        lines.append(f'- "{cat.key}" ({cat.label_fa}): {cat.description} Examples: {examples}.')
    return "\n".join(lines)


_CATEGORIES_DESC = _build_categories_desc()


def load_vocabulary() -> list[dict]:
    """
    Reads the raw vocabulary and merges duplicate terms.

    Why is this merge needed?
        In legal_vocabulary.json, about 90 terms (e.g. "خلاف", "تصرف")
        appear more than once with different definitions. If these
        duplicate records were sent to the LLM independently, they might
        get different labels (since each sees a different definition) —
        and since the checkpoint is keyed by "term", one of these
        results silently overwrites the other (one term, two
        contradictory decisions, only the last one survives). To
        prevent this silent data loss, all definitions of a term are
        merged with "؛" before sending to the LLM, so the model makes a
        single decision informed by both meanings.
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
        print(f"ℹ️ Found and merged {len(duplicates)} duplicate terms in the source "
              f"(example: {list(duplicates.keys())[:3]})")

    return [
        {"واژه": word, "معنی": " ؛ ".join(meanings)}
        for word, meanings in merged.items()
    ]


def load_checkpoint() -> dict[str, dict]:
    """Returns terms already categorized: {term: {"دسته", "ریشه"}}"""
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
    Sends a batch of terms to the LLM and returns a list of
    {"واژه", "دسته", "ریشه"}. If the count or content of the output
    doesn't match the input, the batch is considered invalid and — rather
    than guessing — reported as an error (after retrying), so incorrect
    data doesn't silently enter the closed vocabulary.
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
                    f"Term count/order mismatch "
                    f"(expected {len(expected_words)}, got {len(got_words)})"
                )
            for row in parsed:
                if row["دسته"] not in ALL_VOCAB_LABELS:
                    raise ValueError(f"Invalid label: {row['دسته']}")
                if not row.get("ریشه", "").strip():
                    raise ValueError(f"Empty «root» field for term «{row['واژه']}»")

            return parsed

        except Exception as e:  # noqa: BLE001 — JSON/schema errors should be caught here too
            last_error = e
            if attempt < retries:
                print(f"  ⏳ Batch error, retrying ({attempt + 1}/{retries}): {e}")
                time.sleep(3)

    raise RuntimeError(f"❌ Batch failed after {retries + 1} attempts: {last_error}")


def audit_checkpoint_conflicts() -> dict[str, list[dict]]:
    """
    Reads the raw checkpoint (jsonl, before dict-collapse) and reports
    terms recorded more than once with a different category or root.
    Mainly relevant for checkpoints built with a version predating the
    duplicate-term-merge bugfix (load_vocabulary) — so a manual decision
    can be made about which label is correct before export_categorized().
    """
    if not CHECKPOINT_PATH.exists():
        print("⚠️ Checkpoint file not found.")
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
        print("✅ No conflicts found in the checkpoint.")
    else:
        print(f"⚠️ Found {len(conflicts)} terms with conflicting labels "
              f"(export_categorized currently keeps only the last record):")
        for w, rows in list(conflicts.items())[:20]:
            labels = " | ".join(f'{r["دسته"]}/{r["ریشه"]}' for r in rows)
            print(f"  - {w}: {labels}")
        if len(conflicts) > 20:
            print(f"  ... and {len(conflicts) - 20} more")

    return conflicts


def run(limit: int | None = None):
    vocabulary = load_vocabulary()
    if limit:
        vocabulary = vocabulary[:limit]

    already_done = load_checkpoint()
    remaining = [row for row in vocabulary if row["واژه"] not in already_done]

    print(f"📚 Full vocabulary: {len(vocabulary)} | already categorized: {len(already_done)} "
          f"| remaining: {len(remaining)}")

    if not remaining:
        print("✅ All terms already categorized. Go run export_categorized().")
        return

    total_batches = (len(remaining) + VOCAB_BATCH_SIZE - 1) // VOCAB_BATCH_SIZE
    for i, batch in enumerate(_batch(remaining, VOCAB_BATCH_SIZE), start=1):
        print(f"  🔎 batch {i}/{total_batches} ({len(batch)} terms)...")
        try:
            results = classify_batch(batch)
        except RuntimeError as e:
            print(f"  ⚠️ Skipped, will retry on next run: {e}")
            continue
        append_checkpoint(results)
        time.sleep(0.5)  # small delay to respect rate limits

    print("✅ All batches categorized (or queued for the next run).")


def export_categorized():
    """
    Reads the raw checkpoint (jsonl, one row per raw term) and
    deduplicates based on "root" — i.e. the final output is a list of
    *root* concepts (e.g. "بیع"), not all 2416 raw terms (e.g. "بیع",
    "عقد بیع", "قرارداد بیع" all three). Two files are produced per
    category: the list of roots (for use in extractor.py's prompt) and
    the root→alias mapping (for traceability/debugging).
    """
    done = load_checkpoint()
    if not done:
        print("⚠️ Nothing categorized yet. Run run() first.")
        return

    # buckets[category][root] = [alias1, alias2, ...]
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
    print("\n📊 Final report (after merging synonyms/aliases):")
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
        print(f"  💾 {canon_path.name}: {len(canonical_list)} root concepts "
              f"(from {raw_count} raw terms)")

    with open(OUTPUT_DIR / "skipped.json", "w", encoding="utf-8") as f:
        json.dump(sorted(skipped), f, ensure_ascii=False, indent=2)
    print(f"  💾 skipped.json: {len(skipped)} terms (ignored)")

    print(f"\n🎯 Total root concepts across all categories: {total_roots} "
          f"(suggested rough target: 400-600 — this is only a guide, "
          f"not a rule; use your own judgment looking at the aliases)")


if __name__ == "__main__":
    run()
    export_categorized()