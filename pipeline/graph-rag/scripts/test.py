from features.vocab_resolver import VocabResolver
r = VocabResolver()
print(r.resolve("توقف", "concept"))  # باید None باشه (اگه فقط «توقیف» در واژه‌نامه‌ست)
print(r.resolve("خوانده ردیف اول", "role"))  # باید "خوانده" برگردونه
print(r.resolve("توقیف", "concept"))