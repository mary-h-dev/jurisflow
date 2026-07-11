from django.core.management.base import BaseCommand
from apps.users.models import LegalField


LEGAL_FIELDS = [
    {"name": "حقوق مدنی",       "slug": "civil",       "icon": "scale",    "order": 1},
    {"name": "حقوق کیفری",      "slug": "criminal",    "icon": "gavel",    "order": 2},
    {"name": "حقوق خانواده",    "slug": "family",      "icon": "home",     "order": 3},
    {"name": "حقوق تجاری",      "slug": "commercial",  "icon": "briefcase","order": 4},
    {"name": "حقوق ملکی",       "slug": "property",    "icon": "building", "order": 5},
    {"name": "حقوق کار",        "slug": "labor",       "icon": "users",    "order": 6},
    {"name": "حقوق بین‌الملل",  "slug": "international","icon": "globe",   "order": 7},
    {"name": "حقوق اداری",      "slug": "administrative","icon": "file",   "order": 8},
    {"name": "حقوق جزایی", "slug": "procedural-criminal", "icon": "shield", "order": 9},
]


class Command(BaseCommand):
    help = "ساخت داده‌های اولیه زمینه‌های حقوقی"

    def handle(self, *args, **kwargs):
        created = 0
        updated = 0

        for field in LEGAL_FIELDS:
            obj, is_created = LegalField.objects.update_or_create(
                slug=field["slug"],
                defaults={
                    "name":  field["name"],
                    "icon":  field["icon"],
                    "order": field["order"],
                },
            )
            if is_created:
                created += 1
            else:
                updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"✅ {created} زمینه جدید ساخته شد، {updated} زمینه آپدیت شد."
            )
        )