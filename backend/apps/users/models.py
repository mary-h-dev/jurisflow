from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models, transaction
from django.utils import timezone


class UserManager(BaseUserManager):
    def create_user(self, email: str, password: str, **extra):
        if not email:
            raise ValueError("ایمیل الزامیست.")
        email = self.normalize_email(email)
        user  = self.model(email=email, **extra)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email: str, password: str, **extra):
        extra.setdefault("is_staff", True)
        extra.setdefault("is_superuser", True)
        return self.create_user(email, password, **extra)




class User(AbstractBaseUser, PermissionsMixin):

    PLAN_CHOICES = [
        ("free",       "رایگان"),
        ("pro",        "حرفه‌ای"),
        ("enterprise", "سازمانی"),
    ]

    email      = models.EmailField(unique=True)
    full_name  = models.CharField(max_length=120, blank=True)
    plan       = models.CharField(max_length=20, choices=PLAN_CHOICES, default="free")
    is_active  = models.BooleanField(default=True)
    is_staff   = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    daily_query_count = models.IntegerField(default=0)
    last_query_date   = models.DateField(null=True, blank=True)

    objects = UserManager()

    USERNAME_FIELD  = "email"
    REQUIRED_FIELDS = []

    class Meta:
        verbose_name        = "کاربر"
        verbose_name_plural = "کاربران"

    def __str__(self):
        return self.email

    def can_query_and_increment(self) -> bool:
        from django.conf import settings
        limit = settings.PLAN_LIMITS.get(self.plan, {}).get("daily_queries", 0)
        if limit == -1:
            return True

        today = timezone.now().date()

        with transaction.atomic():
            user = User.objects.select_for_update().get(pk=self.pk)

            if user.last_query_date != today:
                user.daily_query_count = 0
                user.last_query_date   = today

            if user.daily_query_count >= limit:
                return False

            user.daily_query_count += 1
            user.save(update_fields=["daily_query_count", "last_query_date"])

            self.daily_query_count = user.daily_query_count
            self.last_query_date   = user.last_query_date
            return True

    def get_daily_usage(self) -> dict:
        from django.conf import settings
        today = timezone.now().date()
        count = self.daily_query_count if self.last_query_date == today else 0
        limit = settings.PLAN_LIMITS.get(self.plan, {}).get("daily_queries", 0)
        return {
            "used":         count,
            "limit":        limit,
            "remaining":    max(0, limit - count) if limit != -1 else -1,
            "is_unlimited": limit == -1,
        }


class LegalField(models.Model):
    """زمینه‌های حقوقی — dynamic و قابل توسعه"""

    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True)
    icon = models.CharField(max_length=50, blank=True)   # برای frontend
    order = models.PositiveIntegerField(default=0)       # ترتیب نمایش
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name        = "زمینه حقوقی"
        verbose_name_plural = "زمینه‌های حقوقی"
        ordering            = ["order", "name"]

    def __str__(self):
        return self.name


class UserProfile(models.Model):
    """پروفایل کاربر — جدا از User برای Single Responsibility"""

    ROLE_CHOICES = [
        ("lawyer", "وکیل"),
        ("client", "متقاضی مشاوره"),
    ]

    ONBOARDING_STEPS = [
        ("1",    "ثبت‌نام"),
        ("2",    "تعیین نقش"),
        ("3",    "انتخاب زمینه‌ها"),
        ("done", "تکمیل‌شده"),
    ]

    user                  = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    role                  = models.CharField(max_length=20, choices=ROLE_CHOICES, blank=True)
    legal_fields          = models.ManyToManyField(LegalField, blank=True, related_name="users")
    onboarding_step       = models.CharField(max_length=10, choices=ONBOARDING_STEPS, default="1")
    is_onboarding_complete = models.BooleanField(default=False)
    updated_at            = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name        = "پروفایل"
        verbose_name_plural = "پروفایل‌ها"

    def __str__(self):
        return f"پروفایل {self.user.email}"