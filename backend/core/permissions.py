from functools import wraps
from ninja.errors import HttpError
from django.conf import settings

def check_plan_limit(user) -> None:
    """
    بررسی سقف مجاز پرسش‌های روزانه کاربر بر اساس پلن فعال.
    """
    if not user.can_query():
        limit = settings.PLAN_LIMITS.get(user.plan, {}).get("daily_queries", 0)
        raise HttpError(
            429,
            f"سقف روزانه شما ({limit} پرسش) تمام شده است. "
            "برای ادامه لطفاً پلن خود را ارتقا دهید."
        )


def require_plan(minimum_plan: str):
    """
    دکوراتور کنترل سطح دسترسی به APIها بر اساس نوع پلن کاربری.
    """
    PLAN_ORDER = {"free": 0, "pro": 1, "enterprise": 2}

    def decorator(func):
        @wraps(func)  # برای حفظ مستندات و ساختار تابع در Swagger داکیومنت
        def wrapper(request, *args, **kwargs):
            if not request.user or not request.user.is_authenticated:
                raise HttpError(401, "احراز هویت انجام نشده است.")
                
            user_level = PLAN_ORDER.get(request.user.plan, 0)
            required_level = PLAN_ORDER.get(minimum_plan, 0)
            
            if user_level < required_level:
                raise HttpError(
                    403,
                    f"این قابلیت انحصاری است و نیاز به پلن {minimum_plan} یا بالاتر دارد."
                )
            return func(request, *args, **kwargs)
        return wrapper
    return decorator