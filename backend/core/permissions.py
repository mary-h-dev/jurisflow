from ninja.errors import HttpError
from django.conf import settings


def check_plan_limit(user) -> None:
    """
    فقط چک می‌کنه — increment نمی‌کنه.
    increment در api.py با can_query_and_increment انجام میشه.
    """
    from django.utils import timezone
    limit = settings.PLAN_LIMITS.get(user.plan, {}).get("daily_queries", 0)
    if limit == -1:
        return

    today = timezone.now().date()
    count = user.daily_query_count if user.last_query_date == today else 0

    if count >= limit:
        raise HttpError(
            429,
            f"سقف روزانه شما ({limit} پرسش) تمام شده. "
            "برای ادامه پلن خود را ارتقا دهید."
        )


def require_plan(minimum_plan: str):
    PLAN_ORDER = {"free": 0, "pro": 1, "enterprise": 2}

    def decorator(func):
        def wrapper(request, *args, **kwargs):
            user_level     = PLAN_ORDER.get(request.user.plan, 0)
            required_level = PLAN_ORDER.get(minimum_plan, 0)
            if user_level < required_level:
                raise HttpError(
                    403,
                    f"این قابلیت نیاز به پلن {minimum_plan} دارد."
                )
            return func(request, *args, **kwargs)
        return wrapper
    return decorator