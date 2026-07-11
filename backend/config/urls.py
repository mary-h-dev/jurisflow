from django.urls import path
from ninja import NinjaAPI
from django.contrib import admin

from apps.users.api import router as users_router
# from apps.search.api import router as search_router
# from apps.agents.api import router as agents_router

api = NinjaAPI(
    title="JurisFlow API",
    version="1.0.0",
    description="سیستم هوشمند تحلیل قوانین ایران",
)

api.add_router("/auth",   users_router,  tags=["Authentication"])
# api.add_router("/search", search_router, tags=["Search"])
# api.add_router("/agents", agents_router, tags=["Agents"])

urlpatterns = [
    path("api/", api.urls),
    path("admin/", admin.site.urls),
]


