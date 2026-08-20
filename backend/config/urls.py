from django.urls import path
from ninja import NinjaAPI
from django.contrib import admin

from apps.users.api import router as users_router
from apps.search.api import router as search_router
from apps.auditor.api import router as auditor_router
from apps.legal_agents.api import router as agents_router


api = NinjaAPI(
    title="JurisFlow API",
    version="1.0.0",
    description="Intelligent System for Iranian Law Analysis",
)

api.add_router("/auth",   users_router,  tags=["Authentication"])
api.add_router("/search", search_router, tags=["Search"])
api.add_router("/auditor", auditor_router, tags=["Auditor"])
api.add_router("/legal-agents", agents_router, tags=["Agents"])



urlpatterns = [
    path("api/", api.urls),
    path("admin/", admin.site.urls),
]


