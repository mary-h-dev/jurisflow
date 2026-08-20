from django.apps import AppConfig
from django.conf import settings

from apps.legal_agents.nodes._base import init_client
from apps.legal_agents.nodes._base import LLMClient



class LegalAgentsConfig(AppConfig):
    name            = "apps.legal_agents"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self) -> None:
        init_client(
            LLMClient(
                api_key  = settings.OPENROUTER_API_KEY,
                base_url = getattr(
                    settings,
                    "OPENROUTER_BASE_URL",
                    "https://openrouter.ai/api/v1",
                ),
            )
        )


