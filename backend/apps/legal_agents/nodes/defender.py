from __future__ import annotations

from apps.legal_agents.nodes._base import BaseDeliberationAgent
from apps.legal_agents.schemas import AgentOpinion, AuditorOut

_SYSTEM_PROMPT = """\
You are a senior Iranian law defence counsel.
Build the strongest possible case FOR the client using only the verified articles provided.
Be rigorous — only claim a strong verdict when the law genuinely supports it."""


class DefenderAgent(BaseDeliberationAgent):
    role          = "defender"
    system_prompt = _SYSTEM_PROMPT
    temperature   = 0.2

    def _role_instruction(self) -> str:
        return """\
=== YOUR TASK ===
1. Identify the 2-4 strongest legal arguments IN FAVOUR of the client.
2. Cite only the applicable articles listed above.
3. Assign a verdict that honestly reflects how strong the defence is.
4. List any procedural or evidentiary risks the defence must manage."""


def defend(auditor_out: AuditorOut) -> AgentOpinion:
    return DefenderAgent().run(auditor_out)