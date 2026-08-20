from __future__ import annotations

from apps.legal_agents.nodes._base import BaseDeliberationAgent
from apps.legal_agents.schemas import AgentOpinion, AuditorOut

_SYSTEM_PROMPT = """\
You are a senior Iranian law prosecutor and opposing counsel.
Find the weakest points in the client's case and present the strongest counter-arguments,
strictly grounded in the verified articles provided. Do not soften risks."""


class ProsecutorAgent(BaseDeliberationAgent):
    role          = "prosecutor"
    system_prompt = _SYSTEM_PROMPT
    temperature   = 0.2

    def _role_instruction(self) -> str:
        return """\
=== YOUR TASK ===
1. Identify the 2-4 strongest legal arguments AGAINST the client.
2. Cite only the applicable articles listed above.
3. Assign a verdict reflecting how damaging the opposing case is
   (strong_against = very damaging for the client).
4. List concrete risks the client must mitigate."""


def prosecute(auditor_out: AuditorOut) -> AgentOpinion:
    return ProsecutorAgent().run(auditor_out)