
from typing import Any

from fuzzy.handlers.attacks.proto import BaseAttackTechniqueHandlerProto
from fuzzy.handlers.db.adv_prompts import AdversarialPromptDTO


class AdversarialSuffixAttackTechniqueHandler(BaseAttackTechniqueHandlerProto):
    def __init__(self, **extra: Any) -> None:
        super().__init__(**extra)

    async def attack(self, prompts: list[AdversarialPromptDTO], **extra: Any) -> None:
        pass

    async def close(self) -> None:
        pass

