from enum import Enum


class LLMRole(str, Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    USER = "user"