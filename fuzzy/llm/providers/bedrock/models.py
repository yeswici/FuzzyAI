from typing import Optional

from pydantic import BaseModel

from fuzzy.llm.providers.base import BaseLLMMessage
from fuzzy.models.base_models import RemoveNoneModel


class ClaudeGenerateOptions(BaseModel):
    max_tokens: int = 100
    
class ClaudeAnthropicMessage(BaseModel):
    type: str
    text: str

class ClaudeAnthropicMessagesRequest(RemoveNoneModel):
    model: str
    system: Optional[str] = None
    messages: list[BaseLLMMessage]
    max_tokens: int = 100

class ClaudeAnthropicMessagesResponse(BaseModel):
    type: str
    content: list[ClaudeAnthropicMessage]

