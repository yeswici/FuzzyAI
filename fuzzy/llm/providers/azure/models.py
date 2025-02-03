
from pydantic import BaseModel, Field

from fuzzy.enums import LLMRole
from fuzzy.consts import PARAMETER_MAX_TOKENS


class AzureGenerateOptions(BaseModel):
    max_tokens: int = Field(0, alias=PARAMETER_MAX_TOKENS) # type: ignore
    
class AzureMessage(BaseModel):
    role: str = LLMRole.USER
    content: str
    
class AzureRequest(BaseModel):
    messages: list[AzureMessage]
    max_tokens: int = 5
