from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from fuzzy.consts import PARAMETER_MAX_TOKENS
from fuzzy.models.base_models import RemoveNoneModel

OllamaModels = Literal['llama2', 'llama2-uncensored', 'llama2:70b', 'llama3', "dolphin-llama3", "llama3.1", "llama3.2", 'vicuna','mistral', 'mixtral',
                       'gemma', "gemma2", 'zephyr', 'phi', 'phi3', "qwen", "llama32_finetuned", "llama32_finetuned_largeset"]
    
class OllamaOptions(RemoveNoneModel):
    num_predict: int = Field(128, alias=PARAMETER_MAX_TOKENS) # type: ignore
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 40
    num_ctx: int = 2048
    seed: int = 42

class OllamaGenerateRequest(RemoveNoneModel):
    model: OllamaModels
    prompt: str
    stream: bool = False
    system: Optional[str] = None
    raw: bool = False
    options: OllamaOptions = OllamaOptions() # type: ignore

class OllamaGenerateResponse(BaseModel):
    response: str

    @field_validator('response')
    @classmethod
    def strip_value(cls, v: str) -> str:
        return v.strip()

class OllamaChatMessage(BaseModel):
    content: str
    role: str = 'user'

class OllamaChatRequest(RemoveNoneModel):
    model: OllamaModels
    messages: list[OllamaChatMessage]
    stream: bool = False
    options: OllamaOptions = OllamaOptions() # type: ignore

class OllamaChatResponse(BaseModel):
    done: bool
    message: OllamaChatMessage