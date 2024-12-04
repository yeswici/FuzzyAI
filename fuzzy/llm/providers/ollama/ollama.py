import asyncio
import logging
import random
from typing import Any, Optional, get_args

import aiohttp

from fuzzy.consts import OLLAMA_BASE_PORT, ROLE_SYSTEM
from fuzzy.llm.models import BaseLLMProviderResponse
from fuzzy.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException, llm_provider_fm
from fuzzy.llm.providers.enums import LLMProvider
from fuzzy.llm.providers.ollama.models import (OllamaChatMessage, OllamaChatRequest, OllamaChatResponse,
                                               OllamaGenerateRequest, OllamaGenerateResponse, OllamaModels,
                                               OllamaOptions)
from fuzzy.llm.providers.shared.decorators import api_endpoint

logger = logging.getLogger(__name__)

class OllamaProviderException(BaseLLMProviderException):
    pass


@llm_provider_fm.flavor(LLMProvider.OLLAMA)
class OllamaProvider(BaseLLMProvider):
    def __init__(self, llm_address: Optional[str] = None, port: int = OLLAMA_BASE_PORT, 
                 seed: int = 0, **extra: Any):
        super().__init__(**extra)
        
        if llm_address is None:
            llm_address = "localhost"
        
        self._base_url = f"http://{llm_address}:{int(port) + int(seed)}/api"
        
        random.seed(seed)
        self._seed = random.randint(42, 1024)
        self._session = aiohttp.ClientSession()
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        return list(get_args(OllamaModels))

    @api_endpoint("/generate") # type: ignore
    async def generate(self, prompt: str, url: str, system_prompt: Optional[str] = None, raw: bool = False, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        try:
            options = OllamaOptions(seed=self._seed, **extra)
            request = OllamaGenerateRequest(model=self._model_name, prompt=prompt, raw=raw, # type: ignore
                                            options=options)
            if system_prompt is not None:
                request.system = system_prompt

            async with self._session.post(url, json=request.model_dump()) as response:
                raw_response = await response.json()

                if 'error' in raw_response:
                    raise OllamaProviderException(f"Ollama error: {raw_response.get('error')}")
            
                ollama_response = OllamaGenerateResponse(**raw_response)
                return BaseLLMProviderResponse(response=ollama_response.response)
        except OllamaProviderException:
            raise
        except Exception as e:
            logger.error(f'Error generating text: {e}')
            raise OllamaProviderException('Cant generate text')

    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        return asyncio.run(self.generate(prompt, **extra)) # type: ignore
    
    @api_endpoint("/chat") # type: ignore
    async def chat(self, messages: list[BaseLLMMessage], url: str, system_prompt: Optional[str] = None, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        try:
            options = OllamaOptions(seed=self._seed, **extra)
            ollama_messages = [OllamaChatMessage.model_validate(m.model_dump()) for m in messages]

            if system_prompt is not None:
                ollama_messages.insert(0, OllamaChatMessage(role=ROLE_SYSTEM, content=system_prompt))

            request = OllamaChatRequest(model=self._model_name, # type: ignore
                                        messages=ollama_messages,
                                        options=options)
            
            async with self._session.post(url, json=request.model_dump()) as response:
                ollama_response = OllamaChatResponse(**(await response.json()))
                return BaseLLMProviderResponse(response=ollama_response.message.content)
        except Exception as e:
            logger.error(f'Error during chat: {e}')
            raise OllamaProviderException('Error during chat')
    
    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError
        
    async def close(self) -> None:
        logger.debug(f"Closing OllamaProvider {self}")
        await self._session.close()

    def __repr__(self) -> str:
        return f"model: {self._model_name}, base_url: {self._base_url}"
    
    def __str__(self) -> str:
        return f"model: {self._model_name}, base_url: {self._base_url}"
