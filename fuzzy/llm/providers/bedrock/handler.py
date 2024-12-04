import asyncio
import logging
import os
from typing import Any, Optional

import anthropic_bedrock
from anthropic_bedrock import AnthropicBedrock
from anthropic_bedrock.types import Completion

from fuzzy.llm.models import BaseLLMProviderResponse
from fuzzy.llm.providers.anthropic.models import AnthropicGenerateOptions
from fuzzy.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException, llm_provider_fm
from fuzzy.llm.providers.enums import LLMProvider

logger = logging.getLogger(__name__)


class AwsBedrockException(BaseLLMProviderException):
    pass


@llm_provider_fm.flavor(LLMProvider.AWS_BEDROCK)
class AwsBedrockProvider(BaseLLMProvider):
    AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
    AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
    AWS_REGION = "AWS_REGION"

    def __init__(self, model: str = "anthropic.claude-v2:1", **extra: Any):
        super().__init__(model=model, **extra)

        if self.AWS_SECRET_ACCESS_KEY not in os.environ:
            raise AwsBedrockException(f"{self.AWS_SECRET_ACCESS_KEY} not in os.environ")

        if self.AWS_ACCESS_KEY_ID not in os.environ:
            raise AwsBedrockException(f"{self.AWS_ACCESS_KEY_ID} not in os.environ")

        if (aws_region := os.environ.get(self.AWS_REGION)) is None:
            raise AwsBedrockException(f"{self.AWS_REGION} not in os.environ")

        self._client = AnthropicBedrock(
            aws_region=aws_region,
        )

    @classmethod
    def get_supported_models(cls) -> list[str]:
        return [
            "anthropic.claude-v2:1",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
        ]

    async def generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        try:
            options = AnthropicGenerateOptions.model_validate(extra)
            completion: Completion = await asyncio.to_thread(self._client.completions.create, # type: ignore
                                                             model=self._model_name,
                                                             prompt=f"{anthropic_bedrock.HUMAN_PROMPT} {prompt} {anthropic_bedrock.AI_PROMPT}",
                                                             **options
                                                             ) 

            return BaseLLMProviderResponse(response=completion.completion)
        except Exception as e:
            logger.error(f'Error generating text: {e}')
            raise AwsBedrockException('Cant generate text')

    async def chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        if len(messages) > 1:
            logger.warning("claude-aws only supports one message at a time, using the first message")
    
        return await self.generate(messages[0].content, **extra)
    
    
    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError
    
    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)
