from fuzzy.llm.providers.base import BaseLLMProvider, BaseLLMMessage, BaseLLMProviderResponse
from typing import Optional
from fuzzy.consts import ROLE_USER, ROLE_ASSISTANT
import logging

logger = logging.getLogger(__name__)
IMPROVE_PROMPT = "Be more specific about each step"

class RefinementException(Exception):
    """
    Exception for failed refinement of model response.
    """
    ...


class ResponseRefinementHandler:

    def __init__(self, improve_attempts: int = 3):
        self._improve_attempts = improve_attempts

    async def refine_response(self, model: BaseLLMProvider, original_prompt: str, response: Optional[str] = None) -> list[str]:
        chat_messages: list[BaseLLMMessage] = [BaseLLMMessage(role=ROLE_USER, content=original_prompt)]
        all_responses: list[str] = []

        if response is None:
            logger.debug(f"Generating a response to: {original_prompt}")
            generated_response: Optional[BaseLLMProviderResponse] = await model.chat(messages=chat_messages)
            if generated_response is None or generated_response.response is None:
                raise RefinementException("\033[91mFailed generating a response from the model, got: None\033[0m")
            response = generated_response.response

        chat_messages.append(BaseLLMMessage(role=ROLE_ASSISTANT, content=response))

        logger.debug("\033[91mRefining previous response for prompt {original_prompt}\033[0m")
        for _ in range(self._improve_attempts):
            chat_messages.append(BaseLLMMessage(role=ROLE_USER, content=IMPROVE_PROMPT))
            improved_response = await model.chat(messages=chat_messages)
            if improved_response is None or improved_response.response is None:
                raise RefinementException("\033[91mCannot refine response, got no response from the model\033[0m")
            all_responses.append(improved_response.response)
            chat_messages.append(BaseLLMMessage(role=ROLE_ASSISTANT, content=improved_response.response))

        return all_responses