import logging
import os
from typing import Any, Callable, Optional

import requests
from jsonpath_ng import parse

from fuzzy.llm.models import BaseLLMProviderResponse
from fuzzy.llm.providers.base import BaseLLMProvider, BaseLLMProviderException, llm_provider_fm
from fuzzy.llm.providers.enums import LLMProvider
from fuzzy.llm.providers.rest.utils import parse_http_request

PROMPT_TOKEN = "<PROMPT>"

logger = logging.getLogger(__name__)

class RestProviderException(BaseLLMProviderException):
    pass

@llm_provider_fm.flavor(LLMProvider.REST)
class RestProvider(BaseLLMProvider):
    """
    A provider that uses a REST API to communicate with a language model.
    This provider is used to communicate with any language model that exposes a REST API, or a REST API which utilizes an LLM behind the scene.

    caveats:
        * REST API response must be in JSON format.
        * We assume that the raw http request file is in the following format:
            ```
            METHOD /path HTTP/1.1
            key: value
            key: value
            
            {
                "key": "value"
            }
            ``` 

    Args:
        host (str): The host of the REST API.
        raw_http_file (str): The path to the raw HTTP request file.
        response_jsonpath (str): The JSONPath to extract the response from the HTTP response. i.e "$.response".
        prompt_token (str): The token to be replaced with the prompt in the HTTP request body. (default: "<PROMPT>")
        scheme (str): The scheme of the REST API (default: "https").
        **extra (Any): Additional arguments to be passed to the BaseLLMProvider constructor.
    """

    def __init__(self, host: str, raw_http_file: str, response_jsonpath: str, prompt_token: str = PROMPT_TOKEN, scheme: str = "https", **extra: Any):
        super().__init__(name=LLMProvider.REST, **extra)
        self._method: str = str()
        self._path: str = str()
        self._headers: dict[str, Any] = dict()
        self._body: str = str()

        self._prompt_token = prompt_token
        self._response_jsonpath = response_jsonpath
        self._parse_http_file(raw_http_file)

        self._url = f"{scheme}://{host}{self._path}"

        self._session = requests.Session()
        self._session.headers = self._headers

    @classmethod
    def get_supported_models(cls) -> list[str]:
        return []
    
    def _parse_http_file(self, raw_http_file: str) -> None:
        """
        Parse the raw HTTP file to extract the method, url, headers, and body.

        Args:
            raw_http_file (str): The path to the raw HTTP file.

        """
        # Check that the file exists
        if not os.path.exists(raw_http_file):
            raise RestProviderException(f"HTTP file not found: {raw_http_file}")
        
        parsed_http = parse_http_request(raw_http_file)
        self._method = parsed_http["method"]
        self._path = parsed_http["path"]
        if not self._path.startswith("/"):
            self._path = f"/{self._path}"

        self._headers = parsed_http["headers"]
        self._body = parsed_http["body"]


    async def generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        """
        Generates a response from the language model using a REST API.

        Args:
            prompt (str): The input prompt to generate a response.
            **extra (Any): Additional arguments to be passed to the REST API.

        Returns:
            Optional[BaseLLMProviderResponse]: The generated response.

        """
        response: Optional[BaseLLMProviderResponse] = None
        
        logger.debug("Generating prompt: %s", prompt)
        try:
            http_response: requests.Response
            method: Callable[..., requests.Response] = getattr(self._session, self._method.lower())
            payload = self._body.replace(self._prompt_token, prompt)

            async with method(url=self._url, data=payload) as http_response:
                http_response.raise_for_status()
                raw_response = await http_response.json()
                jsonpath_expr = parse(self._response_jsonpath)

                # Extract the data
                result = [match.value for match in jsonpath_expr.find(raw_response)]
                # Since we expect only one result, we can just get the first one
                if result:
                    response = BaseLLMProviderResponse(response=result[0])
                else:
                    logger.warning("No response found in the JSONPath: %s", self._response_jsonpath)
                    response = None
        except requests.HTTPError as e:
            raise e
        except Exception as e:
            logger.error("Error generating response: %s", e)
            raise RestProviderException(f"Error generating prompt: {e}")
        
        logger.debug("Generated response: %s", response)
        return response
