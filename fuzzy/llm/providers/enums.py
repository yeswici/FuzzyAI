from enum import Enum, auto


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    AZURE = "azure"
    OPENAI = "openai"
    AWS_BEDROCK = "aws-bedrock"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    LOCAL_LLAMA2 = "local-llama2"
    GEMINI = "gemini"
    REST = "rest"
    

class LLMProviderExtraParams(str, Enum):
    MAX_LENGTH = "max_length"
    APPEND_LAST_RESPONSE = "append_last_response"