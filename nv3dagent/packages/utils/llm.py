import os
import requests
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA
from langchain_core.language_models import LLM

# load environment variables
load_dotenv()

class RemoteLLMType:
    GPT4 = "gpt-4"
    GPT35TURBO16K = "gpt-3.5-turbo-16k"
    GPT35TURBO = "gpt-3.5-turbo"
    GPT4TURBO = "gpt-4-turbo"
    MIXTRAL = "ai-mixtral-8x7b-instruct"
    MISTRAL_LARGE = "ai-mistral-large"
    MIXTRAL22B = "ai-mixtral-8x22b-instruct"
    LLAMA3_70B = "ai-llama3-70b"

    @staticmethod
    def get_available_backends():
        return [v for k, v in vars(RemoteLLMType).items() if k.isupper()]

    @staticmethod
    def get_max_tokens(self):
        max_tokens = {
            RemoteLLMType.MIXTRAL: 16384,
            RemoteLLMType.MISTRAL_LARGE: 16384,
            RemoteLLMType.MIXTRAL22B: 32768,
            RemoteLLMType.LLAMA3_70B: 2000,
        }
        return max_tokens[self]


GPT_FAMILY = [RemoteLLMType.GPT4, RemoteLLMType.GPT35TURBO, RemoteLLMType.GPT4TURBO, RemoteLLMType.GPT35TURBO16K]
NVCF_FAMILY = [
    RemoteLLMType.MIXTRAL,
    RemoteLLMType.MISTRAL_LARGE,
    RemoteLLMType.MIXTRAL22B,
    RemoteLLMType.LLAMA3_70B,
]


def get_azure_openai_oauth_token() -> str:
    """
    Get the Azure OpenAI token from the OAuth server.
    """
    # Check environment variables

    for env_var in [
        "AZURE_OPENAI_TOKEN_URL",
        "AZURE_OPENAI_CLIENT_ID",
        "AZURE_OPENAI_CLIENT_SECRET",
        "AZURE_OPENAI_CLIENT_SCOPE",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
    ]:
        assert os.getenv(env_var) is not None, f"Set {env_var} in .env file (look at .env.example)"

    # Get the token from the OAuth server
    response = requests.post(
        os.getenv("AZURE_OPENAI_TOKEN_URL"),
        data={
            "grant_type": "client_credentials",
            "client_id": os.getenv("AZURE_OPENAI_CLIENT_ID"),
            "client_secret": os.getenv("AZURE_OPENAI_CLIENT_SECRET"),
            "scope": os.getenv("AZURE_OPENAI_CLIENT_SCOPE"),
        },
    )
    response.raise_for_status()
    token = str(response.json()["access_token"])
    return token


class CustomAzureChatOpenAI(AzureChatOpenAI):
    """
    Custom class to handle token expiration for the invoke method.
    """

    kwargs = {}

    def __init__(self, **kwargs):
        os.environ["AZURE_OPENAI_API_KEY"] = get_azure_openai_oauth_token()
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def invoke(self, *args, **kwargs) -> str:
        try:
            return super().invoke(*args, **kwargs)
        except Exception as e:
            if e.body["message"] == "Unauthorized: The token has expired":
                self.__init__(**self.kwargs)
                return super().invoke(*args, **kwargs)
            else:
                raise e


def get_llm(model: str, temperature: float = 0.0, top_p: float = 0.0) -> LLM:
    """
    Return a langchain LLM given a model from the GPT_FAMILY or NVCF_FAMILY
    """

    if model in GPT_FAMILY:
        return CustomAzureChatOpenAI(model=model, temperature=temperature, model_kwargs={"top_p": top_p})
    elif model in NVCF_FAMILY:
        assert os.getenv("NVIDIA_API_KEY") is not None, "Set NVIDIA_API_KEY in .env file"
        return ChatNVIDIA(
            model=model, temperature=temperature, top_p=top_p, max_tokens=RemoteLLMType.get_max_tokens(model)
        )
    else:
        raise Exception(f"{model} not supported! Supported LLMs are {RemoteLLMType.get_available_backends()}")
