import openai
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.multi_modal_llms import MultiModalLLM

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from planner_core.interfaces import AbstractLlmChat, AbstractLlmChatFactory, AbstractModelFactory
from planner_core.config_handler import ConfigHandler, ConfigPrefix

# region OpenAI Model Factory Implementation
class OpenAiLlmChat(AbstractLlmChat):
    """Example implementation of an LLM chat with OpenAI API"""
    def __init__(
        self,
        client: openai.Client,
        model: str,
        max_tokens: int = 4000,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def get_response(self, system_msg: str, query: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ],
            max_tokens=self._max_tokens,
        )
        content = response.choices[0].message.content
        return content


class OpenAiChatModelFactory(AbstractLlmChatFactory):
    """Example implementation of an LLM chat factory with OpenAI API"""
    def __init__(self, config_handler: ConfigHandler) -> None:
        self.cnf = config_handler.get_config(ConfigPrefix.CHAT)

    def get_llm_chat(self) -> AbstractLlmChat:
        client = openai.Client(
            api_key=self.cnf.api_key,
            organization=self.cnf.org_id if hasattr(self.cnf, 'org_id') else None
        )
        return OpenAiLlmChat(client, self.cnf.llm_model)


class OpenAiModelFactory(AbstractModelFactory):
    """Example implementation of a model factory with OpenAI API"""
    def __init__(self, config_handler: ConfigHandler) -> None:
        self.config_h = config_handler
    
    def get_llm_model(self, config_type: ConfigPrefix) -> BaseLLM:
        cnf = self.config_h.get_config(config_type)
        return OpenAI(
            model=cnf.llm_model,
            api_key=cnf.api_key,
            max_tokens=2000,
            temperature=cnf.temperature,
        )

    def get_multimodal_llm_model(self, config_type: ConfigPrefix) -> MultiModalLLM:
        cnf = self.config_h.get_config(config_type)
        return OpenAIMultiModal(
            model=cnf.llm_model,
            api_key=cnf.api_key,
            max_tokens=1000,
            temperature=cnf.temperature,
        )

    def get_embed_model(self, config_type: ConfigPrefix) -> BaseEmbedding:
        cnf = self.config_h.get_config(config_type)
        return OpenAIEmbedding(
            model=cnf.embed_model,
            api_key=cnf.api_key,
            max_tokens=2000,
        )
# endregion OpenAI Model Factory Implementation

# region Azure OpenAI Model Factory Implementation
class AzureOpenAiLlmChat(AbstractLlmChat):
    """Example implementation of an LLM chat with Azure OpenAI API"""
    def __init__(
        self,
        client: openai.AzureOpenAI,
        deployment: str,
        max_tokens: int = 4000,
    ) -> None:
        self._client: openai.AzureOpenAI = client
        self._deployment_llm: str = deployment
        self._max_tokens: int = max_tokens

    def get_response(self, system_msg: str, query: str) -> str:
        response = self._client.chat.completions.create(
            model=self._deployment_llm,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ],
            max_tokens=4000,
        )
        content = response.choices[0].message.content
        return content


class AzureOpenAiChatModelFactory(AbstractLlmChatFactory):
    """Example implementation of an LLM chat factory with Azure OpenAI API with Azure Identity"""
    def __init__(self, config_handler: ConfigHandler) -> None:
        self.cnf = config_handler.get_config(ConfigPrefix.CHAT)

    def get_llm_chat(self) -> AbstractLlmChat:
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        client = openai.AzureOpenAI(
            azure_endpoint=self.cnf.endpoint,
            azure_deployment=self.cnf.llm_deployment,
            azure_ad_token_provider=token_provider,
            api_version=self.cnf.api_version,
        )
        return AzureOpenAiLlmChat(client, self.cnf.llm_deployment)
    

class AzureOpenAiModelFactory(AbstractModelFactory):
    """Example implementation of a model factory with Azure OpenAI API with Azure Identity"""
    def __init__(self, config_handler: ConfigHandler) -> None:
        self.config_h = config_handler
    
    def get_llm_model(self, config_type: ConfigPrefix) -> BaseLLM:
        cnf = self.config_h.get_config(config_type)
        return AzureOpenAI(
            model=cnf.llm_model,
            deployment_name=cnf.llm_deployment,
            use_azure_ad=True,
            base_url=f"{cnf.endpoint}/openai/deployments/{cnf.llm_deployment}",
            api_version=cnf.api_version,
            max_tokens=2000,
            temperature=cnf.temperature,
        )

    def get_multimodal_llm_model(self, config_type: ConfigPrefix) -> MultiModalLLM:
        cnf = self.config_h.get_config(config_type)
        return AzureOpenAIMultiModal(
            model=cnf.llm_model,
            deployment_name=cnf.llm_deployment,
            use_azure_ad=True,
            azure_endpoint=cnf.endpoint,
            api_version=cnf.api_version,
            max_new_tokens=1000,
            temperature=cnf.temperature,
        )

    def get_embed_model(self, config_type: ConfigPrefix) -> BaseEmbedding:
        cnf = self.config_h.get_config(config_type)
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAIEmbedding(
            model=cnf.embed_model,
            deployment_name=cnf.embed_deployment,
            use_azure_ad=True,
            azure_ad_token_provider=token_provider,
            base_url=f"{cnf.endpoint}/openai/deployments/{cnf.embed_deployment}",
            api_version=cnf.api_version,
            max_tokens=2000,
        )
# endregion Azure OpenAI Model Factory Implementation