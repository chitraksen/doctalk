import os
from transformers import logging
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core import Settings

from doctalk.config import Config

# MISTRAL_API_KEY = os.environ.get("API_KEY")


def getLLM():
    # TODO: implement non mistral LLM capabilities
    config = Config()
    llm = MistralAI(api_key=config.llm_api_key, model=config.llm_name)
    return llm


def getEmbeddingModel():
    # TODO: non huggingface model capabilities
    config = Config()
    logging.set_verbosity_error()
    embed_model = HuggingFaceEmbedding(
        model_name=config.embed_name, trust_remote_code=True
    )
    logging.set_verbosity_warning()
    return embed_model
