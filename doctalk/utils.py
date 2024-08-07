import os
from transformers import logging
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core import Settings

MISTRAL_API_KEY = os.environ.get("API_KEY")


def getLLM():
    # TODO: implement model choice
    llm = MistralAI(api_key=MISTRAL_API_KEY, model="open-mixtral-8x7b")
    return llm


def getEmbeddingModel():
    # TODO: implement model choice
    logging.set_verbosity_error()
    embed_model = HuggingFaceEmbedding(
        model_name="Snowflake/snowflake-arctic-embed-m-v1.5", trust_remote_code=True
    )
    logging.set_verbosity_warning()
    return embed_model
