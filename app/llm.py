from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import torch

def init_llm():
    llm = Ollama(model="llama3.2", request_timeout=300.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model

    return embed_model
