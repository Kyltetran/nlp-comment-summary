from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv


def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model="llama3.2",
    )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings
