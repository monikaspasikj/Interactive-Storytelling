import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings

fmt = "\nCollection '{}' successfully created and operational with vector dimension {}."
client = QdrantClient(host="localhost", port=6333)
collection_name, vector_dim = 'example_vectors', 1536

# Check if the collection exists and print confirmation
if client.has_collection(collection_name):
    print(fmt.format(collection_name, vector_dim))