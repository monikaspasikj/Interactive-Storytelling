import pandas as pd
import openai
import os
import logging
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI embeddings and Qdrant client
embeddings = OpenAIEmbeddings()
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))

# Collection name and configuration
collection_name = "children_stories"
embedding_dimension = embeddings.embedding_dim
if not qdrant_client.has_collection(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

def preprocess_txt_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    parts = data.split('\n\n')
    titles, texts, current_title, current_story = [], [], "", []
    for part in parts:
        part = part.strip()
        if part.isupper() or part.istitle():
            if current_title and current_story:
                titles.append(current_title)
                texts.append(' '.join(current_story))
            current_title = part
            current_story = []
        else:
            current_story.append(part)
    if current_title and current_story:
        titles.append(current_title)
        texts.append(' '.join(current_story))
    return pd.DataFrame({'title': titles, 'text': texts})

# Load and process dataset
file_path = 'cleaned_stories_final.txt'
try:
    df_stories = preprocess_txt_dataset(file_path)
    logger.info(f"Loaded {len(df_stories)} stories.")
except Exception as e:
    logger.error(f"Error processing dataset: {e}")
    raise

# Generate and upload embeddings
try:
    for i, (title, text) in enumerate(zip(df_stories['title'], df_stories['text'])):
        embedding = embeddings.embed_documents([text])[0]
        vector_store.add_documents([{
            "id": str(i),
            "vector": embedding,
            "payload": {"title": title, "text": text}
        }])
    logger.info("Uploaded stories to Qdrant.")
except Exception as e:
    logger.error(f"Error uploading to Qdrant: {e}")