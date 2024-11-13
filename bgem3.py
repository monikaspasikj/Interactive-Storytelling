import pandas as pd
import openai
import os
import logging
from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
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
collection_name = "children_stories_chunks"
embedding_dimension = embeddings.embedding_dim

# Create collection if it doesn't exist
if not qdrant_client.has_collection(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )
vector_store = Qdrant(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

# Function to split a story into chunks
def split_story_into_chunks(story_text, chunk_size=100):
    words = story_text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Load and process dataset in the format provided
def load_stories(file_path):
    titles = []
    texts = []
    current_text = ""

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() and line.strip().endswith('.'):
                if current_text:
                    texts.append(current_text.strip())
                    current_text = ""
                titles.append(line.strip())
            else:
                current_text += line.strip() + " "
        if current_text:
            texts.append(current_text.strip())

    return pd.DataFrame({'title': titles, 'text': texts})

file_path = 'cleaned_stories_final.txt'
try:
    df_stories = load_stories(file_path)
    logger.info(f"Loaded {len(df_stories)} stories.")
except Exception as e:
    logger.error(f"Error processing dataset: {e}")
    raise

# Generate and upload embeddings in chunks
try:
    for i, (title, text) in enumerate(zip(df_stories['title'], df_stories['text'])):
        chunks = split_story_into_chunks(text)
        for j, chunk in enumerate(chunks):
            embedding = embeddings.embed_documents([chunk])[0]
            vector_store.add_documents(
                documents=[
                    {
                        "id": f"{i}-{j}",
                        "vector": embedding,
                        "payload": {
                            "title": title,
                            "chunk_index": j,
                            "text": chunk
                        }
                    }
                ]
            )
            logger.info(f"Uploaded chunk {j} of story '{title}' to Qdrant.")
except Exception as e:
    logger.error(f"Error uploading to Qdrant: {e}")