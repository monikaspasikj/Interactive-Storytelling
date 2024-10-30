from qdrant_client import QdrantClient
from qdrant_client.http import models
import pandas as pd
import openai
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def preprocess_txt_dataset(file_path):
    """Reads the .txt dataset and splits it into titles and stories, ensuring proper formatting."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    parts = data.split('\n\n')

    titles, texts, current_title, current_story = [], [], "", []
    for part in parts:
        part = part.strip()
        if part.isupper() or part.istitle():
            if current_title and current_story:
                titles.append(current_title)
                texts.append(' '.join([line.replace('\n', ' ').strip() for line in current_story if line.strip()]))
            current_title = part
            current_story = []
        else:
            current_story.append(part)
    if current_title and current_story:
        titles.append(current_title)
        texts.append(' '.join([line.replace('\n', ' ').strip() for line in current_story if line.strip()]))
    
    return pd.DataFrame({'title': titles, 'text': texts})

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

def embed_text_openai(text):
    """Generates embeddings using OpenAI API."""
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Load and process dataset
file_path = 'cleaned_stories_final.txt'
try:
    df_stories = preprocess_txt_dataset(file_path)
    logger.info(f"Loaded {len(df_stories)} stories.")
except Exception as e:
    logger.error(f"Error processing dataset: {e}")
    raise

# Create Qdrant collection with 1536 vector size for OpenAI embeddings
try:
    client.recreate_collection(
        collection_name="children_stories",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    logger.info("Collection 'children_stories' created successfully.")
except Exception as e:
    logger.error(f"Error creating Qdrant collection: {e}")
    raise

# Generate and upload embeddings
try:
    for i, (title, text) in enumerate(zip(df_stories['title'], df_stories['text'])):
        embedding = embed_text_openai(text)
        response=client.upload_records(
            collection_name="children_stories",
            records=[models.Record(id=i, vector=embedding, payload={"text": text, "title": title})]
        )
        print(response)
        print(f"Generated embedding for document {i}: {embedding}")
    logger.info("Uploaded stories to Qdrant.")
except Exception as e:
    logger.error(f"Error uploading to Qdrant: {e}")