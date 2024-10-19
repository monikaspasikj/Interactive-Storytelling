from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def preprocess_txt_dataset(file_path):
    """Reads the .txt dataset and splits it into titles and stories, ensuring proper formatting."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Split the dataset into parts based on potential title indicators
    parts = data.split('\n\n')

    titles = []
    texts = []
    current_title = ""
    current_story = []

    for part in parts:
        part = part.strip()  # Remove leading/trailing whitespace
        if part.isupper() or part.istitle():  # If part looks like a title (UPPERCASE or Title Case)
            if current_title and current_story:  # Save previous story
                titles.append(current_title)
                # Join the story parts into a single paragraph without any new lines (to ensure continuity)
                story_text = ' '.join([line.replace('\n', ' ').strip() for line in current_story if line.strip()])
                texts.append(story_text)
            current_title = part
            current_story = []
        else:
            current_story.append(part)

    # Add the last story if exists
    if current_title and current_story:
        titles.append(current_title)
        story_text = ' '.join([line.replace('\n', ' ').strip() for line in current_story if line.strip()])
        texts.append(story_text)

    # Return a DataFrame for easy handling
    df = pd.DataFrame({'title': titles, 'text': texts})
    return df

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Load the dataset
file_path = 'cleaned_stories_final.txt' 
try:
    df_stories = preprocess_txt_dataset(file_path)
    if df_stories.empty:
        raise ValueError("No stories found in the dataset.")
    logger.info(f"Successfully loaded {len(df_stories)} stories from the dataset.")
except Exception as e:
    logger.error(f"Error loading or processing dataset: {e}")
    raise

# Extract text from the dataset
pagesText = df_stories['text'].tolist()  # Get the list of stories

# Initialize LangChain embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a new collection if it doesn't exist
try:
    client.recreate_collection(
        collection_name="children_stories",  # Use the same collection name for consistency
        vectors_config=models.VectorParams(size=embedder.embed_query("test").shape[0], distance=models.Distance.COSINE),
    )
    logger.info("Collection 'children_stories' created successfully.")
except Exception as e:
    logger.error(f"Error creating Qdrant collection: {e}")
    raise

# Generate document embeddings and insert into Qdrant
try:
    docs_embeddings = embedder.embed_documents(pagesText)
    logger.info(f"Generated {len(docs_embeddings)} document embeddings.")

    for i, embedding in enumerate(docs_embeddings):
        client.upload_records(
            collection_name="children_stories",  # Use the same collection name for consistency
            records=[models.Record(id=i, vector=embedding, payload={"text": pagesText[i], "title": df_stories['title'][i]})]
        )
    logger.info("Successfully uploaded embeddings and stories to Qdrant.")
except Exception as e:
    logger.error(f"Error during embedding generation or upload: {e}")

# Search example (optional, for testing)
try:
    query = "What are spindles"  # Example search query
    query_embedding = embedder.embed_query(query)
    search_result = client.search(collection_name="children_stories", query_vector=query_embedding, limit=6, with_payload=True)

    for hit in search_result:
        print(hit.payload['title']) 
        print(hit.payload['text'])  
except Exception as e:
    logger.error(f"Error during search: {e}")
