import os
import logging
import autogen
import pandas as pd
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Qdrant client
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=os.getenv("QDRANT_PORT", "6333"))
logger.debug("Initialized Qdrant client.")

# Check available collections in Qdrant
collections = qdrant_client.get_collections()
st.write("Available collections:", collections)
logger.info(f"Available collections: {collections}")

# Initialize LangChain's embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
logger.debug("Initialized HuggingFace embedding model.")

# Initialize LangChain's LLM
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
logger.debug("Initialized OpenAI model.")

# Function to preprocess the dataset
def preprocess_txt_dataset(file_path):
    """Reads the .txt dataset, splitting it into titles and stories."""
    if not os.path.exists(file_path):
        logger.error(f"The file {file_path} was not found.")
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    logger.info(f"Loading dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    stories = data.split('\n\n')
    stories = [story.strip() for story in stories if story.strip()]

    titles = []
    texts = []
    
    for story in stories:
        lines = story.split('\n')
        title = lines[0].strip()  # First line is the title
        text = ' '.join(lines[1:]).strip()  # The rest is the story content
        titles.append(title)
        texts.append(text)
    
    logger.info("Successfully preprocessed the dataset.")
    return pd.DataFrame({'title': titles, 'text': texts})

# Load the .txt dataset and preprocess it
file_path = 'cleaned_stories_final.txt'
try:
    df_stories = preprocess_txt_dataset(file_path)
    st.write(f"Loaded {len(df_stories)} stories from the dataset.")
except (FileNotFoundError, ValueError) as e:
    st.write(f"Error: {e}")
    logger.error(e)
    df_stories = pd.DataFrame()

if not df_stories.empty:
    st.write("Creating collection in Qdrant...")
    qdrant_client.recreate_collection(
        collection_name="children_stories",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    logger.info("Qdrant collection created.")

    st.write("Generating embeddings for the stories...")
    embeddings = embedder.embed_documents(df_stories['text'].tolist())

    if embeddings is None:
        st.write("Error: Failed to generate embeddings for the stories.")
        logger.error("Failed to generate embeddings.")
    else:
        logger.info(f"Generated {len(embeddings)} embeddings successfully.")

    payload = [{"story_id": i, "title": df_stories['title'][i], "text": df_stories['text'][i]} for i in range(len(embeddings))]

    st.write("Inserting stories into Qdrant...")
    qdrant_client.upload_collection(
        collection_name="children_stories",
        vectors=embeddings,
        payload=payload,
    )
    logger.info("Embeddings and stories successfully inserted into Qdrant.")

else:
    st.write("No stories found to process.")

# Function to search for stories using Qdrant by title
def search_story(search_query):
    """Search for stories related to the user's query."""
    try:
        logger.debug(f"Searching for: {search_query}")
        search_embedding = embedder.embed_query(search_query)
        logger.debug("Generated embedding for search query")

        results = qdrant_client.search(
            collection_name="children_stories",
            query_vector=search_embedding,
            limit=5,
        )
        logger.debug(f"Raw Qdrant search results: {results}")

        if results:
            logger.info(f"Found {len(results)} results")
            return results
        else:
            logger.info("No results found")
            return []

    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return []

# Streamlit UI to input story prompt and show the generated story
st.title("Interactive Storytelling with LLM")

# Input field for the user to type a story prompt
user_story_prompt = st.text_input("Enter a story prompt or request:", key="story_prompt")

# Generate story button
if st.button("Generate Story", key="generate_story_button"):
    if user_story_prompt:
        # Simulate a conversation with the story agent (you can customize the function)
        story_response = f"Simulated response for: {user_story_prompt}"
        st.write("Generated Story:")
        st.write(story_response)
    else:
        st.write("Please type a prompt for the story.")

# Input field for the user to search for a story
search_query = st.text_input("Search for a story:", key="search_story")

# Search story button
if st.button("Search Story", key="search_story_button"):
    if search_query:
        search_results = search_story(search_query)
        if search_results:
            st.write("Search Results:")
            for result in search_results:
                # Ensure each result is displayed correctly
                story_title = result.payload.get('title', 'No title found')
                story_text = result.payload.get('text', 'No text found')
                st.write(f"**Story ID:** {result.id}\n**Title:** {story_title}\n**Story:** {story_text[:300]}...")
        else:
            st.write("No matching stories found.")
    else:
        st.write("Please type a search query.")