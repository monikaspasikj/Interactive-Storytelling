import os
import logging
import pandas as pd
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Qdrant client
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", "6333")))
logger.debug("Initialized Qdrant client.")

# Check available collections in Qdrant
try:
    collections = qdrant_client.get_collections()
    logger.info(f"Available collections: {collections}")
except Exception as e:
    logger.error(f"Failed to retrieve collections: {e}")
    st.error("Could not connect to Qdrant or retrieve collections.")

# Initialize LangChain's embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
logger.debug("Initialized HuggingFace embedding model.")

# Initialize Hugging Face text generation pipeline
text_generator = pipeline("text-generation", model="gpt2")
logger.debug("Initialized Hugging Face text generation model.")

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
    st.error(f"Error: {e}")
    logger.error(e)
    df_stories = pd.DataFrame()

if not df_stories.empty:
    st.write("Creating collection in Qdrant...")

    try:
        qdrant_client.recreate_collection(
            collection_name="children_stories",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        logger.info("Qdrant collection created.")
    except Exception as e:
        logger.error(f"Failed to create collection in Qdrant: {e}")
        st.error("Failed to create collection in Qdrant.")

    st.write("Generating embeddings for the stories...")

    # Filter valid documents and ensure they have valid strings
    valid_documents = df_stories[df_stories['text'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

    if valid_documents.empty:
        st.error("No valid documents found for embedding.")
    else:
        embeddings = embedder.embed_documents(valid_documents['text'].tolist())

        if embeddings is None:
            st.error("Error: Failed to generate embeddings for the stories.")
            logger.error("Failed to generate embeddings.")
        else:
            logger.info(f"Generated {len(embeddings)} embeddings successfully.")

            payload = [{"story_id": i, "title": title, "text": text} for i, (title, text) in enumerate(zip(valid_documents['title'], valid_documents['text']))]
            qdrant_client.upload_collection(
                collection_name="children_stories",
                vectors=embeddings,
                payloads=payload,
            )
            logger.info("Stories uploaded to Qdrant collection.")
            st.success("Stories successfully uploaded to Qdrant!")

# Search function
def search_story(title):
    """Searches for a story based on the title."""
    try:
        query_vector = embedder.embed_query(title)
        print(f"Searching with vector: {query_vector}")  # Debugging

        results = qdrant_client.search(
            collection_name="children_stories",
            query_vector=query_vector,
            limit=5
        )
        if results:
            print(f"Search results: {results}")  # Debugging
        return results
    except Exception as e:
        logger.error(f"Error searching for story: {e}")
        st.error(f"Error searching for story: {e}")
        return []

if st.button("Search Story"):
    title = st.text_input("Enter the title of the story you want to search:")
    if title:
        search_results = search_story(title)
        if search_results:
            st.write("Search Results:")
            for result in search_results:
                st.write(f"Title: {result['payload']['title']}, Text: {result['payload']['text']}")
        else:
            st.write("No valid stories found.")

if st.button("Generate New Story"):
    prompt = st.text_area("Enter a prompt to generate a new story:")
    if prompt:
        # Use Hugging Face to generate the story
        generated_story = text_generator(prompt, max_length=200, num_return_sequences=1)
        st.subheader("Generated Story:")
        st.write(generated_story[0]['generated_text'])