import os
import logging
import pandas as pd
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from gtts import gTTS
import tempfile

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Qdrant client
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", "6333")))
logger.debug("Initialized Qdrant client.")  # Fixed missing parenthesis on line 21

# Set up embeddings
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
logger.debug("Using BGE embeddings for queries.")

# Convert text to audio
def text_to_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Preprocess dataset
def preprocess_txt_dataset(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found.")
        raise FileNotFoundError(f"{file_path} not found.")
    logger.info(f"Loading data from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    stories = data.split('\n\n')
    titles, texts = [], []
    for story in stories:
        lines = story.split('\n')
        titles.append(lines[0].strip())
        texts.append(' '.join(lines[1:]).strip())
    logger.info("Dataset processed successfully.")
    return pd.DataFrame({'title': titles, 'text': texts})

# Load dataset
file_path = 'cleaned_stories_final.txt'
try:
    df_stories = preprocess_txt_dataset(file_path)
    st.write(f"Loaded {len(df_stories)} stories.")
except Exception as e:
    st.error(f"Error: {e}")
    df_stories = pd.DataFrame()

# If dataset loaded, recreate collection with correct vector size and upload embeddings
if not df_stories.empty:
    try:
        # Delete any existing collection to avoid dimension mismatch
        qdrant_client.delete_collection("children_stories")

        # Recreate collection with vector size 1024 for BAAI/bge-large-en embeddings
        qdrant_client.recreate_collection(
            "children_stories",
            models.VectorParams(size=1024, distance=models.Distance.COSINE)
        )

        # Generate embeddings and upload to collection
        embeddings = embedder.embed_documents(df_stories['text'].tolist())
        payload = [{"story_id": i, "title": title, "text": text} for i, (title, text) in enumerate(zip(df_stories['title'], df_stories['text']))]
        qdrant_client.upload_collection("children_stories", vectors=embeddings, payloads=payload)
        st.success("Stories uploaded to Qdrant!")
    except Exception as e:
        logger.error(f"Error uploading to Qdrant: {e}")

# Query function
def search_story(title):
    try:
        query_vector = embedder.embed_query(title)
        results = qdrant_client.search("children_stories", query_vector=query_vector, limit=5)
        return results
    except Exception as e:
        logger.error(f"Error in search: {e}")
        st.error(f"Error in search: {e}")
        return []

# Search and audio playback in Streamlit app
if st.button("Search Story"):
    title = st.text_input("Enter the title of the story you want to search:")
    if title:
        search_results = search_story(title)
        if search_results:
            st.write("Search Results:")
            for result in search_results:
                story_text = result['payload']['text']
                st.write(f"Title: {result['payload']['title']}, Text: {story_text}")

                # Generate and play audio for the story
                audio_file_path = text_to_audio(story_text)
                st.audio(audio_file_path, format="audio/mp3")
        else:
            st.write("No valid stories found.")

if st.button("Generate New Story"):
    prompt = st.text_area("Enter a prompt to generate a new story:")
    if prompt:
        # Use Hugging Face to generate the story
        generated_story = text_generator(prompt, max_length=200, num_return_sequences=1)
        st.subheader("Generated Story:")
        st.write(generated_story[0]['generated_text'])