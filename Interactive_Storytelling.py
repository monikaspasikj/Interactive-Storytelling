import os
import logging
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from gtts import gTTS
import tempfile
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Qdrant client
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", "6333")))

# Text to speech conversion
def text_to_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Embed text using OpenAI
def embed_text_openai(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Load dataset and create collection
file_path = 'cleaned_stories_final.txt'
try:
    df_stories = preprocess_txt_dataset(file_path)
    st.write(f"Loaded {len(df_stories)} stories.")
except Exception as e:
    st.error(f"Error: {e}")

# Upload embeddings to Qdrant
if not df_stories.empty:
    try:
        qdrant_client.recreate_collection("children_stories", models.VectorParams(size=1536, distance=models.Distance.COSINE))
        for i, (title, text) in enumerate(zip(df_stories['title'], df_stories['text'])):
            embedding = embed_text_openai(text)
            qdrant_client.upload_records(
                collection_name="children_stories",
                records=[models.Record(id=i, vector=embedding, payload={"title": title, "text": text})]
            )
        st.success("Stories uploaded to Qdrant!")
    except Exception as e:
        logger.error(f"Error uploading to Qdrant: {e}")
        st.error(f"Failed to upload to Qdrant: {e}")

# Search function
def search_story(title):
    try:
        query_vector = embed_text_openai(title)
        results = qdrant_client.search("children_stories", query_vector=query_vector, limit=5)
        return results
    except Exception as e:
        logger.error(f"Error in search: {e}")
        st.error(f"Error in search: {e}")
        return []

# Streamlit interface
if st.button("Search Story"):
    title = st.text_input("Enter the story title to search:")
    if title:
        search_results = search_story(title)
        if search_results:
            st.write("Search Results:")
            for result in search_results:
                story_text = result.payload['text']
                st.write(f"Title: {result.payload['title']}, Text: {story_text}")
                audio_file_path = text_to_audio(story_text)
                st.audio(audio_file_path, format="audio/mp3")
        else:
            st.write("No valid stories found.")

if st.button("Generate New Story"):
    prompt = st.text_area("Enter a prompt to generate a new story:")
    if prompt:
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=200)
        generated_story = response.choices[0].text.strip()
        st.subheader("Generated Story:")
        st.write(generated_story)
        audio_file_path = text_to_audio(generated_story)
        st.audio(audio_file_path, format="audio/mp3")