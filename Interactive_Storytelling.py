import os
import logging
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from gtts import gTTS
import tempfile
import openai
from qdrant_client import QdrantClient
from qdrant_client import models  # Import for vector configuration

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings and Qdrant client
embeddings = OpenAIEmbeddings()
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))

# Check if the collection exists, create it if not
collection_name = "children_stories"
if not qdrant_client.has_collection(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings.embedding_dim, distance=models.Distance.COSINE)
    )

vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

# Text to speech conversion
def text_to_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Streamlit interface
if st.button("Search Story"):
    title = st.text_input("Enter the story title to search:")
    if title:
        query_vector = embeddings.embed_query(title)
        results = vector_store.similarity_search(query_vector=query_vector, k=5)
        if results:
            st.write("Search Results:")
            for result in results:
                story_text = result['payload']['text']
                st.write(f"Title: {result['payload']['title']}, Text: {story_text}")
                audio_file_path = text_to_audio(story_text)
                st.audio(audio_file_path, format="audio/mp3")
        else:
            st.write("No valid stories found.")

if st.button("Generate New Story"):
    prompt = st.text_area("Enter a prompt to generate a new story:")
    if prompt:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        generated_story = response.choices[0].message['content'].strip()
        st.subheader("Generated Story:")
        st.write(generated_story)
        audio_file_path = text_to_audio(generated_story)
        st.audio(audio_file_path, format="audio/mp3")
