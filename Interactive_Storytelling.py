import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from gtts import gTTS
from io import BytesIO
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings and Qdrant client
embeddings = OpenAIEmbeddings()
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))

# Check if the collection exists, create it if not
collection_name = "children_stories"
embedding_dimension = embeddings.embedding_dim
if not qdrant_client.has_collection(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )

vector_store = Qdrant(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

# Function to convert text to audio in memory
def text_to_audio(text):
    if not text or not isinstance(text, str):
        raise ValueError("Invalid text for TTS conversion.")
    
    tts = gTTS(text, lang='en')
    audio_file = BytesIO()
    tts.save(audio_file)
    audio_file.seek(0)
    return audio_file

# Streamlit interface
st.title("Story Query and Generation")

# Input for searching the story
title = st.text_input("Enter a story title to search:")
if st.button("Search Story"):
    if title:
        query_vector = embeddings.embed_query(title)
        results = vector_store.similarity_search(query_vector=query_vector, k=1)

        if results:
            st.write("Search Result:")
            story_text = results[0]['payload']['text']
            st.write(story_text)

            st.write("Click the button to listen to the story.")
            if st.button("Play Audio"):
                try:
                    audio_file = text_to_audio(story_text)
                    st.audio(audio_file, format="audio/mp3")
                except Exception as e:
                    st.write("Error generating audio:", e)
        else:
            st.write("No stories found with that title.")

# Generate new story
if st.button("Generate New Story"):
    prompt = st.text_area("Enter a prompt to generate a new story:")
    if prompt:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        generated_story = response.choices[0].message['content'].strip()

        st.subheader("Generated Story:")
        st.write(generated_story)

        if st.button("Play Generated Story Audio"):
            try:
                audio_file_path = text_to_audio(generated_story)
                st.audio(audio_file_path, format="audio/mp3")
            except Exception as e:
                st.write("Error generating audio:", e)