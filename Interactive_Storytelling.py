import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from gtts import gTTS
from io import BytesIO
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings and Qdrant client
embeddings = OpenAIEmbeddings()
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))

# Define the collection name and set up vector store for "children_stories_chunks"
collection_name = "children_stories_chunks"
embedding_dimension = embeddings.embedding_dim

# Check if the collection exists, create it if not
if not qdrant_client.has_collection(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )

vector_store = Qdrant(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

# Check if the collection has points
point_count = qdrant_client.count(collection_name)
if point_count.count == 0:
    st.write("Collection is empty. Please upload data to Qdrant.")
else:
    st.write(f"Collection '{collection_name}' has {point_count.count} points.")

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

# Input for searching the story by title or content snippet
title = st.text_input("Enter a story title or content snippet to search:")
if st.button("Search Story"):
    if title:
        query_vector = embeddings.embed_query(title)
        
        # Metadata filtering setup (if querying by title only)
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="title",
                    match=MatchValue(value=title)
                )
            ]
        )
        
        # Search for relevant chunks with filtering on title or by similarity if needed
        results = vector_store.similarity_search(query_vector=query_vector, k=3, filter=filter_condition)

        if results:
            st.write("Search Result:")
            # Concatenate and display all retrieved chunks
            story_text = " ".join([result['payload']['text'] for result in results])
            st.write(story_text)

            # Audio playback for concatenated chunks
            st.write("Click the button to listen to the story.")
            if st.button("Play Audio"):
                try:
                    audio_file = text_to_audio(story_text)
                    st.audio(audio_file, format="audio/mp3")
                except Exception as e:
                    st.write("Error generating audio:", e)
        else:
            st.write("No stories found with that title or content snippet.")

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
                audio_file = text_to_audio(generated_story)
                st.audio(audio_file, format="audio/mp3")
            except Exception as e:
                st.write("Error generating audio:", e)