import time
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance  # Updated import
from dotenv import load_dotenv
from gtts import gTTS
import openai
import os
import tempfile
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings and Qdrant client
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Specify model
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))

# Define collection name and embedding dimension
collection_name = "children_stories"
embedding_dimension = 1536  # Set to match text-embedding-ada-002 dimension

# Function to check if a collection exists
def collection_exists(collection_name):
    try:
        qdrant_client.get_collection(collection_name)
        return True
    except Exception:
        return False

# Check if the collection exists; create it if not
if not collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)  # Updated to use VectorParams
    )

# Initialize the vector store
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

# Convert text to audio function
def text_to_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Main application logic
def main():
    st.sidebar.subheader("Qdrant API Status")
    st.success("Connected to Qdrant!", icon="⚡️")

    st.header("Story Query and Generation")
    
    # Search stories in Qdrant based on title
    story_title = st.text_input("Enter the story title to search:")
    if story_title:
        query_vector = embeddings.embed_query(story_title)
        results = vector_store.similarity_search(query_vector=query_vector, k=5)

        if results:
            st.write("Search Results:")
            for result in results:
                st.write(f"Title: {result['payload']['title']}\n\nText: {result['payload']['text']}")
                audio_file_path = text_to_audio(result['payload']['text'])
                st.audio(audio_file_path, format="audio/mp3")
                os.remove(audio_file_path)  # Clean up audio file
        else:
            st.write("No valid story found for the given title.")

    # Generate a new story based on a prompt using ChatOpenAI
    if st.button("Generate New Story"):
        prompt = st.text_area("Enter a prompt for a new story:")
        if prompt:
            chat_model = ChatOpenAI()
            try:
                response = chat_model.generate(prompt)  # Adjusted to use generate method
                generated_story = response.strip()
                st.subheader("Generated Story:")
                st.write(generated_story)
                audio_file_path = text_to_audio(generated_story)
                st.audio(audio_file_path, format="audio/mp3")
                os.remove(audio_file_path)  # Clean up audio file
            except Exception as e:
                st.error("Error generating story: {}".format(e))

if __name__ == "__main__":
    main()