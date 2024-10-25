import time
import streamlit as st
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, http
from transformers import pipeline, AutoTokenizer, AutoModel
from gtts import gTTS
import os
import random
import tempfile
import base64

# Ensure the embedding model is downloaded
model_name = "BAAI/bge-large-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("Model and tokenizer downloaded successfully.")

# Retry logic for handling rate limits
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 6,
    errors: tuple = (Exception,),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum retries ({max_retries}) exceeded.")
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
    return wrapper

# Connect to Qdrant
def get_qdrant_client():
    qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
    return QdrantClient(host=qdrant_host, port=6333)

# Convert text to audio
def text_to_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Audio playback
def audio_player(file_path):
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

# Main function
def main():
    with st.sidebar:
        tab1, tab2, tab3 = st.tabs(["Qdrant API", "Hugging Face API", "Collections"])
        with tab1:
            st.subheader("Qdrant API 🔑")
            st.success("Connected to Qdrant!", icon="⚡️")
        with tab2:
            st.subheader("Hugging Face API 🔑")
            st.success("Using Hugging Face for text generation!", icon="🔑")
        with tab3:
            st.subheader("Qdrant Collections")
            qdrant_client = get_qdrant_client()
            selection_qdrant = st.selectbox("Choose one:", ("Create Collection", "Get All Collections", "Delete Collection"), placeholder="Waiting...")

            if selection_qdrant == "Create Collection":
                with st.form("Create New Collection", clear_on_submit=True):
                    collection_name = st.text_input("New Collection Name:")
                    submitted = st.form_submit_button("Add Collection!")
                    if submitted:
                        try:
                            vectors_config = http.models.VectorParams(size=1536, distance=http.models.Distance.COSINE)
                            qdrant_client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
                            st.success(f"Collection '{collection_name}' created!", icon="🙌")
                        except Exception as e:
                            st.warning("Collection name exists. Use a unique name.", icon="🚨")

            if selection_qdrant == "Get All Collections":
                collections = qdrant_client.get_collections().dict()["collections"]
                for col in collections:
                    st.write(f"- {col['name']}")

            if selection_qdrant == "Delete Collection":
                collections = qdrant_client.get_collections().dict()["collections"]
                collection_to_delete = st.selectbox("Select a collection to delete:", options=[col['name'] for col in collections])
                delete_button = st.button("Delete!")
                if delete_button:
                    qdrant_client.delete_collection(collection_name=collection_to_delete)
                    st.success(f"Deleted Collection '{collection_to_delete}'", icon="💨")

    st.header("Story Query and Generation")
    qdrant_client = get_qdrant_client()
    collections = qdrant_client.get_collections().dict()["collections"]
    collection_to_store = st.selectbox("Select a collection for query:", options=[col['name'] for col in collections])
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

    vector_store = Qdrant(client=qdrant_client, collection_name=collection_to_store, embeddings=embeddings)

    # Search story by title
    story_title = st.text_input("Enter story title to query:")
    if story_title:
        try:
            search_results = vector_store.similarity_search(story_title, k=1)
            valid_results = [res for res in search_results if res.page_content and isinstance(res.page_content, str) and res.page_content.strip()]

            if valid_results:
                st.subheader("Story:")
                st.write(valid_results[0].page_content)
                audio_file_path = text_to_audio(valid_results[0].page_content)
                audio_player(audio_file_path)
            else:
                st.write("No valid story found for the title.")
        except Exception as e:
            st.error(f"Error connecting to Qdrant: {e}")

if __name__ == "__main__":
    main()