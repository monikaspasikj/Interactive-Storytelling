import time
import streamlit as st
from qdrant_client import QdrantClient, http
from dotenv import load_dotenv
from gtts import gTTS
import openai
import os
import tempfile
import random

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure retry logic for API requests
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 6,
    errors: tuple = (Exception,)
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

# Qdrant connection setup
def get_qdrant_client():
    qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
    return QdrantClient(host=qdrant_host, port=int(os.getenv('QDRANT_PORT', '6333')))

# Convert text to audio function
def text_to_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Main application logic
def main():
    with st.sidebar:
        tab1, tab2 = st.tabs(["Qdrant API", "Collections"])
        
        with tab1:
            st.subheader("Qdrant API 🔑")
            st.success("Connected to Qdrant!", icon="⚡️")

        with tab2:
            st.subheader("Qdrant Collections")
            qdrant_client = get_qdrant_client()
            selection_qdrant = st.selectbox(
                "Choose one:", 
                ("Create Collection", "Get All Collections", "Delete Collection"), 
                placeholder="Select operation..."
            )

            # Collection management operations
            if selection_qdrant == "Create Collection":
                with st.form("Create Collection", clear_on_submit=True):
                    collection_name = st.text_input("New Collection Name:")
                    submitted = st.form_submit_button("Create!")
                    if submitted:
                        try:
                            vectors_config = http.models.VectorParams(size=1536, distance=http.models.Distance.COSINE)
                            qdrant_client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
                            st.success(f"Collection '{collection_name}' created!", icon="🙌")
                        except Exception as e:
                            st.warning("Collection name exists. Use a unique name.", icon="🚨")

            elif selection_qdrant == "Get All Collections":
                collections = qdrant_client.get_collections().dict()["collections"]
                st.write("Available collections:")
                for col in collections:
                    st.write(f"- {col['name']}")

            elif selection_qdrant == "Delete Collection":
                collections = qdrant_client.get_collections().dict()["collections"]
                collection_to_delete = st.selectbox("Select a collection to delete:", options=[col['name'] for col in collections])
                delete_button = st.button("Delete!")
                if delete_button:
                    qdrant_client.delete_collection(collection_name=collection_to_delete)
                    st.success(f"Deleted Collection '{collection_to_delete}'", icon="💨")

    st.header("Story Query and Generation")

    qdrant_client = get_qdrant_client()
    collections = qdrant_client.get_collections().dict()["collections"]
    collection_to_store = st.selectbox("Select a collection to query:", options=[col['name'] for col in collections])

    # Story search by title using OpenAI embeddings
    story_title = st.text_input("Enter the story title to search:")
    if story_title:
        try:
            def embed_text(text):
                response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
                return response['data'][0]['embedding']

            query_vector = embed_text(story_title)
            results = qdrant_client.search(
                collection_name=collection_to_store,
                query_vector=query_vector,
                limit=5
            )
            
            if results:
                st.write("Search Results:")
                for result in results:
                    story_text = result.payload['text']
                    st.write(f"Title: {result.payload['title']}\n\nText: {story_text}")

                    # Convert story text to audio and play
                    audio_file_path = text_to_audio(story_text)
                    st.audio(audio_file_path, format="audio/mp3")
            else:
                st.write("No valid story found for the given title.")
        except Exception as e:
            st.error(f"Error: {e}")

    # Generate a new story based on a prompt using OpenAI's text completion
    if st.button("Generate New Story"):
        prompt = st.text_area("Enter a prompt for a new story:")
        if prompt:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=200
            )
            generated_story = response.choices[0].text.strip()
            st.subheader("Generated Story:")
            st.write(generated_story)
            
            # Convert the generated story to audio
            audio_file_path = text_to_audio(generated_story)
            st.audio(audio_file_path, format="audio/mp3")

if __name__ == "__main__":
    main()