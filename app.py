import time
import streamlit as st
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings  # Can replace with local embeddings
from qdrant_client import QdrantClient, http
import os
import random

# Retry logic for handling rate limits
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 6,
    errors: tuple = (Exception,),
):
    """Retry a function with exponential backoff."""

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


# Function to connect to Qdrant
def get_qdrant_client():
    qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
    return QdrantClient(host=qdrant_host, port=6333)

def main():
    with st.sidebar:
        tab1, tab2, tab3 = st.tabs(["Qdrant API", "OpenAI API", "Collections"])

        with tab1:
            st.subheader("Qdrant API 🔑")
            st.write("No API key is required to connect to the Qdrant service.")
            st.success("Successfully connected to Qdrant!", icon="⚡️")

        with tab3:
            st.subheader("Qdrant Collections")
            qdrant_client = get_qdrant_client()

            selection_qdrant = st.selectbox(
                "Choose one of these options:",
                ("Create Collection", "Get All Collections", "Delete Collection"),
                placeholder="Waiting..."
            )

            if selection_qdrant == "Create Collection":
                with st.form("Create New Collection", clear_on_submit=True):
                    collection_name = st.text_input("Name your new collection: ", placeholder="Has to be a unique name...")
                    submitted = st.form_submit_button("Add Collection!")
                    if submitted:
                        try:
                            vectors_config = http.models.VectorParams(size=1536, distance=http.models.Distance.COSINE)
                            qdrant_client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
                            st.success(f"The Collection titled '{collection_name}' has been created!", icon="🙌")
                        except Exception as e:
                            st.warning(f"This collection name already exists. Please choose a unique name!", icon="🚨")

            if selection_qdrant == "Get All Collections":
                collections = qdrant_client.get_collections().dict()["collections"]
                for i in collections:
                    st.write(f"- {i['name']}")

            if selection_qdrant == "Delete Collection":
                collections = qdrant_client.get_collections().dict()["collections"]
                collection_to_delete = st.selectbox(label="Please choose a collection to delete:", options=[i['name'] for i in collections])
                delete_button = st.button("Delete!")
                if delete_button:
                    qdrant_client.delete_collection(collection_name=collection_to_delete)
                    st.success(f"Collection '{collection_to_delete}' has been deleted!", icon="💨")

    st.header("Story Query and Generation")

    qdrant_client = get_qdrant_client()
    collections = qdrant_client.get_collections().dict()["collections"]
    collection_to_store = st.selectbox(label="Please choose a collection to store the text you wish to query:", options=[i['name'] for i in collections])
    embeddings = OpenAIEmbeddings()  # Can switch to a local embedding model

    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=collection_to_store,
        embeddings=embeddings,
    )

    # Direct search for story in Qdrant
    story_title = st.text_input("Enter the title of the story to query:")
    if story_title:
        try:
            search_results = vector_store.similarity_search(story_title, k=1)  # Search by similarity
            if search_results:
                st.subheader("Story:")
                st.write(search_results[0].page_content)
            else:
                st.write("No story found for the given title.")
        except Exception as e:
            st.error(f"Error connecting to Qdrant: {e}")

    # Generate a new story with OpenAI (add exponential backoff for retries)
    new_story_prompt = st.text_area("Generate a new story based on the existing stories:", height=200)
    if st.button("Generate New Story"):
        try:
            @retry_with_exponential_backoff
            def generate_story(prompt):
                return openai.Completion.create(
                    model="gpt-3.5-turbo",
                    prompt=prompt,
                    max_tokens=200
                )

            if new_story_prompt:
                response = generate_story(new_story_prompt)
                st.subheader("Generated Story:")
                st.write(response['choices'][0]['text'])

        except Exception as e:
            st.error(f"Error generating story: {e}")

if __name__ == "__main__":
    main()