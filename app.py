import os
import tempfile
import whisper
import time
import hashlib
from gtts import gTTS
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
import pandas as pd

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Whisper model for speech-to-text
whisper_model = whisper.load_model("base")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize Qdrant client
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))
collection_name = "children_stories_chunks"
embedding_dimension = 1536

# Ensure collection exists
try:
    qdrant_client.get_collection(collection_name)
    collection_exists = True
except:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )
    collection_exists = False

# Verify point count in the collection to determine if upload is necessary
point_count = qdrant_client.count(collection_name)
if point_count.count == 0:
    print("Collection is empty. Uploading data to Qdrant...")
    collection_exists = False
else:
    print(f"Collection '{collection_name}' has {point_count.count} points.")
    collection_exists = True

# Define Qdrant vector store with LangChain integration
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

# Function to load and upload data if collection is empty
def load_and_upload_data(file_path="cleaned_stories_final.txt"):
    if collection_exists:
        print("Data already exists in the collection. Skipping upload.")
        return

    def split_story_into_chunks(story_text, chunk_size=100):
        words = story_text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Initialize lists for titles and texts
    titles, texts = [], []
    current_title, current_text = None, ""

    # Parse stories from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.endswith('.') and stripped_line == stripped_line.title():
                if current_title and current_text:
                    titles.append(current_title)
                    texts.append(current_text.strip())
                current_title = stripped_line
                current_text = ""
            else:
                current_text += f"{stripped_line} "

        if current_title and current_text:
            titles.append(current_title)
            texts.append(current_text.strip())

    # Upload data to Qdrant if titles and texts are the same length
    if len(titles) != len(texts):
        print("Error: Titles and texts count mismatch. Please check the input file format.")
        return

    for i, (title, text) in enumerate(zip(titles, texts)):
        chunks = split_story_into_chunks(text)
        for j, chunk in enumerate(chunks):
            embedding = embeddings.embed_documents([chunk])[0]
            doc = Document(
                page_content=chunk,
                metadata={"title": title, "chunk_index": j}
            )
            vector_store.add_documents([doc])
    print("Data upload completed.")

# Load and upload data to Qdrant if necessary
load_and_upload_data()

# Function to query by title with metadata filtering
def query_by_title(title_query):
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="title",
                match=MatchValue(value=title_query)
            )
        ]
    )
    results = vector_store.similarity_search(query=title_query, k=5, filter=filter_condition)
    return results

# Streamlit setup
def main():
    st.sidebar.subheader("Qdrant API Status")
    st.success("Connected to Qdrant!", icon="⚡️")
    st.title("Interactive Storytelling App")

    tab1, tab2, tab3 = st.tabs(["Retrieve Story by Title or Content", "Generate New Story", "Speech-to-Text Query"])
    with tab1:
        st.header("Retrieve Story by Title or Content")
        search_query = st.text_input("Enter a story title or part of the content to search:")
        if search_query:
            query_results = query_by_title(search_query)
            if query_results:
                st.write("Story Excerpts:")
                for result in query_results:
                    st.write(result.page_content)
            else:
                st.write("No stories found with that title or content.")
    with tab2:
        # Story Generation Tab
        pass  # The rest of the code goes here

if __name__ == "__main__":
    main()