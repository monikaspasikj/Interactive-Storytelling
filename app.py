import time
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from dotenv import load_dotenv
from gtts import gTTS
import openai
import os
import tempfile
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import uuid  # For generating unique IDs

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings and Qdrant client
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST"), 
    port=int(os.getenv("QDRANT_PORT"))
)

# Define collection name and embedding dimension
collection_name = "children_stories"
embedding_dimension = 1536

# Check if the collection exists; create it if not
try:
    qdrant_client.get_collection(collection_name)
except Exception:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )

# Initialize the vector store
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

# Create a retrieval instance
retriever = vector_store.as_retriever()

# Create the RetrievalQA instance
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever,
)

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
    
    # Query stories using RetrievalQA
    story_title = st.text_input("Enter a story title to search:")
    if story_title:
        result = qa_chain({"query": story_title})
        st.write("Search Result:")
        st.write(result['result'])

    # Input for generating a new story
    prompt = st.text_area("Enter a prompt for a new story:")

    # Generate a new story based on the entered prompt
    if st.button("Generate New Story"):
        if prompt:
            st.write("Generating story...")
            response = qa_chain({"query": prompt})
            generated_story = response['result'].strip()
            st.subheader("Generated Story:")
            st.write(generated_story)

            # Convert generated story to audio
            audio_file_path = text_to_audio(generated_story)
            st.audio(audio_file_path, format="audio/mp3")
            os.remove(audio_file_path)
        else:
            st.write("Please enter a prompt to generate a story.")

if __name__ == "__main__":
    main()