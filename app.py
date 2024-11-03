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
retriever = vector_store.as_retriever()  # Ensure you get the retriever from the vector store

# Create the RetrievalQA instance
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),  # Change model as needed
    chain_type="stuff",  # This can be adjusted based on the desired behavior
    retriever=retriever,  # Pass the retriever here
)

# Function to read stories from the text file
def read_stories_from_file(filename):
    stories = []
    with open(filename, 'r', encoding='utf-8') as file:
        title = None
        text = []
        
        for line in file:
            line = line.strip()
            if not line:  # If the line is blank, continue to the next line
                if title and text:  # If we have a title and text, save the story
                    stories.append({'title': title, 'text': ' '.join(text)})
                    title = None
                    text = []
            elif title is None:  # The first non-blank line should be the title
                title = line
            else:  # Subsequent lines are part of the story
                text.append(line)
        
        # Save the last story if there's no trailing blank line
        if title and text:
            stories.append({'title': title, 'text': ' '.join(text)})
    
    return stories

# Function to insert stories into Qdrant
def insert_stories(stories):
    for story in stories:
        unique_id = str(uuid.uuid4())  # Create a unique ID for each story
        title_embedding = embeddings.embed_query(story['title'])
        point = {
            'id': unique_id,  # Use a UUID as a unique string ID
            'vector': title_embedding,
            'payload': {
                'title': story['title'],
                'text': story['text']
            }
        }
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[point]
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

    # Read stories from file and insert them into Qdrant
    if st.button("Load Stories"):
        stories_to_insert = read_stories_from_file('cleaned_stories_final.txt')
        insert_stories(stories_to_insert)
        st.success("Stories loaded successfully!")

    st.header("Story Query and Generation")
    
    # Query stories using RetrievalQA
    story_title = st.text_input("Enter a story title to search:")
    if story_title:
        result = qa_chain({"query": story_title})
        
        st.write("Search Result:")
        st.write(result['result'])  # The result should contain the text of the story

    # Generate a new story based on a prompt using ChatOpenAI
    if st.button("Generate New Story"):
        prompt = st.text_area("Enter a prompt for a new story:")
        if prompt:
            response = qa_chain({"query": prompt})
            generated_story = response['result'].strip()
            st.subheader("Generated Story:")
            st.write(generated_story)
            audio_file_path = text_to_audio(generated_story)
            st.audio(audio_file_path, format="audio/mp3")
            os.remove(audio_file_path)

if __name__ == "__main__":
    main()