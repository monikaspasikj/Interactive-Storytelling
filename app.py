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
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
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
    print(f"Collection '{collection_name}' exists.")
except Exception as e:
    print(f"Collection not found: {e}. Creating a new collection.")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )
    print(f"Collection '{collection_name}' created successfully.")

# Initialize the vector store
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

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
    
    print(f"Read {len(stories)} stories from file.")
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
        print(f"Inserted story with title: '{story['title']}' and ID: {unique_id}")

# Convert text to audio function
def text_to_audio(text):
    """Convert text to audio and return the path to the saved MP3 file."""
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
    
    # Search stories in Qdrant based on title
    story_title = st.text_input("Enter the story title to search:")
    if story_title:
        results = vector_store.similarity_search(query=story_title, k=5)

        if results:
            st.write("Search Results:")
            for result in results:
                try:
                    # Check if result has the expected attributes
                    story_payload = result.metadata if hasattr(result, 'metadata') else {}
                    title = story_payload.get('title', 'Unknown Title')
                    text = story_payload.get('text', 'No text available.')

                    st.write(f"**Title:** {title}\n\n**Text:** {text}")
                    
                    # Convert text to audio only if text is available
                    if text != 'No text available.':
                        audio_file_path = text_to_audio(text)
                        st.audio(audio_file_path, format="audio/mp3")
                        os.remove(audio_file_path)
                except KeyError as e:
                    st.error(f"Error accessing payload: {e}. Result content: {result}")
        else:
            st.write("No valid story found for the given title.")

    # Generate a new story based on a prompt using ChatOpenAI
    if st.button("Generate New Story"):
        prompt = st.text_area("Enter a prompt for a new story:")
        if prompt:
            chat_model = ChatOpenAI()
            try:
                response = chat_model.generate(prompt)
                generated_story = response.strip()
                st.subheader("Generated Story:")
                st.write(generated_story)
                audio_file_path = text_to_audio(generated_story)
                st.audio(audio_file_path, format="audio/mp3")
                os.remove(audio_file_path)
            except Exception as e:
                st.error("Error generating story: {}".format(e))

if __name__ == "__main__":
    main()