import os
import autogen
import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Initialize Qdrant client (connection details from environment variables)
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=os.getenv("QDRANT_PORT", "6333")
)

# Initialize LangChain's embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LangChain's LLM with OpenAI API
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")  # Use gpt-3.5-turbo for better access

# Setup environment variables and packages
st.write("AutoGen version:", autogen.__version__)

# Initialize AutoGen agents
storytelling_agent = autogen.ConversableAgent(
    "storytelling_agent",
    system_message="You are a storytelling agent that can create and tell stories for children. Use your knowledge from children's books to craft engaging and creative stories.",
    human_input_mode="NEVER",
)

child_user_agent = autogen.ConversableAgent(
    "child_user_agent",
    system_message="You are a child who loves stories. You can ask the storytelling agent to tell you a story, or ask for a new one.",
    human_input_mode="ALWAYS",
)

# Preprocess the dataset from .txt format
def preprocess_txt_dataset(file_path):
    """Reads the .txt dataset and splits it into individual stories."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    stories = data.split('\n\n')
    stories = [story.strip() for story in stories if story.strip()]
    if not stories:
        raise ValueError("The dataset appears to be empty after preprocessing.")
    
    return stories

# Load the .txt dataset and preprocess it
file_path = 'cleaned_merged_fairy_tales_without_eos.txt'
try:
    stories = preprocess_txt_dataset(file_path)
except (FileNotFoundError, ValueError) as e:
    st.write(f"Error: {e}")
    stories = []

# Ensure we have stories before proceeding
if stories:
    st.write(f"Loaded {len(stories)} stories from the dataset.")
    
    # Define Qdrant collection schema and create collection
    qdrant_client.recreate_collection(
        collection_name="children_stories",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    # Generate embeddings for each story using LangChain and insert into Qdrant
    embeddings = np.array([embedder.embed_query(story) for story in stories])

    if embeddings is None:
        raise ValueError("Failed to generate embeddings for the stories.")

    payload = [{"story_id": i, "text": stories[i]} for i in range(len(embeddings))]

    # Insert embeddings into Qdrant
    qdrant_client.upload_collection(
        collection_name="children_stories",
        vectors=embeddings,
        payload=payload,
    )
    st.write("Embeddings successfully inserted into Qdrant.")
else:
    st.write("No stories found to process.")

# Function to generate a story based on a user prompt
def generate_story(story_prompt):
    """Simulate a conversation between the child agent and the storytelling agent."""
    try:
        # Receive the prompt from the child
        child_user_agent.receive(story_prompt, recipient=storytelling_agent)

        # Pass the story prompt to the storytelling agent
        storytelling_prompt = f"Tell me a story about {story_prompt}"
        st.write(f"Storytelling prompt: {storytelling_prompt}")

        storytelling_response = storytelling_agent.receive(storytelling_prompt, recipient=storytelling_agent)
        st.write(f"Storytelling agent response: {storytelling_response}")

        if storytelling_response is None:
            return "Error occurred: No response from the storytelling agent."

        if isinstance(storytelling_response, dict) and 'text' in storytelling_response:
            return storytelling_response['text']
        else:
            return f"Error occurred: Unexpected response format from the agent: {storytelling_response}"

    except Exception as e:
        st.write(f"Exception occurred: {str(e)}")
        return f"Error occurred: {str(e)}"

# Function to search for similar stories using Qdrant
def search_story(search_query):
    """Search for stories related to the user's query."""
    search_embedding = embedder.embed_query(search_query)  # Create an embedding for the search query
    results = qdrant_client.search(
        collection_name="children_stories",
        query_vector=search_embedding,
        limit=5,
    )
    return results

# Streamlit UI to input story prompt and show the generated story
st.title("Interactive Storytelling with LLM")

# Input field for the user to type a story prompt
user_story_prompt = st.text_input("Enter a story prompt or request:")

# Generate story button
if st.button("Generate Story"):
    if user_story_prompt:
        # Generate the story using the LLM based on the input prompt
        story_response = generate_story(user_story_prompt)
        st.write("Generated Story:")
        st.write(story_response)
    else:
        st.write("Please type a prompt for the story.")

# Input field for the user to search for a story
search_query = st.text_input("Search for a story:")

# Search story button
if st.button("Search Story"):
    if search_query:
        search_results = search_story(search_query)
        if search_results:
            st.write("Search Results:")
            for result in search_results:
                st.write(f"Story ID: {result.id}, Score: {result.score}, Story: {result.payload['text']}")
        else:
            st.write("No matching stories found.")
    else:
        st.write("Please type a search query.")
