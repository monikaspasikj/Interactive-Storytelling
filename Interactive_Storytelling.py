import subprocess
import os
import autogen
import pandas as pd
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
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=os.getenv("QDRANT_PORT", "6333"))

# Initialize LangChain's embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LangChain's LLM with OpenAI API
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

# Function to run shell commands (optional for debugging, not needed in Docker)
def run_shell_command(command):
    result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

# Setup environment variables and packages
print("AutoGen version:", autogen.__version__)

# Initialize AutoGen agents using LangChain as LLM interface
storytelling_agent = autogen.ConversableAgent(
    "storytelling_agent",
    system_message="You are a storytelling agent that can create and tell stories for children. Use your knowledge from children's books to craft engaging and creative stories.",
    llm=llm,  # Use the LangChain LLM instance here
    human_input_mode="NEVER",
)

child_user_agent = autogen.ConversableAgent(
    "child_user_agent",
    system_message="You are a child who loves stories. You can ask the storytelling agent to tell you a story, or ask for a new one.",
    llm=llm,  # Use the LangChain LLM instance here
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
    print(f"Error: {e}")
    stories = []

# Ensure we have stories before proceeding
if stories:
    print(f"Loaded {len(stories)} stories from the dataset.")
    
    # Define Qdrant collection schema and create collection
    qdrant_client.recreate_collection(
        collection_name="children_stories",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
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
    print("Embeddings successfully inserted into Qdrant.")
else:
    print("No stories found to process.")

# Function to simulate the interaction and generate the story
def generate_story(prompt):
    """Simulate a conversation between the child agent and the storytelling agent."""
    try:
        print(f"Child agent sent: {prompt}")
        child_user_agent.receive(prompt, sender=child_user_agent)

        storytelling_prompt = f"Tell me a story about {prompt}"
        print(f"Storytelling prompt: {storytelling_prompt}")

        storytelling_response = storytelling_agent.receive(storytelling_prompt, sender=storytelling_agent)
        print(f"Storytelling agent response: {storytelling_response}")

        if storytelling_response is None:
            return "Error occurred: No response from the storytelling agent."

        if isinstance(storytelling_response, dict) and 'text' in storytelling_response:
            return storytelling_response['text']
        else:
            return f"Error occurred: Unexpected response format from the agent: {storytelling_response}"

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return f"Error occurred: {str(e)}"

# Direct story generation without agents (as a fallback)
def generate_story_direct(prompt):
    try:
        storytelling_response = autogen.get_response(
            query=prompt,
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        if storytelling_response:
            return storytelling_response
        else:
            return "Error: No response from the model."

    except Exception as e:
        return f"Error occurred during direct story generation: {str(e)}"

# Example for searching similar stories using Qdrant
search_embedding = embedder.embed_query("Example story to search")
results = qdrant_client.search(
    collection_name="children_stories",
    query_vector=search_embedding,
    limit=5,
)

# Print results from the search
for result in results:
    print(f"Story ID: {result.id}, Distance: {result.distance}")

# Start communication between agents
child_message = "Can you tell me a new story?"
response_from_storytelling_agent = child_user_agent.send_message(child_message)

# Storytelling agent responds with a story
story = storytelling_agent.receive_message(response_from_storytelling_agent)
print("Storytelling Agent:", story)