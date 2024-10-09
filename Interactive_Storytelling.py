import subprocess
import os
import autogen
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables from .env file
load_dotenv()

# Initialize Qdrant client (connection details from environment variables)
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=os.getenv("QDRANT_PORT", "6333")
)

# Initialize LangChain's embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to run shell commands (optional for debugging, not needed in Docker)
def run_shell_command(command):
    result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

# Setup environment variables and packages
print("AutoGen version:", autogen.__version__)

# Initialize AutoGen agents
storytelling_agent = autogen.ConversableAgent(
    "storytelling_agent",
    system_message="You are a storytelling agent that can create and tell stories for children. Use your knowledge from children's books to craft engaging and creative stories. When asked, generate a new story based on existing ones.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",
)

child_user_agent = autogen.ConversableAgent(
    "child_user_agent",
    system_message="You are a child who loves stories. You can ask the storytelling agent to tell you a story, or ask for a new story if you have already heard the previous ones.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]},
    human_input_mode="ALWAYS",
)

# Function to simulate the interaction and generate the story
def generate_story(prompt):
    """Simulate a conversation between the child agent and the storytelling agent."""
    try:
        # Send the prompt to the child agent
        print(f"Child agent sent: {prompt}")
        child_user_agent.receive(prompt, sender=child_user_agent)

        # Construct a prompt for the storytelling agent
        storytelling_prompt = "Tell me a story about " + prompt  # Custom prompt for storytelling agent

        # Send the storytelling prompt to the storytelling agent
        storytelling_response = storytelling_agent.receive(storytelling_prompt, sender=storytelling_agent)

        # Debug: Check the response from the storytelling agent
        print(f"Storytelling agent received: {storytelling_response}")

        # Check if the response is None
        if storytelling_response is None:
            return "Error occurred: No response from the storytelling agent."

        # Ensure it's a dictionary and contains the 'text' key
        if isinstance(storytelling_response, dict) and 'text' in storytelling_response:
            return storytelling_response['text']
        else:
            return f"Error occurred: Unexpected response format from the agent: {storytelling_response}"

    except Exception as e:
        print(f"Exception occurred: {str(e)}")  # Debugging
        return f"Error occurred: {str(e)}"

# Preprocess the dataset from .txt format
def preprocess_txt_dataset(file_path):
    """Reads the .txt dataset and splits it into individual stories."""
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Assuming each story is separated by double newlines
    stories = data.split('\n\n')

    # Filter out any empty stories
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
    test_embedding = embedder.embed_query("test")
    if test_embedding is None:
        raise ValueError("Failed to generate test embedding. Check the embedding model.")

    qdrant_client.recreate_collection(
        collection_name="children_stories",
        vectors_config=models.VectorParams(size=len(test_embedding), distance=models.Distance.COSINE),
    )

    # Generate embeddings for each story and insert into Qdrant
    embeddings = embedder.embed_documents(stories)

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