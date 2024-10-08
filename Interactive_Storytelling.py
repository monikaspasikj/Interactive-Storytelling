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
        # Child sends the initial request (prompt)
        child_user_agent.receive(prompt, sender=child_user_agent)

        # Storytelling agent processes the prompt and generates the story
        storytelling_response = storytelling_agent.receive(prompt, sender=storytelling_agent)

        # Debug: Print the interaction for troubleshooting
        print(f"Child agent sent: {prompt}")
        print(f"Storytelling agent received: {storytelling_response}")

        # Check if the response is None
        if storytelling_response is None:
            return "Error occurred: No response from the storytelling agent."

        # Ensure it's a dictionary and contains the 'text' key
        if isinstance(storytelling_response, dict) and 'text' in storytelling_response:
            return storytelling_response['text']
        else:
            return "Error occurred: Unexpected response format from the agent."

    except Exception as e:
        return f"Error occurred: {str(e)}"

# Define Qdrant collection schema and create collection
test_embedding = embedder.embed_query("test")
if test_embedding is None:
    raise ValueError("Failed to generate test embedding. Check the embedding model.")

qdrant_client.recreate_collection(
    collection_name="children_stories",
    vectors_config=models.VectorParams(size=len(test_embedding), distance=models.Distance.COSINE),
)

# Load and process data with additional error handling
try:
    df = pd.read_csv('cleaned_merged_fairy_tales_without_eos.csv', header=None, on_bad_lines='skip')
    if df is None or df.empty:
        raise ValueError("Failed to load or empty CSV file. Please check the file.")
    print(df.head())  # Debugging: Inspect the first few rows
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)
except ValueError as e:
    print(f"Error: {e}")
    df = None

# Proceed only if the data was successfully loaded
if df is not None:
    # Generate embeddings for each story and insert into Qdrant
    stories = df[0].tolist()  # Assuming stories are in the first column
    embeddings = embedder.embed_documents(stories)
    
    if embeddings is None:
        raise ValueError("Failed to generate embeddings for the stories.")

    payload = [{"story_id": i} for i in range(len(embeddings))]

    # Insert embeddings into Qdrant
    qdrant_client.upload_collection(
        collection_name="children_stories",
        vectors=embeddings,
        payload=payload,
    )
else:
    print("No data to process.")