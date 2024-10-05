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

qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=os.getenv("QDRANT_PORT", "6333"))

# Initialize LangChain's embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to run shell commands (not needed in Docker setup)
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

# Define Qdrant collection schema and create collection
qdrant_client.recreate_collection(
    collection_name="children_stories",
    vectors_config=models.VectorParams(size=embedder.embed_query("test").shape[0], distance=models.Distance.COSINE),
)

# Load and process data with additional error handling
try:
    df = pd.read_csv('cleaned_merged_fairy_tales_without_eos.csv', header=None, on_bad_lines='skip')
    print(df.head())  # Debugging: Inspect the first few rows
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)

# Generate embeddings for each story and insert into Qdrant
embeddings = embedder.embed_documents(df[0].tolist())  # Assuming stories are in the first column
payload = [{"story_id": i} for i in range(len(embeddings))]

# Insert embeddings into Qdrant
qdrant_client.upload_collection(
    collection_name="children_stories",
    vectors=embeddings,
    payload=payload,
)
