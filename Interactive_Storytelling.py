import subprocess
import os
import autogen
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# Install qdrant-client
# pip install qdrant-client

# Initialize Qdrant client (in-memory for testing or persistent for production)


# Load environment variables from .env file
load_dotenv()

qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=os.getenv("QDRANT_PORT", "6333"))  # In-memory for testing, CI/CD
# OR for disk persistence:
# qdrant_client = QdrantClient(path="path/to/db")


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

# Load models
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_text_embedding(text):
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()

# Define Qdrant collection schema and create collection
qdrant_client.recreate_collection(
    collection_name="children_stories",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
)

# Load and process data
df = pd.read_csv('cleaned_merged_fairy_tales_without_eos.csv')
embeddings = np.array(df['story_column'].apply(get_text_embedding).tolist())

# Insert data into Qdrant collection
payload = [{"story_id": i} for i in range(len(embeddings))]
qdrant_client.upload_collection(
    collection_name="children_stories",
    vectors=embeddings,
    payload=payload,
)
