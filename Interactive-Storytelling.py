import subprocess
import os
import autogen
import pandas as pd
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Define Milvus schema and create collection
fields = [
    FieldSchema(name="story_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]

connections.connect("default", host="localhost", port="19530")
schema = CollectionSchema(fields, description="Embeddings на приказните")
collection_name = "children_stories"
collection = Collection(name=collection_name, schema=schema)

# Adjust the path based on Docker setup
df = pd.read_csv('/app/children_stories/cleaned_merged_fairy_tales_without_eos.txt')
embeddings = df['story_column'].apply(get_text_embedding).tolist()

data = [
    [i for i in range(len(embeddings))],
    embeddings
]

collection.insert(data)

index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
