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

# Load environment variables from .env file
load_dotenv()

# Initialize Qdrant client (in-memory for testing or persistent for production)
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=os.getenv("QDRANT_PORT", "6333"))

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

# Function to get text embeddings
def get_text_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)  # Ограничување на 512 токени
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()

# Патека до текстуалниот фајл
file_path = 'cleaned_merged_fairy_tales_without_eos.txt'

# Читање на текстот од фајлот
with open(file_path, 'r', encoding='utf-8') as file:
    text_data = file.read()

# Подели го текстот по празен ред, кој ги дели приказните
stories = text_data.strip().split('\n\n')

# Иницијализирај листи за наслови и приказни
titles = []
contents = []

# Процесирај секоја секција од текстот
for story in stories:
    lines = story.strip().split('\n')
    title = lines[0].strip()  # Првиот ред е наслов
    content = ' '.join(lines[1:]).strip()  # Остатокот е приказната
    titles.append(title)
    contents.append(content)

# Креирај DataFrame со две колони: 'Title' и 'Story'
df = pd.DataFrame({'Title': titles, 'Story': contents})

# Зачувај го DataFrame во CSV фајл
csv_file_path = 'children_stories.csv'
df.to_csv(csv_file_path, index=False, encoding='utf-8')

# Печати порака за успешно создавање на CSV фајл
print(f'CSV file created successfully: {csv_file_path}')

# Define Qdrant collection schema and create collection
qdrant_client.recreate_collection(
    collection_name="children_stories",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
)

# Load and process data
df = pd.read_csv('children_stories.csv', on_bad_lines='skip')
embeddings = np.array(df['Story'].apply(get_text_embedding).tolist())

# Insert data into Qdrant collection
payload = [{"story_id": i} for i in range(len(embeddings))]
qdrant_client.upload_collection(
    collection_name="children_stories",
    vectors=embeddings,
    payload=payload,
)

# Коментар за индексирање (методот не постои)
# Create an index for faster searching
# index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
# qdrant_client.create_index(collection_name="children_stories", field_name="embedding", index_params=index_params)

# Example for searching similar stories
search_embedding = get_text_embedding("Пример приказна за пребарување")
results = qdrant_client.search(
    collection_name="children_stories",
    query_vector=search_embedding,
    limit=5,
)

# Print results
for result in results:
    print(f"Story ID: {result.id}, Distance: {result.distance}")

# Start communication between agents
child_message = "Can you tell me a new story?"
response_from_storytelling_agent = child_user_agent.send_message(child_message)

# Storytelling agent responds with a story
story = storytelling_agent.receive_message(response_from_storytelling_agent)

print("Storytelling Agent:", story)
