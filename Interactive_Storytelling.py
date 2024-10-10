import subprocess
import os
import autogen
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> c831580c1be1ff9b18e37c9f50f49ab792efcc60

# Load environment variables from .env file
load_dotenv()

<<<<<<< HEAD
# Initialize Qdrant client (in-memory for testing or persistent for production)
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=os.getenv("QDRANT_PORT", "6333"))
=======
# Initialize Qdrant client (connection details from environment variables)
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=os.getenv("QDRANT_PORT", "6333")
)

# Initialize LangChain's embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
>>>>>>> c831580c1be1ff9b18e37c9f50f49ab792efcc60

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

<<<<<<< HEAD
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
=======
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
>>>>>>> c831580c1be1ff9b18e37c9f50f49ab792efcc60
