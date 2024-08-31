!git clone https://github.com/LjupchoStefanov/milvus-bge-m3.git
%cd milvus-bge-m3
!curl -fsSL https://get.docker.com -o get-docker.sh
!sh get-docker.sh
!sudo apt-get update
!sudo apt-get install docker-compose -y
!ls
!docker-compose -f docker-compose-v2.4-standalone.yml up -d
!pip install pyautogen
!pip install openai
import os
os.environ['OPENAI_API_KEY'] = 'key'
import autogen
print("autogen version:", autogen.__version__)
from autogen import ConversableAgent
storytelling_agent = ConversableAgent(
    "storytelling_agent",
    system_message="You are a storytelling agent that can create and tell stories for children. Use your knowledge from children's books to craft engaging and creative stories. When asked, generate a new story based on existing ones.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
    human_input_mode="NEVER",  
)
child_user_agent = ConversableAgent(
    "child_user_agent",
    system_message="You are a child who loves stories. You can ask the storytelling agent to tell you a story, or ask for a new story if you have already heard the previous ones.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
    human_input_mode="ALWAYS",  
)
!pip install kaggle
!kaggle datasets download -d edenbd/children-stories-text-corpus
!unzip children-stories-text-corpus.zip -d /content/children_stories
!pip install pymilvus
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
def get_text_embedding(text):
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()
fields = [
    FieldSchema(name="story_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
!docker ps
!docker-compose -f docker-compose-v2.4-standalone.yml up -d
!docker-compose down
!docker-compose up -d
connections.connect("default", host="localhost", port="19530")
schema = CollectionSchema(fields, description="Embeddings на приказните")
collection_name = "children_stories"
collection = Collection(name=collection_name, schema=schema)
df = pd.read_txt('/content/children_stories/cleaned_merged_fairy_tales_without_eos.txt')
embeddings = df['story_column'].apply(get_text_embedding).tolist(
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
