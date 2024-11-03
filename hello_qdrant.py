import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 1536  # Set to the OpenAI embedding dimension size

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()

# Collection name and configuration
collection_name = "hello_qdrant"
if not client.has_collection(collection_name):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embedding_model.embedding_dim, distance=models.Distance.COSINE)
    )
    print(fmt.format(f"Collection {collection_name} created."))
else:
    print(fmt.format(f"Collection {collection_name} already exists."))

# 2. Insert data
print(fmt.format("Start inserting entities"))
entities = []
for i in range(num_entities):
    text = f"This is example text for entity {i}."
    embedding = embedding_model.embed(text)
    
    entities.append({
        "id": str(i),
        "vector": embedding,
        "payload": {"random": np.random.rand()}
    })

client.upload_records(
    collection_name=collection_name, 
    records=[models.Record(id=entity["id"], vector=entity["vector"], payload=entity["payload"]) for entity in entities]
)
print(f"Number of entities inserted into Qdrant: {len(entities)}")

# 3. Search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))
search_vector = entities[-1]["vector"]
start_time = time.time()
result = client.search(collection_name=collection_name, query_vector=search_vector, limit=3, with_payload=True)
end_time = time.time()

for hit in result:
    print(f"hit: {hit}, random field: {hit.payload.get('random')}")
print(search_latency_fmt.format(end_time - start_time))

# 4. Query based on scalar filtering
print(fmt.format("Start querying with random > 0.5"))
query_result = client.query(collection_name=collection_name, expr="random > 0.5", limit=4)
print(f"Query result:\n-{query_result}")

# 5. Hybrid search
print(fmt.format("Start hybrid searching with random > 0.5"))
hybrid_result = client.search(collection_name=collection_name, query_vector=search_vector, limit=3, expr="random > 0.5", with_payload=True)

for hit in hybrid_result:
    print(f"hit: {hit}, random field: {hit.payload.get('random')}")

# 6. Delete entities by ID
print(fmt.format("Start deleting entities by ID"))
ids_to_delete = [entities[0]["id"], entities[1]["id"]]
client.delete(collection_name=collection_name, ids=ids_to_delete)

remaining_result = client.query(collection_name=collection_name, expr=f"id in {ids_to_delete}")
print(f"Remaining entities after delete: {remaining_result}")

# 7. Drop collection
print(fmt.format(f"Drop collection {collection_name}"))
client.delete_collection(collection_name=collection_name)