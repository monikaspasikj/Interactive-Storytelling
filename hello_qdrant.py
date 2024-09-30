import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 8

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# 1. Check if collection exists, create if not
collection_name = "hello_qdrant"
if not client.has_collection(collection_name):
    # Create the collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)  # Adjust size if needed
    )
    print(fmt.format(f"Collection {collection_name} created."))
else:
    print(fmt.format(f"Collection {collection_name} already exists."))

# 2. Insert data
print(fmt.format("Start inserting entities"))
rng = np.random.default_rng(seed=19530)
entities = []
for i in range(num_entities):
    entities.append({
        "id": str(i),
        "random": rng.random(),  # field random
        "embeddings": rng.random(dim).astype(np.float32).tolist()  # field embeddings
    })

# Insert entities into Qdrant
client.upload_records(collection_name=collection_name, records=[
    models.Record(id=entity["id"], vector=entity["embeddings"], payload={"random": entity["random"]})
    for entity in entities
])

print(f"Number of entities inserted into Qdrant: {len(entities)}")

# 3. Search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))
vectors_to_search = entities[-1]["embeddings"]
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

start_time = time.time()
result = client.search(collection_name=collection_name, query_vector=vectors_to_search, limit=3, with_payload=True)
end_time = time.time()

for hit in result:
    print(f"hit: {hit}, random field: {hit.payload.get('random')}")
print(search_latency_fmt.format(end_time - start_time))

# 4. Query based on scalar filtering
print(fmt.format("Start querying with random > 0.5"))
query_result = client.query(collection_name=collection_name, expr="random > 0.5", limit=4)
print(f"Query result:\n-{query_result}")

# 5. Hybrid search (combine vector similarity and scalar filtering)
print(fmt.format("Start hybrid searching with random > 0.5"))
hybrid_result = client.search(collection_name=collection_name, query_vector=vectors_to_search, limit=3, expr="random > 0.5", with_payload=True)

for hit in hybrid_result:
    print(f"hit: {hit}, random field: {hit.payload.get('random')}")

# 6. Delete entities by ID (using Qdrant, this example uses a specific ID)
print(fmt.format("Start deleting entities by ID"))
ids_to_delete = [entities[0]["id"], entities[1]["id"]]  # Example IDs
client.delete(collection_name=collection_name, ids=ids_to_delete)

# Confirm deletion
remaining_result = client.query(collection_name=collection_name, expr=f"id in {ids_to_delete}")
print(f"Remaining entities after delete: {remaining_result}")

# 7. Drop collection
print(fmt.format(f"Drop collection {collection_name}"))
client.delete_collection(collection_name=collection_name)