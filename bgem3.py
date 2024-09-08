from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection
)
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from os import listdir
from os.path import isfile, join


onlyfiles = [f for f in listdir("./pdf_files") if isfile(join("./pdf_files/", f))]
pagesText = []

for file in onlyfiles:
    try:
        reader = PdfReader(join("./pdf_files/", file))
    except PdfReadError:
        print("invalid PDF file")
    else:
        for page in reader.pages: 
            text = page.extract_text() 
            # print(text)
            pagesText.append(text.replace("\n", " ")) 


print(pagesText)

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3', # Specify the model name
    device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)


connections.connect("default", host="127.0.0.1", port="19530")


fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="index", dtype=DataType.INT64)
]

schema = CollectionSchema(fields, description="Document Embeddings Collection")
collection_name = "doc_embeddings"
collection = Collection(name=collection_name, schema=schema)
# utility.drop_collection(collection_name)

if utility.has_collection(collection_name):
    collection.release()


if collection.has_index():
    collection.drop_index()

index_params = {
    "index_type": "IVF_FLAT",  # Inverted File System
    "metric_type": "L2",  # Euclidean distance
    "params": {"nlist": 128}
}

collection.create_index("embedding", index_params)
print("New index created successfully.")

collection.load()

docs_embeddings = bge_m3_ef.encode_documents(pagesText)

print(docs_embeddings)
entities= []

for i in range(len(docs_embeddings["dense"])):
    entities.append({"embedding": docs_embeddings["dense"][i], "index": i})

insert_result = collection.insert(entities)

collection.load()

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

queries = ["What are Spindles"]
query_embeddings = bge_m3_ef.encode_queries(queries)  # Example: search with the first two embeddings
result = collection.search(
    query_embeddings['dense'], 
    "embedding", 
    search_params, 
    limit=6, 
    output_fields=["index"]
)

print(len(pagesText))

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, ")

