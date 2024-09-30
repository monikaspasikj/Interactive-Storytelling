from qdrant_client import QdrantClient
from qdrant_client.http import models
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from os import listdir
from os.path import isfile, join

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Function to read PDFs
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
            pagesText.append(text.replace("\n", " "))

print(pagesText)

# Create a new collection if it doesn't exist
collection_name = "doc_embeddings"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),  # Adjust size if needed
)

# Generate document embeddings (replace this with your actual embedding generation method)
# Assuming bge_m3_ef.encode_documents is a placeholder for your embedding generation
docs_embeddings = []  # Replace this with your method to get embeddings

# Insert documents into Qdrant
for i, text in enumerate(pagesText):
    embedding = ...  # Compute the embedding for the current document text
    client.upload_records(
        collection_name=collection_name,
        records=[models.Record(
            id=i,  # or some unique ID
            vector=embedding,
            payload={"text": text}  # or other metadata
        )]
    )

# Example search
queries = ["What are Spindles"]
query_embeddings = ...  # Compute embeddings for queries

search_result = client.search(
    collection_name=collection_name,
    query_vector=query_embeddings,
    limit=6,
    with_payload=True  # To include payload in the results
)

for hit in search_result:
    print(f"hit: {hit}, ")

print(len(pagesText))