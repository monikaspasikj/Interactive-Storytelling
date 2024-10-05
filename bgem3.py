from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.embeddings import HuggingFaceEmbeddings
from os import listdir
from os.path import isfile, join

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Function to read PDFs and extract text
onlyfiles = [f for f in listdir("./pdf_files") if isfile(join("./pdf_files/", f))]
pagesText = []

# Process each PDF and extract text
for file in onlyfiles:
    reader = PdfReader(join("./pdf_files/", file))
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pagesText.append(text.replace("\n", " "))  # Normalize newlines

# Initialize LangChain embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a new collection if it doesn't exist
client.recreate_collection(
    collection_name="doc_embeddings",
    vectors_config=models.VectorParams(size=embedder.embed_query("test").shape[0], distance=models.Distance.COSINE),
)

# Generate document embeddings and insert into Qdrant
docs_embeddings = embedder.embed_documents(pagesText)

for i, embedding in enumerate(docs_embeddings):
    client.upload_records(
        collection_name="doc_embeddings",
        records=[models.Record(id=i, vector=embedding, payload={"text": pagesText[i]})]
    )

# Search example
query = "What are spindles"
query_embedding = embedder.embed_query(query)
search_result = client.search(collection_name="doc_embeddings", query_vector=query_embedding, limit=6, with_payload=True)

for hit in search_result:
    print(hit.payload['text'])
