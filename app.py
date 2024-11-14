import os
import tempfile
import whisper
import hashlib
from gtts import gTTS
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Whisper model for speech-to-text
whisper_model = whisper.load_model("base")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize Qdrant client
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))
collection_name = "children_stories_chunks"
embedding_dimension = 1536

# Ensure collection exists
try:
    qdrant_client.get_collection(collection_name)
    collection_exists = True
except:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )
    collection_exists = False

# Define Qdrant vector store with LangChain integration
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever
)

# Function to load and upload data if collection is empty
def load_and_upload_data(file_path="cleaned_stories_final.txt"):
    if collection_exists:
        print("Data already exists in the collection. Skipping upload.")
        return

    def split_story_into_chunks(story_text, chunk_size=100):
        words = story_text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    titles, texts = [], []
    current_title, current_text = None, ""

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.endswith('.') and stripped_line == stripped_line.title():
                if current_title and current_text:
                    titles.append(current_title)
                    texts.append(current_text.strip())
                current_title = stripped_line
                current_text = ""
            else:
                current_text += f"{stripped_line} "

        if current_title and current_text:
            titles.append(current_title)
            texts.append(current_text.strip())

    if len(titles) != len(texts):
        print("Error: Titles and texts count mismatch. Please check the input file format.")
        return

    for title, text in zip(titles, texts):
        chunks = split_story_into_chunks(text)
        for j, chunk in enumerate(chunks):
            embedding = embeddings.embed_documents([chunk])[0]
            doc = Document(
                page_content=chunk,
                metadata={"title": title, "chunk_index": j}
            )
            vector_store.add_documents([doc])
    print("Data upload completed.")

load_and_upload_data()

# Function to retrieve the entire story by title
def retrieve_story_by_title(title):
    filter_condition = {"must": [{"key": "title", "match": {"value": title}}]}
    chunks = vector_store.similarity_search_with_score(title, k=100, filter=filter_condition)
    story = ' '.join(chunk.page_content for chunk, _ in chunks)
    return story

# Function to retrieve the best result based on a search phrase
def retrieve_best_match(phrase):
    results = vector_store.similarity_search_with_score(phrase, k=5)
    best_result, best_score = max(results, key=lambda item: item[1])
    return best_result.page_content

# Convert text to audio
def text_to_audio(text):
    filename = hashlib.md5(text.encode('utf-8')).hexdigest() + ".mp3"
    filepath = os.path.join(tempfile.gettempdir(), filename)
    tts = gTTS(text)
    tts.save(filepath)
    return filepath

# Streamlit setup
def main():
    st.sidebar.subheader("Qdrant API Status")
    st.success("Connected to Qdrant!", icon="⚡️")
    st.title("Interactive Storytelling App")

    tab1, tab2, tab3 = st.tabs(["Conversational Story Query", "Generate New Story", "Speech-to-Text Query"])

    # Conversational Story Query Tab
    with tab1:
        st.header("Conversational Story Query")
        search_query = st.text_input("Ask about a story, phrase, or title:")

        if search_query:
            if search_query.istitle():  # If query is a title, retrieve the full story
                story_text = retrieve_story_by_title(search_query)
                if story_text:
                    st.subheader("Full Story:")
                    st.write(story_text)
                    audio_file = text_to_audio(story_text)
                    st.audio(audio_file, format="audio/mp3")
                    os.remove(audio_file)
                else:
                    st.write("No story found with that title.")

            elif len(search_query.split()) < 6:  # Check if it might be a story phrase
                best_snippet = retrieve_best_match(search_query)
                if best_snippet:
                    st.subheader("Best Matching Snippet:")
                    st.write(best_snippet)
                    audio_file = text_to_audio(best_snippet)
                    st.audio(audio_file, format="audio/mp3")
                    os.remove(audio_file)
                else:
                    st.write("No matching content found.")
                    
            else:  # Otherwise, generate a conversational response for general queries
                conversational_prompt = f"Answer this question conversationally based on the story collection: '{search_query}'."
                st.write("Generating conversational response...")
                response = qa_chain({"query": conversational_prompt})
                conversational_response = response["result"].strip()

                st.subheader("Conversational Response:")
                st.write(conversational_response)

                audio_file_path = text_to_audio(conversational_response)
                st.audio(audio_file_path, format="audio/mp3")
                os.remove(audio_file_path)

    # Generate New Story Tab
    with tab2:
        st.header("Generate New Story")
        prompt = st.text_area("Enter a prompt for a new story:")
        
        if st.button("Generate New Story"):
            if prompt:
                full_prompt = f"Tell me a creative story about: {prompt}. Make it engaging and magical."
                st.write("Generating story...")
                response = qa_chain({"query": full_prompt})
                generated_story = response["result"].strip()
                
                if "I don't have a specific story" in generated_story:
                    fallback_prompt = f"Imagine a magical world and tell a story about: {prompt}."
                    response = qa_chain({"query": fallback_prompt})
                    generated_story = response["result"].strip()

                st.subheader("Generated Story:")
                st.write(generated_story)

                audio_file_path = text_to_audio(generated_story)
                st.audio(audio_file_path, format="audio/mp3")
                os.remove(audio_file_path)
            else:
                st.write("Please enter a prompt to generate a story.")

    # Speech-to-Text Query Tab
    with tab3:
        st.header("Speech-to-Text Query")
        audio_value = st.audio_input("Record a voice message")
        
        if audio_value:
            st.write("Audio recorded! Processing...")
            transcribed_text = audio_to_text(audio_value)
            st.write("Transcribed Text:")
            st.write(transcribed_text)
            
            story_prompt = f"Create a detailed, imaginative story set in a magical place based on the theme: '{transcribed_text}'. Include interesting characters, an exciting plot, and an engaging ending."
            st.write("Generating a new story based on transcribed text...")
            response = qa_chain({"query": story_prompt})
            generated_story = response["result"].strip()
            
            if "I don't have" in generated_story or len(generated_story) < 100:
                fallback_prompt = f"Imagine an adventure story inspired by the concept: '{transcribed_text}'."
                response = qa_chain({"query": fallback_prompt})
                generated_story = response["result"].strip()
            
            st.subheader("Generated Story from Audio Query:")
            st.write(generated_story)

if __name__ == "__main__":
    main()