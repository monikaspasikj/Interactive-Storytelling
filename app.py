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

# Function to retrieve the full story by title
def retrieve_story_by_title(title):
    normalized_title = title.strip().lower()
    filter_condition = {"must": [{"key": "title", "match": {"value": normalized_title}}]}
    chunks = vector_store.similarity_search_with_score(normalized_title, k=100, filter=filter_condition)
    if not chunks:
        results = vector_store.similarity_search(normalized_title, k=5)
        if results:
            return ' '.join([result.page_content for result in results])
        return None
    story = ' '.join(chunk.page_content for chunk, _ in chunks)
    return story

# Function to retrieve the best snippet for a phrase
def retrieve_best_match(phrase):
    results = vector_store.similarity_search_with_score(phrase, k=5)
    best_result, best_score = max(results, key=lambda item: item[1])
    return best_result.page_content

# Function to generate a story from a prompt
def generate_story_from_prompt(prompt):
    full_prompt = (
        f"Write a detailed and engaging story about: {prompt}. "
        "Make it imaginative, with interesting characters, an exciting plot, and a satisfying conclusion. "
        "Include rich descriptions and dialogue where appropriate."
    )
    response = qa_chain({"query": full_prompt})
    generated_story = response["result"].strip()
    if not generated_story or "I don't know" in generated_story.lower():
        fallback_prompt = (
            f"Imagine a magical or futuristic world and craft a unique story based on: {prompt}. "
            "Include a central conflict, character growth, and a resolution."
        )
        response = qa_chain({"query": fallback_prompt})
        generated_story = response["result"].strip()
    return generated_story

# Convert text to audio
def text_to_audio(text):
    filename = hashlib.md5(text.encode('utf-8')).hexdigest() + ".mp3"
    filepath = os.path.join(tempfile.gettempdir(), filename)
    tts = gTTS(text)
    tts.save(filepath)
    return filepath

def audio_to_text(audio_file):
    audio_data = audio_file.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file.close()
        result = whisper_model.transcribe(temp_audio_file.name)
        os.remove(temp_audio_file.name)
        return result["text"]

# Streamlit setup
def main():
    st.sidebar.subheader("Qdrant API Status")
    st.success("Connected to Qdrant!", icon="⚡️")
    st.title("Interactive Storytelling App")

    tab1, tab2, tab3 = st.tabs(["Conversational Story Query", "Generate New Story", "Speech-to-Text Query"])

    # Conversational Story Query Tab
    with tab1:
        st.markdown("### Conversational Story Query")
        search_query = st.text_input("Ask about a story, phrase, or title:")

        if search_query:
            if search_query.istitle():  # Full story for titles
                story_text = retrieve_story_by_title(search_query)
                if story_text:
                    st.markdown("#### Full Story:")
                    st.markdown(story_text)
                    audio_file = text_to_audio(story_text)
                    st.audio(audio_file, format="audio/mp3")
                    os.remove(audio_file)
                else:
                    st.markdown("**No story found with that title.**")
            elif len(search_query.split()) < 6:  # Specific phrase match
                best_snippet = retrieve_best_match(search_query)
                if best_snippet:
                    st.markdown("#### Best Matching Snippet:")
                    st.markdown(best_snippet)
                    audio_file = text_to_audio(best_snippet)
                    st.audio(audio_file, format="audio/mp3")
                    os.remove(audio_file)
                else:
                    st.markdown("**No matching content found.**")
            else:  # General conversational queries
                conversational_prompt = f"Provide a detailed and engaging response based on the story collection for the query: '{search_query}'. Include relevant details to expand the answer."
                st.write("Generating conversational response...")
                conversational_response = generate_story_from_prompt(conversational_prompt)

                st.markdown("#### Conversational Response:")
                st.markdown(conversational_response)

                audio_file_path = text_to_audio(conversational_response)
                st.audio(audio_file_path, format="audio/mp3")
                os.remove(audio_file_path)

    # Generate New Story Tab
    with tab2:
        st.header("Generate New Story")
        prompt = st.text_area("Enter a prompt for a new story:")
        if st.button("Generate New Story"):
            if prompt:
                st.write("Generating story...")
                generated_story = generate_story_from_prompt(prompt)
                st.subheader("Generated Story:")
                st.write(generated_story)

                audio_file_path = text_to_audio(generated_story)
                st.audio(audio_file_path, format="audio/mp3")
                os.remove(audio_file_path)
            else:
                st.write("Please enter a prompt to generate a story.")

    # Speech-to-Text Query Tab
    with tab3:
        st.markdown("### Speech-to-Text Query")
        audio_value = st.audio_input("Record a voice message")
        
        if audio_value:
            st.write("Audio recorded! Processing...")
            transcribed_text = audio_to_text(audio_value)
            st.markdown("#### Transcribed Text:")
            st.markdown(transcribed_text)
            
            story_prompt = f"Create a detailed, imaginative story set in a magical place based on the theme: '{transcribed_text}'. Include interesting characters, an exciting plot, and an engaging ending."
            st.write("Generating a new story based on transcribed text...")
            response = qa_chain({"query": story_prompt})
            generated_story = response["result"].strip()
            
            if "I don't have" in generated_story or len(generated_story) < 100:
                fallback_prompt = f"Imagine an adventure story inspired by the concept: '{transcribed_text}'."
                response = qa_chain({"query": fallback_prompt})
                generated_story = response["result"].strip()
            
            st.markdown("#### Generated Story from Audio Query:")
            st.markdown(generated_story)

if __name__ == "__main__":
    main()