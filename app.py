import tempfile
import os
import whisper
from gtts import gTTS
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Initialize Whisper model for speech-to-text
whisper_model = whisper.load_model("base")

# Initialize OpenAI embeddings and Qdrant client
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT"))
)

# Define collection name and embedding dimension
collection_name = "children_stories"
embedding_dimension = 1536

# Check if the collection exists; create it if not
try:
    qdrant_client.get_collection(collection_name)
    collection_exists = True
except:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )
    collection_exists = False

vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)
if not collection_exists:
    with open("cleaned_stories_files.txt", "r") as f:
        stories = f.readlines()
    for story in stories:
        vector_store.add_texts([story])

retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo"),  # Use ChatOpenAI instead of OpenAI
    chain_type="stuff",
    retriever=retriever,
)

def text_to_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

def audio_to_text(audio_file):
    # Get the bytes from the audio file
    audio_data = audio_file.getvalue()

    # Save the audio input to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file.close()
        
        # Use Whisper to transcribe the audio file
        result = whisper_model.transcribe(temp_audio_file.name)
        
        # Remove the temporary audio file after transcription
        os.remove(temp_audio_file.name)
        
        return result["text"]

def main():
    st.sidebar.subheader("Qdrant API Status")
    st.success("Connected to Qdrant!", icon="⚡️")

    st.title("Interactive Storytelling App")

    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Retrieve Story by Title", "Generate New Story", "Speech-to-Text Query"])

    # Tab 1: Retrieve Story by Title
    with tab1:
        st.header("Retrieve Story by Title")
        story_title = st.text_input("Enter a story title to search:")
    
        if story_title:
            # Embed the title and perform a direct similarity search in Qdrant
            query_vector = embeddings.embed_query(story_title)
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=1
            )

            if results:
                # Retrieve the full story from the first search result
                story_text = results[0].payload.get("text", "No story text found.")
            
                # Display the full story text
                st.write("Full Story:")
                st.write(story_text)
            
                # Convert the retrieved story to audio
                audio_file_path_story = text_to_audio(story_text)
                st.audio(audio_file_path_story, format="audio/mp3")
                os.remove(audio_file_path_story)
            else:
                st.write("No stories found with that title.")

    # Tab 2: Generate New Story
    with tab2:
        st.header("Generate New Story")
        prompt = st.text_area("Enter a prompt for a new story:")
        if st.button("Generate Story", key="generate_story"):
            if prompt:
                st.write("Generating story...")

                # Correctly invoke ChatOpenAI to get a response
                chat_openai = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.8)
                response = chat_openai([{"role": "user", "content": prompt}])

                # Access the generated story content
                generated_story = response.content.strip()

                st.write("Generated Story:")
                st.write(generated_story)

                # Convert the generated story to audio and display
                audio_file_path = text_to_audio(generated_story)
                st.audio(audio_file_path, format="audio/mp3")
                os.remove(audio_file_path)

    # Tab 3: Speech-to-Text Query
    with tab3:
        st.header("Speech-to-Text Story Query")
        audio_value = st.audio_input("Record a voice message")

        if audio_value:
            st.write("Audio recorded! Processing...")

            # Convert the recorded audio to text
            transcribed_text = audio_to_text(audio_value)

            # Display the transcribed text
            st.write("Transcribed Text:")
            st.write(transcribed_text)

            # Search in Qdrant collection for the query
            query_vector = embeddings.embed_query(transcribed_text)
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=1
            )

            if results:
                story_text = results[0].payload.get("text", "No story text found.")
                st.write("Story Result:")
                st.write(story_text)

                # Convert the retrieved story to audio
                audio_file_path_story = text_to_audio(story_text)
                st.audio(audio_file_path_story, format="audio/mp3")
                os.remove(audio_file_path_story)
            else:
                st.write("No matching story found.")

if __name__ == "__main__":
    main()
