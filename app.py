import os
import tempfile
import time
import uuid  # For generating unique IDs
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse
from gtts import gTTS
import openai
import whisper
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
except UnexpectedResponse:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )
    collection_exists = False

# Initialize the vector store only if the collection was just created
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)
if not collection_exists:
    # Load your dataset of cleaned stories only once
    with open("cleaned_stories_files.txt", "r") as f:
        stories = f.readlines()
    for story in stories:
        vector_store.add_texts([story])

# Create a retrieval instance
retriever = vector_store.as_retriever()

# Create the RetrievalQA instance
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever,
)

# Convert text to audio function
def text_to_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Audio Processor for capturing microphone input
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv_audio(self, frames):
        self.audio_frames.extend(frames)
        return frames

# Speech-to-Text processing using Whisper
def audio_to_text(audio_frames):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
        audio_file.write(b"".join(audio_frames))
        audio_file_path = audio_file.name

    # Transcribe the audio using Whisper
    result = whisper_model.transcribe(audio_file_path)
    os.remove(audio_file_path)
    return result["text"]

# Main application logic
def main():
    st.sidebar.subheader("Qdrant API Status")
    st.success("Connected to Qdrant!", icon="⚡️")

    st.header("Story Query and Generation")

    # Microphone input section
    st.subheader("Speak to Search for a Story")
    audio_ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun1.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True},
        audio_processor_factory=AudioProcessor,
    )

    if audio_ctx and audio_ctx.state.playing:
        if st.button("Transcribe and Search"):
            # Retrieve frames and transcribe them
            audio_frames = audio_ctx.audio_processor.audio_frames
            if audio_frames:
                transcribed_text = audio_to_text(audio_frames)
                st.write("Transcribed Text:")
                st.write(transcribed_text)

                # Query based on transcribed text
                result = qa_chain({"query": transcribed_text})
                story_result = result['result'].strip()
                st.write("Story Result:")
                st.write(story_result)

                # Convert story result to audio
                audio_file_path = text_to_audio(story_result)
                st.audio(audio_file_path, format="audio/mp3")
                os.remove(audio_file_path)

    # Additional text-based query for stories
    story_title = st.text_input("Enter a story title to search:")
    if story_title:
        result = qa_chain({"query": story_title})
        st.write("Search Result:")
        st.write(result['result'])

    # Input for generating a new story
    prompt = st.text_area("Enter a prompt for a new story:")
    if st.button("Generate New Story"):
        if prompt:
            st.write("Generating story...")
            response = qa_chain({"query": prompt})
            generated_story = response['result'].strip()
            st.subheader("Generated Story:")
            st.write(generated_story)

            # Convert generated story to audio
            audio_file_path = text_to_audio(generated_story)
            st.audio(audio_file_path, format="audio/mp3")
            os.remove(audio_file_path)
        else:
            st.write("Please enter a prompt to generate a story.")

if __name__ == "__main__":
    main()