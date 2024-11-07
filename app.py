import os
import tempfile
import time
import uuid
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from gtts import gTTS
from langchain_openai import ChatOpenAI
import whisper
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

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

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv_audio(self, frames):
        self.audio_frames.extend(frames)
        return frames

def audio_to_text(audio_frames):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
        audio_file.write(b"".join(audio_frames))
        audio_file_path = audio_file.name
    result = whisper_model.transcribe(audio_file_path)
    os.remove(audio_file_path)
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
        audio_ctx = webrtc_streamer(
            key="audio",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun1.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True},
            audio_processor_factory=AudioProcessor,
        )

        if audio_ctx and audio_ctx.state.playing:
            if st.button("Transcribe and Search", key="transcribe_search"):
                audio_frames = audio_ctx.audio_processor.audio_frames
                if audio_frames:
                    transcribed_text = audio_to_text(audio_frames)
                    st.write("Transcribed Text:")
                    st.write(transcribed_text)

                    result = qa_chain({"query": transcribed_text})
                    story_result = result['result'].strip()
                    st.write("Story Result:")
                    st.write(story_result)

                    audio_file_path_story = text_to_audio(story_result)
                    st.audio(audio_file_path_story, format="audio/mp3")
                    os.remove(audio_file_path_story)

if __name__ == "__main__":
    main()