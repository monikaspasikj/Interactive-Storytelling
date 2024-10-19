import time
import streamlit as st
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient, http
import os

# Функција за поврзување со Qdrant
def get_qdrant_client():
    qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
    return QdrantClient(host=qdrant_host, port=6333)

def main():
    with st.sidebar:
        tab1, tab2, tab3 = st.tabs(["Qdrant API", "OpenAI API", "Collections"])

        with tab1:
            st.subheader("Qdrant API 🔑")
            st.write("No API key is required to connect to the Qdrant service.")
            st.success("Successfully connected to Qdrant!", icon="⚡️")

        with tab2:
            st.subheader("OpenAI API 🗝️")
            open_ai_api = st.text_input("Please enter your OpenAI Api Key: ", type="password")
            if open_ai_api:
                os.environ['OPENAI_API_KEY'] = open_ai_api
                st.success("OpenAI API Key set!", icon="⚡️")

        with tab3:
            st.subheader("Qdrant Collections")
            qdrant_client = get_qdrant_client()

            selection_qdrant = st.selectbox(
                "Choose one of these options:",
                ("Create Collection", "Get All Collections", "Delete Collection"),
                placeholder="Waiting..."
            )

            if selection_qdrant == "Create Collection":
                with st.form("Create New Collection", clear_on_submit=True):
                    collection_name = st.text_input("Name your new collection: ", placeholder="Has to be a unique name...")
                    submitted = st.form_submit_button("Add Collection!")
                    if submitted:
                        try:
                            vectors_config = http.models.VectorParams(size=1536, distance=http.models.Distance.COSINE)
                            qdrant_client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
                            st.success(f"The Collection titled '{collection_name}' has been created!", icon="🙌")
                        except Exception as e:
                            st.warning(f"This collection name already exists. Please choose a unique name!", icon="🚨")

            if selection_qdrant == "Get All Collections":
                collections = qdrant_client.get_collections().dict()["collections"]
                for i in collections:
                    st.write(f"- {i['name']}")

            if selection_qdrant == "Delete Collection":
                collections = qdrant_client.get_collections().dict()["collections"]
                collection_to_delete = st.selectbox(label="Please choose a collection to delete:", options=[i['name'] for i in collections])
                delete_button = st.button("Delete!")
                if delete_button:
                    qdrant_client.delete_collection(collection_name=collection_to_delete)
                    st.success(f"Collection '{collection_to_delete}' has been deleted!", icon="💨")

    st.header("Story Query and Generation")

    if open_ai_api:
        try:
            qdrant_client = get_qdrant_client()
            collections = qdrant_client.get_collections().dict()["collections"]
            collection_to_store = st.selectbox(label="Please choose a collection to store the text you wish to query:", options=[i['name'] for i in collections])
            embeddings = OpenAIEmbeddings()

            vector_store = Qdrant(
                client=qdrant_client,
                collection_name=collection_to_store,
                embeddings=embeddings,
            )

            # Запрашување на приказна
            story_title = st.text_input("Enter the title of the story to query:")
            if story_title:
                retrieval_qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=vector_store.as_retriever())
                story = retrieval_qa.run(story_title)

                st.subheader("Story:")
                st.write(story)

            # Генерирање нова приказна
            new_story_prompt = st.text_area("Generate a new story based on the existing stories:", height=200)
            if st.button("Generate New Story"):
                generated_story = OpenAI().run(new_story_prompt)
                st.subheader("Generated Story:")
                st.write(generated_story)

        except Exception as e:
            st.error(f"Error connecting to Qdrant: {e}")

if __name__ == "__main__":
    main()
