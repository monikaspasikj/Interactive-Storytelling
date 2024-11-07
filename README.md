# Interactive Storytelling App

This project is an interactive storytelling application that leverages AI to provide a dynamic user experience. The app allows users to retrieve stories, generate new ones, and use voice input for interactive story creation. It integrates OpenAI for language generation, Qdrant for vector storage, and Streamlit for the user interface.

## Features

- **Retrieve Story by Title**: Users can input a title of a story to retrieve the full text and audio from a pre-stored collection of children’s stories.
- **Generate a New Story**: Users can input a prompt to generate a unique story with text and audio output.
- **Speech-to-Text Story Query**: Users can use voice input to request stories, and the app transcribes and responds with the full story and audio.

## Architecture

- **OpenAI GPT-3.5-turbo**: For generating and summarizing stories based on user input or prompts.
- **Qdrant**: Vector database used to store and retrieve stories based on title or user prompts.
- **LangChain**: Framework to streamline interactions between Qdrant, OpenAI, and other components.
- **Streamlit**: Web-based front-end framework that presents the app's interface.

## File Structure

- **`app.py`**: Main file for Streamlit app setup and functionality implementation.
- **`bgem3.py`**: Script to process and upload cleaned stories into the Qdrant vector database.
- **`Interactive_Storytelling.py`**: Core functionality for story retrieval, generation, and audio output.
- **`docker-compose.yml`**: Docker configuration for running the app and Qdrant container.
- **`requirements.txt`**: Python dependencies needed for the project.
- **`.env`**: Environment variables to store API keys and host configurations.

## Setup and Installation

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key
- Qdrant database setup

### Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
