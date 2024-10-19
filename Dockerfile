# Use the official Python image from the Docker Hub
FROM python:3.10

# Set environment variables to non-interactive mode (avoid prompts during package installation)
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Copy only the requirements file first, to cache dependencies
COPY requirements.txt /app/requirements.txt

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    unzip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt first
RUN pip install torch==2.4.1 --no-cache-dir
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Install additional Python dependencies
RUN pip install pyautogen openai kaggle qdrant-client transformers whisper gtts streamlit python-dotenv scikit-learn \
    langchain langchain-community sentence-transformers flaml[automl]

# Install updated LangChain community packages
RUN pip install -U langchain-community langchain-huggingface langchain-openai  # Added new dependencies

# Copy the rest of the application files into the container
COPY . /app

# Copy the .env file to the container
COPY .env /app/.env

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
