# Use the official Python image from the Docker Hub
FROM python:3.10

# Set environment variables to non-interactive mode (avoid prompts during package installation)
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    unzip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt first for better caching
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Install additional Python dependencies that are not listed in requirements.txt
RUN pip install pyautogen openai kaggle qdrant-client transformers whisper gtts streamlit python-dotenv scikit-learn \
    langchain langchain-community sentence-transformers flaml[automl]

# Install updated LangChain community packages
RUN pip install -U langchain-community langchain-huggingface langchain-openai  # Added new dependencies

# Copy the rest of the application files into the container
COPY . /app

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]