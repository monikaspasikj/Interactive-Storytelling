# Use the compatible Python image from the Docker Hub
FROM python:3.10  

# Set environment variables to non-interactive mode (avoid prompts during package installation)
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    unzip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from requirements.txt using cache
RUN --mount=type=cache,target=/root/.cache \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Install updated LangChain packages for HuggingFace and Qdrant
RUN pip install langchain-huggingface langchain-qdrant

# Copy the rest of the application files into the container
COPY . /app

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]