# Use the official Python image from the Docker Hub
FROM python:3.10

# Set environment variables to non-interactive mode (avoid prompts during package installation)
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Copy all the contents of the current directory to /app in the container
COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    unzip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional Python dependencies
RUN pip install pyautogen openai kaggle qdrant-client transformers torch whisper gtts streamlit python-dotenv scikit-learn

# Copy the .env file to the container
COPY .env /app/.env

# Ensure the Interactive_Storytelling.py file is in the container
COPY Interactive_Storytelling.py /app/Interactive_Storytelling.py

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
