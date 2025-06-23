# Start from Ollama image (Ubuntu with Ollama preinstalled)
FROM ollama/ollama:latest

# Install Python & pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Set workdir
WORKDIR /app

# Copy your app files
COPY . .

# Install requirements
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# (Optional) Preload a model
RUN ollama pull mistral

# Expose ports: Ollama (11434) and Streamlit (8501)
EXPOSE 11434 8501

# Run both services: Ollama + Streamlit
CMD ollama serve & streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
