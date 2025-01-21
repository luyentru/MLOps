# Base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set work directory to /app
WORKDIR /app

# Copy necessary files
COPY requirements.txt pyproject.toml README.md ./
COPY src/ src/
COPY data/ data/
COPY reports/ reports/
COPY bentoml_api/ bentoml_api/
COPY models/ models/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Set work directory to root directory to load the model
WORKDIR /app
RUN python bentoml_api/src/load_model.py

# Set the work directory to bentoml to build the server
WORKDIR /app/bentoml_api
RUN bentoml build

# Set the entrypoint to serve the BentoML service
EXPOSE 5000
ENTRYPOINT ["bentoml", "serve", "src.service:svc", "--reload", "--host", "0.0.0.0", "--port", "5000"]
