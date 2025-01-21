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
COPY requirements_frontend.txt pyproject.toml README.md ./
COPY src/ src/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements_frontend.txt

# Set work directory to root directory
WORKDIR /app

# Expose port 9000 for frontend application
EXPOSE 9000

# Set the entrypoint to serve the BentoML service
ENTRYPOINT ["streamlit", "run", "src/streamlit/frontend.py", "--server.port=9000"]
