# Base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install gsutil (part of google-cloud-sdk)
RUN curl https://sdk.cloud.google.com | bash -s -- --disable-prompts
ENV PATH="/root/google-cloud-sdk/bin:$PATH"

# Copy necessary files
COPY requirements.txt pyproject.toml README.md gcloud_container_command.sh ./
COPY src/ src/

# Set work directory
WORKDIR /

# Install Python dependencies
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install .

ENTRYPOINT ["chmod", "+x", "gcloud_container_command.sh"]

