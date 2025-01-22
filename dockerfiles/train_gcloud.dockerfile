# Stage 1: Base Image with CUDA and Python 3.11
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS cuda-base

# Configuring TZ
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        software-properties-common \
        build-essential \
        gcc \
        git \
        curl && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py --output get-pip.py && \
    python3.11 get-pip.py && rm get-pip.py && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as an alternative
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

# Stage 2: Final Image with Python and Dependencies
FROM cuda-base AS final

# Install google-cloud-sdk from APT repository
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the PATH for google-cloud-sdk
ENV PATH="/root/google-cloud-sdk/bin:$PATH"

# Create working directory with secure permissions
RUN mkdir -p /workspace && chmod 755 /workspace

# Copy necessary files into the container
COPY requirements.txt pyproject.toml README.md gcloud_container_command.sh data.dvc vertex_train.dvc /workspace/
COPY src/ /workspace/src/

# Ensure the script is executable
RUN chmod +x /workspace/gcloud_container_command.sh

# Set working directory
WORKDIR /workspace

# Install Python dependencies in a single layer
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt && \
    python3.11 -m pip install --no-cache-dir .

# Set the entrypoint to execute the script
ENTRYPOINT ["bash", "/workspace/gcloud_container_command.sh"]
