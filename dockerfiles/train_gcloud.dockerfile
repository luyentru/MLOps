# Stage 1: Base Image with CUDA and Python 3.11
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS cuda-base

# Configuring TZ
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN apt-get update && \
    apt-get install -y tzdata && \
    # 2. Link your chosen timezone to localtime so dpkg won't prompt
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install Python 3.11 and other system dependencies
RUN apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        gcc \
        git \
        curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as an alternative
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Stage 2: Final Image with Python and Dependencies
FROM cuda-base AS final

# Install google-cloud-sdk
RUN curl https://sdk.cloud.google.com | bash -s -- --disable-prompts
ENV PATH="/root/google-cloud-sdk/bin:$PATH"

# Copy necessary files into the container
COPY requirements.txt pyproject.toml README.md gcloud_container_command.sh ./
COPY src/ src/

# Ensure the script is executable
RUN chmod +x gcloud_container_command.sh

# Set working directory
WORKDIR /

# Install Python dependencies via the Python 3.11 interpreter
RUN python3.11 -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip python3.11 -m pip install -r requirements.txt
RUN python3.11 -m pip install .

# Set the entrypoint to execute the script
ENTRYPOINT ["bash", "./gcloud_container_command.sh"]
