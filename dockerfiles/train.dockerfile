# Base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy necessary files
COPY requirements.txt pyproject.toml README.md ./
COPY src/ src/
# COPY data/ data/ # Removed due to causing problems with cloud build
COPY reports/ reports/

# Set work directory
WORKDIR /

# Install Python dependencies
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install .


# Set the entrypoint
ENTRYPOINT ["python", "-u", "src/pet_fac_rec/train.py"]
