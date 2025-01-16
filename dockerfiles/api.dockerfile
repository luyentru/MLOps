# Use a specific Python version as base
FROM python:3.11-slim AS base

# Set the working directory in the container
WORKDIR /MLOPS

# Copy requirements.txt into the container
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# Copy the application code into the container
COPY src ./src
COPY models ./models
COPY data ./data

# Set the command to run the API
CMD ["uvicorn", "src.pet_fac_rec.api:app", "--host", "0.0.0.0", "--port", "80"]
