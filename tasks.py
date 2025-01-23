import os
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor
import warnings
from pathlib import Path

from invoke import Context
from invoke import task


WINDOWS = os.name == "nt"
PROJECT_NAME = "pet_fac_rec"
PYTHON_VERSION = "3.11"


# Helper functions
def check_directory_contents(directory_path: str, num_items: int = 0) -> bool:
    """
    Check if directory has the same number of items than specified threshold.
    Returns True if safe to proceed, False if not.
    """
    if not os.path.exists(directory_path):
        return False
        
    items = os.listdir(directory_path)
    if len(items) == num_items:
        return False
    return True


def sync_directories(dir_a: str, dir_b: str):
    """
    Recursively copies files/directories that exist in dir_a but not in dir_b to dir_b
    """
    try:
        # Convert to Path objects for easier handling
        src_path = Path(dir_a)
        dst_path = Path(dir_b)
        
        # Walk through all files and directories in source
        for src_file in src_path.rglob('*'):
            # Get relative path
            rel_path = src_file.relative_to(src_path)
            dst_file = dst_path / rel_path
            
            # Skip if destination exists
            if dst_file.exists():
                continue
                
            # Create parent directories if they don't exist
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file or directory
            if src_file.is_file():
                shutil.copy2(src_file, dst_file)
            elif src_file.is_dir():
                dst_file.mkdir(parents=True, exist_ok=True)
                
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return -1
 
 
def delete_yaml_files(directory: str) -> None:
    """
    Deletes only .yaml or .yml files from the specified directory.
    """
    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        return

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                path = os.path.join(root, file)
                os.remove(path)
                
                
# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for the project and install invoke."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"conda run -n {PROJECT_NAME} pip install invoke",
        echo=True,
        pty=not WINDOWS,
    )


@task
def delete_environment(ctx: Context) -> None:
    """Delete the conda environment for project."""
    ctx.run("conda deactivate", echo=True, pty=not WINDOWS)
    ctx.run(f"conda remove --name {PROJECT_NAME} --all --yes", echo=True, pty=not WINDOWS)


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run(
        f"{sys.executable} -m pip install -U pip setuptools wheel",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data", echo=True, pty=not WINDOWS)


@task
def traineffnet(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def trainresnet(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py --model-name resnet50", echo=True, pty=not WINDOWS)


@task
def trainvgg(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py --model-name vgg16", echo=True, pty=not WINDOWS)


@task
def evaluateeffnet(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(
        f"python src/{PROJECT_NAME}/evaluate.py --model-name efficientnet --model-checkpoint models/efficientnet.pth",
        echo=True,
        pty=not WINDOWS,
    )


@task
def evaluateresnet(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(
        f"python src/{PROJECT_NAME}/evaluate.py --model-name resnet50 --model-checkpoint models/resnet50.pth",
        echo=True,
        pty=not WINDOWS,
    )


@task
def evaluatevgg(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(
        f"python src/{PROJECT_NAME}/evaluate.py --model-name vgg16 --model-checkpoint models/vgg16.pth",
        echo=True,
        pty=not WINDOWS,
    )


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Cloud commands
@task
def gcloud_build_image(ctx: Context) -> None:
    """Build image in gcloud artifact registry."""
    ctx.run(
        "gcloud builds submit --config=cloudbuild.yaml . "
        "--service-account=projects/pet-fac-rec/serviceAccounts/trigger-builder@pet-fac-rec.iam.gserviceaccount.com",
        echo=True,
        pty=not WINDOWS,
    )


@task
def gcloud_train(ctx, machine: str = "gpu") -> None:
    """
    Run a custom training job in Vertex AI using gcloud.

    Args:
        machine: Machine type, such as "cpu" or "gpu".
    """
    
    # Add early return if directory check fails
    queue_dir = os.path.join("vertex_train", "queue")
    if check_directory_contents(queue_dir, num_items=2):
        warnings.warn("Directory can only have 1 config file and 1 placeholder file.")
        return

    # Copy config files to GCS
    ctx.run("gsutil cp ./vertex_train/queue/* gs://pet-fac-rec-bucket/vertex_train/queue/", echo=True, pty=not WINDOWS)
    print("Config file queued successfully.")
    
    if machine == "cpu":
        config = "vertex_config/vertex_set_secrets_cpu.yaml"
    else:
        config = "vertex_config/vertex_set_secrets_gpu.yaml"
        
    # Submit build
    ctx.run(
        f"gcloud builds submit --config={config} . "
        "--service-account=projects/pet-fac-rec/serviceAccounts/trigger-builder@pet-fac-rec.iam.gserviceaccount.com",
        echo=True,
        pty=not WINDOWS,
    )
    delete_yaml_files(queue_dir)
    print("Training job submitted successfully.")


@task
def gcloud_check(ctx: Context) -> None:
    """Fetch the output of Vertex AI jobs."""
    # Create directory if it doesn't exist
    if not os.path.exists("vertex_temp"):
        os.makedirs("vertex_temp")
    ctx.run(
        'gsutil cp -r gs://pet-fac-rec-bucket/vertex_train/completed/* ./vertex_temp/', 
        echo=True, 
        pty=not WINDOWS
    )
    sync_directories("vertex_temp", "vertex_train/completed")
    shutil.rmtree("vertex_temp")


@task
def gcloud_login(ctx: Context) -> None:
    """Login to gcloud."""
    ctx.run("gcloud auth application-default login", echo=True, pty=not WINDOWS)
    ctx.run("gcloud config set project pet-fac-rec", echo=True, pty=not WINDOWS)


# Backend Tasks


@task
def loadbentomodel(ctx: Context) -> None:
    """Load .onnx model into bento"""
    ctx.run("python bentoml_api/src/load_model.py")


@task
def runbento(ctx: Context) -> None:
    """Start a bentoml server"""
    with ctx.cd("bentoml_api"):
        ctx.run("bentoml build")
        ctx.run("bentoml serve src.service:svc --reload")


@task
def buildbentoimage(ctx: Context) -> None:
    """Build a docker image for a bentoml API"""
    ctx.run("docker build -t bento-image -f dockerfiles/bento.dockerfile .")


@task
def runbentocontainer(ctx: Context, name: str = "backend") -> None:
    """Run bento API as docker container. Access locally via localhost:8080. Args: name (str): Docker Container Name"""
    ctx.run(f"docker run -p 8080:5000 --name {name} bento-image")


# Frontend Tasks


@task
def buildstreamlitimage(ctx: Context) -> None:
    """Build a docker image for the streamlit frontend"""
    ctx.run("docker build -t frontend-image -f dockerfiles/frontend.dockerfile .")


@task
def runstreamlitcontainer(ctx: Context, name: str = "frontend") -> None:
    """Run a container with the streamlit frontend, accessible via localhost:9000"""
    ctx.run(f"docker run -p 9000:9000 --name {name} frontend-image")


# Run Frontend and Backend in one Network
# Make sure to delete any containers named "frontend" or "backend" before running
@task
def runfrontendbackend(ctx: Context) -> None:
    # Create Docker Network
    ctx.run("docker network create pet_fac_network")

    # Run Backend Tasks
    def run_backend():
        ctx.run("docker build -t bento-image -f dockerfiles/bento.dockerfile .")
        ctx.run("docker run --name backend --network pet_fac_network bento-image")

    # Run Frontend Tasks
    def run_frontend():
        ctx.run("docker build -t frontend-image -f dockerfiles/frontend.dockerfile .")
        ctx.run("docker run -p 9000:9000 --name frontend --network pet_fac_network frontend-image")

    # Run Backend and Frontend in parallel
    with ThreadPoolExecutor() as executor:
        executor.submit(run_backend)
        executor.submit(run_frontend)


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task  # Run with: invoke git --message "My commit message"
def git(ctx, message):
    ctx.run("git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
