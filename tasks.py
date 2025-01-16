import os
import sys

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "pet_fac_rec"
PYTHON_VERSION = "3.11"


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
    ctx.run(f"conda deactivate", echo=True, pty=not WINDOWS)
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
def trainEffNet(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def trainResnet(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py --model-name resnet50", echo=True, pty=not WINDOWS)


@task
def trainVgg(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py --model-name vgg16", echo=True, pty=not WINDOWS)


@task
def evaluateEffNet(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(
        f"python src/{PROJECT_NAME}/evaluate.py --model-name efficientnet --model-checkpoint models/efficientnet.pth",
        echo=True,
        pty=not WINDOWS,
    )


@task
def evaluateResnet(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(
        f"python src/{PROJECT_NAME}/evaluate.py --model-name resnet50 --model-checkpoint models/resnet50.pth",
        echo=True,
        pty=not WINDOWS,
    )


@task
def evaluateVgg(ctx: Context) -> None:
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
        f"gcloud builds submit --config=cloudbuild.yaml . "
        f"--service-account=projects/pet-fac-rec/serviceAccounts/trigger-builder@pet-fac-rec.iam.gserviceaccount.com",
        echo=True,
        pty=not WINDOWS,
    )


@task
def gcloud_train(ctx, machine: str = "cpu", epochs: int = 10) -> None:
    """
    Run a custom training job in Vertex AI using gcloud.

    Args:
        machine: Machine type, such as "cpu" or "gpu".
    """
    if machine == "gpu":
        i = 1
    elif machine == "cpu":
        i = 4
    else:
        raise ValueError("Machine must be 'cpu' or 'gpu'.")
    
    command = (
        f"gcloud ai custom-jobs create "
        f"--region=europe-west{i} "
        f"--display-name=test-run "
        f"--config=config_{machine}.yaml "
        f"--command 'python' "
        f"--args 'src/pet_fac_rec/train.py' "
        f"--args '--epochs' "
        f"--args '{epochs}' "
    )
    ctx.run(command, echo=True, pty=not WINDOWS)

    
@task
def gcloud_login(ctx: Context) -> None:
    """Login to gcloud."""
    ctx.run(f"gcloud auth application-default login", echo=True, pty=not WINDOWS)
    
        
@task
def gcloud_dvc_push(ctx: Context) -> None:
    """Push data to dvc remote."""
    ctx.run(f"dvc add ./data/")
    ctx.run(f"git add ./data.dvc")
    ctx.run(f"dvc push --no-run-cache", echo=True, pty=not WINDOWS)
    
    
# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task # Run with: invoke git --message "My commit message"
def git(ctx, message):
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")

@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
