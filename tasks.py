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
def gcloud_queue(ctx, configname: str) -> None:
    """
    Queue config files for training in Vertex AI. Run before gcloud_train.

    Args:
        configname: Name of the config file.
    """
    ctx.run(f"mkdir -p vertex_train\\completed\\{configname}")
    ctx.run(f"dvc add ./vertex_train", echo=True, pty=not WINDOWS)
    ctx.run(f"dvc push ./vertex_train --no-run-cache", echo=True, pty=not WINDOWS)
    print("Data pushed to dvc remote.")
    

@task
def gcloud_train(ctx, configname: str, machine: str = "gpu") -> None:
    """
    Run a custom training job in Vertex AI using gcloud.

    Args:
        configname: Name of the config file.
        machine: Machine type, such as "cpu" or "gpu".
    """
    
    # Paths and variables
    config_file = f"vertex_config/config_{machine}.yaml"
    output_file = f"vertex_config/temp/{configname}_config_{machine}.yaml"
    image_uri = "europe-west1-docker.pkg.dev/pet-fac-rec/pet-fac-rec-image-storage/train:latest"
    storage_uri = "gs://pet-fac-rec-bucket"

    # Create the yq command
    yq_command = (
        f"bash -c \"yq eval "
        f"'.workerPoolSpecs.containerSpec.imageUri = \\\"{image_uri}\\\" | "
        f".workerPoolSpecs.containerSpec.env[0].value = \\\"{storage_uri}\\\" | "
        f".workerPoolSpecs.containerSpec.env[1].value = \\\"{configname}\\\"' "
        f"{config_file} > {output_file}\""
    )
    
    region = "europe-west4" if machine == "cpu" else "europe-west1"
    gcloud_command = (
        f"gcloud ai custom-jobs create "
        f"--region={region} "
        f"--display-name={configname}-run "
        f"--config={output_file} "
    )
    
    ctx.run(yq_command, echo=True, pty=False)
    ctx.run(gcloud_command, echo=True, pty=False)
    print("Training job submitted successfully.")
    ctx.run(f"rm {output_file}", echo=True, pty=False)


@task
def gcloud_check(ctx: Context) -> None:
    """Check the status of the Vertex AI jobs."""
    ctx.run(f"dvc pull ./vertex_train --no-run-cache", echo=True, pty=not WINDOWS)

    
@task
def gcloud_login(ctx: Context) -> None:
    """Login to gcloud."""
    ctx.run(f"gcloud auth application-default login", echo=True, pty=not WINDOWS)
    ctx.run(f"gcloud config set project pet-fac-rec", echo=True, pty=not WINDOWS)
    
        
@task
def gcloud_data_push(ctx: Context) -> None:
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
