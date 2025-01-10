# pet_fac_rec

A short description of the project.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Project Description

### Goal of the Project
This project aims to develop a machine learning (ML) algorithm capable of classifying facial expressions of animals. Based on a prelabelled dataset of 1000 animal faces, we will experiment with different machine learning approaches to correctly classify between the states "angry", "happy", "sad", and "other". While this project is primarily exploratory and intended to demonstrate the application of machine learning in an interesting context, it also opens up some relevant application scenarios, e.g., in animal welfare, for improving training methods, or for supporting advancements in animal-assisted therapy.

### Dataset: Pet's Facial Expression Image Dataset
The dataset that is used for the project is the "Pet's Facial Expression Image Dataset" which was obtained from Kaggle [\[1\]](#references). The dataset contains 1000 images of pets' faces, including dogs, cats, rabbits, hamsters, sheep, horses, and birds. The images capture four different types of emotions: happiness, anger, sadness, and other emotions. Each image is named according to their image ID and is labelled according to the emotion it displays.

### Framework and Models: PyTorch Image Models
To achieve the project goals, we utilize PyTorch Image Models (TIMM) [\[2\]](#references), a versatile and widely adopted deep learning framework. PyTorch's image modeling capabilities allow us to efficiently process and analyze the dataset of animal faces. Specifically, we will look into pre-trained convolutional neural networks (CNNs) such as ResNet and EfficientNet, which provide a strong foundation for image classification tasks. Our approach involves fine-tuning these pre-trained models on the labeled dataset. This includes adapting the final fully connected layers to match the number of output classes and optimizing the models using appropriate loss functions and metrics. Furthermore, PyTorch's flexibility allows us to experiment with different architectures and hyperparameters, such as learning rates and batch sizes, to optimize the performance of our classification model. This process enables us to identify the most effective configurations for accurately classifying animal facial expressions.

### References
1. Ansh Tanwar. *Pet's Facial Expression Image Dataset*. Kaggle, 2023. DOI: [10.34740/KAGGLE/DS/3546787](https://doi.org/10.34740/KAGGLE/DS/3546787). [Dataset Link](https://www.kaggle.com/ds/3546787)
2. Ross Wightman. *PyTorch Image Models*. GitHub, 2019. DOI: [10.5281/zenodo.4414861](https://doi.org/10.5281/zenodo.4414861). [Repository Link](https://github.com/rwightman/pytorch-image-models)

