# Classifying Facial Expressions of Pets: pet_fac_rec

## Link to Deployed Application

We have deployed our application via Cloud Run (Google Cloud) and it should be accessible via the following links:

Frontend: https://frontend-image-73164234676.europe-west1.run.app/
Backend: https://backend-bento-73164234676.europe-west1.run.app/

## Authors
- Beat Weichsler (s244469)
- Paula Barho (s242926)
- Rong Jet Cheong (s241961)
- Trung Kien Luyen (s243416)


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


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
│       └── deploy_docs.yaml
│       └── stage_model.yaml
│       └── lint.yaml
├── data/                     
│   ├── data.csv
│   └── test
│   └── train
│   └── valid
├── dockerfiles/              
│   ├── api.Dockerfile
│   └── train.Dockerfile
│   ├── bento.Dockerfile
│   └── frontend.Dockerfile
│   └── train_gcloud.Dockerfile
├── docs/                     
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   
├── reports/                 
│   └── figures/
│   └── report.py
│   └── README.md
├── src/                      
│   ├── pet_fac_rec/
│   │   ├── configs/
│   │   │   ├── config.yaml
│   │   │   ├── experiments/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   ├── train.py
│   │   ├── utils.py
│   │   └── visualize.py
│   ├── streamlit/
│   │   └── frontend.py
└── tests/                    
│   ├── integrationtests/
│   ├── performancetests/
│   ├── unittests/
│   ├── generate_dummy_model.py
│   └── testimage_happy.jpg
├── .gitignore
├── .coveragerc
├── .glcoudignore
├── .dvcignore
├── .pre-commit-config.yaml
├── LICENSE
├── cloudbild.yaml           
├── data.dcv         
├── gcloud_container_command.sh     
├── pyproject.toml            
├── README.md                 
├── requirements.txt          
├── requirements_dev.txt
├── requirements_frontend.txt
├── requirements_bentoapi.txt
├── requirements_gpu.txt
├── requirements_tests.txt  
└── tasks.py                 
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
