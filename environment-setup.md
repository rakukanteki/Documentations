# Go to Anaconda Command Prompt.

## If you didn't download the Anaconda/Miniconda:
Go to this link and download:
- [Download](https://www.anaconda.com/download)

## Creating new Virtual Environmnet:
```
conda create -n your_venv_name python=the_version_you_want
```

## Activate the Environment:
```
conda activate your_venv_name
```

## Deactivate the Environment:
```
conda deactivate
```

## Delete Environment:
```
conda remove --name general_use --all
```

## Check the environment lists:
```
conda env list
```

## Install Packages:
```
conda install numpy pandas
```
```
pip install numpy pandas
```

## Remove Package:
```
conda remove numpy pandas
```

## Update Package:
```
conda update numpy pandas
```

## How to use Jupyter Notebook:
Create virtual environment
```
conda create -n env_name
```
Activate the environment,
```
conda activate env_name
```
If `Jupyter Notebook` is not installed, if installed then skip,
```
conda install jupyter
```
Launch the notebook,
```
jupyter notebook
```