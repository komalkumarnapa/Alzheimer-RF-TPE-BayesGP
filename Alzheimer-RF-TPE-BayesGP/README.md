# Alzheimer's Disease Prediction using Alternative Feature Selection and Model-Based Hyperparameter Optimization

## Overview

The proposed framework integrates three feature selection approaches:

- ReliefF
- Boruta
- Recursive Feature Elimination with Cross Validation (RFECV)

with two Bayesian optimization algorithms:

- Tree-Structured Parzen Estimator (TPE)
- Bayesian Optimization using Gaussian Processes (BayesGP)

to optimize Random Forest classifiers for Alzheimer's disease prediction.

---

## Dataset

The dataset used in this study is publicly available from Kaggle.

Alzheimer's Disease Dataset

https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

Download the dataset and place

alzheimers_disease_data.csv

inside

data/

---

## Repository Structure

src/
contains all source codes.

notebooks/
contains the complete notebook.

results/
stores all generated tables.

figures/
stores all publication-quality figures.

supplementary/
contains supplementary tables.

---

## Installation

Create a virtual environment

```bash
conda create -n alz python=3.11
conda activate alz

git clone https://github.com/username/Alzheimer-RF-TPE-BayesGP.git

cd Alzheimer-RF-TPE-BayesGP

pip install -r requirements.txt

python src/run_pipeline.py