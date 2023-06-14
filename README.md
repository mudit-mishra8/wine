 <img src="https://img.onmanorama.com/content/dam/mm/en/food/in-season/images/2019/11/8/wine.jpg" alt="Image 1" width="800" height="500">
 
# Wine Quality and Type Prediction

This repository contains the Wine Quality and Type Prediction project where machine learning algorithms are used to predict the quality and type of wine. Here is the link to the [Jupyter Notebook](<insert-link-to-ipynb-file-here>).

## Introduction
This project consists of two separate parts, both focusing on predicting different aspects of wine using various machine learning models. The dataset contains different chemical properties of wines. The first part aims to classify wines into two categories based on quality - good and bad, while the second part predicts the type of wine.

Notebook Link for Wine Type Prediction: 👉 [Wine Type Prediction Notebook](https://github.com/mudit-mishra8/wine/blob/main/Project_Wine_Type_0.99.ipynb)


Notebook Link for Wine Quality Prediction: 👉 [Wine Quality Prediction Notebook](https://github.com/mudit-mishra8/wine/blob/main/Project_Wine_Quality_2_class_0.93.ipynb)

## Data Processing

### Reading the Data
The datasets were read into two separate Pandas DataFrames for analysis and manipulation.

### Descriptive Statistics
Performed basic descriptive statistics to understand the data distribution and the key statistics of the features.

### Cleaning Column Headers
Column headers were cleaned by making them more consistent, for example, by converting to lowercase, replacing spaces with underscores, etc.

### Converting to Binary Classification
In the first part, originally the dataset contained multi-class labels for wine quality. It was transformed into a binary classification problem. For this, a new column 'type' was created that contains 1 if the wine is Red and 0 otherwise.

### Creating New Features
A new feature called `molecular_sulfur_dioxide` was created using existing features - `free_sulfur_dioxide` and `pH`.

### Density Plots
Plotted density plots to observe the distribution of the features.

### Outlier Removal and Missing Value Imputation
Used `IterativeImputer` from scikit-learn to handle missing values in the datasets.

## Data Preparation

### Train-Test Split
Split the data into training and testing sets for model evaluation in both projects.

### Binary and Discrete Data Preparation
Vectorized all numerical features to make them suitable for machine learning algorithms in both projects.

## Modeling and Evaluation

### Wine Quality Prediction

#### Random Forest
Used Random Forest as one ofthe classification models for predicting wine quality.

#### Neural Networks
Implemented Neural Networks for the classification task in wine quality prediction.

#### XGBoost
Utilized XGBoost, a gradient boosting framework for the classification problem in wine quality prediction.

#### Logistic Regression
Implemented Logistic Regression as one of the classifiers for predicting wine quality.

#### K-Nearest Neighbors (KNN)
Used K-Nearest Neighbors algorithm for classification in wine quality prediction.

#### Evaluation - Quality Prediction
Generated a classification report that includes precision, recall, f1-score, and support for the classes. The Random Forest classifier achieved the best results with an accuracy of 0.94 and an AUC of 0.932. 

### Wine Type Prediction

#### AdaBoost Classifier
Used AdaBoost Classifier as one of the classification models for predicting wine type.

#### Evaluation - Type Prediction
Generated a classification report for wine type prediction. AdaBoost Classifier achieved the highest AUC of 0.99 and an accuracy of 0.99 for wine type prediction.
