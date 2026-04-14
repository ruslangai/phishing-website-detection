# Phishing Website Detection Based on Machine Learning and Deep Learning

## Overview
This project develops a phishing website detection system using Machine Learning (ML) and Deep Learning (DL) techniques. Five classification models were trained and evaluated on a publicly available dataset of labelled phishing and legitimate URLs.

## Models
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Deep Neural Network (DNN)

## Results

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | 0.9306 | 0.9303 |
| Decision Tree | 0.9353 | 0.9354 |
| Random Forest | 0.9644 | 0.9644 |
| SVM | 0.9469 | 0.9468 |
| Deep Neural Network | 0.9394 | 0.9385 |

**Best Model: Random Forest (F1 = 0.9644)**

## ROC-AUC Scores

| Model | AUC |
|---|---|
| Logistic Regression | 0.9786 |
| Decision Tree | 0.9353 |
| Random Forest | 0.9932 |
| SVM | 0.9868 |
| Deep Neural Network | 0.9852 |

## Dataset
Dataset available at Kaggle:
https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset

- 11,430 instances
- 89 features
- Binary classification: phishing (1) / legitimate (0)

## How to Install

```
pip install -r requirements.txt
```

## How to Run

```
python project_phishing.py
```

## Project Structure

```
├── project_phishing.py        # Main code
├── best_ml_model.pkl          # Best ML model (Random Forest)
├── phishing_dnn_model.keras   # Deep Neural Network model
├── scaler.pkl                 # StandardScaler
├── selected_features.pkl      # Selected features list
├── requirements.txt           # Required libraries
└── README.md                  # Project description
```

## Technologies
- Python 3.12
- Scikit-learn
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
