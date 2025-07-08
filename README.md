# Credit Card Fraud Detection using Autoencoders
A machine learning project that detects fraudulent credit card transactions using an autoencoder-based anomaly detection system combined with a Logistic Regression classifier.

## 🚀 Overview
This project uses a deep autoencoder trained on non-fraudulent transactions to learn the normal behavior of credit card usage. It then flags transactions with high reconstruction error as anomalies (potential frauds).
The encoded features are further classified using Logistic Regression for improved fraud detection.
The model was trained and evaluated on the popular credit card transactions dataset (with anonymized PCA features V1–V28).

## 🛠️ Tools & Technologies
Python
Pandas, NumPy
Matplotlib, Seaborn
scikit-learn
TensorFlow / Keras
Google Colab

## 📊 Dataset
The dataset contains 284,807 transactions with 31 features:
V1–V28: PCA components of original features (anonymized)
Amount: Transaction amount
Time: Time elapsed since first transaction
Class: Target variable (0 = Non-Fraud, 1 = Fraud)
Link to the dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 🏗️ Project Workflow
### Data Preprocessing
Dropped Time feature
Standardized Amount and normalized all features
### Autoencoder Training
Trained only on non-fraudulent data
Encoded latent representations of transactions
### Anomaly Detection
Reconstruction error identifies suspicious transactions
### Classification
Logistic Regression trained on latent features to classify fraud
### Evaluation
Achieved 95% accuracy, 100% precision, and 66% recall for fraud detection.

