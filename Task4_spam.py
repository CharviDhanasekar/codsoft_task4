# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 22:15:32 2024

@author: jdcha
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
dataset = pd.read_csv("C:/Users/jdcha/Downloads/archive (8)/spam.csv", encoding="ISO-8859-1")

# Select the relevant columns
dataset = dataset[['v1', 'v2']]
dataset.columns = ['label', 'message']

# Encode the labels
label_encoder = LabelEncoder()
dataset['label'] = label_encoder.fit_transform(dataset['label'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['message'], dataset['label'], test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Initialize dictionaries to store the results
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': []
}

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_results = evaluate_model(nb_model, X_test_tfidf, y_test)
results['Model'].append('Naive Bayes')
results['Accuracy'].append(nb_results[0])
results['Precision'].append(nb_results[1])
results['Recall'].append(nb_results[2])
results['F1-score'].append(nb_results[3])

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
lr_results = evaluate_model(lr_model, X_test_tfidf, y_test)
results['Model'].append('Logistic Regression')
results['Accuracy'].append(lr_results[0])
results['Precision'].append(lr_results[1])
results['Recall'].append(lr_results[2])
results['F1-score'].append(lr_results[3])

# Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_results = evaluate_model(svm_model, X_test_tfidf, y_test)
results['Model'].append('SVM')
results['Accuracy'].append(svm_results[0])
results['Precision'].append(svm_results[1])
results['Recall'].append(svm_results[2])
results['F1-score'].append(svm_results[3])

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print results
print(results_df)