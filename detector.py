# --- PROJECT INTRO ---
# This is a spam detector tool made using  Machine Learning.
# This software is based on informations given by THM platform during the 2023 Cyber Advent (Day 15).
# Our data comes from .csv files structured as 'Classification' (defines ham or spam emails), and 'Message' (defines emails content) columns.
# Libraries: numpy, pandas and scikit-learn.
# Training Model: Naive Bayes Classifier, a probabilistic classifier based on Bayes Theorem with an assumption of independence between features. 
#   Particularly suited for high-dimensional text data.
# Language: Python.

# --- ATTENTION! --- This current version requires the Train_Data, Test_Data, Model_Data and Spam_Data folders to work properly! Don't forget to pip install requirements.txt libs too!

# --- CODE START POINT, LETS GO! ---
# Numpy library for data formatting.
import numpy as np
# Pandas library for data processing, data reading and formatting data structures.
import pandas as pd
# This will split our data to datasets.
from sklearn.model_selection import train_test_split, cross_val_score
# This will preprocess our data, text needs to be transformed to numerical format.
from sklearn.feature_extraction.text import CountVectorizer
# This is our Naive Bayes training model.
from sklearn.naive_bayes import MultinomialNB
# This is our model evaluation library.
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# This library is used to save and load model.
import joblib
# Logging library, nothing more to say... I know, need to implement logging better...
import logging
# Import sys functions, do I need to explain?
import sys
# Import useless comment.
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

# Defining file paths.
training_data_path = "Train_Data/emails_dataset.csv"
testing_data_path = "Test_Data/test_emails.csv"
output_results_path = "Spam_Data/spam_results.csv"
model_data_path = "Model_Data/model_data.pk1"

# Centralised function for data loading and checking.
def load_and_preprocess_data(file_path):
    try:
        # Loading data
        data = pd.read_csv(file_path)
        # Dropping rows with missing values in the 'Message' column.
        data = data.dropna(subset=['Message'])
        print("\nTraining Data Head:") if "Train_Data" in file_path else print("\nTesting Data Head:")
        print(data.head())
        # Log data loading and checking.
        logging.info("\nData checked and loaded!")
        return data
    except FileNotFoundError:
        print(f"\nError: File not found at path {file_path}. Exiting program.")
        # Log data loading and checking error, FileNotFoundError.
        logging.info("\nHey... Where is my Data file?!")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"\nError: Empty data in file at path {file_path}. Exiting program.")
        # Log data loading and checking error, EmptyDataError.
        logging.info("\nHmmm... My data looks empty...")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Exiting program.")
        # Log data loading and checking error, generic Exception.
        logging.info("\nSomething went wrong... let's call Batman!")
        sys.exit(1)

# Model training function using Naive Bayes Classifier.
def train_naive_bayes_model(X_train, y_train, model_path):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Save model in Model_Path directory.
    if model_path:
        joblib.dump(clf, model_path)
        logging.info(f"\nModel saved to {model_path}")

    # Log Naive Bayes Classifier model.
    logging.info("\nModel trained with Naive Bayes Classifier!")
    return clf

# Loading and checking collected training data.
logging.info("\nChecking training data from Train_Data folder.")
data = load_and_preprocess_data(training_data_path)
# Loading and checking collected testing data.
logging.info("\nChecking testing data from Test_Data folder.")
test_data = load_and_preprocess_data(testing_data_path)

# Preprocessing our training data.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Message'])
y = data['Classification']
print("\nVectorized Data:")
print(X)

# Preprocessing our testing data.
X_new = vectorizer.transform(test_data['Message'])
print("\nVectorized Test Data:")
print(X_new)

# Splitting our train/test dataset using cross-validation (80/20 splitting).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# Function call to Machine Learning Classifier model.
try:
    clf = joblib.load(model_data_path)
    logging.info(f"\nModel loaded from {model_data_path}")
except FileNotFoundError:
    logging.info("\nNo saved model found. Let's train a new one! ")
    clf = train_naive_bayes_model(X_train, y_train, model_data_path)

# Model performance evaluation using cross-validation.
cv_scores = cross_val_score(clf, X, y, cv=5) # Number of folds may need to be adjusted.
print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Model performance evaluation on test set.
y_pred = clf.predict(X_test)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label='spam'))
print("Recall:", recall_score(y_test, y_pred, pos_label='spam'))
print("F1 Score:", f1_score(y_test, y_pred, pos_label='spam'))

# Generating predictions for test dataset
new_predictions = clf.predict(X_new)
# Log predictions for test dataset.
logging.info("\nPredictions for testing dataset generated!")

# Filter the DataFrame to show only spam emails.
results_data = pd.DataFrame({'Message': test_data['Message'], 'Prediction': new_predictions})
spam_results = results_data[results_data['Prediction'] == 'spam']
print("\nSpam emails found:")
print(spam_results)
# Log spam emails found.
logging.info("\nSpam emails found yay! AI is cool!")

# Save spam_results on spam_results.csv file
try:
    # Check Spam_Data directory existence before saving csv file.
    os.makedirs(os.path.dirname(output_results_path), exist_ok = True)

    spam_results.to_csv(output_results_path, index=False)
    print("\nSpam emails found --> spam_results.csv generated!")
    # Log spam emails csv generated.
    logging.info("\nSpam emails csv generated! That was so EZ.")
except FileNotFoundError:
    print(f"\nError: Unable to write to file at path {output_results_path}. Exiting program.")
    # Log spam emails csv generation error, FileNotFoundError.
    logging.info("\nWhere do you suppose I should save my data bro?")
    sys.exit(1)
except Exception as e:
    print(f"\nAn unexpected error occurred while saving the CSV file: {e}. Exiting program.")
    # Log spam emails csv generation error, generic Exception.
    logging.info("\nSpam emails csv NOT generated WTF!")
    sys.exit(1)
