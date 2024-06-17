import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):

    ## Load the data from a CSV file.
    
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully with shape {data.shape}")
        return data
    except FileNotFoundError:
        logging.error("File not found. Please check the file path.")
        return None

def preprocess_data(data):
    
    ## Preprocess the data by separating features and target variable.

    X = data.drop(columns='target', axis=1)
    Y = data['target']
    return X, Y

def split_data(X, Y, test_size=0.25, random_state=2):
    
    ## Split the data into training and testing sets.

    return train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)

def train_model(x_train, y_train):
    
    ## Train a logistic regression model.
    
    model = LogisticRegression(max_iter=200)  # Increased max_iter to ensure convergence
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_train, y_train, x_test, y_test):
    
    ## Evaluate the model on training and testing data.
    # Training data evaluation
    x_train_predict = model.predict(x_train)
    x_train_accu = accuracy_score(y_train, x_train_predict)
    logging.info(f"Accuracy on training data: {x_train_accu}")

    # Testing data evaluation
    x_test_predict = model.predict(x_test)
    x_test_accu = accuracy_score(y_test, x_test_predict)
    logging.info(f"Accuracy on test data: {x_test_accu}")

def predict_inputs(model, inputs):
    
    ## Predict the class for given test inputs.
    
    for i, test_input in enumerate(inputs, start=1):
        test_input = np.asarray(test_input).reshape(1, -1)
        try:
            prediction = model.predict(test_input)
            logging.info(f'Prediction for input {i}: {prediction}')
        except Exception as e:
            logging.error(f"Error predicting input {i}: {e}")

def main():
    
    ## Main function to load data, preprocess, train model, evaluate and predict.
    
    file_path = 'heart_disease_data_final.csv'
    data = load_data(file_path)
    if data is not None:
        X, Y = preprocess_data(data)
        x_train, x_test, y_train, y_test = split_data(X, Y)
        model = train_model(x_train, y_train)
        evaluate_model(model, x_train, y_train, x_test, y_test)

        # Test inputs for prediction
        test_inputs = [
            (50, 1, 0, 128, 0, 2.6, 1, 0, 3),
            (58, 0, 2, 172, 0, 0.0, 2, 0, 2)
        ]
        predict_inputs(model, test_inputs)

if __name__ == "__main__":
    main()