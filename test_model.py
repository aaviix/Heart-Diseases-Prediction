import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('heart_disease_data_final.csv')

print(data.shape)
# (303, 14)

# Dividing data into X and Y
X = data.drop(columns='target', axis=1)
Y = data['target']

# Splitting the Data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=2, stratify=Y)

# Model Training
model = LogisticRegression()
model.fit(x_train, y_train)

x_train_predict = model.predict(x_train)
x_train_accu = accuracy_score(x_train_predict, y_train)
print("Accuracy on training data: ", x_train_accu)
# Accuracy on training data:  0.8458149779735683

x_test_predict = model.predict(x_test)
x_test_accu = accuracy_score(x_test_predict, y_test)
print("Accuracy on test data: ", x_test_accu)
# Accuracy on test data:  0.8421052631578947

test_input1 = (50, 1, 0, 128, 0, 2.6, 1, 0, 3)
test_input2 = (58, 0, 2, 172, 0, 0.0, 2, 0, 2)

test_input1 = np.asarray(test_input1)
test_input2 = np.asarray(test_input2)

test_input1 = test_input1.reshape(-1, 9)
test_input2 = test_input2.reshape(-1, 9)

predict1 = model.predict(test_input1)
predict2 = model.predict(test_input2)

print(f'predict1: {predict1}')
print(f'predict2: {predict2}')
