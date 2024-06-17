import pandas as pd

# Read the CSV file
data = pd.read_csv('heart_disease_data.csv')

# List of columns to remove
drop_columns = ['trestbps', 'chol', 'fbs', 'restecg']

# Remove the specified columns
data = data.drop(columns=drop_columns)

# Save the modified data to a new file
data.to_csv('heart_disease_data_final.csv', index=False)
