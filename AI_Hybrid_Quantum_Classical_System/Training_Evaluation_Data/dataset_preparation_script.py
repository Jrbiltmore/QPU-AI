
# /Training_Evaluation_Data/dataset_preparation_script.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    # Load dataset from a CSV file
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Example preprocessing steps
    df = df.dropna()  # Remove missing values
    df = pd.get_dummies(df, drop_first=True)  # Convert categorical variables to dummy variables
    return df

def split_data(df, test_size=0.2):
    # Split dataset into training and testing sets
    return train_test_split(df, test_size=test_size, random_state=42)

# Example usage
if __name__ == '__main__':
    df = load_data('path_to_your_dataset.csv')
    df_preprocessed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_preprocessed)
    print(X_train.head())
