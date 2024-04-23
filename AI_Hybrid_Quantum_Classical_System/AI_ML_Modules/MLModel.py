# AI_ML_Modules/MLModel.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle

class MLModel:
    """
    A sophisticated machine learning model that processes player data to adapt game content dynamically.
    This model leverages random forest algorithms to predict player behaviors and incorporates 
    feature engineering to improve prediction accuracy.
    """
    def __init__(self, model_path=None):
        """
        Initialize the MLModel, optionally loading a pre-trained model from a specified path.
        
        Parameters:
            model_path (str): Path to a pre-trained model file (optional).
        """
        self.model = None
        if model_path:
            self.load_model(model_path)

    def preprocess_data(self, data):
        """
        Preprocess the input data to fit the requirements of the machine learning model.
        
        Parameters:
            data (DataFrame): The raw data as a Pandas DataFrame.
        
        Returns:
            DataFrame: The processed data ready for training or prediction.
        """
        # Example preprocessing steps (could include normalization, encoding categorical variables, etc.)
        if "age" in data.columns:
            data['age'] = data['age'].fillna(data['age'].mean())
        return data

    def train(self, data):
        """
        Train the model on provided data using a RandomForest algorithm. Includes a validation phase.
        
        Parameters:
            data (DataFrame): A Pandas DataFrame containing the training data.
        """
        processed_data = self.preprocess_data(data)
        X = processed_data.drop('target', axis=1)
        y = processed_data['target']

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Validate the model
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {accuracy}")

    def predict(self, player_profile):
        """
        Use the trained model to predict player behavior based on their profile.
        
        Parameters:
            player_profile (dict): A dictionary containing player attributes.
        
        Returns:
            float: A prediction score reflecting certain behaviors or preferences.
        """
        player_data = pd.DataFrame([player_profile])
        processed_data = self.preprocess_data(player_data)
        prediction = self.model.predict_proba(processed_data)[0, 1]
        return prediction

    def save_model(self, path):
        """
        Save the trained model to a file.
        
        Parameters:
            path (str): The path where the model should be saved.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        """
        Load a pre-trained model from a file.
        
        Parameters:
            path (str): The path to the model file.
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
