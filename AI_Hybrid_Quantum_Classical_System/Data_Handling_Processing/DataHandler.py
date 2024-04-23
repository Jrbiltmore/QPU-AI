# Data_Handling_Processing/DataHandler.py

import pandas as pd
import numpy as np
import json
from typing import Any, Dict

class DataHandler:
    """
    Handles data operations necessary for ML model training and real-time interactions in VR.
    Manages efficient data fetching, preprocessing, and storage to support dynamic content adjustment.
    """

    def __init__(self, database_connection):
        """
        Initialize the DataHandler with a database connection or path to data store.
        
        Parameters:
            database_connection (Any): Connection object or path to the database/data store.
        """
        self.database_connection = database_connection

    def fetch_player_data(self, player_id: str) -> pd.DataFrame:
        """
        Fetch detailed data for a specific player from the database.
        
        Parameters:
            player_id (str): The unique identifier for the player.
        
        Returns:
            pd.DataFrame: A DataFrame containing the player's data.
        """
        # Example: SQL query to fetch player data
        query = f"SELECT * FROM player_data WHERE player_id = '{player_id}'"
        return pd.read_sql(query, self.database_connection)

    def store_interaction_data(self, interaction_data: Dict[str, Any]) -> None:
        """
        Store interaction data to improve ML model accuracy and enhance personalization.
        
        Parameters:
            interaction_data (Dict[str, Any]): A dictionary containing interaction details.
        """
        # Convert dictionary to JSON for storage
        interaction_json = json.dumps(interaction_data)
        cursor = self.database_connection.cursor()
        query = f"INSERT INTO interaction_logs (data) VALUES ({interaction_json})"
        cursor.execute(query)
        self.database_connection.commit()

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data before it is used for model training or predictions.
        
        Parameters:
            data (pd.DataFrame): Raw data to be processed.
        
        Returns:
            pd.DataFrame: Processed data ready for use.
        """
        # Example preprocessing steps: fill missing values, encode categorical variables, etc.
        if 'age' in data.columns:
            data['age'] = data['age'].fillna(data['age'].mean())
        if 'category' in data.columns:
            data['category'] = pd.Categorical(data['category']).codes
        return data

    def update_player_profile(self, player_id: str, updates: Dict[str, Any]) -> None:
        """
        Update player profile data based on new information or interactions.
        
        Parameters:
            player_id (str): The unique identifier for the player.
            updates (Dict[str, Any]): A dictionary with updated player attributes.
        """
        # Example: SQL update statement
        update_statement = ", ".join([f"{key} = {json.dumps(value)}" for key, value in updates.items()])
        query = f"UPDATE player_data SET {update_statement} WHERE player_id = '{player_id}'"
        cursor = self.database_connection.cursor()
        cursor.execute(query)
        self.database_connection.commit()
