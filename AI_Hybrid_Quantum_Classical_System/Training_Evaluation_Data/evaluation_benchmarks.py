
# /Training_Evaluation_Data/evaluation_benchmarks.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(predictions, actuals):
    # Compute various evaluation metrics
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='macro')
    recall = recall_score(actuals, predictions, average='macro')
    f1 = f1_score(actuals, predictions, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Example usage
if __name__ == '__main__':
    # Example data
    predictions = np.random.randint(0, 2, 100)
    actuals = np.random.randint(0, 2, 100)
    results = evaluate_model(predictions, actuals)
    print(results)
