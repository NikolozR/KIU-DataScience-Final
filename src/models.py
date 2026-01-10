import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)


def load_processed_data():
    """Load the processed data."""
    data_path = 'data/processed/student_processed.csv'
  
    df = pd.read_csv(data_path)
    return df

def split_data_regression(df, test_size=0.2, random_state=42):
    """
    Splits data for regression task (predicting G3).
    Prevents data leakage by dropping G1, G2, and passed.
    """
    X = df.drop(columns=['G3'])
    
    # Drop G1, G2 to avoid using grades from previous years
    if 'G1' in X.columns:
        X = X.drop(columns=['G1'])
    if 'G2' in X.columns:
        X = X.drop(columns=['G2'])
    
    # Drop 'passed' to prevent data leakage (derived from G3)
    if 'passed' in X.columns:
        X = X.drop(columns=['passed'])
    
    y = df['G3']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test

def split_data_classification(df, test_size=0.2, random_state=42):
    """
    Splits data for classification task (predicting pass/fail).
    Prevents data leakage by dropping G3.
    """
    X = df.drop(columns=['passed'])
    
    # Drop G3 to prevent data leakage (passed is derived from G3)
    if 'G3' in X.columns:
        X = X.drop(columns=['G3'])
    
    y = df['passed']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test

def evaluate_regression(model, X_test, y_test, model_name):
    """
    Evaluates regression model with some metrics.
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"\n{model_name} metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def evaluate_classification(model, X_test, y_test, model_name):
    """
    Evaluates classification model with comprehensive metrics.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }