import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Loads data from csv file. Uses semicolon seperator cuz thats what the dataset uses.
    Returns None if file not found.
    """
    try:
        # We use sep=';' because the dataset uses it
        df = pd.read_csv(filepath, sep=';')
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None



def load_and_merge_data(mat_path, por_path):
    """
    Loads both Math and Portuguese datasets and combines them.
    We concat them instead of merge to get more data. Adds a 'subject' column 
    so we know which is which.
    """
    df_mat = load_data(mat_path)
    df_por = load_data(por_path)
    
    if df_mat is None or df_por is None:
        return None
        
    df_mat['subject'] = 'Math'
    df_por['subject'] = 'Portuguese'
    
    # Concatenating both datasets
    df_final = pd.concat([df_mat, df_por], ignore_index=True)
    print(f"Merged shape: {df_final.shape}")
    return df_final


def clean_data(df):
    """
    Cleans the data by removing duplicates and filling missing values.
    Also creates new fetures (avg_alcohol and passed) for better predictions.
    
    - Duplicates: removed becuase they're useless
    - Missing vals: filled with mode (most common value)
    - New features: avg of alcohol consumption and binary pass/fail
    
    This gives us the Feature Engineering bonus point!
    """
    data = df.copy()
    
    # Check for duplicates
    duplicates = data.duplicated().sum()
    print(f"Found {duplicates} duplicates.")
    if duplicates > 0:
        data = data.drop_duplicates()
        print("Duplicates removed.")
        
    # Check for missing values
    missing = data.isnull().sum().sum()
    if missing > 0:
        # In this dataset usually there are no missing values but just in case
        print(f"Found {missing} missing values. Filling with mode.")
        for col in data.columns:
            if data[col].isnull().any():
                data[col].fillna(data[col].mode()[0], inplace=True)
                
    # Feature Engineering (Bonus point!)
    # Create average alcohol consumption
    data['avg_alcohol'] = (data['Dalc'] + data['Walc']) / 2
    
    # Create boolean for if student passed G3
    # Usually passing is >= 10
    data['passed'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)
    
    return data



def detect_outliers_iqr(df, columns=None):
    """
    Finds outliers using IQR method
    Outliers are values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR.
    Returns a dict with column names and their outlier indices.
    """
    if columns is None:
        # Select only numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers_dict = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outlier indices
        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        
        if outlier_indices:
            outliers_dict[col] = outlier_indices
            print(f"{col}: Found {len(outlier_indices)} outliers (range: {lower_bound:.2f} to {upper_bound:.2f})")
    
    return outliers_dict


def handle_outliers(df, threshold=1.5):
    """
    Handles outliers by capping them at IQR boundries. 
    This way we keep all the data but limit extreme values.
    """
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude binary and small-range columns from outlier handling
    exclude_cols = ['passed']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Cap outliers at boundries
        outliers_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        if outliers_count > 0:
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"{col}: Capped {outliers_count} outliers")
    
    return data

def encode_categorical(df):
    """
    Converts categorical (text) columns to numbers using one-hot encoding.
    ML models need numbers not strings. drop_first=True to avoid multicolinearity.
    """
    # Identify categorical columns (object type)
    cat_cols = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns to encode: {list(cat_cols)}")
    
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded


def preprocess_pipeline(mat_path, por_path, output_path='data/processed/student_processed.csv'):
    """
    Complete preprocessing pipeline
    1. Load and merge data from both files
    2. Clean data (duplicates, missing vals, feature engineering)
    3. Detect and handle outliers
    4. Encode categorical variables
    5. Save processed data
    
    Returns the final cleaned dataframe ready for ML models.
    """
    print("=" * 67)
    print("STARTING COMPLETE PREPROCESSING PIPELINE")
    print("=" * 67)
    print()
    
    # Step 1: Load and merge data
    print("Step 1/5: Loading and merging data sources...")
    df_raw = load_and_merge_data(mat_path, por_path)
    if df_raw is None:
        raise ValueError("Failed to load data files")
    print("Data loaded successfully\n")
    
    # Step 2: Clean data
    print("Step 2/5: Cleaning data (duplicates, missing values, feature engineering)...")
    df_cleaned = clean_data(df_raw)
    print("Data cleaning completed\n")
    
    # Step 3: Handle outliers
    print("Step 3/5: Detecting and handling outliers...")
    outliers = detect_outliers_iqr(df_cleaned)
    print(f"Detected outliers in {len(outliers)} columns")
    df_no_outliers = handle_outliers(df_cleaned)
    print("Outlier handling completed\n")
    
    # Save pre-encoded version for visualizations
    viz_path = 'data/processed/student_for_viz.csv'
    df_no_outliers.to_csv(viz_path, index=False)
    print(f"  Saved visualization-ready data to: {viz_path}\n")
    
    # Step 4: Encode categorical variables
    print("Step 4/5: Encoding categorical variables...")
    df_encoded = encode_categorical(df_no_outliers)
    print(f"Encoding completed. Final shape: {df_encoded.shape}\n")
    
    # Step 5: Save processed data
    print("Step 5/5: Saving processed data...")
    df_encoded.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}\n")
    
    print("=" * 67)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 67)
    print(f"Final dataset shape: {df_encoded.shape}")
    print(f"Ready for exploratory data analysis and machine learning!")
    
    return df_encoded




mat_path = './data/raw/student-mat.csv'
por_path = './data/raw/student-por.csv'
output_path = './data/processed/student_processed.csv'

df_processed = preprocess_pipeline(mat_path, por_path, output_path)
print(df_processed.head())