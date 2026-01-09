import pandas as pd

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

df = load_and_merge_data('data/raw/student-mat.csv', 'data/raw/student-por.csv')
print(df.head())