import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def preprocess_iris_data(input_path, output_dir):
    """
    Preprocess the Iris dataset and save the results.
    
    Args:
        input_path (str): Path to the raw Iris CSV file
        output_dir (str): Directory to save preprocessed files
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(input_path)
    df = df.drop('Id', axis=1)
    
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])
    
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df
    
    for column in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
        df = remove_outliers(df, column)
    
    scaler = StandardScaler()
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    df[features] = scaler.fit_transform(df[features])
    
    X = df.drop('Species', axis=1)
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"Data yang telah dipreproses disimpan di {output_dir}")

if __name__ == "__main__":
    input_path = 'D:/submissions_SML/Eksperimen_SML_MGalangPangestuNH/dataset/Iris.csv'
    output_dir = 'D:/submissions_SML/Eksperimen_SML_MGalangPangestuNH/preprocessing/iris_preprocessed'
    preprocess_iris_data(input_path, output_dir)