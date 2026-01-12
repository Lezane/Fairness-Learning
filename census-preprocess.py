import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def preprocess_adult_data(file_path=None):
    # Define columns for the dataset
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'income']

    # Load the dataset
    if file_path is None:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        df = pd.read_csv(url, names=columns, na_values='?', skipinitialspace=True)
    else:
        df = pd.read_csv(file_path, names=columns, na_values='?', skipinitialspace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Strip whitespace from string columns to ensure clean matching
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Create binary target 'y': 1 if '>50K', 0 if '<=50K'
    y = (df['income'] == '>50K').astype(int)

    # Create boolean masks
    # L0: Sex is 'Female' and Income is '>50K' (y=1)
    l0_mask = (df['sex'] == 'Female') & (y == 1)
    # L1: The rest of the population
    l1_mask = ~l0_mask

    # Drop the 'income' column from the dataframe to prepare features
    df = df.drop(columns=['income'])

    # Define numerical columns to normalize
    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # Initialize StandardScaler and apply to numerical columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # One-hot encode categorical variables
    X_df = pd.get_dummies(df)

    # Filter X_df to include only numeric columns to prevent TypeError
    numeric_cols_X_df = X_df.select_dtypes(include=np.number).columns
    X_df_numeric = X_df[numeric_cols_X_df]

    # Convert features and target to PyTorch tensors
    X = torch.tensor(X_df_numeric.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

    # Convert masks to tensors
    l0_mask_tensor = torch.tensor(l0_mask.values)
    l1_mask_tensor = torch.tensor(l1_mask.values)

    # Save the PyTorch tensors to files
    torch.save(X, 'features.pt')
    torch.save(y_tensor, 'target.pt')
    torch.save(l0_mask_tensor, 'l0_mask.pt')
    torch.save(l1_mask_tensor, 'l1_mask.pt')

    print("Tensors saved as features.pt, target.pt, l0_mask.pt, and l1_mask.pt")

    return X, y_tensor, l0_mask_tensor, l1_mask_tensor

if __name__ == '__main__':
    print("Running preprocessing script...")
    
    X, y_tensor, l0_mask_tensor, l1_mask_tensor = preprocess_adult_data()
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y_tensor.shape}")
