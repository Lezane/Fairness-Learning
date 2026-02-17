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

    


    # Set a random seed for reproducibility
    torch.manual_seed(42)

    # Determine the size of the dataset
    dataset_size = X.shape[0]

    # Define the split ratios
    train_ratio = 0.9
    val_ratio = 0.1

    # Calculate the number of samples for each set
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Generate random indices for splitting
    indices = torch.randperm(dataset_size)

    # Split the indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Split the data, target, and masks into training and validation sets
    X_train = X[train_indices]
    y_train = y_tensor[train_indices]
    l0_mask_train = l0_mask_tensor[train_indices]

    X_val = X[val_indices]
    y_val = y_tensor[val_indices]
    l0_mask_val = l0_mask_tensor[val_indices]

    # Save the PyTorch tensors to files
    torch.save(X_train, 'features_train.pt')
    torch.save(y_train, 'target_train.pt')
    torch.save(l0_mask_train, 'l0_mask_train.pt')
    torch.save(X_val, 'features_val.pt')
    torch.save(y_val, 'target_val.pt')
    torch.save(l0_mask_val, 'l0_mask_val.pt')

    print("Tensors saved as features_train, target_train.pt, l0_mask_train.pt features_val.pt target_val.pt l0_mask_val.pt")

    return X_train, y_train, l0_mask_train, X_val, y_val, l0_mask_val

if __name__ == '__main__':
    print("Running preprocessing script...")
    
    X_train, y_train, l0_mask_train, X_val,y_val, l0_mask_val = preprocess_adult_data()
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"L0_mask_train shape: {l0_mask_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"L0_mask_val shape: {l0_mask_val.shape}")
