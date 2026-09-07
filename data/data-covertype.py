import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_covtype

class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y, s0): 
        self.x_num, self.x_cat, self.y, self.s0 = x_num, x_cat, y, s0
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x_num[idx], self.x_cat[idx], self.y[idx], self.s0[idx]

def get_covertype_dataloaders(batch_size, random_seed):
    # Force scikit-learn to cache inside your project folder
    df = fetch_covtype(data_home='./data', as_frame=True).frame
    df['target_bin'] = (df['Cover_Type'] == 2).astype(int)
    
    w_cols = [c for c in df.columns if 'Wilderness' in c]
    s_cols = [c for c in df.columns if 'Soil_Type' in c]
    
    df['Wilderness_Area'] = np.argmax(df[w_cols].values, axis=1)
    df['Soil_Type'] = np.argmax(df[s_cols].values, axis=1)
    df.drop(columns=w_cols + s_cols, inplace=True)
    
    num_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    cat_cols = ['Wilderness_Area', 'Soil_Type']
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed, stratify=df['target_bin'])
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    
    minority_class = df_train['Cover_Type'].value_counts().index[-2]
    df_train['S0'] = (df_train['Cover_Type'] == minority_class).astype(int)
    df_test['S0'] = (df_test['Cover_Type'] == minority_class).astype(int)
    
    cat_cardinalities = []
    for col in cat_cols:
        uniq = df_train[col].astype(str).unique()
        v2i = {v: i for i, v in enumerate(uniq)}
        df_train[col] = df_train[col].astype(str).map(v2i)
        df_test[col] = df_test[col].astype(str).map(v2i).fillna(len(uniq)).astype(int)
        cat_cardinalities.append(len(uniq) + 1)
        
    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])
    
    X_num_tr, X_num_te = torch.tensor(df_train[num_cols].values).float(), torch.tensor(df_test[num_cols].values).float()
    X_cat_tr, X_cat_te = torch.tensor(df_train[cat_cols].values).long(), torch.tensor(df_test[cat_cols].values).long()
    y_tr, y_te = torch.tensor(df_train['target_bin'].values).long(), torch.tensor(df_test['target_bin'].values).long()
    s0_tr, s0_te = torch.tensor(df_train['S0'].values).bool(), torch.tensor(df_test['S0'].values).bool()
    
    return (
        DataLoader(TabularDataset(X_num_tr, X_cat_tr, y_tr, s0_tr), batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(TabularDataset(X_num_tr, X_cat_tr, y_tr, s0_tr), batch_size=batch_size, shuffle=False),
        DataLoader(TabularDataset(X_num_te, X_cat_te, y_te, s0_te), batch_size=batch_size, shuffle=False), 
        len(num_cols), cat_cardinalities
    )
