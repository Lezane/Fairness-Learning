import torch
import torch.nn as nn
import rtdl

class FTTransformerWrapper(nn.Module):
    def __init__(self, num_features, cat_cardinalities):
        super().__init__()
        self.model = rtdl.FTTransformer.make_baseline(
            n_num_features=num_features, cat_cardinalities=cat_cardinalities,
            d_token=32, d_out=32, n_blocks=3, attention_dropout=0.2, ffn_d_hidden=64,
            ffn_dropout=0.1, residual_dropout=0.0, last_layer_query_idx=[-1]
        )
        self.head = nn.Linear(32, 2)
        
    def forward(self, x_num, x_cat):
        x = self.model(x_num if x_num.shape[1] > 0 else None, x_cat if x_cat.shape[1] > 0 else None)
        return self.head(x.squeeze(1) if x.dim() == 3 else x)

class MLPWrapper(nn.Module):
    def __init__(self, num_features, cat_cardinalities, embed_dim=16, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cat_cardinalities])
        in_dim = num_features + len(cat_cardinalities) * embed_dim
        layers = []
        for h in hidden_dims: 
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 2)
        
    def forward(self, x_num, x_cat):
        x_embeds = [self.embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])] if x_cat is not None and x_cat.shape[1] > 0 else []
        x = x_num
        if x_embeds:
            x_cat_concat = torch.cat(x_embeds, dim=1)
            x = torch.cat([x, x_cat_concat], dim=1) if x is not None else x_cat_concat
        return self.head(self.mlp(x))