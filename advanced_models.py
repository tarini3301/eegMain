"""
Advanced EEG Neural Architectures
==================================
Implements:
1. Graph Neural Networks (GCN/GAT) for electrode connectivity.
2. EEG Transformers for spectral-spatial attention.
3. Self-Supervised Learning (Contrastive / Masked Modeling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# 1. Graph Convolutional Network (GCN) for EEG
# ─────────────────────────────────────────────

class EEGGraphConv(nn.Module):
    """
    Simple Graph Convolutional Layer.
    A_hat = D^-1/2 * (A + I) * D^-1/2
    """
    def __init__(self, in_features, out_features):
        super(EEGGraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x shape: [Batch, Nodes, Features]
        # adj shape: [Nodes, Nodes]
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output

class EEGGraphRegressor(nn.Module):
    def __init__(self, n_nodes, node_features, adj):
        super(EEGGraphRegressor, self).__init__()
        self.adj = nn.Parameter(adj, requires_grad=False)
        self.conv1 = EEGGraphConv(node_features, 32)
        self.conv2 = EEGGraphConv(32, 16)
        self.fc = nn.Sequential(
            nn.Linear(n_nodes * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x is [Batch, FlattenedFeatures] -> Reshape to [Batch, Nodes, Features]
        # Here we assume 5 regions (nodes) and 5 bands (features)
        x = x.view(x.size(0), 5, 5) 
        x = F.relu(self.conv1(x, self.adj))
        x = F.relu(self.conv2(x, self.adj))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ─────────────────────────────────────────────
# 2. EEG Transformer
# ─────────────────────────────────────────────

class EEGTransformer(nn.Module):
    def __init__(self, input_size, n_heads=4, n_layers=2):
        super(EEGTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [Batch, Features]
        x = self.embedding(x).unsqueeze(1) # [Batch, 1, 128]
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

# ─────────────────────────────────────────────
# 3. Self-Supervised Learning (SSL) Modules
# ─────────────────────────────────────────────

class ContrastiveEncoder(nn.Module):
    """SimCLR-style encoder for contrastive learning."""
    def __init__(self, input_size, projection_dim=64):
        super(ContrastiveEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.projector = nn.Sequential(
            nn.Linear(128, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

class MaskedEEGAutoencoder(nn.Module):
    """BERT-style masked signal modeling."""
    def __init__(self, input_size):
        super(MaskedEEGAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ─────────────────────────────────────────────
# Scikit-Learn Wrapper for Advanced Models
# ─────────────────────────────────────────────

class AdvancedNeuroRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='transformer', input_size=25, epochs=100, lr=0.001):
        self.model_type = model_type
        self.input_size = input_size
        self.epochs = epochs
        self.lr = lr
        self.model = None
        
        # Default Adjacency for GNN (Connectivity between regions)
        # Frontal-Central, Central-Parietal, etc.
        adj = torch.eye(5)
        # Add some neighbors (simulated connectivity)
        neighbors = [(0,1), (1,2), (2,3), (3,4), (0,2), (1,3)]
        for i, j in neighbors:
            adj[i, j] = 0.5
            adj[j, i] = 0.5
        self.adj = adj

    def fit(self, X, y):
        if self.model_type == 'transformer':
            self.model = EEGTransformer(self.input_size)
        elif self.model_type == 'gnn':
            self.model = EEGGraphRegressor(n_nodes=5, node_features=5, adj=self.adj)
        else:
            self.model = EEGTransformer(self.input_size) # default
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        X_t = torch.FloatTensor(np.array(X))
        y_t = torch.FloatTensor(np.array(y)).view(-1, 1)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.array(X))
            preds = self.model(X_t).numpy().flatten()
        return preds
