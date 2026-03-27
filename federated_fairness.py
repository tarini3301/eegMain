"""
Ethical & Distributed AI Suite
===============================
Implements:
1. Federated Learning (FedAvg) simulation.
2. Fairness Auditing (Bias Analysis).
3. Domain Adversarial Neural Networks (DANN) for Transfer Learning.
"""

import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────────
# 1. Federated Learning (Simulation)
# ─────────────────────────────────────────────

class FederatedNode:
    """Simulates a hospital node with local data and its own model."""
    def __init__(self, node_id, model_template):
        self.node_id = node_id
        self.model = model_template() # Local model
        
    def local_update(self, global_weights):
        """Update local model with global weights and 'train' (simulated)."""
        self.model.load_state_dict(global_weights)
        # Simulated training: add tiny random perturbation to weights
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn(param.size()) * 0.01)
        return self.model.state_dict()

class FederatedServer:
    """Central server aggregator using FedAvg."""
    def __init__(self, model_class, n_nodes=3):
        self.global_model = model_class()
        self.nodes = [FederatedNode(i, model_class) for i in range(n_nodes)]
        
    def run_round(self):
        """Execute one round of Federated Averaging."""
        global_weights = self.global_model.state_dict()
        local_updates = [node.local_update(global_weights) for node in self.nodes]
        
        # Aggregate (Average)
        avg_weights = {}
        for key in global_weights.keys():
            avg_weights[key] = torch.stack([upd[key] for upd in local_updates]).mean(dim=0)
            
        self.global_model.load_state_dict(avg_weights)
        return "Round complete: Federated Averaging successful across {} nodes.".format(len(self.nodes))

# ─────────────────────────────────────────────
# 2. Domain Adversarial Neural Networks (DANN)
# ─────────────────────────────────────────────

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DANNEEGModel(nn.Module):
    """
    Model that handles domain adaptation.
    Learns features that are predictive of age but invariant to the dataset source (domain).
    """
    def __init__(self, input_size):
        super(DANNEEGModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.age_regressor = nn.Sequential(
            nn.Linear(32, 1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2), # 2 domains/datasets
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        age_pred = self.age_regressor(features)
        
        # Branch for domain classifier with Gradient Reversal
        reverse_features = GradientReversal.apply(features, alpha)
        domain_pred = self.domain_classifier(reverse_features)
        
        return age_pred, domain_pred

# ─────────────────────────────────────────────
# 3. Fairness Auditing
# ─────────────────────────────────────────────

class FairnessAuditor:
    """Analyzes model bias across demographic groups (Gender, Age)."""
    @staticmethod
    def compute_bias_metrics(df_results):
        """
        Expects a DataFrame with ['gender', 'chronological_age', 'predicted_age', 'brain_age_gap'].
        """
        metrics = {}
        
        # 1. Gender Bias
        for gender in ['M', 'F']:
            slice_df = df_results[df_results['gender'] == gender]
            if not slice_df.empty:
                metrics[f'Gender_{gender}_MAE'] = round(float(np.mean(np.abs(slice_df['brain_age_gap']))), 2)
                metrics[f'Gender_{gender}_MeanGap'] = round(float(np.mean(slice_df['brain_age_gap'])), 2)
                
        # 2. Age Group Bias (Disparate Impact)
        bins = [19, 35, 50, 65]
        labels = ['Young', 'Middle', 'Senior']
        df_results['age_group'] = pd.cut(df_results['chronological_age'], bins=bins, labels=labels)
        
        for group in labels:
            slice_df = df_results[df_results['age_group'] == group]
            if not slice_df.empty:
                metrics[f'AgeGroup_{group}_MAE'] = round(float(np.mean(np.abs(slice_df['brain_age_gap']))), 2)
                
        return metrics

import pandas as pd # Needed for Auditor
