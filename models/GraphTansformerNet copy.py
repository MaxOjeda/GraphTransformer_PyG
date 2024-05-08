import torch
import torch.nn as nn
from torch_geometric.nn.aggr import MultiAggregation
from layers import GTConv, MLP
from typing import Optional, List

class GraphTNET(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: Optional[int] = None , pe_dim: Optional[int] = None, batch_norm=True, layer_norm=False,
                 num_layers: int = 4, aggregators: List[str] = ["sum"]):
        super().__init__()

        self.node_emb = nn.Linear(in_dim, hidden_dim, bias=False)

        if edge_dim:
            self.edge_emb = nn.Linear(edge_dim, hidden_dim, bias=False)
        else:
            self.register_parameter("edge_emb", None)
        
        if pe_dim:
            self.pe_emb = nn.Linear(pe_dim, hidden_dim, bias=False)
        else:
            self.register_parameter("pe_dim", None)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GTConv())
        self.global_pool = MultiAggregation(aggregators, mode='cat')

        self.mu_mlp = MLP()

        self.log_var_mlp = MLP()

        self.reset_parameters()

    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb)
        if self.edge_emb is not None:
            nn.init.xavier_uniform_(self.edge_emb)
        if self.pe_emb is not None:
            nn.init.xavier_uniform_(self.pe_emb)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        if self.pe_emb is not None:
            x = x + self.pe_emb(x)
        
        if self.edge_emb is not None:
            edge_attr = self.edge_emb(edge_attr)

        for layer in self.layers:
            (x, edge_index) = layer(x, edge_index, edge_attr)
        
        x = self.global_pool(x, batch)

        mu = self.mu_mlp(x)
        log_var = self.log_var_mlp(x)
        std = torch.exp(0.5, log_var)

        if self.training:
            eps = torch.randn_like(std)
            return mu + std *eps, std
        return mu, std
