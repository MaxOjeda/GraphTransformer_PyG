import torch
import torch.nn as nn
from torch_geometric.nn.aggr import MultiAggregation
from typing import Optional, List

from layers import GTConv, MLP

class GraphTransformerNet(nn.Module):
    def __init__(self, in_dim, edge_in_dim=None, pe_in_dim: Optional[int] = None, hidden_dim: int = 128,
                 batch_norm=True, layer_norm=False, gate=False, qkv_bias=False,
                 num_layers: int = 4, num_heads: int = 8, gt_aggregators: List[str] = ["sum"],
                 aggregators: List[str] = ["sum"], dropout: float = 0.0):
        super(GraphTransformerNet).__init__()

        self.node_emb = nn.Linear(in_dim, hidden_dim, bias=False)

        if edge_in_dim:
            self.edge_emb = nn.Linear(edge_in_dim, hidden_dim, bias=False)
        
        if pe_in_dim:
            self.pe_emb = nn.Linear(pe_in_dim, hidden_dim, bias=False)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GTConv(
                node_in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                edge_in_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                norm="bn",
                gate=gate,
                qkv_bias=qkv_bias,
                aggregators=gt_aggregators,
            ))

        self.global_pool = MultiAggregation(aggregators, mode="cat")

        num_aggrs = len(aggregators)
        self.mu_mlp = MLP(
            input_dim=num_aggrs * hidden_dim,
            output_dim = 1,
            hidden_dim=hidden_dim,
            num_layers=1,
            dropout=dropout
        )
        self.log_var_mlp = MLP(
            input_dim=num_aggrs * hidden_dim,
            output_dim = 1,
            hidden_dim=hidden_dim,
            num_layers=1,
            dropout=dropout
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        if self.edge_emb is not None:
            nn.init.xavier_uniform_(self.edge_emb.weight)
        if self.pe_emb is not None:
            nn.init.xavier_uniform_(self.pe_emb.weight)

    def forward(self, x, edge_index, edge_attr, pe, batch):

        x = self.node_emb(x)
        if self.pe_emb is not None:
            x = x + self.pe_emb(x)
        if self.edge_emb:
            edge_attr = self.edge_emb(edge_attr)
        
        for layer in self.layers:
            (x, edge_attr) = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = self.global_pool(x, batch)
        mu = self.mu_mlp(x)
        log_var = self.log_var_mlp(x)
        std = torch.exp(0.5 * log_var)

        if self.training:
            eps = torch.randn_like(std)
            return mu + std * eps, std
        return mu, std