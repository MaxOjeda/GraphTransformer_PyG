import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import MultiAggregation
from typing import Optional, List

from layers.mlp import MLPReadout
from layers.GTConv import GraphTransformerLayer

class GraphTransformerNet(nn.Module):
    def __init__(self, node_dim, edge_dim=None, pe_dim: Optional[int] = None, hidden_dim: int = 128,
                 batch_norm=True,
                 num_layers: int = 4, num_heads: int = 8, gt_aggregators: List[str] = ["sum"],
                 aggregators: List[str] = ["sum"], in_feat_dropout: float=0.0, dropout: float = 0.0):
        super().__init__()

        """
        Graph Transformer Convolution (GTConv) module.

        Args:
            node_dim (int): Dimensionality of the input node features -> d_n.
            edge_dim (int, optional): Dimensionality of the input edge features -> d_e.
            pe_dim (int, optional): Dimensionality of the positional encodings -> d_k.
            hidden_dim (int): Dimensionality of the hidden representations -> d.
            num_heads (int, optional): Number of attention heads. Default is 8. 
            dropout (float, optional): Dropout probability. Default is 0.0.
            gate (bool, optional): Use a gate attantion mechanism.
                                   Default is False
            qkv_bias (bool, optional): Bias in the attention mechanism.
                                       Default is False
            norm (str, optional): Normalization type. Options: "bn" (BatchNorm), "ln" (LayerNorm).
                                  Default is "bn".
            act (str, optional): Activation function name. Default is "relu".
            aggregators (List[str], optional): Aggregation methods for the messages aggregation.
                                               Default is ["sum"].
        """
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.pe_dim = pe_dim

        # h0_i = A0 * alpha_i + a0
        self.node_emb = nn.Linear(node_dim, hidden_dim, bias=False)
        if edge_dim:
            # e0_ij = B0 * betha_ij + b0
            self.edge_emb = nn.Linear(edge_dim, hidden_dim, bias=False)
        
        if pe_dim:
            # lambda0 = C0 * lambda_i + c0
            self.pe_emb = nn.Linear(pe_dim, hidden_dim, bias=False)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphTransformerLayer(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                edge_dim=self.edge_dim,
                n_heads=num_heads,
                dropout=dropout,                
                aggregators=gt_aggregators,
                batch_norm=batch_norm
            ))

        self.global_pool = MultiAggregation(aggregators, mode="cat")
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        num_aggrs = len(aggregators)
        self.mlp_readout = MLPReadout(hidden_dim, 1)

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        if self.edge_dim is not None:
            nn.init.xavier_uniform_(self.edge_emb.weight)
        if self.pe_dim is not None:
            nn.init.xavier_uniform_(self.pe_emb.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, pe: Tensor, batch: Batch):

        x = self.node_emb(x)
        x = self.in_feat_dropout(x)
        if self.pe_dim is not None:
            x = x + self.pe_emb(pe)

        if self.edge_dim is not None:
            # print(edge_attr.shape)
            edge_attr = self.edge_emb(edge_attr)
        # print(f"X: {x.shape}")
        #print(f"edge: {edge_attr.shape}")

        for layer in self.layers:
            (x, edge_attr) = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = self.global_pool(x, batch)
        x = self.mlp_readout(x)

        return x