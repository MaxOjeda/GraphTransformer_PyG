import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import MultiAggregation
from typing import Optional, List

from layers.mlp import MLPReadout
from layers.graph_transformer_layer import GraphTransformerLayer

class GraphTransformerNet(nn.Module):
    def __init__(self, node_dim: int, edge_dim: Optional[int] = None, pe_dim: Optional[int] = None, 
                 hidden_dim: int = 128, out_dim: int = 1, batch_norm: bool = True, num_layers: int = 4, 
                 num_heads: int = 8, aggregators: List[str] = ["sum"], in_feat_dropout: float = 0.0,
                 dropout: float = 0.0, graph_classification_task: bool = True):
        """
        Initializes the GraphTransformerNet model.
        
        Args:
            node_dim (int): Dimension of node features.
            edge_dim (Optional[int]): Dimension of edge features.
            pe_dim (Optional[int]): Dimension of positional encoding features.
            hidden_dim (int): Dimension of hidden layers.
            out_dim (int): Dimension of output layer.
            batch_norm (bool): Whether to use batch normalization. If false, norm = layer normalization
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            aggregators (List[str]): Global pooling aggregation methods.
            in_feat_dropout (float): Dropout rate for input features.
            dropout (float): Dropout rate for transformer layers.
            graph_classification_task (bool): Whether the task is graph classification/regression.
        """
        super(GraphTransformerNet, self).__init__()

        self.edge_dim = edge_dim
        self.pe_dim = pe_dim
        self.graph_classification_task = graph_classification_task

        # Node embedding layer
        self.node_emb = nn.Linear(node_dim, hidden_dim, bias=False)

        # Edge embedding layer (if edge features are provided)
        if edge_dim:
            self.edge_emb = nn.Linear(edge_dim, hidden_dim, bias=False)

        # Positional encoding embedding layer (if positional encodings are provided)
        if pe_dim:
            self.pe_emb = nn.Linear(pe_dim, hidden_dim, bias=False)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphTransformerLayer(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                n_heads=num_heads,
                dropout=dropout,
                aggregators=aggregators,
                batch_norm=batch_norm
            ))

        # Global pooling layer
        self.global_pool = MultiAggregation(aggregators, mode="cat")

        # Dropout for input features
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # MLP readout layer
        self.mlp_readout = MLPReadout(hidden_dim, out_dim)

        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initializes the parameters of the model.
        """
        nn.init.xavier_uniform_(self.node_emb.weight)
        if self.edge_dim is not None:
            nn.init.xavier_uniform_(self.edge_emb.weight)
        if self.pe_dim is not None:
            nn.init.xavier_uniform_(self.pe_emb.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None, 
                pos_enc: Optional[Tensor] = None, batch: Optional[Batch] = None) -> Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            edge_attr (Optional[Tensor]): Edge features.
            pos_enc (Optional[Tensor]): Positional encodings.
            batch (Optional[Batch]): Batch information for graph classification.
        
        Returns:
            Tensor: Output of the model.
        """
        # Node embedding
        h = self.node_emb(x)
        h = self.in_feat_dropout(h)
        if self.pe_dim is not None:
            h += self.pe_emb(pos_enc)

        # Edge embedding
        if self.edge_dim is not None:
            edge_attr = self.edge_emb(edge_attr)

        # Transformer layers
        for layer in self.layers:
            h, edge_attr = layer(x=h, edge_index=edge_index, edge_attr=edge_attr)

        # Global pooling for graph classification
        if self.graph_classification_task:
            h = self.global_pool(h, batch)

        # MLP readout
        h = self.mlp_readout(h)

        return h
