import torch
import numpy as np
import torch.nn as nn
from typing import List, Optional
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class MultiHeadAttentionLayer(MessagePassing):
    def __init__(self, in_dim:int, out_dim:int, edge_dim=None, n_heads:int=4, use_edges=False):
        super().__init__(node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.eij = None

        self.WQ = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.WK = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.WV = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        if edge_dim is not None:
            self.WE = nn.Linear(in_dim, out_dim * n_heads, bias=False)


    def forward(self, x, edge_index, edge_attr=None):
        Q_h, K_h, V_h = self.WQ(x), self.WK(x), self.WV(x)

        Q = Q_h.view(-1, self.n_heads, self.out_dim)
        K = K_h.view(-1, self.n_heads, self.out_dim)
        V = V_h.view(-1, self.n_heads, self.out_dim)
        if self.edge_dim is not None:
            E_h = self.WE(edge_attr)
            E = E_h.view(-1, self.n_heads, self.out_dim)
        else:
            E = None
        h = self.propagate(edge_index=edge_index, Q=Q, K=K, V=V, E=E, size=None)
        return h, self.eij

    def message(self, Q_i, K_j, V_j, E):
        scores = (K_j * Q_i) / np.sqrt(self.out_dim)
        if E is not None:
            scores = scores * E
            self.eij = scores
        else:
            self.eij = None
        alpha = torch.exp(scores.sum(-1, keepdim=True).clamp(-5,5))
        #alpha = scores.softmax(dim=-1)
        h = alpha * V_j
        return h



class GraphTransformerLayer(MessagePassing):
    def __init__(self, in_dim, hidden_dim:int=128, edge_dim=None, n_heads:int=8, dropout:float=0.0, aggregators: List[str]=['sum'], use_edges=False, batch_norm=False):
        super().__init__(aggr="add")
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.aggregators = aggregators
        self.dropout = dropout
        self.num_aggrs = len(aggregators)
        self.use_edges = use_edges

        self.attention = MultiHeadAttentionLayer(in_dim=in_dim, out_dim=hidden_dim // n_heads, edge_dim=self.edge_dim, n_heads=n_heads, use_edges=self.use_edges)
        self.WO = nn.Linear(self.num_aggrs * hidden_dim, hidden_dim)
        if edge_dim is not None:
            self.WOe = nn.Linear(hidden_dim, hidden_dim)

            self.ffn1e = nn.Linear(hidden_dim, 2 * hidden_dim)
            self.ffn2e = nn.Linear(2 * hidden_dim, hidden_dim)

            if batch_norm:
                self.norm1e = nn.BatchNorm1d(hidden_dim)
                self.norm2e = nn.BatchNorm1d(hidden_dim)
            else:
                self.norm1e = nn.LayerNorm(hidden_dim)
                self.norm2e = nn.LayerNorm(hidden_dim)

        if batch_norm:
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.ffn2 = nn.Linear(2 * hidden_dim, hidden_dim)


        self.reset_parameters()


    def reset_parameters(self):
        # Inicialización de parámetros para las capas lineales
        nn.init.xavier_uniform_(self.attention.WQ.weight)
        nn.init.xavier_uniform_(self.attention.WK.weight)
        nn.init.xavier_uniform_(self.attention.WV.weight)
        nn.init.xavier_uniform_(self.WO.weight)
        if self.edge_dim is not None:
            nn.init.xavier_uniform_(self.attention.WE.weight)
            nn.init.xavier_uniform_(self.WOe.weight)
        nn.init.xavier_uniform_(self.ffn1.weight)
        nn.init.xavier_uniform_(self.ffn2.weight)
        if self.edge_dim is not None:
            nn.init.xavier_uniform_(self.ffn1e.weight)
            nn.init.xavier_uniform_(self.ffn2e.weight)


    def forward(self, x, edge_index, edge_attr=None):
        h_in1 = x # for residual
        e_in1 = edge_attr
        attn_h, attn_e = self.attention(x, edge_index, edge_attr)
        h = attn_h.view(-1, self.hidden_dim * self.num_aggrs)  # concatenation
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.WO(h)
        h = h + h_in1 # Residual 1
        h = self.norm1(h)
        
        # FFN
        h_in2 = h
        h = self.ffn1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.ffn2(h)
        h = h + h_in2
        h = self.norm2(h)
        if self.edge_dim is None:
            e_ij = None
        else:
            e_ij = attn_e.view(-1, self.hidden_dim)
            e_ij = F.dropout(e_ij, self.dropout, training=self.training)
            e_ij = self.WOe(e_ij) + e_in1  # Residual connection
            e_ij = self.norm1e(e_ij)

            # FFN--edges
            eij_in = e_ij
            e_ij = self.ffn1e(e_ij)
            e_ij = F.relu(e_ij)
            e_ij = F.dropout(e_ij, self.dropout, training=self.training)
            e_ij = self.ffn2e(e_ij)
            e_ij = self.norm2e(eij_in + e_ij)

        return h, e_ij
    

