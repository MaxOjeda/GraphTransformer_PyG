import torch
import math
import torch.nn as nn
from typing import List, Optional
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class MultiHeadAttentionLayer(MessagePassing):
    def __init__(self, in_dim:int, out_dim:int, n_heads:int=4):
        super().__init__(node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.WQ = nn.Linear(in_dim, out_dim, bias=False)
        self.WK = nn.Linear(in_dim, out_dim, bias=False)
        self.WV = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index, edge_attr=None):
        Q_h, K_h, V_h = self.WQ(x), self.WK(x), self.WV(x)

        Q = Q_h.view(-1, self.n_heads, self.out_dim // self.n_heads)
        K = K_h.view(-1, self.n_heads, self.out_dim // self.n_heads)
        V = V_h.view(-1, self.n_heads, self.out_dim // self.n_heads)
  
        out = self.propagate(edge_index=edge_index, Q=Q, K=K, V=V, edge_attr=edge_attr, size=None)
        return out

    def message(self, Q_i, K_j, V_j, edge_attr=None):
        scores = torch.matmul(Q_i, K_j.transpose(-1, -2)) / math.sqrt(self.head_dim)
        alpha = torch.softmax(scores, dim=-1)
        out_alpha = torch.matmul(alpha, V_j)
        return out_alpha



class GraphTransformerLayer(MessagePassing):
    def __init__(self, in_dim, hidden_dim:int=128, edge_dim=None, n_heads:int=8, dropout:float=0.0, aggregators: List[str]=['sum'], qkv_bias=False,
                 batch_norm=False):
        super().__init__(aggr="add")
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.aggregators = aggregators
        self.dropout = dropout
        self.num_aggrs = len(aggregators)

        self.attention = MultiHeadAttentionLayer(in_dim, hidden_dim, n_heads)
        self.WO = nn.Linear(self.num_aggrs * hidden_dim, hidden_dim, bias=False)

        if batch_norm is not None:
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn1 = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.ffn2 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)

        #self.reset_parameters()


    # def reset_parameters(self):
    #     """
    #     Note: The output of the Q-K-V layers does not pass through the activation layer (as opposed to the input),
    #             so the variance estimation should differ by a factor of two from the default
    #             kaiming_uniform initialization.
    #     """
    #     nn.init.xavier_uniform_(self.WQ.weight)
    #     nn.init.xavier_uniform_(self.WK.weight)
    #     nn.init.xavier_uniform_(self.WV.weight)
    #     nn.init.xavier_uniform_(self.WO.weight)
    #     if self.edge_dim is not None:
    #         nn.init.xavier_uniform_(self.WE.weight)
    #         nn.init.xavier_uniform_(self.WOe.weight)


    def forward(self, x, edge_index, edge_attr=None):
        x_ = x # for residual
        edge_index_ = edge_index
        attn_out = self.attention(x, edge_index)
        h = attn_out.view(-1, self.hidden_dim * self.num_aggrs)  # concatenation
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.WO(h)
        h = h + x_ # Residual 1
        h = self.norm1(h)
        
        # FFN
        h_ = h
        h = self.ffn1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.ffn2(h)
        h = self.norm2(h + h_)

        # if self.edge_in_dim is None:
        #     out_eij = None
        # else:
        #     out_eij = self._eij
        #     self._eij = None
        #     out_eij = out_eij.view(-1, self.hidden_dim)

        #     # EDGES
        #     out_eij = self.dropout_layer(out_eij)
        #     out_eij = self.WOe(out_eij) + edge_attr  # Residual connection
        #     out_eij = self.norm1e(out_eij)
        #     # FFN--edges
        #     ffn_eij_in = out_eij
        #     out_eij = self.ffn_e(out_eij)
        #     out_eij = self.norm2e(ffn_eij_in + out_eij)

        return h
    

