import torch
import math
import torch.nn as nn
from typing import List, Optional
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax

class MultiHeadAttentionLayer(MessagePassing):
    def __init__(self, in_dim:int, out_dim:int, edge_dim=None, n_heads:int=4, use_edges=False):
        super().__init__(node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.WQ = nn.Linear(in_dim, out_dim, bias=False)
        self.WK = nn.Linear(in_dim, out_dim, bias=False)
        self.WV = nn.Linear(in_dim, out_dim, bias=False)
        if edge_dim is not None:
            self.WE = nn.Linear(in_dim, out_dim, bias=False)


    def forward(self, x, edge_index, edge_attr=None):
        Q_h, K_h, V_h = self.WQ(x), self.WK(x), self.WV(x)

        Q = Q_h.view(-1, self.n_heads, self.out_dim // self.n_heads)
        K = K_h.view(-1, self.n_heads, self.out_dim // self.n_heads)
        V = V_h.view(-1, self.n_heads, self.out_dim // self.n_heads)
        print(f"x: {x.shape}")
        #print(f"edge: {edge_attr.shape}")
        if self.edge_dim is not None:
            E_h = self.WE(edge_attr)
            E = E_h.view(-1, self.n_heads, self.out_dim // self.n_heads)
        else:
            E = None
        temp = torch.matmul(Q, K.transpose(-1, -2))
        # print(f"Q shape: {Q.shape}")
        # print(f"K shape: {K.shape}")
        # print(f"K transpose shape: {K.transpose(-1, -2).shape}")
        # print(f"scores shape: {temp.shape}")
        # print(f"E shape: {E.shape}")
        # print(f"E T shape: {E.transpose(-1, -2).shape}")
        h = self.propagate(edge_index=edge_index, Q=Q, K=K, V=V, E=E, size=None)
        #h = self.propagate(edge_index=edge_index, Q=Q, K=K, V=V, edge_attr=edge_attr, size=None)
        return h

    def message(self, Q_i, K_j, V_j, E):
        print(f"Q_i: {Q_i.shape}")
        print(f"K_j: {K_j.shape}")
        print(f"V_j: {V_j.shape}")
        #scores = torch.matmul(Q_i, K_j.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = (K_j * Q_i) / math.sqrt(self.head_dim)
        print(f"scores: {scores.shape}")
        if E is not None:
            print(f"E shape: {E.shape}")
            #scores = torch.matmul(scores, E)
            scores = scores * E

            print(f"Scores Edges: {scores.shape}")
            self._eij = scores 
        else:
            self._eij = None
        alpha = torch.exp((scores.sum(-1, keepdim=True)).clamp(-5,5))
        print(f"Alpha: {alpha.shape}")
        # if E is not None:
        #     V_j = V_j.transpose(-1, -2)        
        #h = torch.matmul(alpha, V_j)
        h = alpha * V_j
        print(f"H: {h.shape}")
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

        self.attention = MultiHeadAttentionLayer(in_dim=in_dim, out_dim=hidden_dim, edge_dim=self.edge_dim, n_heads=n_heads, use_edges=self.use_edges)
        self.WO = nn.Linear(self.num_aggrs * hidden_dim, hidden_dim, bias=False)
        if edge_dim is not None:
            self.WOe = nn.Linear(hidden_dim, hidden_dim, bias=True)

            self.ffn1e = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
            self.ffn2e = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)

            if batch_norm is not None:
                self.norm1e = nn.BatchNorm1d(hidden_dim)
                self.norm2e = nn.BatchNorm1d(hidden_dim)
            else:
                self.norm1e = nn.LayerNorm(hidden_dim)
                self.norm2e = nn.LayerNorm(hidden_dim)

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
        edge_attr_ = edge_attr
        attn = self.attention(x, edge_index, edge_attr)
        h = attn.view(-1, self.hidden_dim * self.num_aggrs)  # concatenation
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
        if self.edge_dim is None:
            e_ij = None
        else:
            e_ij = self._eij.view(-1, self.hidden_dim)
            e_ij = F.dropout(e_ij, self.dropout, training=self.training)
            e_ij = self.WOe(e_ij) + edge_attr_  # Residual connection
            e_ij = self.norm1e(e_ij)

            # FFN--edges
            eij_in = e_ij
            e_ij = self.ffn1e(e_ij)
            e_ij = F.relu(e_ij)
            e_ij = F.dropout(e_ij, self.dropout, training=self.training)
            e_ij = self.ffn2e(e_ij)
            e_ij = self.norm2e(eij_in + e_ij)

        return h, e_ij
    

