import torch
import math
import torch.nn as nn
from typing import List, Optional
from torch_geometric.nn import MessagePassing


class MultiHeadAttentionLayer(MessagePassing):
    def __init__(self, in_dim:int, out_dim:int, n_heads:int=4):
        super().__init__()
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

    def message(self, edge_index, Q_i, K_j, V_j, edge_attr=None):
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
        self.num_aggrs = len(aggregators)

        # self.WQ = nn.Linear(in_dim, hidden_dim, bias=False)
        # self.WK = nn.Linear(in_dim, hidden_dim, bias=False)
        # self.WV = nn.Linear(in_dim, hidden_dim, bias=False)
        self.attention = MultiHeadAttentionLayer(in_dim, hidden_dim, n_heads)
        self.WO = nn.Linear(self.num_aggrs * hidden_dim, in_dim, bias=False)


        if batch_norm is not None:
            self.norm1 = nn.BatchNorm1d(in_dim)
            self.norm2 = nn.BatchNorm1d(in_dim)
        else:
            self.norm1 = nn.LayerNorm(in_dim)
            self.norm2 = nn.LayerNorm(in_dim)


        self.dropout_layer = nn.Dropout(p=dropout)

        #self.ff = MLP()

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
        x_ = x
        edge_attr_ = edge_attr
        attn_out = self.attention(x, edge_index)
        print(attn_out)
        # Q_h = self.WQ(x)
        # K_h = self.WK(x)
        # V_h = self.WV(x)

        # Q = Q_h.view(-1, self.n_heads, self.hidden_dim // self.n_heads)
        # K = K_h.view(-1, self.n_heads, self.hidden_dim // self.n_heads)
        # V = V_h.view(-1, self.n_heads, self.hidden_dim // self.n_heads)
        # #a = torch.matmul(Q_h, K_h.transpose(-1, -2)).squeeze(-1).squeeze(-1)
        # scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # alpha = torch.softmax(scores, dim=-1)
        # a = torch.matmul(alpha, V)
        # print(scores)
        # print(alpha)
        # print(a)
        # print(scores.shape)
        # print(alpha.shape)
        # print(a.shape)
        # print(V.shape)


        # print(f"x shape: {x.shape}")
        # print(f"Qh shape: {Q_h.shape}")
        # print(f"Q shape: {Q.shape}")
        # print(f"Q size: {Q.size(-1)}")

        # out = self.propagate(
        #     edge_index, Q=Q, K=K, V=V, edge_attr=edge_attr, size=None
        # )
        out = out.view(-1, self.hidden_dim * self.num_aggrs)  # concatenation

        # NODES
        out = self.dropout_layer(out)
        out = self.WO(out) + x_
        out = self.norm1(out)
        # FFN--nodes
        ffn_in = out
        out = self.ffn(out)
        out = self.norm2(ffn_in + out)

        if self.edge_in_dim is None:
            out_eij = None
        else:
            out_eij = self._eij
            self._eij = None
            out_eij = out_eij.view(-1, self.hidden_dim)

            # EDGES
            out_eij = self.dropout_layer(out_eij)
            out_eij = self.WOe(out_eij) + edge_attr_  # Residual connection
            out_eij = self.norm1e(out_eij)
            # FFN--edges
            ffn_eij_in = out_eij
            out_eij = self.ffn_e(out_eij)
            out_eij = self.norm2e(ffn_eij_in + out_eij)

        return (out, out_eij)
    
    def message(self, Q_i, K_j, V_j, G_j, index, edge_attr=None):
        d_k = Q_i.size(-1)
        qijk = (Q_i * K_j) / math.sqrt(d_k)
        if self.edge_in_dim is not None:
            assert edge_attr is not None
            E = self.WE(edge_attr).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
            qijk = E * qijk
            self._eij = qijk
        else:
            self._eij = None


        qijk = (Q_i * K_j).sum(dim=-1) / math.sqrt(d_k)

        alpha = torch.softmax(qijk, index)  # Log-Sum-Exp trick used. No need for clipping (-5,5)

        if self.gate:
            V_j_g = torch.mul(V_j, torch.sigmoid(G_j))
        else:
            V_j_g = V_j

        return alpha.view(-1, self.num_heads, 1) * V_j_g

