import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torch_scatter import scatter
from utils import negate_edge_index

class MultiHeadAttention(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, fake_edge_emb, use_bias) -> None:
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph = full_graph

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        if self.full_graph:
            self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.fake_edge_emb = fake_edge_emb

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]  # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]  # (num real edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        if self.full_graph:
            fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)
            src_2 = batch.K_2h[fake_edge_index[0]]  # (num fake edges) x num_heads x out_dim
            dest_2 = batch.Q_2h[fake_edge_index[1]]  # (num fake edges) x num_heads x out_dim
            score_2 = torch.mul(src_2, dest_2)

            # Scale scores by sqrt(d)
            score_2 = score_2 / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim

        if self.full_graph:
            # E_2 is 1 x num_heads x out_dim and will be broadcast over dim=0
            score_2 = torch.mul(score_2, batch.E_2)

        if self.full_graph:
            # softmax and scaling by gamma
            score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1
            score_2 = torch.exp(score_2.sum(-1, keepdim=True).clamp(-5, 5))  # (num fake edges) x num_heads x 1
            score = score / (self.gamma + 1)
            score_2 = self.gamma * score_2 / (self.gamma + 1)
        else:
            score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.full_graph:
            # Attention via fictional edges
            msg_2 = batch.V_h[fake_edge_index[0]] * score_2
            # Add messages along fake edges to destination nodes
            scatter(msg_2, fake_edge_index[1], dim=0, out=batch.wV, reduce='add')

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, batch.edge_index[1], dim=0, out=batch.Z, reduce='add')
        if self.full_graph:
            scatter(score_2, fake_edge_index[1], dim=0, out=batch.Z, reduce='add')
        
    
    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        E = self.E(batch.edge_attr)

        if self.full_graph:
            Q_2h = self.Q_2(batch.x)
            K_2h = self.K_2(batch.x)
            # One embedding used for all fake edges; shape: 1 x emb_dim
            dummy_edge = self.fake_edge_emb(batch.edge_index.new_zeros(1))
            E_2 = self.E_2(dummy_edge)

        V_h = self.V(batch.x)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)

        if self.full_graph:
            batch.Q_2h = Q_2h.view(-1, self.num_heads, self.out_dim)
            batch.K_2h = K_2h.view(-1, self.num_heads, self.out_dim)
            batch.E_2 = E_2.view(-1, self.num_heads, self.out_dim)

        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch)

        h_out = batch.wV / (batch.Z + 1e-6)

        return h_out




class GraphTransformer(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph,
                 fake_edge_emb, dropout=0.0, layer_norm=False, batch_norm=True, residual=True) -> None:
        super().__init__()

        self.gamma = gamma
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.full_graph = full_graph
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.residual = residual

        self.attention = MultiHeadAttention()

        self.O_h = nn.Linear(out_dim, out_dim)

        # Primero se normaliza, puede ser layer o batch normalization
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # Para por una MLP (dos capas fc)
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2) # W1 -> R 2dxd
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim) # W2 -> R dx2d

        # Se vuelve a normalizar
        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    
    def forward(self, batch):
        h = batch.x
        h_in1 = h # Pre residual
        
        ### Self attention
        # Multihead outputs
        h_attn_out = self.attention(batch)
        # Concatenar outputs
        h = h_attn_out.view(-1, self.out_dim)

        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O_h(h)

        # Residual connection
        if self.residual:
            h = h + h_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        elif self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h # Pre residual
        
        # Feed Forward
        h = self.FFN_h_layer1(h)
        h = nn.ReLU(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        elif self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch
        

