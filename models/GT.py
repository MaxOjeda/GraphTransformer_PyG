import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from layers.gt_layer import GraphMultiHeadAttentionLayer, MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = 28
        num_bond_type = 4
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        #self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DEVICE2: {self.device}")
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphMultiHeadAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphMultiHeadAttentionLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # input embedding
        x = self.embedding_h(x)
        x = self.in_feat_dropout(x)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            x = x + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            x = x + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            edge_attr = torch.ones(edge_attr.size(0),1).to(self.device)
        edge_attr = self.embedding_e(edge_attr)   

        # convnets
        for conv in self.layers:
            x = conv(x, edge_index)
        # print(x)
        # print(x.shape)
        # print(batch)
        # print(batch.shape)

        # if self.readout == "sum":
        #     out = global_mean_pool(x, batch)
        # elif self.readout == "max":
        #     # You can use global_max_pool instead if available
        #     raise NotImplementedError("Global max pooling not implemented in torch_geometric.")
        # elif self.readout == "mean":
        #     out = global_mean_pool(x, batch)
        # else:
        #     out = global_mean_pool(x, batch)  # default readout is mean nodes
            
        return self.MLP_layer(x)

    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss
