import torch
from torch_geometric.data import Data
from layers.GTConv import  GraphTransformerLayer

num_nodes = 10
num_node_features = 3
num_edges = 20
num_edge_features = 2

# Generate random node features
x = torch.randn(num_nodes, num_node_features)

# Generate random edge indices
edge_index = torch.randint(high=num_nodes, size=(2, num_edges))

# Generate random edge attributes (optional)
edge_attr = torch.randn(num_edges, num_edge_features)

gt = GraphTransformerLayer(in_dim=num_node_features, 
            #edge_dim=num_edge_features,
            hidden_dim=16, 
            n_heads=2)
gt(x=x, edge_index=edge_index, edge_attr=edge_attr)

# tensor([[1, 0, 2, 1, 0, 2, 0, 2, 2, 2],
#        [0, 2, 2, 1, 2, 1, 2, 1, 0, 1]])