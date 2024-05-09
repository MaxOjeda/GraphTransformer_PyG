import torch
from torch_geometric.data import Data
from layers.GTConv import  GraphTransformerLayer

num_nodes = 10
num_node_features = 16
num_edges = 20
num_edge_features = 2

# Generate random node features
x = torch.randn(num_nodes, num_node_features)

# Generate random edge indices
#edge_index = torch.randint(high=num_nodes, size=(2, num_edges), dtype=torch.long)
edge_index = torch.tensor([[0,1],
                           [1,0],
                           [1,2],
                           [2,1],
                           [2,4],
                           [3,6],
                           [3,5],
                           [4,2],
                           [4,7],
                           [5,6],
                           [5,3],
                           [6,3],
                           [6,8],
                           [6,9],
                           [7,3],
                           [7,4],
                           [8,7],
                           [8,6],
                           [9,6]], dtype=torch.long)


# Generate random edge attributes (optional)
edge_attr = torch.randn(num_edges, num_edge_features)

gt = GraphTransformerLayer(in_dim=num_node_features, 
            #edge_dim=num_edge_features,
            hidden_dim=16, 
            n_heads=2)

data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
out = gt(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
print(out)
# tensor([[1, 0, 2, 1, 0, 2, 0, 2, 2, 2],
#        [0, 2, 2, 1, 2, 1, 2, 1, 0, 1]])