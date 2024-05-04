import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple, Optional
from torch import Tensor
#from torch_sparse import SparseTensor, set_diag
from torch.nn import Parameter, Linear

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class GraphMultiHeadAttentionLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_dim, out_dim, heads=1, negative_slope=0.2, dropout=0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GraphMultiHeadAttentionLayer, self).__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if bias:
            self.Q = nn.Linear(in_dim, out_dim * heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * heads, bias=False)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.Q.weight)
        glorot(self.K.weight)
        glorot(self.V.weight)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):

        Q_h = self.Q(x)
        K_h = self.K(x)
        V_h = self.V(x)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        Q_h = Q_h.view(-1, self.heads, self.out_dim).unsqueeze(2)  # (batch, H, 1, C)
        K_h = K_h.view(-1, self.heads, self.out_dim).unsqueeze(2)
        V_h = V_h.view(-1, self.heads, self.out_dim)

        attention_scores = torch.matmul(Q_h, K_h.transpose(-1, -2)).squeeze(-1).squeeze(-1)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.out_dim).float())

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = Q_h.size(0)
                if K_h is not None:
                    num_nodes = min(num_nodes, K_h.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            #elif isinstance(edge_index, SparseTensor):
            #    edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=(V_h),
                             alpha=(attention_scores), size=size)

        alpha = self._alpha
        self._alpha = None

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            #elif isinstance(edge_index, SparseTensor):
            #    return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_dim,
                                             self.out_channels, self.heads)


class GraphTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, batch_norm=False, residual=True, use_bias=True):
        super.__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.residual = residual
        self.use_bias = use_bias
        self.batch_norm = batch_norm

        self.attention = GraphMultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads)

        self.O = nn.Linear(out_dim, out_dim)

        if self.batch_norm:
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            self.norm = nn.LayerNorm(out_dim)

        self.ffn = nn.Linear(out_dim, out_dim*2)
        self.fnn2 = nn.Linear(out_dim * 2, out_dim)

        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(out_dim)
        else:
            self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, g, x):
        h_in1 = x
        attn_out = self.attention(g, x)
        h = attn_out.view(-1, self.out_dim)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)

        if self.residual:
            h = h + h_in1
        
        # Norm
        h = self.norm(h)
        h_in2 = h

        # FFN
        h = nn.ReLU(self.fnn(h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.fnn2(h)

        if self.residual:
            h = h + h_in2

        h = self.norm2(h)
        return h

        