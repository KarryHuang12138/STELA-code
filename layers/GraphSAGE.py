import torch
import torch.nn.functional as F
import scipy.sparse as sp

from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dense_to_sparse

class GraphSAGE_TimeSeries(torch.nn.Module):
    def __init__(self, num_nodes, in_channels=1, out_channels=1, hidden_dim=16, adj_matrix=None):
        super(GraphSAGE_TimeSeries, self).__init__()
        self.num_nodes = num_nodes
        self.adj_matrix = adj_matrix
        self.edge_index, self.edge_weight = self.process_adj_matrix()
        self.sage = SAGEConv(in_channels, out_channels)

    def process_adj_matrix(self):
        """转换邻接矩阵为 edge_index 和 edge_weight"""
        if isinstance(self.adj_matrix, torch.Tensor):
            adj_tensor = self.adj_matrix
        else:
            adj_tensor = torch.tensor(self.adj_matrix.toarray(), dtype=torch.float32)
        edge_index, edge_weight = dense_to_sparse(adj_tensor)
        edge_weight = (edge_weight - edge_weight.mean()) / edge_weight.std()
        edge_weight = torch.clamp(edge_weight, min=0, max=1)
        return edge_index, edge_weight

    def forward(self, x):
        batchsize, seqlen, numnodes = x.shape
        x = x.unsqueeze(-1)
        x_out = []
        for t in range(seqlen):
            xt = x[:, t, :, :].reshape(batchsize * numnodes, -1)
            xt = self.sage(xt, self.edge_index)
            xt = xt.reshape(batchsize, numnodes, -1)
            x_out.append(xt)
        x_out = torch.stack(x_out, dim=1)
        return x_out.squeeze(-1)
