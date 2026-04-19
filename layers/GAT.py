import torch
import torch.nn.functional as F
import scipy.sparse as sp

from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

class GAT_TimeSeries(torch.nn.Module):
    def __init__(self, num_nodes, in_channels=1, out_channels=768, hidden_dim=256, heads=8, adj_matrix=None):
        super(GAT_TimeSeries, self).__init__()
        self.num_nodes = num_nodes
        self.adj_matrix = adj_matrix
        self.edge_index, self.edge_weight = self.process_adj_matrix()
        self.gat1 = GATConv(in_channels, hidden_dim, heads=heads, dropout=0.1, concat=True, add_self_loops=False)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1, concat=True, add_self_loops=False)
        self.gat3 = GATConv(hidden_dim * heads, out_channels, heads=1, dropout=0.1, concat=False, add_self_loops=False)

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
        """
        x: (batch_size, seq_len, num_nodes)
        output: (batch_size, num_nodes, llmdim)
        """
        batch_size, seq_len, num_nodes = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * num_nodes, seq_len)
        x = F.relu(self.gat1(x, self.edge_index, self.edge_weight))
        x = F.relu(self.gat2(x, self.edge_index, self.edge_weight))
        x = self.gat3(x, self.edge_index, self.edge_weight)
        x = x.reshape(batch_size, num_nodes, -1)
        return x
