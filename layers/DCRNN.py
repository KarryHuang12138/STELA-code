import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_supports(adj_mx, max_diffusion_step):
    """
    Computes diffusion supports up to max_diffusion_step using random walk
    """
    adj = adj_mx + torch.eye(adj_mx.size(0), device=adj_mx.device)
    d = adj.sum(1)
    d_inv = torch.diag(torch.pow(d, -1))
    d_inv[torch.isinf(d_inv)] = 0.
    rw_matrix = torch.mm(d_inv, adj)
    supports = [torch.eye(adj.size(0), device=adj.device), rw_matrix]
    for k in range(2, max_diffusion_step + 1):
        supports.append(torch.matmul(rw_matrix, supports[-1]))
    return supports

class DiffusionConv(nn.Module):
    def __init__(self, input_dim, output_dim, adj_mx, max_diffusion_step):
        super(DiffusionConv, self).__init__()
        self.supports = compute_supports(adj_mx, max_diffusion_step)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, output_dim)) for _ in self.supports
        ])
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        out = 0
        for support, weight in zip(self.supports, self.weights):
            x_support = torch.einsum("ij,bjk->bik", support, x)
            x_support = x_support.to(torch.bfloat16)
            out += torch.matmul(x_support, weight)
        return out + self.bias

class DCGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj_mx, max_diffusion_step):
        super(DCGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.dc_gate = DiffusionConv(input_dim + hidden_dim, 2 * hidden_dim, adj_mx, max_diffusion_step)
        self.dc_candidate = DiffusionConv(input_dim + hidden_dim, hidden_dim, adj_mx, max_diffusion_step)

    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=-1)
        gate_output = self.dc_gate(combined)
        r, u = torch.chunk(torch.sigmoid(gate_output), 2, dim=-1)
        c = torch.tanh(self.dc_candidate(torch.cat([x, r * hidden], dim=-1)))
        h_new = u * hidden + (1 - u) * c
        return h_new

class DCRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj_mx, max_diffusion_step, num_layers):
        super(DCRNN, self).__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            DCGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim, adj_mx, max_diffusion_step)
            for i in range(num_layers)
        ])

    def forward(self, x):
        batch_size, seq_len, num_nodes, input_dim = x.shape
        device = x.device
        hidden_states = [torch.zeros(batch_size, num_nodes, self.cells[0].hidden_dim, device=device)
                         for _ in range(self.num_layers)]
        for t in range(seq_len):
            input_t = x[:, t, :, :]
            for i, cell in enumerate(self.cells):
                hidden_states[i] = cell(input_t, hidden_states[i])
                input_t = hidden_states[i]
        return input_t.to(torch.bfloat16)
