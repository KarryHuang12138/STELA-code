import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import GPT2Model, LlamaModel

from layers.Graph_Conv import Static_Graph_Conv
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def forward(self, conv_feat, gnn_feat):
        gate = torch.sigmoid(self.W1(conv_feat) + self.W2(gnn_feat))
        return gate * conv_feat + (1 - gate) * gnn_feat


class GraphFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_static, x_adapt):
        fusion = torch.cat([x_static, x_adapt], dim=-1)
        gate = torch.sigmoid(self.proj(fusion))
        return gate * x_static + (1 - gate) * x_adapt


class PFA_GPT2(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=2, model_name_or_path="gpt2"):
        super().__init__()
        self.device = device
        self.U = U
        self.gpt2 = GPT2Model.from_pretrained(
            model_name_or_path,
            output_attentions=True,
            output_hidden_states=True,
        ).to(device).to(torch.float32)
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x, return_attn=False):
        output = self.gpt2(inputs_embeds=x, output_attentions=return_attn)
        if return_attn:
            return output.last_hidden_state, output.attentions
        return output.last_hidden_state


class PFA_LLaMA(nn.Module):
    def __init__(self, device="cuda:0", llama_layers=32, U=4, llama_path=""):
        super().__init__()
        self.device = device
        self.U = U
        self.llama_layers = llama_layers
        self.llama = LlamaModel.from_pretrained(
            llama_path,
            output_attentions=True,
            output_hidden_states=True,
        ).to(device).to(torch.float32)
        if hasattr(self.llama, "model"):
            self.llama.model.layers = nn.ModuleList(list(self.llama.model.layers)[:llama_layers])
            self.llama_layers_ref = self.llama.model.layers
        else:
            self.llama.layers = nn.ModuleList(list(self.llama.layers)[:llama_layers])
            self.llama_layers_ref = self.llama.layers
        self.finetuned_layer_ids = None
        for param in self.llama.parameters():
            param.requires_grad = False

    def set_finetune_layers(self, layer_ids):
        """Unfreeze selected transformer blocks while keeping core norms trainable."""
        self.finetuned_layer_ids = layer_ids
        for idx, layer in enumerate(self.llama_layers_ref):
            for name, param in layer.named_parameters():
                if idx in layer_ids:
                    param.requires_grad = True
                else:
                    if "norm" in name or "embed" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        for name, param in self.llama.embed_tokens.named_parameters():
            param.requires_grad = True
        if hasattr(self.llama, "norm"):
            for _, param in self.llama.norm.named_parameters():
                param.requires_grad = True

    def forward(self, x, return_attn=False):
        output = self.llama(inputs_embeds=x, output_attentions=return_attn)
        if return_attn:
            return output.last_hidden_state, output.attentions
        return output.last_hidden_state


class STransformer(nn.Module):
    """Project temporal features into the LLM space and refine them with self-attention."""

    def __init__(
        self,
        seq_len: int,
        num_nodes: int,
        d_model: int = 4096,
        nhead: int = 32,
        nlayers: int = 2,
        dim_ff: int = 4096,
        bf16: bool = True,
    ):
        super().__init__()
        dtype = torch.bfloat16 if bf16 else torch.float32
        self.time_proj = nn.Linear(seq_len, d_model, bias=False, dtype=dtype)
        self.pos = nn.Parameter(torch.zeros(1, num_nodes, d_model, dtype=dtype))
        self.layers = nn.ModuleList(
            [_TransformerBlock(d_model, nhead, dim_ff, dtype) for _ in range(nlayers)]
        )

    def forward(self, x):
        if x.dtype != self.time_proj.weight.dtype:
            x = x.to(self.time_proj.weight.dtype)
        x = self.time_proj(x.permute(0, 2, 1))
        x = x + self.pos
        for blk in self.layers:
            x = blk(x)
        return x


class _TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dtype):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            bias=False,
            dtype=dtype,
        )
        self.ln2 = nn.LayerNorm(d_model, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff, bias=False, dtype=dtype),
            nn.GELU(),
            nn.Linear(dim_ff, d_model, bias=False, dtype=dtype),
        )

    def forward(self, x):
        qkv = self.ln1(x)
        attn_out, _ = self.attn(qkv, qkv, qkv, need_weights=False, attn_mask=None)
        x = x + attn_out
        ff = self.ffn(self.ln2(x))
        x = x + ff
        return x


class AdaptiveGraphModule(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        _, _, N = x.shape
        assert N == self.num_nodes
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        A = torch.matmul(self.nodevec1, self.nodevec2)
        A = F.relu(A)
        A = F.softmax(A, dim=1)
        x = torch.matmul(A, x)
        x = torch.matmul(x, self.weight)
        return x


class Model(nn.Module):
    def __init__(self, configs, adj_tensor, patch_len=16, stride=8):
        super().__init__()
        self.pred_len = configs.pred_len
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        num_nodes = adj_tensor.shape[0]
        self.SG_Conv = Static_Graph_Conv(adj=adj_tensor, hidden_dim=self.d_llm, seq_len=self.seq_len)
        self.graph_adaptive = AdaptiveGraphModule(
            num_nodes=configs.enc_in,
            input_dim=self.seq_len,
            hidden_dim=self.d_llm,
        )
        self.graph_fusion = GraphFusion(hidden_dim=self.d_llm)
        self.llm_model = configs.llm_model
        self.llm_path = getattr(configs, "llm_path", "")
        self.fusion = FeatureFusion(dim=self.d_llm)
        self.finalfusion = FeatureFusion(dim=self.d_llm)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpt2_layer = 12
        self.U = 2
        self.llm_layers = configs.llm_layers
        self.llm_train = configs.llm_train

        if self.llm_model == "GPT2":
            model_name_or_path = self.llm_path or "gpt2"
            self.llm_model = PFA_GPT2(
                device=self.device,
                gpt_layers=self.gpt2_layer,
                U=self.U,
                model_name_or_path=model_name_or_path,
            )
        else:
            if not self.llm_path:
                raise ValueError("Please provide --llm_path when --llm_model is LLAMA.")
            self.llm_model = PFA_LLaMA(
                device=self.device,
                llama_layers=self.llm_layers,
                U=self.llm_train,
                llama_path=self.llm_path,
            )

        self.local_conv = nn.Conv2d(self.seq_len, self.d_llm, kernel_size=(1, 1))
        self.time_transformer = STransformer(seq_len=self.seq_len, num_nodes=num_nodes)
        self.regression_layer = nn.Conv2d(self.d_llm, self.pred_len, kernel_size=(1, 1))
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_attn=False):
        x_enc = self.normalize_layers(x_enc, "norm")
        conv_feat = self.local_conv(x_enc.unsqueeze(-1).to(torch.bfloat16))
        conv_feat = conv_feat.permute(0, 2, 1, 3).squeeze(-1)
        x_adapt = self.graph_adaptive(x_enc.to(torch.bfloat16))
        enc_in = self.time_transformer(x_enc.to(torch.bfloat16))
        x_enc = self.SG_Conv(x_enc).to(torch.bfloat16)
        x_enc = self.graph_fusion(x_enc, x_adapt)
        x_enc = self.fusion(enc_in, x_enc)
        x_enc = self.finalfusion(x_enc, conv_feat)
        if return_attn:
            x_enc, attentions = self.llm_model(x_enc, return_attn=True)
            return x_enc, attentions

        x_enc = self.llm_model(x_enc)
        x_enc = x_enc.permute(0, 2, 1).unsqueeze(-1)
        x_enc = self.regression_layer(x_enc)
        x_enc = x_enc.squeeze(-1)
        x_enc = self.normalize_layers(x_enc, "denorm")
        return x_enc
