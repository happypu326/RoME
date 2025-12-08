from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from model.gcn import GNNEncoder, GNNDecoder


class Expert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
        )
        self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GatingNetwork(nn.Module):
    def __init__(self, dim: int, num_dedicate_experts: int,
                 top_k: int, temperature: float = 0.5, bias_lr: float = 1e-3):
        super().__init__()
        self.num_experts = num_dedicate_experts
        self.top_k = top_k
        self.temperature = temperature
        self.bias_lr = bias_lr

        self.gate_proj = nn.Linear(dim, num_dedicate_experts)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.expert_biases = nn.Parameter(torch.zeros(num_dedicate_experts))

    def forward(self, g_emb: torch.Tensor) -> torch.Tensor:
        # g_emb: [B, dim]
        base_logits = self.gate_proj(g_emb) * self.alpha / self.temperature
        logits = base_logits + self.expert_biases.unsqueeze(0)

        # Hard top-k
        _, topk_idx = logits.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)

        weights = logits.softmax(dim=-1) * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        return weights

    def update_biases(self, route_weights: torch.Tensor):
        if not self.training:
            return

        expert_load = route_weights.detach().mean(dim=0)
        ideal = expert_load.mean()
        delta = expert_load - ideal

        update = -self.bias_lr * torch.sign(delta)
        self.expert_biases.data.add_(update).clamp_(-1.5, 1.5)

class StructTokenMemory(nn.Module):
    """
    QKV-style structural codebook:
    - tokens_K: keys for structural prototypes
    - tokens_V: values for structural tokens
    - q_proj: GNN node embedding -> query vector
    Supports:
      - soft mode: each node performs softmax over all tokens
      - hard/top-k mode: each node performs softmax only on top-k tokens (sparse activation)
    """
    def __init__(
        self,
        in_dim: int,
        num_tokens: int = 64,
        token_dim: int | None = None,
        hard_token_routing: bool = False,  
        token_topk: int = 8,             
    ):
        super().__init__()
        self.in_dim = in_dim
        self.token_dim = token_dim or in_dim
        self.num_tokens = num_tokens

        self.hard_token_routing = hard_token_routing
        self.token_topk = token_topk

        # Key/Value codebook
        self.tokens_K = nn.Parameter(
            torch.randn(num_tokens, self.token_dim) * 0.02
        )
        self.tokens_V = nn.Parameter(
            torch.randn(num_tokens, self.token_dim) * 0.02
        )

        # Query projection: from v_emb -> token space
        self.q_proj = nn.Linear(in_dim, self.token_dim)

    def forward(self, v_emb: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        v_emb: [N, d] node-level embedding (variable nodes)
        batch_idx: [N] which graph each node belongs to (0..B-1)
        Returns:
            struct_emb: [B, token_dim] structural representation for each graph
        """
        if v_emb.numel() == 0:
            B = 0 if batch_idx.numel() == 0 else int(batch_idx.max().item()) + 1
            return torch.zeros(B, self.token_dim, device=v_emb.device, dtype=v_emb.dtype)

        # Node-level Query
        Q = self.q_proj(v_emb)  # [N, d_t]

        # 2) QK^T attention (query codebook)
        scores = torch.matmul(Q, self.tokens_K.t()) / math.sqrt(self.token_dim)  # [N, T]

        if self.hard_token_routing:
            # Top-K sparse activation
            k = min(self.token_topk, self.num_tokens)
            topk_val, topk_idx = scores.topk(k=k, dim=-1)        # [N, k], [N, k]

            # Mask non-top-k positions with -inf, then apply softmax
            mask = scores.new_full(scores.shape, float("-inf"))  # [N, T]
            mask.scatter_(dim=-1, index=topk_idx, src=topk_val)  

            weights = mask.softmax(dim=-1)                       # [N, T]，only top-k are non-zero
        else:
            # Original soft mode
            weights = scores.softmax(dim=-1)                     # [N, T]

        # 3) Node-level structural embedding
        node_struct = torch.matmul(weights, self.tokens_V)       # [N, d_t]

        # 4) Graph-level pooling
        struct_emb = global_mean_pool(node_struct, batch_idx)    # [B, d_t]

        return struct_emb


class MoEPolicy(nn.Module):
    def __init__(
        self,
        emb_size: int = 64,
        constraint_nfeats: int = 4,
        edge_nfeats: int = 1,
        variable_nfeats: int = 6,
        num_shared_experts: int = 2,
        num_dedicate_experts: int = 16,
        top_k: int = 4,
        gate_temperature: float = 0.6,
        bias_lr: float = 1e-3,
        dropout: float = 0.1,
        use_dro: bool = True,
        eps_wasserstein: float = 0.2,
        dro_perturb_type: str = "gaussian",
        use_struct_tokens: bool = True,
        num_struct_tokens: int = 64,
        struct_token_dim: int | None = None,
        hard_token_routing: bool = False,  
        token_topk: int = 8, 
    ):
        super().__init__()
        self.dim = emb_size
        self.Ne = num_dedicate_experts
        self.Ks = num_shared_experts
        self.top_k = top_k

        self.use_dro = use_dro
        self.eps = eps_wasserstein
        self.dro_perturb_type = dro_perturb_type

        # GNN Encoder
        self.encoder = GNNEncoder(
            emb_size, constraint_nfeats, edge_nfeats, variable_nfeats
        )

        # MoE Experts
        self.shared_experts = nn.ModuleList(
            [Expert(emb_size) for _ in range(self.Ks)]
        )
        self.dedicate_experts = nn.ModuleList(
            [Expert(emb_size) for _ in range(self.Ne)]
        )

        # Structural token memory
        self.use_struct_tokens = use_struct_tokens
        self.struct_token_dim = struct_token_dim or emb_size
        self.num_struct_tokens = num_struct_tokens
        self.hard_token_routing = hard_token_routing
        self.token_topk = token_topk

        if self.use_struct_tokens:
            self.struct_mem = StructTokenMemory(
                in_dim=emb_size,
                num_tokens=self.num_struct_tokens,
                token_dim=self.struct_token_dim,
                hard_token_routing=self.hard_token_routing,
                token_topk=self.token_topk,
            )
            gate_in_dim = emb_size + self.struct_token_dim
        else:
            self.struct_mem = None
            gate_in_dim = emb_size

        # Gating network
        self.gate = GatingNetwork(
            gate_in_dim,
            num_dedicate_experts,
            top_k,
            temperature=gate_temperature,
            bias_lr=bias_lr,
        )

        # Task head
        self.task_head = GNNDecoder(emb_size)

    # ------------------------------------------------
    # Shared / Dedicated Experts
    # ------------------------------------------------
    def _shared_experts_forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [exp(x) for exp in self.shared_experts]
        return torch.stack(outs, dim=0).mean(0)

    def _dedicate_experts_forward(self, x, route_weights, batch_idx):
        """
        x: [N, d]
        route_weights: [B, Ne]
        batch_idx: [N]
        """
        # [Ne, N, d] → [N, Ne, d]
        expert_out = torch.stack(
            [exp(x) for exp in self.dedicate_experts], dim=0
        ).permute(1, 0, 2)  # [N, Ne, d]

        # [B, Ne] → [N, Ne]
        batch_weights = route_weights[batch_idx]  # [N, Ne]

        # einsum: (N, Ne) × (N, Ne, d) → (N, d)
        fused = torch.einsum("ne,ned->nd", batch_weights, expert_out)
        return fused

    # ------------------------------------------------
    # DRO part: perturb gate_input
    # ------------------------------------------------
    def _generate_wasserstein_perturbation(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        if self.dro_perturb_type == "gaussian":
            delta = torch.randn_like(z)
        else:
            delta = torch.randn_like(z)
            delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-12)
            delta *= torch.rand(B, 1, device=z.device)

        norm = delta.norm(dim=-1, keepdim=True)
        delta = delta * (self.eps / (norm + 1e-12))
        return delta

    def _dro_robust_loss(
        self,
        gate_input: torch.Tensor,
        route_weights: torch.Tensor,
        v_emb: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_dro or self.eps <= 0:
            return torch.tensor(0.0, device=v_emb.device)

        delta = self._generate_wasserstein_perturbation(gate_input)
        z_tilde = (gate_input + delta).detach()

        adv_route = self.gate(z_tilde)
        adv_fused = self._dedicate_experts_forward(v_emb, adv_route, batch_idx)
        orig_fused = self._dedicate_experts_forward(v_emb, route_weights, batch_idx)
        return F.mse_loss(adv_fused, orig_fused)

    def forward(
        self,
        c_feat: torch.Tensor,
        edge_idx: torch.Tensor,
        edge_attr: torch.Tensor,
        v_feat: torch.Tensor,
        batch_idx: torch.Tensor,
        is_training: bool = False,
    ):
        # GNN encoding
        v_emb, c_emb = self.encoder(c_feat, edge_idx, edge_attr, v_feat, batch_idx)

        # Graph-level embedding
        g_emb = global_mean_pool(v_emb, batch_idx)  # [B, d]

        # Structural token embedding: lookup + combination on QKV codebook
        if self.use_struct_tokens and self.struct_mem is not None:
            struct_emb = self.struct_mem(v_emb, batch_idx)      # [B, d_t]
            gate_input = torch.cat([g_emb, struct_emb], dim=-1) # [B, d + d_t]
        else:
            gate_input = g_emb

        # Gating & MoE
        route_weights = self.gate(gate_input)                   # [B, Ne]
        dedicate_fused = self._dedicate_experts_forward(
            v_emb, route_weights, batch_idx
        )  # [N, d]
        shared_fused = self._shared_experts_forward(v_emb)      # [N, d]
        
        combined = v_emb + shared_fused + dedicate_fused        # [N, d]

        # Task output
        logits = self.task_head(combined).squeeze(-1)           # [N]

        if is_training:
            # Update gate biases
            self.gate.update_biases(route_weights)

            # DRO: apply Wasserstein perturbation to gate_input
            aux_loss = self._dro_robust_loss(
                gate_input, route_weights, v_emb, batch_idx
            )
            return logits, aux_loss

        return logits, None
