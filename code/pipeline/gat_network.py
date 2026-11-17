"""
GNN/GAT-based Policy Networks for RL-SCED
=========================================
Replaces MLP backbone with Graph Attention Network (GATv2) to encode
power system topology into dispatch decisions.

Design:
  - Node features: per-unit state (on/off, duration, power ratio, mask, fixed value)
  - Global features: system-level context (load, reserve, time, line status, etc.)
  - Edge index: built from PTDF matrix (units coupled by shared line flow sensitivity)
  - Architecture: GATv2Conv × 2 layers → global mean pool → concat(global) → actor/critic heads
  - Interface: identical to UCNetwork/EDNetwork (forward, forward_actor, forward_critic)

Compatible with: Stable-Baselines3 PPO, existing BC pre-training pipeline.

Usage (in ED_RL_BC_relaxfix.py):
    from gat_network import GATUCNetwork, GATEDNetwork, build_unit_graph

    # Replace UCNetwork with GATUCNetwork
    # mlp_extractor = GATUCNetwork(obs_dim, n_units, edge_index)
    # Same for EDPolicy → GATEDNetwork
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ==============================================================
# Graph Construction Utilities
# ==============================================================

def build_unit_graph_from_ptdf(
    ptdf_data: np.ndarray,
    n_units: int,
    unit_bus_map: Optional[np.ndarray] = None,
    k_neighbors: int = 5,
    ptdf_threshold: float = 0.05,
    self_loops: bool = False,
) -> torch.LongTensor:
    """
    Build unit-to-unit adjacency (edge_index) from PTDF matrix.

    Two construction strategies (combined via OR):
      1. Bus-based: units on same or physically connected buses are linked.
      2. PTDF-coupling: units with strong joint influence on any line are linked.

    Args:
        ptdf_data: shape (N_LINES, N_BUSES), PTDF matrix from data files
        n_units: number of generator units (N_UNITS)
        unit_bus_map: optional (n_units,) int array mapping each unit to its bus ID
                      If None, falls back to pure PTDF-coupling strategy
        k_neighbors: for each unit, connect to top-k most PTDF-correlated neighbors
        ptdf_threshold: minimum |PTDF correlation| to create an edge
        self_loops: whether to include self-loops

    Returns:
        edge_index: LongTensor (2, E) in COO format (PyTorch Geometric compatible)
    """
    n_lines, n_buses = ptdf_data.shape

    if unit_bus_map is not None:
        # Strategy 1: Physical bus connectivity
        bus_edges = _build_bus_based_edges(unit_bus_map, n_units, self_loops)
    else:
        bus_edges = None

    # Strategy 2: PTDF coupling between units
    # Approximate each unit's "line signature" by its bus's PTDF column
    # If we don't have unit_bus_map, distribute units evenly across buses
    if unit_bus_map is not None:
        unit_ptdf = ptdf_data[:, unit_bus_map]  # (N_LINES, N_UNITS)
    else:
        # Fallback: assign each unit to a unique or shared bus pattern
        # Use a synthetic mapping based on unit index modulo n_buses
        synth_bus = np.arange(n_units) % min(n_buses, n_units)
        unit_ptdf = ptdf_data[:, synth_bus]  # (N_LINES, N_UNITS)

    ptdf_edges = _build_ptdf_coupling_edges(unit_ptdf, n_units, k_neighbors, ptdf_threshold, self_loops)

    # Combine both edge sets and deduplicate
    if bus_edges is not None:
        all_edges = np.concatenate([bus_edges, ptdf_edges], axis=1)  # (2, E1+E2)
    else:
        all_edges = ptdf_edges

    # Deduplicate
    all_edges = np.unique(all_edges, axis=1)

    return torch.LongTensor(all_edges)


def _build_bus_based_edges(
    unit_bus_map: np.ndarray,
    n_units: int,
    self_loops: bool,
) -> np.ndarray:
    """Connect units sharing the same bus."""
    edges_src, edges_dst = [], []
    for b in np.unique(unit_bus_map):
        units_on_bus = np.where(unit_bus_map == b)[0]
        for i in range(len(units_on_bus)):
            for j in range(i + (0 if self_loops else 1), len(units_on_bus)):
                edges_src.append(units_on_bus[i])
                edges_dst.append(units_on_bus[j])
    return np.array([edges_src, edges_dst], dtype=np.int64)


def _build_ptdf_coupling_edges(
    unit_ptdf: np.ndarray,
    n_units: int,
    k: int,
    threshold: float,
    self_loops: bool,
) -> np.ndarray:
    """
    Connect units whose PTDF signatures are highly correlated.

    For each pair (i,j), compute cosine similarity of their PTDF columns
    across all lines. If similarity > threshold or in top-k, add edge.
    """
    # Normalize PTDF columns for cosine similarity
    norms = np.linalg.norm(unit_ptdf, axis=0, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    unit_ptdf_normed = unit_ptdf / norms  # (N_LINES, N_UNITS)

    # Pairwise cosine similarity matrix
    sim_matrix = unit_ptdf_normed.T @ unit_ptdf_normed  # (N_UNITS, N_UNITS)

    edges_src, edges_dst = [], []
    for i in range(n_units):
        # Get similarities for unit i (excluding self if no self-loops)
        row = sim_matrix[i].copy()
        if not self_loops:
            row[i] = -1.0

        # Top-k neighbors
        top_k_idx = np.argsort(row)[-k:] if k < n_units else np.arange(n_units)
        for j in top_k_idx:
            if (self_loops or j != i) and abs(sim_matrix[i, j]) >= threshold:
                edges_src.append(i)
                edges_dst.append(j)

    result = np.array([edges_src, edges_dst], dtype=np.int64)
    # Remove duplicates (i,j) and (j,i) — keep only one direction, GAT handles both
    if result.size > 0:
        sorted_edges = np.sort(result, axis=0)
        _, uniq_idx = np.unique(sorted_edges, axis=1, return_index=True)
        result = result[:, np.sort(uniq_idx)]
    return result


# ==============================================================
# GAT Layer Implementations (Pure PyTorch, no PyG dependency)
# ==============================================================

class GATv2ConvLayer(nn.Module):
    """
    Single GATv2 (static) attention layer implemented in pure PyTorch.
    
    GATv2 key difference from vanilla GAT: attention computation is applied
    AFTER the linear transformation, making it "dynamic" attention.
    
    Reference: Brody et al., "How Attentive Are Graph Attention Networks?"
             (arXiv 2022)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        bias: bool = True,
        share_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.share_weights = share_weights

        # Linear transformation for source nodes
        self.W_src = nn.Linear(in_features, heads * out_features, bias=False)
        # Linear transformation for target nodes
        self.W_dst = nn.Linear(in_features, heads * out_features, bias=bias) \
            if not share_weights else self.W_src
        # Attention scoring parameter (scalar per head)
        self.a = nn.Parameter(torch.zeros(1, heads, out_features))
        # Skip connection projection (optional, helps training stability)
        self.skip_proj = (
            nn.Linear(in_features, heads * out_features * (out_features if concat else 1), bias=False)
            if in_features != (heads * out_features * (out_features if concat else 1)) else None
        )

        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.W_src.weight, gain=gain)
        if self.W_dst is not self.W_src:
            nn.init.xavier_uniform_(self.W_dst.weight, gain=gain)
        nn.init.normal_(self.a, std=0.02)

    def forward(
        self,
        x: torch.Tensor,          # (B*N, F_in) or (N, F_in) for single graph
        edge_index: torch.Tensor,  # (2, E)
        batch: Optional[torch.Tensor] = None,  # (N,) node batch assignment
    ) -> torch.Tensor:
        """
        Forward pass for GATv2 layer.

        Args:
            x: node feature matrix, shape (num_nodes_total, in_features)
               For batched input: can be (batch_size * num_nodes, in_features)
            edge_index: COO format edge list, shape (2, num_edges)
            batch: optional vector indicating which graph each node belongs to

        Returns:
            Updated node features, shape (num_nodes_total, heads*out_features) if concat
                                               or (num_nodes_total, out_features) otherwise
        """
        num_nodes = x.size(0)

        # Linear transformations
        h_src = self.W_src(x)  # (N, H*D_out)
        h_dst = self.W_dst(x)  # (N, H*D_out)

        # Reshape for multi-head: (N, H, D_out)
        h_src = h_src.view(-1, self.heads, self.out_features)
        h_dst = h_dst.view(-1, self.heads, self.out_features)

        # Compute attention scores: LeakyReLU(a^T [W_src*h_i || W_dst*h_j])
        edge_src_idx = edge_index[0]   # (E,)
        edge_dst_idx = edge_index[1]   # (E,)

        # Get source and target representations for each edge
        h_src_j = h_src[edge_src_idx]  # (E, H, D_out) — messages FROM src
        h_dst_i = h_dst[edge_dst_idx]  # (E, H, D_out) — messages TO dst

        # Concatenate and apply attention vector
        e_ij = torch.cat([h_src_j, h_dst_i], dim=-1)  # (E, H, 2*D_out)
        e_ij = (e_ij * self.a).sum(dim=-1)              # (E, H) — raw attention scores
        e_ij = F.leaky_relu(e_ij, negative_slope=0.2)     # (E, H)

        # Softmax normalization over incoming edges for each destination node
        e_ij = self._sparse_softmax(e_ij, edge_dst_idx, num_nodes)

        # Dropout on attention weights
        if self.training and self.dropout > 0:
            e_ij = F.dropout(e_ij, p=self.dropout, training=True)

        # Weighted message aggregation: sum over incoming edges
        # For each destination node i: out_i = sum_j(alpha_ij * W_src * h_j)
        out = torch.zeros_like(h_dst)  # (N, H, D_out)
        out.scatter_add_0(edge_dst_idx, e_ij.unsqueeze(-1) * h_src_j)

        # Residual connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x).view(-1, self.heads, self.out_features if self.concat else 1)
            out = out + skip

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_features)  # (N, H*D_out)
        else:
            out = out.mean(dim=1)  # (N, D_out)

        return out

    def _sparse_softmax(
        self,
        scores: torch.Tensor,       # (E, H)
        dst_idx: torch.Tensor,      # (E,)
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute softmax over groups defined by destination node index."""
        scores_max = torch.full(
            (num_nodes, scores.size(1)), float('-inf'),
            device=scores.device, dtype=scores.dtype,
        )
        scores_max.scatter_reduce_(
            dst_idx.unsqueeze(-1).expand_as(scores),
            scores, reduce='amax', include_self=True,
        )
        scores_stable = scores - scores_max[dst_idx]

        exp_scores = torch.exp(scores_stable)
        exp_sum = torch.zeros(num_nodes, scores.size(1), device=scores.device, dtype=scores.dtype)
        exp_sum.scatter_add_0(dst_idx.unsqueeze(-1).expand_as(exp_scores), exp_scores)

        return exp_scores / (exp_sum[dst_idx] + 1e-10)

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized attention weights (E, H) for visualization."""
        h_src = self.W_src(x).view(-1, self.heads, self.out_features)
        h_dst = self.W_dst(x).view(-1, self.heads, self.out_features)
        h_src_j = h_src[edge_index[0]]
        h_dst_i = h_dst[edge_index[1]]
        e_ij = (torch.cat([h_src_j, h_dst_i], dim=-1) * self.a).sum(dim=-1)
        e_ij = F.leaky_relu(e_ij, 0.2)
        attn = self._sparse_softmax(e_ij, edge_index[1], x.size(0))
        return attn.detach()


# ==============================================================
# GAT-based UC Network (replaces UCNetwork for SCUC task)
# ==============================================================

class GATUCNetwork(nn.Module):
    """
    GNN-based policy network for Unit Commitment (SCUC).

    Replaces MLP backbone with 2-layer GATv2 + readout.
    Maintains identical interface to UCNetwork for SB3 compatibility.

    Observation parsing (internal):
      - obs vector is split into:
        node_feat: (n_units × node_dim) — per-unit features
        global_feat: (global_dim,) — system-level features
      - These are reassembled into a graph structure internally

    Architecture:
      node_feat (N×5) ──→ GATv2Conv(N=4,H=128) ──→ GATv2Conv(N=4,H=128) ──→ mean_pool ──┐
                                                                                          ├→ concat ──→ Actor Head (512→256→N_units) → Bernoulli logits
      global_feat (20) ───────────────────────────────────────────────────────┤
                                                                                          └──→ Critic Head (512→256→1) → V(s)
    """

    # Feature layout constants for SCUC observation vector (obs_dim = 5*n_units + 20)
    NODE_DIM_UC = 5   # on_off, on_dur/T, off_dur/T, active_mask, fixed_value
    GLOBAL_DIM_UC = 20  # t/T, cur_dem, cur_rsv, future_load×4, future_rsv×4, line_obs×3, ...

    def __init__(
        self,
        obs_dim: int,
        n_units: int,
        edge_index: Optional[torch.LongTensor] = None,
        gat_heads: int = 4,
        gat_hidden: int = 128,
        gat_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_units = n_units
        self.obs_dim = obs_dim
        self.latent_dim_pi = 512
        self.latent_dim_vf = 512

        # Store edge index (graph topology) — registered as buffer so it moves with model
        if edge_index is None:
            # Default: complete graph (every unit connected to every other unit)
            src, dst = [], []
            for i in range(n_units):
                for j in range(n_units):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            edge_index = torch.LongTensor([src, dst])
        self.register_buffer('edge_index', edge_index, persistent=True)

        # --- GAT Backbone ---
        self.gat_layers = nn.ModuleList()
        in_dim = self.NODE_DIM_UC
        for l in range(gat_layers):
            is_last = (l == gat_layers - 1)
            self.gat_layers.append(GATv2ConvLayer(
                in_features=in_dim,
                out_features=gat_hidden,
                heads=gat_heads,
                concat=(not is_last),
                dropout=dropout if not is_last else 0.0,
            ))
            in_dim = gat_hidden * gat_heads if (not is_last) else gat_hidden

        self.gat_output_dim = in_dim  # final output dimension after GAT

        # --- Readout & Fusion ---
        # Global feature projector: project global (20-dim) to match GAT output dimension space
        self.global_proj = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM_UC, 128), nn.ReLU(),
            nn.Linear(128, self.gat_output_dim),
        )

        # Fusion: concatenate GAT readout + projected global features
        fusion_dim = self.gat_output_dim + self.gat_output_dim  # graph_readout | global_proj
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.ReLU(),
        )

        # --- Output Heads (same as UCNetwork) ---
        self.on_off_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_units),
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def _parse_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse flat observation vector into structured (node_features, global_features).

        Args:
            obs: (B, obs_dim) flat observation tensor

        Returns:
            node_feat: (B*N, node_dim) — stacked per-unit features
            global_feat: (B, global_dim) — system-level features
        """
        B = obs.size(0)
        node_feat = obs[:, :self.n_units * self.NODE_DIM_UC]  # (B, N*5)
        global_feat = obs[:, self.n_units * self.NODE_DIM_UC:]  # (B, 20)

        node_feat = node_feat.reshape(B * self.n_units, self.NODE_DIM_UC)  # (B*N, 5)

        return node_feat, global_feat

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SB3 interface: returns (actor_feat, critic_feat) tuple."""
        feat = self._gat_forward(obs)
        return feat, feat

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Return backbone features for action head (for UCPolicy)."""
        return self._gat_forward(obs)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return value estimate V(s)."""
        return self.value_head(self._gat_forward(obs))

    def _gat_forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Core GNN forward: obs → parse → GAT layers → readout → fuse → latent (B, 512).
        """
        B = obs.size(0)
        node_feat, global_feat = self._parse_obs(obs)

        # Expand edge_index for batch: offset each graph's node indices by B*i
        edge_index_batch = self.edge_index  # (2, E_base)
        if B > 1:
            offsets = torch.arange(B, device=obs.device).unsqueeze(1) * self.n_units  # (B, 1)
            edge_index_batch = torch.cat([
                edge_index_batch[0:1].unsqueeze(0).expand(B, -1, -1) + offsets,
                edge_index_batch[1:2].unsqueeze(0).expand(B, -1, -1) + offsets,
            ], dim=0).reshape(2, -1)  # (2, B*E_base)

        # GAT message passing
        h = node_feat  # (B*N, 5)
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index_batch)  # (B*N, D_out)

        # Global mean pooling per graph (per sample in batch)
        h_pooled = h.reshape(B, self.n_units, -1).mean(dim=1)  # (B, D_gat)

        # Fuse with global features
        g_proj = self.global_proj(global_feat)  # (B, D_gat)
        fused = torch.cat([h_pooled, g_proj], dim=-1)  # (B, 2*D_gat)

        # Final projection to latent space
        latent = self.fusion_net(fused)  # (B, 512)

        return latent

    def get_attention_map(self, obs: torch.Tensor) -> dict:
        """
        Extract attention weights from first GAT layer for visualization.

        Returns:
            dict with keys 'edge_index' (2,E), 'attention' (E,heads),
                   'node_labels' (list of str)
        """
        if len(self.gat_layers) == 0:
            return {}
        node_feat, _ = self._parse_obs(obs[:1])  # use first sample only
        attn = self.gat_layers[0].get_attention_weights(node_feat, self.edge_index)
        return {
            'edge_index': self.edge_index.cpu().numpy(),
            'attention': attn.cpu().numpy(),
            'node_labels': [f'Unit_{i}' for i in range(self.n_units)],
        }


# ==============================================================
# GAT-based ED Network (replaces EDNetwork for SCED task)
# ==============================================================

class GATEDNetwork(nn.Module):
    """
    GNN-based policy network for Economic Dispatch (SCED).

    Replaces MLP backbone with 2-layer GATv2 + readout.
    Maintains identical interface to EDNetwork for SB3 compatibility.

    Architecture differences vs GATUCNetwork:
      - NODE_DIM_ED = 4 (or 5 for PaperSCEDEnv with q_est): includes pwr_ratio
      - GLOBAL_DIM_ED = 24 (or 28 for PaperSCEDEnv)
      - Has learnable log_std parameter (Gaussian policy)
      - Includes He initialization and Dropout
      - 3-layer GAT backbone (matching original's 3-layer MLP depth)
    """

    NODE_DIM_ED = 4    # on_off, on_dur/T, off_dur/T, pwr_ratio
    GLOBAL_DIM_ED = 24  # t/T, load, reserve, cap_min/max, future_load×4, etc.

    # PaperSCEDEnv variant dimensions
    NODE_DIM_PAPER = 5   # adds q_est
    GLOBAL_DIM_PAPER = 28

    def __init__(
        self,
        obs_dim: int,
        n_units: int,
        edge_index: Optional[torch.LongTensor] = None,
        gat_heads: int = 4,
        gat_hidden: int = 128,
        gat_layers: int = 2,
        dropout: float = 0.1,
        mode: str = 'standard',  # 'standard' (SCEDEnv) or 'paper' (PaperSCEDEnv)
    ):
        super().__init__()
        self.n_units = n_units
        self.obs_dim = obs_dim
        self.mode = mode
        self.latent_dim_pi = 256
        self.latent_dim_vf = 256
        self._n_units = n_units

        # Determine feature dimensions based on mode
        if mode == 'paper':
            self.node_dim = self.NODE_DIM_PAPER
            self.global_dim = self.GLOBAL_DIM_PAPER
        else:
            self.node_dim = self.NODE_DIM_ED
            self.global_dim = self.GLOBAL_DIM_ED

        # Edge index
        if edge_index is None:
            src, dst = [], []
            for i in range(n_units):
                for j in range(n_units):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            edge_index = torch.LongTensor([src, dst])
        self.register_buffer('edge_index', edge_index, persistent=True)

        # --- GAT Backbone (3 layers matching original EDNetwork depth) ---
        self.gat_layers = nn.ModuleList()
        in_dim = self.node_dim
        for l in range(gat_layers):
            is_last = (l == gat_layers - 1)
            self.gat_layers.append(GATv2ConvLayer(
                in_features=in_dim,
                out_features=gat_hidden,
                heads=gat_heads,
                concat=(not is_last),
                dropout=dropout if not is_last else 0.0,
            ))
            in_dim = gat_hidden * gat_heads if (not is_last) else gat_hidden
        self.gat_output_dim = in_dim

        # --- Readout & Fusion ---
        self.global_proj = nn.Sequential(
            nn.Linear(self.global_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, self.gat_output_dim),
        )
        fusion_dim = self.gat_output_dim + self.gat_output_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.LayerNorm(256), nn.ReLU(),
        )

        # --- Actor Branch (Gaussian policy) ---
        self.policy_net = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_units),
        )
        self.log_std = nn.Parameter(torch.full((n_units,), -0.5))

        # --- Critic Branch ---
        self.value_net = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _parse_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = obs.size(0)
        node_feat = obs[:, :self.n_units * self.node_dim]
        global_feat = obs[:, self.n_units * self.node_dim:]
        node_feat = node_feat.reshape(B * self.n_units, self.node_dim)
        return node_feat, global_feat

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self._gat_forward(obs)
        return feat, feat

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self._gat_forward(obs))

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(self._gat_forward(obs))

    def _gat_forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.size(0)
        node_feat, global_feat = self._parse_obs(obs)

        edge_index_batch = self.edge_index
        if B > 1:
            offsets = torch.arange(B, device=obs.device).unsqueeze(1) * self.n_units
            edge_index_batch = torch.cat([
                self.edge_index[0:1].unsqueeze(0).expand(B, -1, -1) + offsets,
                self.edge_index[1:2].unsqueeze(0).expand(B, -1, -1) + offsets,
            ], dim=0).reshape(2, -1)

        h = node_feat
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index_batch)

        h_pooled = h.reshape(B, self.n_units, -1).mean(dim=1)
        g_proj = self.global_proj(global_feat)
        fused = torch.cat([h_pooled, g_proj], dim=-1)
        latent = self.fusion_net(fused)
        return latent

    def get_attention_map(self, obs: torch.Tensor) -> dict:
        if len(self.gat_layers) == 0:
            return {}
        node_feat, _ = self._parse_obs(obs[:1])
        attn = self.gat_layers[0].get_attention_weights(node_feat, self.edge_index)
        return {
            'edge_index': self.edge_index.cpu().numpy(),
            'attention': attn.cpu().numpy(),
            'node_labels': [f'Unit_{i}' for i in range(self.n_units)],
        }
