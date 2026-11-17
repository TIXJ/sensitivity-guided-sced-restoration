# -*- coding: utf-8 -*-
"""
GATv10 Trainer: Train GAT predictor on Complete SCED GT (N-1 secure + AC-PF valid).

Key differences from v8:
  - Data source: case300_gt_v10_dataset.json (260 complete SCED samples)
    vs v8's 183 OPF samples (missing N-1)
  - GT labels from Gurobi QP Economic Dispatch + N-1 correction
    — true SCED optimal with full security constraints
  - Filters to is_complete_sced=True only (260/270 samples)
  - Additional global features: N-1 correction iterations, cost context
"""
import sys, os, time, json, copy, warnings, argparse, numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

CODE_DIR = Path(__file__).resolve().parent
PROJ = CODE_DIR.parents[1]
RESULTS = PROJ / "gatv10_results"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJ))
from sced_gat_rl_pipeline import GATv2ConvLayer


# ============================================================
# 1. Load case300 network parameters for feature engineering
# ============================================================
def get_case300_gen_features():
    """Extract generator features from pandapower case300."""
    from pandapower.networks import case300
    net = case300()

    n_gen = len(net.gen)
    gen_bus = net.gen.bus.values.astype(float)
    gen_pmax = net.gen.max_p_mw.values.astype(float)
    gen_vm = net.gen.vm_pu.values.astype(float)

    # Cost features
    cost_df = net.poly_cost[net.poly_cost["et"] == "gen"]
    cost_map = {}
    for _, row in cost_df.iterrows():
        cost_map[int(row["element"])] = {
            "cp1": float(row.get("cp1_eur_per_mw", 0)),
            "cp2": float(row.get("cp2_eur_per_mw2", 0)),
        }

    cp1_vec = np.zeros(n_gen)
    cp2_vec = np.zeros(n_gen)
    for i in range(n_gen):
        if i in cost_map:
            cp1_vec[i] = cost_map[i]["cp1"]
            cp2_vec[i] = cost_map[i]["cp2"]

    # Normalize features
    pmax_norm = (gen_pmax - gen_pmax.min()) / (gen_pmax.max() - gen_pmax.min() + 1e-8)
    bus_norm = (gen_bus - gen_bus.min()) / (gen_bus.max() - gen_bus.min() + 1e-8)
    vm_norm = (gen_vm - gen_vm.min()) / (gen_vm.max() - gen_vm.min() + 1e-8)

    # Cost rank (lower cost = better = higher priority in merit order)
    if cp1_vec.max() > cp1_vec.min():
        cost_rank = np.argsort(np.argsort(cp1_vec)).astype(float) / max(n_gen - 1, 1)
    else:
        cost_rank = np.ones(n_gen) * 0.5

    # Capacity fraction
    cap_frac = gen_pmax / (gen_pmax.sum() + 1e-8)

    # Large generator flag (> 300 MW)
    is_large = (gen_pmax > 300).astype(np.float32)

    total_cap = float(gen_pmax.sum())
    base_load = float(net.load.p_mw.sum())

    print(f"Case300 Gen Features:")
    print(f"  N={n_gen}, Total Pmax={total_cap:.0f} MW, BaseLoad={base_load:.0f} MW")
    print(f"  Pmax range: {gen_pmax.min():.0f}-{gen_pmax.max():.0f} MW")
    print(f"  Cost cp1 range: {cp1_vec.min():.2f}-{cp1_vec.max():.2f}")
    print(f"  Large gens(>300MW): {int(is_large.sum())}/{n_gen}")

    return {
        "n_gen": n_gen,
        "total_cap": total_cap,
        "base_load": base_load,
        "gen_bus": gen_bus,
        "gen_pmax": gen_pmax,
        "gen_vm": gen_vm,
        "pmax_norm": pmax_norm,
        "bus_norm": bus_norm,
        "vm_norm": vm_norm,
        "cost_rank": cost_rank,
        "cap_frac": cap_frac,
        "is_large": is_large,
        "cp1_vec": cp1_vec,
        "cp2_vec": cp2_vec,
    }


# ============================================================
# 2. Observation Builder for v10 (same as v8 + optional N-1 hints)
# ============================================================
class ObservationBuilderV10:
    """
    Build node+global features for GATv10.
    Same static features as v8; v10 adds N-1 correction context.
    """

    # Per-node: 6 static features + 2 per-sample cost features.
    NODE_STATIC_DIM = 6
    NODE_DIM = 8

    # Global: demand context + N-1 hint
    GLOBAL_DIM = 8  # same as v8: demand_norm, load_scale, reserve_ratio, log_demand,
                    # demand_sq, tier_16, tier_20, tier_24

    def __init__(self, gen_feats):
        self.gen_feats = gen_feats
        self.N = gen_feats["n_gen"]
        self.total_cap = gen_feats["total_cap"]
        self.base_load = gen_feats["base_load"]

        # Static per-generator features (N, 6) — never changes
        self.static_per_unit = np.column_stack([
            gen_feats["pmax_norm"],
            gen_feats["bus_norm"],
            gen_feats["vm_norm"],
            gen_feats["cost_rank"],
            gen_feats["cap_frac"],
            gen_feats["is_large"],
        ]).astype(np.float32)
        assert self.static_per_unit.shape == (self.N, self.NODE_STATIC_DIM)

    def get_obs_dim(self):
        return self.N * self.NODE_DIM + self.GLOBAL_DIM

    def build(self, demand_mw, sample=None):
        """Build observation for a given demand level."""
        if sample is not None and "sample_cp1" in sample and "sample_cp2" in sample:
            cp1 = np.asarray(sample["sample_cp1"], dtype=np.float32)
            cp2 = np.asarray(sample["sample_cp2"], dtype=np.float32)
        else:
            cp1 = np.asarray(self.gen_feats["cp1_vec"], dtype=np.float32)
            cp2 = np.asarray(self.gen_feats["cp2_vec"], dtype=np.float32)
        cp1_norm = cp1 / (float(np.mean(cp1)) + 1e-8)
        cp2_norm = cp2 / (float(np.mean(cp2)) + 1e-8)
        node_feat = np.column_stack([
            self.static_per_unit,
            cp1_norm,
            cp2_norm,
        ]).astype(np.float32)

        # Global features: demand context
        d_norm = demand_mw / self.total_cap
        ls = demand_mw / self.base_load
        reserve = max(0.0, 1.0 - d_norm)
        log_d = np.log(demand_mw + 1e-3) / np.log(self.total_cap + 1e-3)
        d_sq = d_norm ** 2

        # Demand tier one-hot
        tier16 = 1.0 if 15.0 <= demand_mw < 18.0 else 0.0
        tier20 = 1.0 if 19.0 <= demand_mw < 22.0 else 0.0
        tier24 = 1.0 if 23.0 <= demand_mw <= 26.0 else 0.0

        global_feat = np.array([
            d_norm, ls, reserve, log_d, d_sq,
            tier16, tier20, tier24,
        ], dtype=np.float32)

        info = {"demand": demand_mw, "n_on": self.N}
        return node_feat, global_feat, info


# ============================================================
# 3. GATv10 Predictor Model (identical architecture to v8)
# ============================================================
class GATv10Predictor(nn.Module):
    """GAT Predictor for 68-gen case300 SCED dispatch prediction."""

    NODE_DIM_V10 = 8
    GLOBAL_DIM_V10 = 8  # demand context

    def __init__(self, n_units, edge_index, gat_heads=4, gat_hidden=128):
        super().__init__()
        self.n_units = n_units

        if edge_index is None:
            src, dst = [], []
            for i in range(n_units):
                for j in range(n_units):
                    if i != j:
                        src.append(i); dst.append(j)
            edge_index = torch.LongTensor([src, dst])
        self.register_buffer('edge_index', edge_index)

        # GAT backbone (2-layer, same as v8)
        self.gat1 = GATv2ConvLayer(self.NODE_DIM_V10, gat_hidden, heads=gat_heads, concat=True)
        self.gat2 = GATv2ConvLayer(gat_hidden * gat_heads, gat_hidden, heads=gat_heads, concat=False)
        gat_out_dim = gat_hidden

        # Global feature projector
        self.global_proj = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM_V10, 128), nn.ReLU(),
            nn.Linear(128, gat_out_dim),
        )

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(gat_out_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_units),
        )

    def forward(self, node_feat, global_feat):
        B = global_feat.size(0)
        N = self.n_units

        if node_feat.dim() == 3:
            node = node_feat.reshape(B * N, -1)
        else:
            node = node_feat

        ei = self.edge_index
        if B > 1:
            offsets = torch.arange(B, device=global_feat.device).unsqueeze(1) * N
            ei = torch.cat([
                ei[0:1].expand(B, -1) + offsets,
                ei[1:2].expand(B, -1) + offsets,
            ], dim=0).reshape(2, -1)

        h = self.gat1(node, ei)
        h = F.relu(h)
        h = self.gat2(h, ei)

        h_pool = h.reshape(B, N, -1).mean(dim=1)
        g_proj = self.global_proj(global_feat)
        fused = torch.cat([h_pool, g_proj], dim=-1)

        return self.fusion(fused)


# ============================================================
# 4. Dataset: load from v10 JSON (complete SCED only!)
# ============================================================
class SCEDv10Dataset(Dataset):
    """Dataset from case300_gt_v10_dataset.json, filtered to complete SCED only."""

    def __init__(self, json_path, gen_feats, split="train",
                 val_ratio=0.15, test_ratio=0.15, seed=42,
                 require_complete_sced=True, split_ids=None):
        with open(json_path, "r") as f:
            data = json.load(f)

        # Filter to complete SCED samples only
        all_samples = data["samples"]
        if require_complete_sced:
            samples = [s for s in all_samples if s.get("is_complete_sced", False)]
            n_filtered = len(all_samples) - len(samples)
            print(f"  [Filter] {len(samples)}/{len(all_samples)} samples "
                  f"are complete SCED ({n_filtered} excluded)")
        else:
            samples = all_samples

        if split_ids is not None:
            key = {"train": "train_ids", "val": "val_ids", "test": "test_ids"}.get(split)
            if key is None:
                raise ValueError(f"Unknown split: {split}")
            self.indices = np.asarray(split_ids[key], dtype=int)
        else:
            np.random.seed(seed)
            indices = np.arange(len(samples))
            np.random.shuffle(indices)

            n_total = len(samples)
            n_val = int(n_total * val_ratio)
            n_test = int(n_total * test_ratio)
            n_train = n_total - n_val - n_test

            if split == "train":
                self.indices = indices[:n_train]
            elif split == "val":
                self.indices = indices[n_train:n_train + n_val]
            elif split == "test":
                self.indices = indices[n_train + n_val:]
            else:
                raise ValueError(f"Unknown split: {split}")

        self.samples = [samples[i] for i in self.indices]
        self.gen_feats = gen_feats
        self.N = gen_feats["n_gen"]
        self.obs_builder = ObservationBuilderV10(gen_feats)

        print(f"  [{split}] {len(self.samples)} samples (demand range: "
              f"{min(s['demand_mw'] for s in self.samples):.0f}-"
              f"{max(s['demand_mw'] for s in self.samples):.0f} GW)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        dispatch = np.array(sample["dispatch_68"], dtype=np.float32)
        demand = sample["demand_mw"]

        node_feat, global_feat, _info = self.obs_builder.build(demand, sample=sample)

        return (
            torch.from_numpy(node_feat),     # (N, 6)
            torch.from_numpy(global_feat),    # (8,)
            torch.from_numpy(dispatch),       # (N,) label
            demand,                          # scalar metadata
        )


def collate_fn(batch):
    nodes = torch.stack([b[0] for b in batch])
    globs = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    demands = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return nodes, globs, labels, demands


# ============================================================
# 5. Training Loop
# ============================================================
def train_gatv10(args=None):
    args = args or argparse.Namespace(
        split_file=None,
        output_dir=str(RESULTS),
        dataset=str(PROJ / "case300_gt_v10_dataset.json"),
    )
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)
    print("=" * 70)
    print("GATv10 Training: Complete SCED GT (N-1 Secure) on case300")
    print("=" * 70)
    t0 = time.time()

    # --- Network features ---
    gen_feats = get_case300_gen_features()
    N = gen_feats["n_gen"]
    obs_builder = ObservationBuilderV10(gen_feats)

    # Test observation builder
    nf, gf, info = obs_builder.build(20000.0)
    print(f"\nTest obs: node_feat={nf.shape}, global_feat={gf.shape}, D={info['demand']:.0f}")
    assert nf.shape == (N, 8), f"Expected ({N}, 8), got {nf.shape}"
    assert gf.shape == (8,), f"Expected (8,), got {gf.shape}"

    # --- Load dataset ---
    json_path = Path(args.dataset)
    split_ids = None
    if args.split_file:
        with open(args.split_file, "r", encoding="utf-8") as f:
            split_ids = json.load(f)
        print(f"Using master split: {args.split_file}")

    train_ds = SCEDv10Dataset(json_path, gen_feats, split="train",
                               require_complete_sced=True, split_ids=split_ids)
    val_ds = SCEDv10Dataset(json_path, gen_feats, split="val",
                             require_complete_sced=True, split_ids=split_ids)
    test_ds = SCEDv10Dataset(json_path, gen_feats, split="test",
                              require_complete_sced=True, split_ids=split_ids)

    BATCH_SIZE = 16
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    print(f"\nDatasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Total complete SCED samples: {len(train_ds)+len(val_ds)+len(test_ds)}")
    print(f"Batch size: {BATCH_SIZE}, Steps/epoch: {len(train_loader)}")

    # --- Edge index (fully connected for N=68) ---
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i); dst.append(j)
    edge_index = torch.LongTensor([src, dst])
    print(f"Edge index: fully-connected, {edge_index.shape[1]} edges")

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATv10Predictor(n_units=N, edge_index=edge_index, gat_heads=4, gat_hidden=128)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,} on {device}")

    # --- Optimizer & Loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)
    mse_loss = nn.MSELoss()

    # --- Training config ---
    MAX_EPOCHS = 300
    PATIENCE = 40
    best_val_mae = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_mae": [], "val_corr": [], "lr": []}

    print(f"\nTraining: {MAX_EPOCHS} epochs max, batch={BATCH_SIZE}, patience={PATIENCE}")
    print("-" * 70)

    for epoch in range(MAX_EPOCHS):
        # --- Train ---
        model.train()
        train_losses = []
        for nodes, globs, labels, _demands in train_loader:
            nodes = nodes.to(device)
            globs = globs.to(device)
            labels = labels.to(device)

            pred = model(nodes, globs)
            loss = mse_loss(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        all_pred, all_label = [], []
        with torch.no_grad():
            for nodes, globs, labels, _demands in val_loader:
                nodes = nodes.to(device)
                globs = globs.to(device)
                pred = model(nodes, globs)
                all_pred.append(pred.cpu().numpy())
                all_label.append(labels.cpu().numpy())

        all_pred = np.concatenate(all_pred)
        all_label = np.concatenate(all_label)
        val_mae = np.mean(np.abs(all_pred - all_label))

        corrs = []
        for i in range(len(all_pred)):
            c = np.corrcoef(all_pred[i], all_label[i])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
        val_corr = np.mean(corrs) if corrs else 0.0

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_mae)

        history["train_loss"].append(avg_train_loss)
        history["val_mae"].append(val_mae)
        history["val_corr"].append(val_corr)
        history["lr"].append(current_lr)

        # --- Logging ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{MAX_EPOCHS}: "
                  f"Loss={avg_train_loss:.4f}, "
                  f"Val_MAE={val_mae:.1f}, Val_Corr={val_corr:.3f}, lr={current_lr:.6f}")

        # --- Best model checkpoint ---
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            save_path = results_dir / "gatv10_predictor_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "val_corr": val_corr,
                "config": {
                    "node_dim": GATv10Predictor.NODE_DIM_V10,
                    "global_dim": GATv10Predictor.GLOBAL_DIM_V10,
                    "n_units": N,
                    "n_gen": N,
                    "gat_heads": 4,
                    "gat_hidden": 128,
                    "total_cap": gen_feats["total_cap"],
                    "base_load": gen_feats["base_load"],
                    "gen_pmax": gen_feats["gen_pmax"].tolist(),
                    "gen_bus": gen_feats["gen_bus"].tolist(),
                    "data_source": str(json_path.resolve()),
                    "split_file": args.split_file,
                    "n_complete_sced": len(train_ds) + len(val_ds) + len(test_ds),
                },
            }, str(save_path))
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1} (patience={PATIENCE})")
            break

    # --- Final evaluation on TEST set ---
    print("\n" + "=" * 70)
    print("Training Complete! Loading best model for final eval...")
    print("=" * 70)

    sd_best = torch.load(str(save_path), map_location=device, weights_only=False)
    model.load_state_dict(sd_best["model_state_dict"])
    model.eval()

    # Eval on test set
    final_preds, final_labels, final_demands = [], [], []
    with torch.no_grad():
        for nodes, globs, labels, demands in test_loader:
            nodes, globs = nodes.to(device), globs.to(device)
            pred = model(nodes, globs)
            final_preds.append(pred.cpu().numpy())
            final_labels.append(labels.cpu().numpy())
            final_demands.append(demands.numpy())

    fp = np.concatenate(final_preds)
    fl = np.concatenate(final_labels)
    fd = np.concatenate(final_demands)

    mae_per_sample = np.mean(np.abs(fp - fl), axis=1)
    corr_per_sample = [np.corrcoef(fp[i], fl[i])[0, 1] for i in range(len(fp))]
    corr_per_sample = [c for c in corr_per_sample if not np.isnan(c)]

    # HHI computation
    hhi_gt = []
    hhi_pred = []
    for i in range(len(fl)):
        gt_sum = fl[i].sum() + 1e-8
        pr_sum = fp[i].sum() + 1e-8
        hhi_gt.append(((fl[i] / gt_sum) ** 2).sum() * 10000)
        hhi_pred.append(((fp[i] / pr_sum) ** 2).sum() * 10000)

    # Per-demand-level analysis
    level_stats = {}
    for d_level in sorted(np.unique(fd)):
        mask = np.isclose(fd, d_level)
        if mask.sum() > 0:
            mae_d = np.mean(np.abs(fp[mask] - fl[mask]))
            corr_d_list = [np.corrcoef(fp[mask][i], fl[mask][i])[0, 1]
                           for i in range(mask.sum())]
            corr_d = np.mean([c for c in corr_d_list if not np.isnan(c)])
            level_stats[f"{float(d_level) / 1000.0:g}GW"] = {
                "n": int(mask.sum()),
                "MAE": round(float(mae_d), 2),
                "Corr": round(float(corr_d) if not np.isnan(corr_d) else 0, 4),
            }

    results = {
        "best_epoch": sd_best["epoch"] + 1,
        "data_source": str(json_path.resolve()),
        "n_samples_total": len(train_ds) + len(val_ds) + len(test_ds),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "MAE_mean": round(float(np.mean(mae_per_sample)), 2),
        "MAE_std": round(float(np.std(mae_per_sample)), 2),
        "MAE_median": round(float(np.median(mae_per_sample)), 2),
        "Corr_mean": round(float(np.mean(corr_per_sample)), 4),
        "Corr_std": round(float(np.std(corr_per_sample)), 4),
        "Corr_median": round(float(np.median(corr_per_sample)), 4),
        "HHI_GT_mean": round(float(np.mean(hhi_gt)), 1),
        "HHI_GAT_mean": round(float(np.mean(hhi_pred)), 1),
        "target_MAE_achieved": bool(np.mean(mae_per_sample) < 50),
        "target_Corr_achieved": bool(np.mean(corr_per_sample) > 0.85),
        "per_demand_level": level_stats,
        "history": {k: [round(v, 6) if isinstance(v, float) else v for v in vals]
                     for k, vals in history.items()},
        "training_time_s": round(time.time() - t0, 1),
    }

    summary = (
        f"\n{'='*60}\n"
        f"  GATv10 Training Results Summary\n"
        f"  (Complete SCED GT with N-1 Security)\n"
        f"{'='*60}\n"
        f"  Best epoch: {results['best_epoch']}\n"
        f"  Train/Val/Test: {results['n_train']} / {results['n_val']} / {results['n_test']}\n\n"
        f"  MAE (MW/gen): mean={results['MAE_mean']}, std={results['MAE_std']}, median={results['MAE_median']}\n"
        f"  Correlation:   mean={results['Corr_mean']}, std={results['Corr_std']}, median={results['Corr_median']}\n\n"
        f"  HHI_GT:  mean={results['HHI_GT_mean']:.1f}\n"
        f"  HHI_GAT: mean={results['HHI_GAT_mean']:.1f}\n\n"
        f"  Target MAE<50:   {'PASS' if results['target_MAE_achieved'] else 'FAIL'}\n"
        f"  Target Corr>0.85: {'PASS' if results['target_Corr_achieved'] else 'FAIL'}\n\n"
        f"  Per-demand-level:\n"
    )
    for lvl, ls in level_stats.items():
        summary += f"    {lvl}: n={ls['n']}, MAE={ls['MAE']}, Corr={ls['Corr']}\n"
    summary += (
        f"  Training time: {results['training_time_s']:.0f}s\n"
        f"{'='*60}\n"
    )
    print(summary)

    res_path = results_dir / "gatv10_training_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f"Results saved: {res_path}")

    hist_path = results_dir / "gatv10_training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=float)

    return results, model, obs_builder, gen_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(PROJ / "case300_gt_v10_dataset_lodf_corrected_277.json"),
        help="Corrected linear-SCED dataset used for GAT supervision.",
    )
    parser.add_argument("--split-file", default=None,
                        help="Shared master split JSON with train_ids/val_ids/test_ids.")
    parser.add_argument("--output-dir", default=str(RESULTS),
                        help="Directory for gatv10_predictor_best.pth and training outputs.")
    args = parser.parse_args()
    results, model, obs_builder, gen_feats = train_gatv10(args)
