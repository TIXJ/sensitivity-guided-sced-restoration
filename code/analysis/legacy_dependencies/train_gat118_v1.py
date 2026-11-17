# -*- coding: utf-8 -*-
"""
train_gat118_v1.py — Train GAT Predictor for IEEE 118-bus SCED dispatch.

Adapted from train_gat_v10.py for case118 (53 generators, 118 buses).
Same GATv2Conv architecture (dimension-independent), adapted features.
"""
import sys, os, time, json, warnings, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

warnings.filterwarnings("ignore")

PROJ = Path(r".")
RESULTS = PROJ / "gatv118_results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJ))
from sced_gat_rl_pipeline import GATv2ConvLayer


# ============================================================
# 1. Feature extraction for case118
# ============================================================
def get_case118_gen_features():
    from pandapower.networks import case118
    net = case118()
    n_gen = len(net.gen)
    gen_bus = net.gen.bus.values.astype(float)
    gen_pmax = net.gen.max_p_mw.values.astype(float)
    gen_vm = net.gen.vm_pu.values.astype(float)

    cost_df = net.poly_cost[net.poly_cost["et"] == "gen"]
    cost_map = {}
    for _, row in cost_df.iterrows():
        cost_map[int(row["element"])] = {
            "cp1": float(row.get("cp1_eur_per_mw", 0)),
            "cp2": float(row.get("cp2_eur_per_mw2", 0)),
        }
    cp1_vec = np.zeros(n_gen); cp2_vec = np.zeros(n_gen)
    for i in range(n_gen):
        if i in cost_map:
            cp1_vec[i] = cost_map[i]["cp1"]
            cp2_vec[i] = cost_map[i]["cp2"]

    pmax_norm = (gen_pmax - gen_pmax.min()) / (gen_pmax.max() - gen_pmax.min() + 1e-8)
    bus_norm = (gen_bus - gen_bus.min()) / (gen_bus.max() - gen_bus.min() + 1e-8)
    vm_norm = (gen_vm - gen_vm.min()) / (gen_vm.max() - gen_vm.min() + 1e-8)

    if cp1_vec.max() > cp1_vec.min():
        cost_rank = np.argsort(np.argsort(cp1_vec)).astype(float) / max(n_gen - 1, 1)
    else:
        cost_rank = np.ones(n_gen) * 0.5

    cap_frac = gen_pmax / (gen_pmax.sum() + 1e-8)
    # Adjusted threshold for case118: >150MW (was >300MW for case300)
    is_large = (gen_pmax > 150).astype(np.float32)
    total_cap = float(gen_pmax.sum())
    base_load = float(net.load.p_mw.sum())

    print(f"Case118 Gen Features:")
    print(f"  N={n_gen}, Total Pmax={total_cap:.0f} MW, BaseLoad={base_load:.0f} MW")
    print(f"  Pmax range: {gen_pmax.min():.0f}-{gen_pmax.max():.0f} MW")
    print(f"  Large gens(>150MW): {int(is_large.sum())}/{n_gen}")

    return {
        "n_gen": n_gen, "total_cap": total_cap, "base_load": base_load,
        "gen_bus": gen_bus, "gen_pmax": gen_pmax, "gen_vm": gen_vm,
        "pmax_norm": pmax_norm, "bus_norm": bus_norm, "vm_norm": vm_norm,
        "cost_rank": cost_rank, "cap_frac": cap_frac, "is_large": is_large,
        "cp1_vec": cp1_vec, "cp2_vec": cp2_vec,
    }


# ============================================================
# 2. Observation Builder for case118
# ============================================================
class ObsBuilder118:
    NODE_STATIC_DIM = 6  # pmax_norm, bus_norm, vm_norm, cost_rank, cap_frac, is_large
    NODE_DIM = NODE_STATIC_DIM
    GLOBAL_DIM = 8

    def __init__(self, gen_feats):
        self.N = gen_feats["n_gen"]
        self.total_cap = gen_feats["total_cap"]
        self.base_load = gen_feats["base_load"]
        self.static_per_unit = np.column_stack([
            gen_feats["pmax_norm"], gen_feats["bus_norm"], gen_feats["vm_norm"],
            gen_feats["cost_rank"], gen_feats["cap_frac"], gen_feats["is_large"],
        ]).astype(np.float32)

    def build(self, demand_mw):
        node_feat = self.static_per_unit.copy()
        d_norm = demand_mw / self.total_cap
        ls = demand_mw / self.base_load
        reserve = max(0.0, 1.0 - d_norm)
        log_d = np.log(demand_mw + 1e-3) / np.log(self.total_cap + 1e-3)
        d_sq = d_norm ** 2
        # Demand tier one-hot (adapted for case118: ~3.0-5.0 GW)
        tier_low = 1.0 if demand_mw < 3500 else 0.0
        tier_mid = 1.0 if 3500 <= demand_mw < 4300 else 0.0
        tier_high = 1.0 if demand_mw >= 4300 else 0.0
        global_feat = np.array([
            d_norm, ls, reserve, log_d, d_sq,
            tier_low, tier_mid, tier_high,
        ], dtype=np.float32)
        return node_feat, global_feat


# ============================================================
# 3. GATv118 Predictor (same architecture, parameterized by n_units)
# ============================================================
class GATv118Predictor(nn.Module):
    NODE_DIM = 6; GLOBAL_DIM = 8

    def __init__(self, n_units, edge_index, gat_heads=4, gat_hidden=128):
        super().__init__()
        self.n_units = n_units
        if edge_index is None:
            src, dst = [], []
            for i in range(n_units):
                for j in range(n_units):
                    if i != j: src.append(i); dst.append(j)
            edge_index = torch.LongTensor([src, dst])
        self.register_buffer('edge_index', edge_index)
        self.gat1 = GATv2ConvLayer(self.NODE_DIM, gat_hidden, heads=gat_heads, concat=True)
        self.gat2 = GATv2ConvLayer(gat_hidden * gat_heads, gat_hidden, heads=gat_heads, concat=False)
        gat_out_dim = gat_hidden
        self.global_proj = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM, 128), nn.ReLU(), nn.Linear(128, gat_out_dim))
        self.fusion = nn.Sequential(
            nn.Linear(gat_out_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_units))

    def forward(self, node_feat, global_feat):
        B = global_feat.size(0); N = self.n_units
        if node_feat.dim() == 3: node = node_feat.reshape(B * N, -1)
        else: node = node_feat
        ei = self.edge_index
        if B > 1:
            offsets = torch.arange(B, device=global_feat.device).unsqueeze(1) * N
            ei = torch.cat([ei[0:1].expand(B, -1) + offsets,
                           ei[1:2].expand(B, -1) + offsets], dim=0).reshape(2, -1)
        h = F.relu(self.gat1(node, ei))
        h = self.gat2(h, ei)
        h_pool = h.reshape(B, N, -1).mean(dim=1)
        g_proj = self.global_proj(global_feat)
        return self.fusion(torch.cat([h_pool, g_proj], dim=-1))


# ============================================================
# 4. Dataset for case118 GT
# ============================================================
class SCED118Dataset(Dataset):
    def __init__(self, json_path, gen_feats, split="train",
                 val_ratio=0.15, test_ratio=0.15, seed=42):
        with open(json_path, "r") as f: data = json.load(f)
        all_samples = data["samples"]
        samples = [s for s in all_samples if s.get("is_complete_sced")]
        print(f"  [Filter] {len(samples)}/{len(all_samples)} complete SCED samples")

        np.random.seed(seed); indices = np.arange(len(samples)); np.random.shuffle(indices)
        n_total = len(samples); n_val = int(n_total * val_ratio); n_test = int(n_total * test_ratio)
        if split == "train": self.indices = indices[:n_total - n_val - n_test]
        elif split == "val": self.indices = indices[n_total - n_val - n_test:n_total - n_test]
        elif split == "test": self.indices = indices[n_total - n_test:]
        self.samples = [samples[i] for i in self.indices]
        self.gen_feats = gen_feats; self.N = gen_feats["n_gen"]
        self.obs_builder = ObsBuilder118(gen_feats)
        print(f"  [{split}] {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        dispatch = np.array(s["dispatch"], dtype=np.float32)
        node_feat, global_feat = self.obs_builder.build(s["demand_mw"])
        return (torch.from_numpy(node_feat), torch.from_numpy(global_feat),
                torch.from_numpy(dispatch), s["demand_mw"])

def collate_fn(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.tensor([b[3] for b in batch], dtype=torch.float32))


# ============================================================
# 5. Training
# ============================================================
def train_gat118():
    print("=" * 70)
    print("GATv118 Training: IEEE 118-bus Complete SCED GT")
    print("=" * 70)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_feats = get_case118_gen_features(); N = gen_feats["n_gen"]
    json_path = PROJ / "case118_routeb_rescuable_dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found. Run paper_results/code/screen_route_b_smallcase_rescuable.py --case 118 first."
        )
    train_ds = SCED118Dataset(json_path, gen_feats, split="train")
    val_ds = SCED118Dataset(json_path, gen_feats, split="val")
    test_ds = SCED118Dataset(json_path, gen_feats, split="test")

    BATCH_SIZE = 16
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j: src.append(i); dst.append(j)
    edge_index = torch.LongTensor([src, dst])

    model = GATv118Predictor(n_units=N, edge_index=edge_index).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    mse_loss = nn.MSELoss()

    MAX_EPOCHS = 300; PATIENCE = 40; best_val_mae = float('inf'); patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for nodes, globs, labels, _ in train_loader:
            nodes, globs, labels = nodes.to(device), globs.to(device), labels.to(device)
            pred = model(nodes, globs); loss = mse_loss(pred, labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

        model.eval(); all_pred, all_label = [], []
        with torch.no_grad():
            for nodes, globs, labels, _ in val_loader:
                nodes, globs = nodes.to(device), globs.to(device)
                all_pred.append(model(nodes, globs).cpu().numpy())
                all_label.append(labels.cpu().numpy())
        ap = np.concatenate(all_pred); al = np.concatenate(all_label)
        val_mae = np.mean(np.abs(ap - al))
        corrs = [np.corrcoef(ap[i], al[i])[0,1] for i in range(len(ap))]
        val_corr = np.mean([c for c in corrs if not np.isnan(c)])

        scheduler.step(val_mae)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: Val_MAE={val_mae:.2f}, Val_Corr={val_corr:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae; patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_mae": val_mae, "val_corr": val_corr,
                        "config": {"n_units": N, "n_gen": N, "gat_heads": 4, "gat_hidden": 128,
                                   "total_cap": gen_feats["total_cap"], "base_load": gen_feats["base_load"]}},
                       str(RESULTS / "gatv118_predictor_best.pth"))
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}"); break

    # Test eval
    best_sd = torch.load(str(RESULTS / "gatv118_predictor_best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(best_sd["model_state_dict"]); model.eval()
    fp_all, fl_all = [], []
    with torch.no_grad():
        for nodes, globs, labels, _ in test_loader:
            nodes, globs = nodes.to(device), globs.to(device)
            fp_all.append(model(nodes, globs).cpu().numpy()); fl_all.append(labels.cpu().numpy())
    fp = np.concatenate(fp_all); fl = np.concatenate(fl_all)
    test_mae = np.mean(np.abs(fp - fl))
    test_corr = np.mean([np.corrcoef(fp[i], fl[i])[0,1] for i in range(len(fp))
                         if not np.isnan(np.corrcoef(fp[i], fl[i])[0,1])])

    print(f"\n{'='*60}")
    print(f"  GATv118 Final: Test MAE={test_mae:.2f} MW/gen, Corr={test_corr:.4f}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_gat118()
