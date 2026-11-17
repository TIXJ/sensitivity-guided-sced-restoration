"""GAT predictor for IEEE 39-bus (9 generators)."""
import sys, os, time, json, warnings, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader; from pathlib import Path
warnings.filterwarnings("ignore")
PROJ = Path(r"."); sys.path.insert(0, str(PROJ))
from sced_gat_rl_pipeline import GATv2ConvLayer
RESULTS = PROJ / "gat39_results"; RESULTS.mkdir(exist_ok=True)

def get_features():
    from pandapower.networks import case39; net = case39()
    n = len(net.gen); pmax = net.gen.max_p_mw.values.astype(float); pmin = net.gen.min_p_mw.values.astype(float)
    bus = net.gen.bus.values.astype(float); vm = net.gen.vm_pu.values.astype(float)
    cost_df = net.poly_cost[net.poly_cost["et"] == "gen"]
    cp1 = np.zeros(n)
    for _, row in cost_df.iterrows():
        g = int(row["element"])
        if 0 <= g < n: cp1[g] = float(row.get("cp1_eur_per_mw", 30))
    pmax_n = (pmax-pmax.min())/(pmax.max()-pmax.min()+1e-8)
    bus_n = (bus-bus.min())/(bus.max()-bus.min()+1e-8)
    vm_n = (vm-vm.min())/(vm.max()-vm.min()+1e-8)
    cr = np.argsort(np.argsort(cp1)).astype(float)/max(n-1,1) if cp1.max()>cp1.min() else np.ones(n)*0.5
    cf = pmax/(pmax.sum()+1e-8); il = (pmax>300).astype(np.float32)
    tc = float(pmax.sum()); bl = float(net.load.p_mw.sum())
    print(f"case39: N={n}, cap={tc:.0f}MW, load={bl:.0f}MW, large={int(il.sum())}")
    return {"n_gen":n,"total_cap":tc,"base_load":bl,"pmax_norm":pmax_n,"bus_norm":bus_n,"vm_norm":vm_n,
            "cost_rank":cr,"cap_frac":cf,"is_large":il,"cp1_vec":cp1,"gen_pmax":pmax,"gen_bus":bus}

class ObsBuilder:
    NODE_DIM = 6; GLOBAL_DIM = 8
    def __init__(self, gf):
        self.N = gf["n_gen"]; self.tc = gf["total_cap"]; self.bl = gf["base_load"]
        self.static = np.column_stack([gf["pmax_norm"],gf["bus_norm"],gf["vm_norm"],gf["cost_rank"],gf["cap_frac"],gf["is_large"]]).astype(np.float32)
    def build(self, dmw):
        nf = self.static.copy(); dn = dmw/self.tc; ls = dmw/self.bl; rsv = max(0.0,1-dn)
        ld = np.log(dmw+1e-3)/np.log(self.tc+1e-3); ds = dn**2
        t_low = 1.0 if dmw<6500 else 0.0; t_mid = 1.0 if 6500<=dmw<7000 else 0.0; t_hi = 1.0 if dmw>=7000 else 0.0
        gf_arr = np.array([dn,ls,rsv,ld,ds,t_low,t_mid,t_hi], dtype=np.float32)
        return nf, gf_arr

class GATPredictor(nn.Module):
    ND=6; GD=8
    def __init__(self, n_units, edge_index, heads=4, hidden=128):
        super().__init__(); self.nu = n_units
        if edge_index is None:
            src=[]; dst=[]
            for i in range(n_units):
                for j in range(n_units):
                    if i!=j: src.append(i); dst.append(j)
            edge_index = torch.LongTensor([src,dst])
        self.register_buffer('ei', edge_index)
        self.g1 = GATv2ConvLayer(self.ND, hidden, heads=heads, concat=True)
        self.g2 = GATv2ConvLayer(hidden*heads, hidden, heads=heads, concat=False)
        self.gp = nn.Sequential(nn.Linear(self.GD,128),nn.ReLU(),nn.Linear(128,hidden))
        self.fusion = nn.Sequential(nn.Linear(hidden*2,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,n_units))
    def forward(self, nf, gf):
        B=gf.size(0); N=self.nu
        node = nf.reshape(B*N,-1) if nf.dim()==3 else nf
        ei = self.ei
        if B>1:
            off = torch.arange(B,device=gf.device).unsqueeze(1)*N
            ei = torch.cat([ei[0:1].expand(B,-1)+off, ei[1:2].expand(B,-1)+off],dim=0).reshape(2,-1)
        h = F.relu(self.g1(node,ei)); h = self.g2(h,ei)
        hp = h.reshape(B,N,-1).mean(dim=1); gp = self.gp(gf)
        return self.fusion(torch.cat([hp,gp],dim=-1))

class SCDataset(Dataset):
    def __init__(self, jp, gf, split="train", vr=0.15, tr=0.15, seed=42):
        with open(jp) as f: data = json.load(f)
        samples = data["samples"]
        n_total = len(samples); n_val = int(n_total*vr); n_test = int(n_total*tr)
        np.random.seed(seed); idx = np.arange(n_total); np.random.shuffle(idx)
        if split=="train": self.idx = idx[:n_total-n_val-n_test]
        elif split=="val": self.idx = idx[n_total-n_val-n_test:n_total-n_test]
        else: self.idx = idx[n_total-n_test:]
        self.samples = [samples[i] for i in self.idx]; self.N = gf["n_gen"]
        self.ob = ObsBuilder(gf)
        print(f"  [{split}] {len(self.samples)} samples")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]; disp = np.array(s["dispatch"], dtype=np.float32)
        nf, gf = self.ob.build(s["demand_mw"])
        return torch.from_numpy(nf), torch.from_numpy(gf), torch.from_numpy(disp), s["demand_mw"]

def collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), torch.tensor([b[3] for b in batch], dtype=torch.float32))

def train():
    print("="*60); print("GAT39 Training"); print("="*60)
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gf = get_features(); N = gf["n_gen"]
    jp = PROJ / "case39_routeb_rescuable_dataset.json"
    if not jp.exists():
        raise FileNotFoundError(
            f"{jp} not found. Run paper_results/code/screen_route_b_smallcase_rescuable.py --case 39 first."
        )
    tr_ds = SCDataset(jp, gf, "train"); vl_ds = SCDataset(jp, gf, "val"); ts_ds = SCDataset(jp, gf, "test")
    B = 16
    tr_ld = DataLoader(tr_ds,B,shuffle=True,collate_fn=collate,drop_last=True)
    vl_ld = DataLoader(vl_ds,B,shuffle=False,collate_fn=collate)
    ts_ld = DataLoader(ts_ds,B,shuffle=False,collate_fn=collate)
    src=[]; dst=[]
    for i in range(N):
        for j in range(N):
            if i!=j: src.append(i); dst.append(j)
    ei = torch.LongTensor([src,dst])
    model = GATPredictor(n_units=N, edge_index=ei).to(dev)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=15)
    mse = nn.MSELoss(); best_mae = float('inf'); patience = 0; MAXE = 300; PAT = 40
    for ep in range(MAXE):
        model.train()
        for nf, gf, lb, _ in tr_ld:
            nf, gf, lb = nf.to(dev), gf.to(dev), lb.to(dev)
            pred = model(nf, gf); loss = mse(pred, lb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); ap=[]; al=[]
        with torch.no_grad():
            for nf, gf, lb, _ in vl_ld:
                nf, gf = nf.to(dev), gf.to(dev); ap.append(model(nf,gf).cpu().numpy()); al.append(lb.cpu().numpy())
        ap=np.concatenate(ap); al=np.concatenate(al); vmae=np.mean(np.abs(ap-al))
        sch.step(vmae)
        if (ep+1)%20==0 or ep==0: print(f"  E{ep+1:3d}: Val_MAE={vmae:.2f}")
        if vmae < best_mae:
            best_mae=vmae; patience=0
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),"val_mae":vmae,"config":{"n_units":N}},
                       str(RESULTS/"gat39_best.pth"))
        else: patience+=1
        if patience>=PAT: print(f"  Early stop at {ep+1}"); break
    sd = torch.load(str(RESULTS/"gat39_best.pth"), map_location=dev, weights_only=False)
    model.load_state_dict(sd["model_state_dict"]); model.eval()
    fp=[]; fl=[]
    with torch.no_grad():
        for nf, gf, lb, _ in ts_ld:
            nf, gf = nf.to(dev), gf.to(dev); fp.append(model(nf,gf).cpu().numpy()); fl.append(lb.cpu().numpy())
    fp=np.concatenate(fp); fl=np.concatenate(fl)
    tmae=np.mean(np.abs(fp-fl))
    print(f"\nTest MAE={tmae:.2f} MW/gen, Time={time.time()-t0:.0f}s")
if __name__=="__main__": train()
