import argparse, json, random
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Generator labels: [L, beta2, D, gamma, loss_dB, P0, T0, pulse_type]
# D is derived from beta2, so the MDN predicts six independent continuous labels.
Y_IDX = [0, 1, 3, 4, 5, 6]
Y_NAMES = ["L_km", "beta2_ps2_km", "gamma_W_km", "loss_dB_km", "P0_W", "T0_ps"]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_data(path):
    with h5py.File(path, "r") as f:
        names = ["input_temporal_intensity", "input_spectral_intensity",
                 "output_temporal_intensity", "output_spectral_intensity"]
        x = np.stack([f[n][:] for n in names], axis=1).astype(np.float32)
        labels = f["meta_parameters"][:].astype(np.float32)
        time = f["time_vector"][:].astype(np.float32)
    if labels.ndim != 2 or labels.shape[1] != 8:
        raise ValueError(f"Expected meta_parameters shape (N,8); got {labels.shape}")
    if not np.isfinite(x).all() or not np.isfinite(labels).all():
        raise ValueError("NaN or infinite values found in data.")
    y = labels[:, Y_IDX]
    pulse_type = labels[:, 7].astype(np.int64)
    if not np.isin(pulse_type, [0, 1]).all():
        raise ValueError("pulse_type must be 0 (Gaussian) or 1 (sech).")
    return x, y, pulse_type, time

def split_ids(n, seed):
    ids = np.random.default_rng(seed).permutation(n)
    ntr, nval = int(.8*n), int(.1*n)
    return ids[:ntr], ids[ntr:ntr+nval], ids[ntr+nval:]

def normalize_x(x, train_ids):
    # One mean/std per channel, calculated on training samples only.
    mean = x[train_ids].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(x[train_ids].std(axis=(0, 2), keepdims=True), 1e-6)
    return (x-mean)/std, mean, std

def normalize_y(y, train_ids):
    mean = y[train_ids].mean(axis=0, keepdims=True)
    std = np.maximum(y[train_ids].std(axis=0, keepdims=True), 1e-8)
    return (y-mean)/std, mean, std

class Encoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 16, 9, stride=2, padding=4), nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.BatchNorm1d(96), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(96, feature_dim), nn.GELU(), nn.Dropout(.1))
    def forward(self, x): return self.net(x)

class MDN(nn.Module):
    def __init__(self, n_targets=6, n_components=5, feature_dim=128):
        super().__init__()
        self.k, self.c = n_components, n_targets
        self.encoder = Encoder(feature_dim)
        self.logits = nn.Linear(feature_dim, self.k)
        self.means = nn.Linear(feature_dim, self.k*self.c)
        self.log_scales = nn.Linear(feature_dim, self.k*self.c)
        self.type_head = nn.Linear(feature_dim, 2)
    def forward(self, x):
        h = self.encoder(x); b = x.shape[0]
        logits = self.logits(h)
        means = self.means(h).view(b, self.k, self.c)
        log_scales = self.log_scales(h).view(b, self.k, self.c).clamp(-5, 3)
        return logits, means, log_scales, self.type_head(h)

def mdn_nll(y, logits, means, log_scales):
    # Diagonal-Gaussian mixture negative log likelihood.
    z = (y[:, None, :] - means) * torch.exp(-log_scales)
    log_pdf = -.5*z.square() - log_scales - .5*np.log(2*np.pi)
    log_pdf = log_pdf.sum(dim=-1)
    log_mix = torch.log_softmax(logits, dim=-1)
    return -torch.logsumexp(log_mix + log_pdf, dim=-1).mean()

def run_eval(model, loader, device, class_weight=.2):
    model.eval(); totals = np.zeros(4, dtype=np.float64); count = 0
    with torch.no_grad():
        for x, y, typ in loader:
            x, y, typ = x.to(device), y.to(device), typ.to(device)
            logits, means, scales, type_logits = model(x)
            lm = mdn_nll(y, logits, means, scales)
            lc = nn.functional.cross_entropy(type_logits, typ)
            n = len(x); totals += n*np.array([lm.item()+class_weight*lc.item(),
                lm.item(), lc.item(), (type_logits.argmax(1)==typ).float().mean().item()])
            count += n
    return dict(loss=totals[0]/count, mdn_nll=totals[1]/count,
                type_ce=totals[2]/count, type_accuracy=totals[3]/count)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="nlse_tnn_dataset.h5")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="mdn_run")
    args = p.parse_args()
    seed_all(args.seed); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    x, y, typ, time = load_data(args.data)
    tr, va, te = split_ids(len(x), args.seed)
    xn, xm, xs = normalize_x(x, tr); yn, ym, ys = normalize_y(y, tr)
    def loader(ids, shuffle):
        ds = TensorDataset(torch.from_numpy(xn[ids]).float(),
            torch.from_numpy(yn[ids]).float(), torch.from_numpy(typ[ids]).long())
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)
    train, val, test = loader(tr, True), loader(va, False), loader(te, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDN(n_components=args.components).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best = float("inf"); history = []; class_weight = .2
    print(f"{len(x)} samples; train/val/test={len(tr)}/{len(va)}/{len(te)}; device={device}")
    print("MDN continuous targets:", Y_NAMES)
    for epoch in range(1, args.epochs+1):
        model.train(); total = 0.; count = 0
        for xb, yb, tb in train:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, means, scales, type_logits = model(xb)
            lm = mdn_nll(yb, logits, means, scales)
            lc = nn.functional.cross_entropy(type_logits, tb)
            loss = lm + class_weight*lc
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step(); total += loss.item()*len(xb); count += len(xb)
        vm = run_eval(model, val, device, class_weight)
        row = {"epoch": epoch, "train_loss": total/count, **{"val_"+k:v for k,v in vm.items()}}
        history.append(row)
        if vm["loss"] < best:
            best = vm["loss"]
            torch.save({"model_state_dict":model.state_dict(), "n_components":args.components,
                "y_indices":Y_IDX, "y_names":Y_NAMES, "x_mean":xm, "x_std":xs,
                "y_mean":ym, "y_std":ys, "seed":args.seed}, out/"best_mdn.pt")
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d}: train={total/count:.4f}, val={vm['loss']:.4f}, "
                  f"val pulse-type accuracy={vm['type_accuracy']:.3f}")
    metrics = run_eval(model, test, device, class_weight)
    (out/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out/"test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out/"preprocessing.json").write_text(json.dumps({
        "continuous_target_names":Y_NAMES, "continuous_target_indices":Y_IDX,
        "input_channels":["input_temporal_intensity","input_spectral_intensity",
                          "output_temporal_intensity","output_spectral_intensity"],
        "pulse_type":{"0":"Gaussian","1":"sech"},
        "time_vector_start":float(time[0]), "time_vector_end":float(time[-1]),
        "time_vector_length":int(len(time))}, indent=2), encoding="utf-8")
    print("Held-out test metrics:", metrics)
    print("Saved model/logs to", out.resolve())
    print("Baseline only: physics losses have not been added yet.")

if __name__ == "__main__":
    main()
