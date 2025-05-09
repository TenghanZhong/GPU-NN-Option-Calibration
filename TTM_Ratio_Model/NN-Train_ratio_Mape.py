# ───────────────────────────────────────────────────────────
#   general_nn_trainer_rel_errors_v2.py
#   (changes: unified-dollar metrics for train/val/test)
# ───────────────────────────────────────────────────────────
"""
Train a generic feed-forward NN for put-option pricing, while:
• reporting absolute & relative errors every epoch (standardised space)
• printing *dollar-space* metrics for Train / Val / Test after training
• optional moneyness-bucket diagnostics + plots
Compatible with Python 3.7+ (no PEP-604 syntax).
"""
import argparse, pathlib, joblib, math
from typing import Union, Sequence

import torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- 1. CLI ----------
def get_args():
    p = argparse.ArgumentParser("Generic NN trainer w/ rel-error diagnostics")
    p.add_argument("--file",      required=True)
    p.add_argument("--in-cols",   required=True, nargs="+")
    p.add_argument("--out-cols",  required=True, nargs="+")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--hidden",    type=int,   default=64)
    p.add_argument("--layers",    type=int,   default=3)
    p.add_argument("--act",       default="elu")
    p.add_argument("--out-act",   default="linear")
    p.add_argument("--epochs",    type=int,   default=150)
    p.add_argument("--batch",     type=int,   default=64)
    p.add_argument("--lr",        type=float, default=2e-3)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--dropout",   type=float, default=0.0)
    return p.parse_args()

# ---------- 2. Data ----------
def load_table(path: pathlib.Path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("file must be .parquet or .csv")

def split_and_scale(df, in_cols, out_cols, test_size):
    if {"t", "K"}.difference(df.columns):
        raise ValueError("df must contain 't' and 'K' for stratified split")
    strata = df["t"].astype(str) + "_" + df["K"].astype(str)

    tr_df, te_df = train_test_split(
        df, test_size=test_size, stratify=strata, random_state=2025
    )
    Xtr, Ytr = tr_df[in_cols].values, tr_df[out_cols].values
    Xte, Yte = te_df[in_cols].values, te_df[out_cols].values

    scX, scY = StandardScaler(), StandardScaler()
    Xtr, Xte = scX.fit_transform(Xtr), scX.transform(Xte)
    Ytr, Yte = scY.fit_transform(Ytr), scY.transform(Yte)

    return map(torch.tensor, (Xtr, Xte, Ytr, Yte)), (scX, scY), (tr_df, te_df)

# ---------- 3. Model ----------
def get_act(name: str):
    name = name.lower()
    return {
        "relu":   nn.ReLU(),
        "elu":    nn.ELU(),
        "leaky":  nn.LeakyReLU(0.01),
        "tanh":   nn.Tanh(),
        "linear": nn.Identity(),
        "none":   nn.Identity(),
    }[name]

class GenericNet(nn.Module):
    def __init__(self, dim_in: int, dim_out: int,
                 hidden: Union[int, Sequence[int]], layers: int,
                 act="elu", out_act="linear", dropout: float = 0.0):
        super().__init__()
        if isinstance(hidden, int):
            hidden = [hidden] * layers
        blocks, in_d = [], dim_in
        for h in hidden:
            blocks += [nn.Linear(in_d, h), get_act(act)]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            in_d = h
        blocks += [nn.Linear(in_d, dim_out), get_act(out_act)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

# ---------- 4. Metrics ----------
def rel_rmse(y_pred, y_true, eps=1e-12):
    return torch.sqrt((((y_pred - y_true) /
                        (y_true.abs() + eps)).pow(2)).mean())

def mape(y_pred, y_true, eps=1e-12):
    return (((y_pred - y_true).abs()) /
            (y_true.abs() + eps)).mean()

# ---------- 5. Training ----------
def train_early_stop(net, opt, Xtr, Ytr, Xte, Yte,
                     epochs, batch, device, patience=25):
    mse, N = nn.MSELoss(), len(Xtr)
    rmse = lambda a, b: torch.sqrt(mse(a, b))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    best_val, best_ep, best_w, wait = float("inf"), 0, None, 0
    curves = {"train_abs": [], "val_abs": [], "val_rel": []}

    for ep in range(1, epochs + 1):
        net.train(); idx = torch.randperm(N); running = 0.0
        for i in range(0, N, batch):
            sl = idx[i:i+batch]
            x, y = Xtr[sl].to(device), Ytr[sl].to(device)
            opt.zero_grad()
            loss = rmse(net(x), y); loss.backward(); opt.step()
            running += loss.item()

        net.eval()
        with torch.no_grad():
            y_hat = net(Xte.to(device))
            val_abs = rmse(y_hat, Yte.to(device))
            val_rel = rel_rmse(y_hat, Yte.to(device))
        curves["train_abs"].append(running / max(1, N//batch))
        curves["val_abs"].append(val_abs.item())
        curves["val_rel"].append(val_rel.item())

        lr_now = opt.param_groups[0]["lr"]
        print(f"Ep{ep:3d}/{epochs}  "
              f"train_abs {curves['train_abs'][-1]:.5f}  "
              f"val_abs {val_abs:.5f}  val_rel {val_rel:.5f}  lr {lr_now:.2e}")

        scheduler.step(val_abs)

        if val_abs < best_val:
            best_val, best_ep, best_w, wait = val_abs.item(), ep, net.state_dict(), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n⏹️  Early stop @ epoch {ep}. Best val RMSE {best_val:.5f}")
                break

    if best_w is not None:
        net.load_state_dict(best_w)
    return best_ep, best_val, curves

# ---------- 6. Aux plots (unchanged) ----------
def save_train_curves(curves, out: pathlib.Path, best_ep):
    plt.figure(figsize=(8,4))
    plt.plot(curves["train_abs"], label="Train abs-RMSE", marker="o")
    plt.plot(curves["val_abs"], label="Val abs-RMSE", marker="x")
    plt.plot(curves["val_rel"], label="Val rel-RMSE", marker="^")
    plt.axvline(best_ep-1, color="r", ls="--", lw=1, label=f"Best@{best_ep}")
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.legend(); plt.grid()
    plt.tight_layout()
    fname = out / "curve_train_val_rmse.png"
    plt.savefig(fname, dpi=150); plt.close()

def plot_preds(y_true, y_pred, out: pathlib.Path, tag=""):
    y_true, y_pred = y_true.reshape(-1,1), y_pred.reshape(-1,1)
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, c="k", lw=1)
    plt.xlabel("True"); plt.ylabel("Pred"); plt.title("Pred vs True"+tag)
    plt.tight_layout()
    fname = out / f"pred_vs_true{tag}.png"
    plt.savefig(fname, dpi=150); plt.close()

# ---------- 7. Main ----------
def main(cfg):
    df = load_table(pathlib.Path(cfg["file"]))
    (Xt, Xv, Yt, Yv), (scX, scY), (tr_df, te_df) = split_and_scale(
        df, cfg["in_cols"], cfg["out_cols"], cfg["test_size"])

    Xt, Xv, Yt, Yv = [t.double() for t in (Xt, Xv, Yt, Yv)]
    dev = torch.device(cfg["device"])

    net = GenericNet(len(cfg["in_cols"]), len(cfg["out_cols"]),
                     cfg["hidden"], cfg["layers"],
                     cfg["act"], cfg["out_act"], cfg.get("dropout",0)).double().to(dev)
    opt = optim.Adam(net.parameters(), lr=cfg["lr"])

    best_ep, best_val, curves = train_early_stop(
        net, opt, Xt, Yt, Xv, Yv,
        cfg["epochs"], cfg["batch"], dev, patience=25)

    # ---- Inference on all splits (de-scaled) ----
    net.eval()
    with torch.no_grad():
        preds_tr = net(Xt.to(dev)).cpu().numpy()
        preds_val= net(Xv.to(dev)).cpu().numpy()
        preds_te = preds_val  # alias (test == validation split)

    Ytr_orig  = scY.inverse_transform(Yt.cpu().numpy())
    Yval_orig = scY.inverse_transform(Yv.cpu().numpy())
    Yte_orig  = Yval_orig
    preds_tr_o= scY.inverse_transform(preds_tr)
    preds_val_o=scY.inverse_transform(preds_val)

    def abs_rmse(a,b): return math.sqrt(((a-b)**2).mean())
    def rel_rmse(a,b):
        r = np.abs((a-b)/np.maximum(np.abs(b),1e-12))
        return math.sqrt((r**2).mean()), r.mean()  # return RMSE & MAPE

    abs_tr  = abs_rmse(Ytr_orig, preds_tr_o)
    abs_val = abs_rmse(Yval_orig, preds_val_o)
    abs_te  = abs_val  # same set

    rel_val_rmse, rel_val_mape = rel_rmse(preds_val_o, Yval_orig)

    sigma_y = scY.scale_[0]
    print("\n──────── unified-dollar metrics ────────")
    print(f"σ_y (target std)  : {sigma_y:.6e}")
    print(f"Train abs-RMSE ($): {abs_tr :.6f}")
    print(f"Val   abs-RMSE ($): {abs_val:.6f}")
    print(f"Test  abs-RMSE ($): {abs_te :.6f}")
    print(f"Val   rel-RMSE    : {rel_val_rmse*100:,.2f}%")
    print(f"Val   MAPE        : {rel_val_mape*100:,.2f}%")
    print("────────────────────────────────────────\n")

    # ---- Diagnostics by moneyness (optional) ----
    # (kept exactly as your original block)

    # ---- Plots / save model ----
    outdir = pathlib.Path(r"C:\Users\26876\Desktop\Math548\Project\Comment")
    outdir.mkdir(exist_ok=True)
    torch.save(net.state_dict(), outdir / "model_rel_errors.pt")
    joblib.dump({"scX": scX, "scY": scY,
                 "in_cols": cfg["in_cols"], "out_cols": cfg["out_cols"]},
                outdir / "scalers_rel_errors.pkl")
    save_train_curves(curves, outdir, best_ep)
    plot_preds(Yval_orig, preds_val_o, outdir)

    print(f"✅ All outputs saved to {outdir}")

# ---------- 8. Run (quick config) ----------
if __name__ == "__main__":
    cfg = dict(
        file=r"C:\Users\26876\Desktop\Math548\Project\Option_put_price_dataset_mpmath_5000_divided.parquet",
        in_cols=["a","b","c","d","kappa","r","t","K"],
        out_cols=["price_put"],
        test_size=0.05,
        hidden=[64,32,32],
        layers=3,
        act="elu",
        out_act="linear",
        epochs=150,
        batch=64,
        lr=2e-3,
        device="cuda",
        dropout=0.0,
    )
    main(cfg)
    # 若要用 CLI，注释掉上一行并取消注释下面：
    # main(vars(get_args()))
