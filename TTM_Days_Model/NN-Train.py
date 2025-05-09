# ─────────── general_nn_trainer.py (with test‑set plotting) ───────────
import argparse, pathlib, joblib, math
import torch, torch.nn as nn, torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# ---------- 1.  CLI ----------

def get_args():
    p = argparse.ArgumentParser("Generic param→output NN trainer")
    p.add_argument("--file",      required=True)
    p.add_argument("--in-cols",   required=True, nargs="+")
    p.add_argument("--out-cols",  required=True, nargs="+")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--hidden",    type=int,   default=30)   # 单 int → 均匀层
    p.add_argument("--layers",    type=int,   default=3)
    p.add_argument("--act",       default="elu")            # 隐藏层激活
    p.add_argument("--out-act",   default="linear")         # 输出层激活
    p.add_argument("--epochs",    type=int,   default=200)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--device",    default="cpu")
    return p.parse_args()

# ---------- 2.  Data ----------

def load_table(path: pathlib.Path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("file must be .parquet or .csv")


def split_and_scale(df, in_cols, out_cols, test_size):
    # 先构造分层标签：t_K组合
    if "t" not in df.columns or "K" not in df.columns:
        raise ValueError("分层采样需要 df 中包含 't' 和 'K' 列")

    strata = df["t"].astype(str) + "_" + df["K"].astype(str)

    # 执行分层划分
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=strata, random_state=2025
    )

    # 拆出输入输出
    X_tr = train_df[in_cols].values
    Y_tr = train_df[out_cols].values
    X_te = test_df[in_cols].values
    Y_te = test_df[out_cols].values

    # 标准化
    scX, scY = StandardScaler(), StandardScaler()
    X_tr = scX.fit_transform(X_tr);
    X_te = scX.transform(X_te)
    Y_tr = scY.fit_transform(Y_tr);
    Y_te = scY.transform(Y_te)

    return map(torch.tensor, (X_tr, X_te, Y_tr, Y_te)), (scX, scY)


# ---------- 3.  Model ----------

def get_act(name):
    name = name.lower()
    if name in ("linear", "none"):
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "leaky":
        return nn.LeakyReLU(0.01)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown act {name}")


class GenericNet(nn.Module):
    def __init__(self, dim_in, dim_out, hidden, layers, act="elu", out_act="linear"):
        super().__init__()
        if isinstance(hidden, int):
            hidden = [hidden] * layers
        blocks, in_d = [], dim_in
        for h in hidden:
            blocks += [nn.Linear(in_d, h), get_act(act)]
            in_d = h
        blocks += [nn.Linear(in_d, dim_out), get_act(out_act)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# ---------- 4.  Train ----------

def train(net, opt, Xtr, Ytr, Xte, Yte, epochs, batch, device):
    mse, N = nn.MSELoss(), len(Xtr)
    rmse = lambda a, b: torch.sqrt(mse(a, b))
    for ep in range(1, epochs + 1):
        net.train(); idx = torch.randperm(N); run = 0.0
        for i in range(0, N, batch):
            sl = idx[i : i + batch]
            x, y = Xtr[sl].to(device), Ytr[sl].to(device)
            opt.zero_grad(); loss = rmse(net(x), y); loss.backward(); opt.step()
            run += loss.item()
        net.eval()
        with torch.no_grad():
            val = rmse(net(Xte.to(device)), Yte.to(device)).item()
        print(f"Ep{ep:3d}/{epochs}  train {run / (N // batch):.5f}  val {val:.5f}")


def train_early_stopping(net, opt, Xtr, Ytr, Xte, Yte, epochs, batch, device, patience=20):
    mse, N = nn.MSELoss(), len(Xtr)
    rmse = lambda a, b: torch.sqrt(mse(a, b))

    # 加 ReduceLROnPlateau 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    best_val = float("inf")
    best_weights = None
    best_epoch = 0
    wait = 0
    train_curve = []
    val_curve = []

    for ep in range(1, epochs + 1):
        net.train()
        idx = torch.randperm(N)
        run = 0.0
        for i in range(0, N, batch):
            sl = idx[i: i + batch]
            x, y = Xtr[sl].to(device), Ytr[sl].to(device)
            opt.zero_grad()
            loss = rmse(net(x), y)
            loss.backward()
            opt.step()
            run += loss.item()

        avg_train_rmse = run / (N // batch)
        train_curve.append(avg_train_rmse)

        net.eval()
        with torch.no_grad():
            val_rmse = rmse(net(Xte.to(device)), Yte.to(device)).item()
        val_curve.append(val_rmse)

        # 打印本轮信息 + 当前学习率
        current_lr = opt.param_groups[0]['lr']
        print(f"Ep{ep:3d}/{epochs}  train {avg_train_rmse:.5f}  val {val_rmse:.5f}  lr {current_lr:.2e}")

        # 调整学习率
        scheduler.step(val_rmse)

        # Early Stopping
        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = ep
            best_weights = net.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n⏹️ Early stopping triggered at epoch {ep}.")
                print(f"✅ Best validation RMSE = {best_val:.5f} at epoch {best_epoch}.")
                break

    # ── 恢复最佳权重 ──
    if best_weights is not None:
        net.load_state_dict(best_weights)
        print(f"📦 Restored best model weights from epoch {best_epoch}.")

    # ── 保存最终画图 ──
    out_dir = pathlib.Path(r"C:\Users\26876\Desktop\Math548\Project")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_curve, label="Train RMSE", marker="o")
    plt.plot(val_curve, label="Validation RMSE", marker="x")
    plt.scatter(best_epoch - 1, best_val, color="red", s=80, label="Best Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training and Validation RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "train_val_rmse_curve.png", dpi=150)
    plt.show()

    return train_curve, val_curve


# ---------- 5.  Plot ----------

def plot_results(y_true, y_pred, out_dir: pathlib.Path, title="NN prediction vs. true on test‑set"):
    """Scatter/line plot comparing y_true and y_pred for each output dimension."""
    y_true, y_pred = map(lambda t: t.reshape(-1, t.shape[-1]), (y_true, y_pred))
    n_dims = y_true.shape[1]

    plt.figure(figsize=(10, 4 * n_dims))
    for d in range(n_dims):
        ax = plt.subplot(n_dims, 1, d + 1)
        ax.plot(y_true[:, d], label=f"True dim {d}", marker="o", linestyle="", alpha=0.7)
        ax.plot(y_pred[:, d], label=f"Pred dim {d}", marker="x", linestyle="", alpha=0.7)
        ax.set_title(f"Dimension {d}")
        ax.legend()
        ax.grid(True)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = out_dir / "test_pred_vs_true_5000.png"
    plt.savefig(fname, dpi=150)
    plt.show(block=False)
    print(f"📈 plot saved to {fname}")
import numpy as np

def plot_results_small_sample(y_true, y_pred, out_dir: pathlib.Path, n_samples=100, title="NN Prediction vs True (Small Sample)"):
    """随机选 n_samples 个点，画预测 vs 真值的对比图。"""
    assert y_true.shape == y_pred.shape
    n_total = y_true.shape[0]
    idx = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)

    y_true_sample = y_true[idx]
    y_pred_sample = y_pred[idx]

    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_true_sample)), y_true_sample, label="True", marker="o", alpha=0.7)
    plt.scatter(range(len(y_pred_sample)), y_pred_sample, label="Predicted", marker="x", alpha=0.7)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Put Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fname = out_dir / "test_pred_vs_true_small_sample_5000.png"
    plt.savefig(fname, dpi=150)
    plt.show(block=False)
    print(f"📈 Small sample plot saved to {fname}")

# ---------- 6.  Main ----------

def main(cfg):
    df = load_table(pathlib.Path(cfg["file"]))
    (Xt, Xv, Yt, Yv), (scX, scY) = split_and_scale(
        df, cfg["in_cols"], cfg["out_cols"], cfg["test_size"]
    )
    Xt, Xv, Yt, Yv = [t.double() for t in (Xt, Xv, Yt, Yv)]
    dev = torch.device(cfg["device"])
    net = GenericNet(
        len(cfg["in_cols"]),
        len(cfg["out_cols"]),
        cfg["hidden"],
        cfg["layers"],
        cfg["act"],
        cfg["out_act"],
    ).double().to(dev)
    opt = optim.Adam(net.parameters(), lr=cfg["lr"])
    train_curve, val_curve = train_early_stopping(net, opt, Xt, Yt, Xv, Yv, cfg["epochs"], cfg["batch"], dev, patience=25)

    # ── 保存路径 ──
    out = pathlib.Path(r"C:\Users\26876\Desktop\Math548\Project")
    out.mkdir(exist_ok=True)

    # 保存模型权重
    torch.save(net.state_dict(), out / "model_5000_ratio.pt")

    # 保存标准化器及列信息
    joblib.dump(
        {
            "scX": scX,
            "scY": scY,
            "in_cols": cfg["in_cols"],
            "out_cols": cfg["out_cols"],
        },
        out / "scalers_5000_ratio.pkl",
    )

    # ── 预测 & 反标准化 ──
    net.eval()
    with torch.no_grad():
        preds = net(Xv.to(dev)).cpu().numpy()
    Y_true = Yv.cpu().numpy()

    Y_true_orig = scY.inverse_transform(Y_true)
    preds_orig = scY.inverse_transform(preds)

    # ── 画图 ──
    plot_results(Y_true_orig, preds_orig, out)

    print("✅ Model & plots saved to C:/Users/26876/Desktop/Math548/Project/")

    # ── 原本画全部点
    plot_results(Y_true_orig, preds_orig, out)

    # 🆕 新加一行 —— 只画小样本
    plot_results_small_sample(Y_true_orig, preds_orig, out, n_samples=100)

    # ── 计算最终 test set RMSE ──
    final_test_rmse = math.sqrt(((Y_true_orig - preds_orig) ** 2).mean())
    print(f"🏁 Final Test Set RMSE (after inverse transform): {final_test_rmse:.6f}")

# ---------- 7.  Run ----------

if __name__ == "__main__":
    # ——用 CLI 时注释掉下面 cfg 并调用 main(get_args())——
    cfg = dict(
        file=r"C:\Users\26876\Desktop\Math548\Project\Option_put_price_dataset_mpmath_5000.parquet",
        in_cols=["a", "b", "c", "d", "kappa", "r", "t", "K"],
        out_cols=["price_put"],
        test_size=0.05,
        hidden=[64, 32, 32],
        layers=3,
        act="elu",
        out_act="linear",
        epochs=150,
        batch=64,
        lr=2e-3,
        device="cuda",
        dropout=0,
    )
    main(cfg)  # ← 直接运行
    # main(get_args())        # ← 改成这行即可走命令行
'''
 从 parquet 文件加载 θ, t, K → price_put
   → 数据归一化
   → 构建 NN 模型：f(θ, t, K) → price_put
   → MSE 训练并保存模型
   → 可视化 NN 在 test set 上的预测能力
'''