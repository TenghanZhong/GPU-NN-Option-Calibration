# ───────── option_cali_nn_cma_lbfgs.py ─────────
"""
2-Stage Calibration (升级版):
1. CMA-ES               —— 全局搜索
2. L-BFGS-B             —— 局部精细化
输出进度图 + 拟合散点图
"""

import pathlib, joblib, numpy as np, pandas as pd, torch
import torch.nn as nn
import cma                              # ← pip install cma
import multiprocessing as mp, sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------- 1.  Paths ----------
PROJ  = pathlib.Path(r"C:\Users\26876\Desktop\Math548\Project")
MODEL = PROJ / "model.pt"
SCAL  = PROJ / "scalers.pkl"
T_LIST = np.array([258,100,48,20], dtype=float)

# ---------- 2.  Network ----------
def act(name):
    d = dict(linear=nn.Identity, relu=nn.ReLU, elu=nn.ELU,
             leaky=lambda: nn.LeakyReLU(0.01), tanh=nn.Tanh)
    return d[name]() if name in d else nn.Identity()

class GenericNet(nn.Module):
    def __init__(self, dim_in, dim_out, hidden, layers, act_f="elu"):
        super().__init__()
        hidden = [hidden]*layers if isinstance(hidden,int) else hidden
        seq, d = [], dim_in
        for h in hidden:
            seq += [nn.Linear(d,h), act(act_f)]; d = h
        seq += [nn.Linear(d, dim_out)]
        self.net = nn.Sequential(*seq)
    def forward(self,x): return self.net(x)

def load_net(device="cpu"):
    saved = joblib.load(SCAL)
    net = GenericNet(len(saved["in_cols"]),1,[64,64,32],3,"elu").double().to(device)
    net.load_state_dict(torch.load(MODEL,map_location=device))
    net.eval()
    return net, saved

# ---------- 3. helpers ----------
def make_X(df, θ):
    a,b,c,d,kappa,*r = θ
    rvec = np.choose(df["T"].map({t:i for i,t in enumerate(T_LIST)}).values,r,mode='clip')
    base = np.repeat(np.array([a,b,c,d,kappa])[None,:], len(df),0)
    return np.column_stack([base, rvec[:,None], df[["T","k"]]])

def mse_factory(df, net, scX, scY, device):
    y = df["opt_price"].values
    def f(θ):
        Xn = torch.tensor(scX.transform(make_X(df,θ))).double().to(device)
        with torch.no_grad():
            pred = net(Xn).cpu().numpy()
        y_hat = scY.inverse_transform(pred).ravel()
        return float(((y_hat - y)**2).mean())
    return f

# ---------- 4. main calibration ----------
def calibrate(df, device="cpu",
              sigma0=1.0, pop_size=None,
              max_cma_iter=300):

    net, saved      = load_net(device)
    scX, scY        = saved["scX"], saved["scY"]
    loss            = mse_factory(df, net, scX, scY, device)

    # ---- bounds & x0 ----
    lows  = np.array([0,0,1e-4,0.5,0,  -5,-5,-5,-5])
    highs = np.array([5,5,0.9999,0.999,5, 5, 5, 5, 5])
    x0    = (lows+highs)/2

    # ---- CMA-ES global optimisation ----
    es = cma.CMAEvolutionStrategy(x0, sigma0,
            {'bounds':[lows,highs],
             'popsize': pop_size or 4+int(3*np.log(len(x0))),
             'maxiter':max_cma_iter,
             'verb_disp':1})
    history=[]
    while not es.stop():
        xs = es.ask()
        fs = [loss(x) for x in xs]
        es.tell(xs,fs); history.append(es.best.f)

    θ_cma, f_cma = es.result.xbest, es.result.fbest
    print(f"🌍 CMA-ES  best MSE = {f_cma:.6e}")

    # ---- L-BFGS-B local refinement ----
    res = minimize(loss, θ_cma, method="L-BFGS-B",
                   bounds=list(zip(lows,highs)),
                   options=dict(maxiter=500, ftol=1e-10, disp=True))
    θ_opt, f_opt = res.x, res.fun
    print(f"🔎 L-BFGS-B final MSE = {f_opt:.6e}")

    # ---------- 5. plot progress ----------
    plt.figure(figsize=(8,4))
    plt.semilogy(history,label="CMA-ES best so far")
    plt.axhline(f_opt,color="r",ls="--",label="After L-BFGS-B")
    plt.title("Calibration progress"); plt.xlabel("CMA-ES iteration")
    plt.ylabel("MSE (log scale)"); plt.grid(); plt.legend(); plt.tight_layout()
    plt.savefig("cali_progress.png",dpi=150); plt.close()

    # 预测值
    Xn = torch.tensor(scX.transform(make_X(df, θ_opt))).double().to(device)
    with torch.no_grad():
        pred = net(Xn).cpu().numpy()

    y_hat = scY.inverse_transform(pred).ravel()  # NN预测还原
    y_true = df["opt_price"].values  # 真实Market价格

    m, M = min(y_true.min(), y_hat.min()), max(y_true.max(), y_hat.max())

    # — ① 真实数据 Market vs Market 散点图 —
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_true, alpha=0.7, edgecolors='k', label="Market (True)", color="grey")
    plt.plot([m, M], [m, M], 'r--', label="Perfect Fit")
    plt.xlabel("Market")
    plt.ylabel("Market")
    plt.title("Market vs Market (Reference)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("market_vs_market.png", dpi=150)
    plt.close()

    # — ② 模型数据 Model vs Market 散点图 —
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_hat, alpha=0.7, edgecolors='k', label="Model (Predicted)", color="blue")
    plt.plot([m, M], [m, M], 'r--', label="Perfect Fit")
    plt.xlabel("Market")
    plt.ylabel("Model")
    plt.title("Model vs Market (Fit after Calibration)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("model_vs_market.png", dpi=150)
    plt.show()

    print("📈 Scatter plots saved: 'market_vs_market.png' and 'model_vs_market.png'.")

    return θ_opt, f_opt

# ---------- 7. run ----------
if __name__ == "__main__":
    data = (pd.read_excel(r"C:\Users\26876\Desktop\Math548\Project\02_jan_put.xlsx")
              .rename(columns={"Strike_Put":"k","TTM_Put":"T","mid_price_Put":"opt_price"}))
    data["opt_price"] /= 100
    calibrate(data, device="cuda")       # 改成 "cpu" 亦可
