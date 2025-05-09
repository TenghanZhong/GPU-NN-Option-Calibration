# ─── Option_cali_NN_genetic_2stage_relative.py ────────────────────────────
"""
2-Stage calibration (GA + L-BFGS-B) using **relative-RMSE**
"""

import pathlib, joblib, numpy as np, pandas as pd, torch
import torch.nn as nn, pygad, multiprocessing as mp
import matplotlib.pyplot as plt
from  scipy.optimize import minimize

# ── paths & consts ─────────────────────────────────────────────────────────
PROJECT_DIR = pathlib.Path(r"C:\Users\26876\Desktop\Math548\Project")
MODEL_W     = PROJECT_DIR / "model_5000_days.pt"
SCALERS_PKL = PROJECT_DIR / "scalers_5000_days.pkl"
T_LIST      = np.array([258, 100, 48, 20], dtype=float)
EPS         = 1e-4                       # small number for stability
history     = []

# ── NN helper ──────────────────────────────────────────────────────────────
def get_act(name): return dict(linear=nn.Identity(), none=nn.Identity(),
                               relu=nn.ReLU(),  elu=nn.ELU(),
                               leaky=nn.LeakyReLU(0.01), tanh=nn.Tanh())[name]

class GenericNet(nn.Module):
    def __init__(self, d_in, d_out, hidden, layers, act="elu"):
        super().__init__()
        if isinstance(hidden,int): hidden=[hidden]*layers
        net, inp = [], d_in
        for h in hidden: net += [nn.Linear(inp,h), get_act(act)]; inp = h
        net.append(nn.Linear(inp,d_out));  self.net = nn.Sequential(*net)
    def forward(self,x): return self.net(x)

def load_nn(device="cpu"):
    saved     = joblib.load(SCALERS_PKL)
    net       = GenericNet(len(saved["in_cols"]),1,[64,32,32],3,"elu").double().to(device)
    net.load_state_dict(torch.load(MODEL_W,map_location=device)); net.eval()
    return net, saved["scX"], saved["scY"]

# ── input builder ──────────────────────────────────────────────────────────
def make_inputs(df, θ):
    a,b,c,d,k,*r = θ
    r_vec = np.choose(df["T"].map({t: i for i, t in enumerate(T_LIST)}).astype(int),
                      r, mode='clip').to_numpy()
    base  = np.repeat(np.array([a,b,c,d,k])[None,:], len(df), 0)
    return np.column_stack([base, r_vec.reshape(-1,1), df[["T","k"]].values])

# ── relative-RMSE loss ────────────────────────────────────────────────────
def build_rel_rmse(df, net, scX, scY, device):
    y = df["opt_price"].values
    def rel_rmse(θ):
        Xn = torch.tensor(scX.transform(make_inputs(df,θ))).double().to(device)
        with torch.no_grad(): p_std = net(Xn).cpu().numpy()
        p  = scY.inverse_transform(p_std).ravel()
        rel = (p - y) / (y + EPS)
        return float(np.sqrt((rel**2).mean()))
    return rel_rmse

def record(ga):
    best = -ga.best_solution()[1]; history.append(best)
    print(f"Gen {ga.generations_completed:4d} | best rel-RMSE {best:.4e}")

# ── main calibrator ────────────────────────────────────────────────────────
def calibrate(df, device="cuda", pop=400, gens=3000, seed=None):
    net, scX, scY = load_nn(device)
    loss          = build_rel_rmse(df, net, scX, scY, device)
    fit_func      = lambda ga,sol,idx: -loss(sol)

    lows  = np.array([0,0,1e-4,0.5,0,-.25,-.25,-.25,-.25])
    highs = np.array([5,5,0.9999,0.999,10,.25,.25,.25,.25])
    space = [{'low':lo,'high':hi} for lo,hi in zip(lows,highs)]

    ga = pygad.GA(fitness_func=fit_func, num_genes=9, sol_per_pop=pop,
                  num_generations=gens, gene_space=space,
                  num_parents_mating=pop//2, parent_selection_type="rank",
                  crossover_type="two_points", mutation_type="adaptive",
                  mutation_percent_genes=[40,15],
                  parallel_processing=["thread", max(1,mp.cpu_count()-2)],
                  random_seed=seed, on_generation=record)
    ga.run(); θ_ga = ga.best_solution()[0]

    res = minimize(loss, θ_ga, method="L-BFGS-B",
                   bounds=list(zip(lows,highs)),
                   options=dict(maxiter=2000,ftol=1e-12,gtol=1e-12))
    θ_opt = res.x
    print(f"Final relative-RMSE = {res.fun:.4e}")

    # loss curve
    plt.figure(figsize=(8,4))
    plt.plot(history,marker='o'); plt.title("GA progress (relative-RMSE)")
    plt.xlabel("Generation"); plt.ylabel("Best relative-RMSE"); plt.grid(True)
    plt.tight_layout(); plt.savefig("rel_rmse_curve.png",dpi=150); plt.close()
    print("📈 loss curve saved to rel_rmse_curve.png")

    return θ_opt

# ── run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = (pd.read_excel(r"C:\Users\26876\Desktop\Math548\Project\02_jan_put.xlsx")
            .rename(columns={"Strike_Put":"k","TTM_Put":"T","mid_price_Put":"opt_price"}))
    df["opt_price"] /= 100
    calibrate(df, device="cuda")
