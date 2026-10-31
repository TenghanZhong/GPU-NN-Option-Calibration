# Two-stage surrogate calibration with absolute MSE
"""
2-Stage Calibration:
1. Genetic Algorithm (GA) for global search.
2. L-BFGS-B for bounded local refinement.
Compatible with PyGAD 3.x.x
"""

import os as _os
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _Path(_os.environ.get("GPU_NN_PROJECT_ROOT", str(_REPO_ROOT / "Data_Results"))).expanduser()

import pathlib, joblib, numpy as np, pandas as pd, torch
import torch.nn as nn
import pygad, multiprocessing as mp, sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import matplotlib.cm as cm

# —— 1. Paths & Constants ——
PROJECT_DIR = pathlib.Path(str(_PROJECT_ROOT / "TTM_Ratio_Results"))
MODEL_W     = PROJECT_DIR / "model_5000_ratio.pt"
SCALERS_PKL = PROJECT_DIR / "scalers_5000_ratio.pkl"
T_LIST = np.array([258, 100, 48, 20], dtype=float)/252

history = []

def record_history(ga_instance):
    global history
    best_fitness = -ga_instance.best_solution()[1]
    history.append(best_fitness)
    print(f"Generation {ga_instance.generations_completed:4d} | Best MSE: {best_fitness:.6e}")

# —— 2. NN Definition ——
def get_act(name: str):
    name = name.lower()
    if name in ("linear", "none"): return nn.Identity()
    if name == "relu": return nn.ReLU()
    if name == "elu": return nn.ELU()
    if name == "leaky": return nn.LeakyReLU(0.01)
    if name == "tanh": return nn.Tanh()
    raise ValueError(f"unknown activation: {name}")

class GenericNet(nn.Module):
    def __init__(self, dim_in, dim_out, hidden, layers, act="elu", out_act="linear"):
        super().__init__()
        hidden = [hidden] * layers if isinstance(hidden, int) else hidden
        blocks, in_d = [], dim_in
        for h in hidden:
            blocks += [nn.Linear(in_d, h), get_act(act)]
            in_d = h
        blocks += [nn.Linear(in_d, dim_out), get_act(out_act)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

# —— 3. Helpers ——
# ── 3. Helpers ────────────────────────────────────────────────────────────
def load_nn(device="cpu"):
    saved = joblib.load(SCALERS_PKL)
    scX, scY, in_cols = saved["scX"], saved["scY"], saved["in_cols"]
    net = (GenericNet(len(in_cols), 1, [64, 32, 32], 3, "elu", "linear")
           .double().to(device))
    net.load_state_dict(torch.load(MODEL_W, map_location=device))
    net.eval()
    return net, scX, scY, in_cols                 # Include the training column order

# Build inputs in the training column order
def make_inputs(df: pd.DataFrame, theta: np.ndarray, *, in_cols):
    a, b, c, d, kappa, *rates = theta

    nearest_idx = np.abs(df["T"].values[:, None] - T_LIST[None, :]).argmin(axis=1)
    r_vec = np.take(rates, nearest_idx)

    tmp = df.copy()
    tmp["a"], tmp["b"], tmp["c"], tmp["d"], tmp["kappa"] = a, b, c, d, kappa
    tmp["r"] = r_vec

    # Align column names with the training data
    tmp = tmp.rename(columns={"T": "t", "k": "K"})

    X = tmp[in_cols].values
    return X.astype(np.float64)



# —— 4. Fitness Function ——
def fitness_factory(df, net, scX, scY, in_cols, device):
    def mse(theta):
        X = make_inputs(df, theta, in_cols=in_cols)
        Xn = torch.tensor(scX.transform(X)).double().to(device)
        with torch.no_grad():
            preds = net(Xn).cpu().numpy()
        preds_orig = scY.inverse_transform(preds)
        err = preds_orig.ravel() - df["opt_price"].values
        return float(np.mean(err**2))

    def fitness_func(ga_instance, solution, solution_idx):
        return -mse(solution)

    return fitness_func


# —— 5. Calibration Entry ——
def calibrate(df: pd.DataFrame, device="cpu",
              pop_size=600, num_generations=5000, seed=2024):
    start_time = time.time()

    net, scX, scY, in_cols = load_nn(device)
    fitness_func = fitness_factory(df, net, scX, scY, in_cols, device)

    lows  = np.array([0.0, 0.0, 1e-4, 0.50, 0.0, -0.25, -0.25, -0.25, -0.25])
    highs = np.array([3.0, 5.0, 0.9999, 0.999, 10.0, 0.25, 0.25, 0.25, 0.25])
    bounds = list(zip(lows, highs))
    gene_space = [{'low': lo, 'high': hi} for lo, hi in bounds]
    n_cores = max(1, mp.cpu_count()-2)

    history.clear()

    ga = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=int(pop_size * 0.5),
        fitness_func=fitness_func,
        sol_per_pop=pop_size,
        num_genes=len(gene_space),
        gene_space=gene_space,
        parent_selection_type="rank",#保守设置sss steady state selection
        crossover_type="two_points",
        mutation_type="adaptive",
        mutation_percent_genes=[40, 15],
        parallel_processing=["thread", n_cores],
        random_seed=seed,
        on_generation=record_history
    )
    ga.run()

    ga_solution, ga_fitness, _ = ga.best_solution()

    print("\n🔍 GA finished.")
    print(f"GA best MSE: {-ga_fitness:.6e}")

    def mse(theta):
        X = make_inputs(df, theta, in_cols=in_cols)
        Xn = torch.tensor(scX.transform(X)).double().to(device)
        with torch.no_grad():
            preds = net(Xn).cpu().numpy()
        preds_orig = scY.inverse_transform(preds)
        err = preds_orig.ravel() - df["opt_price"].values
        return float(np.mean(err**2))

    result = minimize(
        mse,
        ga_solution,
        method='L-BFGS-B',# or Nelder-Mead
        bounds=bounds,
        options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': True}
    )

    solution = result.x
    fitness  = result.fun

    keys = ["a","b","c","d","kappa","r1","r2","r3","r4"]
    params = {k: round(v, 6) for k, v in zip(keys, solution)}
    print("\n✅ Final optimisation finished.")
    print("Final MSE =", fitness)
    print("📌 Calibrated parameters:")
    for k,v in params.items():
        print(f"  {k:5s}= {v}")

    with open(PROJECT_DIR / "best_params.txt", "w") as f:
        for k,v in params.items():
            f.write(f"{k}={v}\n")

    plt.figure(figsize=(8,4))
    plt.plot(history, marker='o')
    plt.title("GA Calibration Progress")
    plt.xlabel("Generation")
    plt.ylabel("Best MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "progress.png")
    plt.close()
    X_final = make_inputs(df, solution, in_cols=in_cols)
    Xn_final = torch.tensor(scX.transform(X_final)).double().to(device)
    with torch.no_grad():
        preds_final_std = net(Xn_final).cpu().numpy()
    preds_final = scY.inverse_transform(preds_final_std).ravel()

    true_prices = df["opt_price"].values

    # == 画散点 & 参考线 ==
    plt.figure(figsize=(7, 6))

    # ① 市场价 → 模型价   （蓝色 ×）
    plt.scatter(true_prices, preds_final,
                marker="x", s=60, linewidths=1.5, color="#1f77b4",
                label="Model (Predicted)")

    # ② 市场价 → 市场价   （灰色 ○，空心更醒目）
    plt.scatter(true_prices, true_prices,
                marker="o", s=40, facecolors="none", edgecolors="gray",
                alpha=0.8, label="Market (Reference)")

    plt.xlabel("Market Prices")
    plt.ylabel("Model Prices")
    plt.title("Market vs Model Prices After Calibration")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # == 以 Strike 为横坐标，按 T 分类上色，画 Market vs Model Price ==
    plt.figure(figsize=(7, 5))

    unique_T = np.sort(df["T"].unique())
    cmap_market = cm.get_cmap('Oranges')
    cmap_model = cm.get_cmap('Blues')

    # 市场价用橙色系列，模型价用蓝色系列
    for i, t in enumerate(unique_T):
        mask = df["T"] == t
        label_T = f"T = {t:.3f}y"

        color_market = cmap_market(0.4 + 0.6 * i / max(1, len(unique_T) - 1))
        color_model = cmap_model(0.4 + 0.6 * i / max(1, len(unique_T) - 1))

        # Market price: ○ 空心
        plt.scatter(df.loc[mask, "k"], df.loc[mask, "opt_price"],
                    marker='o', facecolors='none', edgecolors=color_market,
                    label=f"Market {label_T}")

        # Model price: + 加号
        plt.scatter(df.loc[mask, "k"], preds_final[mask.values],
                    marker='+', color=color_model, label=f"Model {label_T}")

    plt.xlabel("Strike Price (k)")
    plt.ylabel("Put Option Price")
    plt.title("Put Price vs Strike (by Maturity, Colored)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "price_vs_strike_by_T_colored.png", dpi=150)
    plt.show()

    print("📈 Price vs Strike plot saved to 'price_vs_strike_by_T_colored.png'.")

    elapsed = time.time() - start_time  # ⏱️ 结束计时
    print(f"⏱️ Total calibration time: {elapsed:.2f} seconds")

    return params, ga, result

# —— 6. Script Entry ——
if __name__ == "__main__":
    df = (pd.read_excel(str(_PROJECT_ROOT / "02_jan_put.xlsx"))
            .rename(columns={"Strike_Put":"k",
                             "TTM_Put":"T",
                             "mid_price_Put":"opt_price"}))
    df["opt_price"] /= 100
    df["T"] /= 365
    calibrate(df, device="cuda")
