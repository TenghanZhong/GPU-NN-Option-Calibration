# ──────────────────────────────────────────────────────────────────────────────
# 改进版 Option_cali_NN_genetic_2stage_relative.py
# -----------------------------------------------------------------------------
"""
2‑Stage Calibration with **Relative MSE (rMSE)** loss
1. Genetic Algorithm (GA) → global search (PyGAD ≥ 3.x)
2. L‑BFGS‑B → local refinement
   ‑ rMSE = mean( ((pred − true) / (true + EPS))² )
   ‑ EPS avoids division‑by‑zero for tiny option prices
Compatible with CUDA inference, multiprocessing for GA fitness, and
produces progress & diagnostic plots.
"""

import pathlib, joblib, numpy as np, pandas as pd, torch
import torch.nn as nn
import pygad, multiprocessing as mp, sys, time
import matplotlib.pyplot as plt, matplotlib.cm as cm
from scipy.optimize import minimize

# ─── 1. Paths & Constants ─────────────────────────────────────────────────────
PROJECT_DIR = pathlib.Path(r"C:\Users\26876\Desktop\Math548\Project\TTM_Ratio_Results")
MODEL_W     = PROJECT_DIR / "model_5000_ratio.pt"
SCALERS_PKL = PROJECT_DIR / "scalers_5000_ratio.pkl"
T_LIST      = np.array([258, 100, 48, 20], dtype=float) / 252  # trading‑days → years
EPS         = 1e-8      # small number to stabilise relative error when true≈0

history: list[float] = []

# ─── 2. NN Definition ─────────────────────────────────────────────────────────

def get_act(name: str):
    name = name.lower()
    if name in ("linear", "none"): return nn.Identity()
    if name == "relu":   return nn.ReLU()
    if name == "elu":    return nn.ELU()
    if name == "leaky":  return nn.LeakyReLU(0.01)
    if name == "tanh":   return nn.Tanh()
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

# ─── 3. Helpers ───────────────────────────────────────────────────────────────

def load_nn(device="cpu"):
    saved = joblib.load(SCALERS_PKL)
    scX, scY, in_cols = saved["scX"], saved["scY"], saved["in_cols"]
    net = (GenericNet(len(in_cols), 1, [64, 32, 32], 3, "elu", "linear")
           .double().to(device))
    net.load_state_dict(torch.load(MODEL_W, map_location=device))
    net.eval()
    return net, scX, scY, in_cols


def make_inputs(df: pd.DataFrame, theta: np.ndarray, *, in_cols):
    """Create input matrix with calibrated parameters inserted."""
    a, b, c, d, kappa, *rates = theta
    nearest_idx = np.abs(df["T"].values[:, None] - T_LIST[None, :]).argmin(axis=1)
    r_vec = np.take(rates, nearest_idx)

    tmp = df.copy()
    tmp["a"], tmp["b"], tmp["c"], tmp["d"], tmp["kappa"] = a, b, c, d, kappa
    tmp["r"] = r_vec
    tmp = tmp.rename(columns={"T": "t", "k": "K"})  # align training col names
    return tmp[in_cols].values.astype(np.float64)


# ─── 4. Fitness Function ──────────────────────────────────────────────────────

def fitness_factory(df, net, scX, scY, in_cols, device):
    def r_mse(theta):
        X  = make_inputs(df, theta, in_cols=in_cols)
        Xn = torch.tensor(scX.transform(X)).double().to(device)
        with torch.no_grad():
            preds = net(Xn).cpu().numpy().ravel()
        preds_orig = scY.inverse_transform(preds.reshape(-1,1)).ravel()
        true = df["opt_price"].values
        rel_err = (preds_orig - true) / (true + EPS)
        return float(np.mean(rel_err**2))

    def fitness_func(_, solution, __):  # GA maximises → return negative rMSE
        return -r_mse(solution)

    return fitness_func


def record_history(ga_instance):
    global history
    best_rMSE = -ga_instance.best_solution()[1]
    history.append(best_rMSE)
    print(f"Generation {ga_instance.generations_completed:4d} | Best relative MSE: {best_rMSE:.6e}")

# ─── 5. Calibration Entry ─────────────────────────────────────────────────────

def calibrate(df: pd.DataFrame, device="cpu", *,
              pop_size=600, num_generations=4000, seed=2024):
    start_time = time.time()

    net, scX, scY, in_cols = load_nn(device)
    fitness_func = fitness_factory(df, net, scX, scY, in_cols, device)

    lows  = np.array([0.0, 0.0, 1e-4, 0.50, 0.0, -0.25, -0.25, -0.25, -0.25])
    highs = np.array([5.0, 5.0, 0.9999, 0.999, 10.0, 0.25, 0.25, 0.25, 0.25])
    bounds = list(zip(lows, highs))
    gene_space = [{'low': lo, 'high': hi} for lo, hi in bounds]
    n_cores = max(1, mp.cpu_count() - 2)

    history.clear()

    ga = pygad.GA(num_generations=num_generations,
                  num_parents_mating=int(pop_size * 0.5),
                  fitness_func=fitness_func,
                  sol_per_pop=pop_size,
                  num_genes=len(gene_space),
                  gene_space=gene_space,
                  parent_selection_type="rank",
                  crossover_type="two_points",
                  mutation_type="adaptive",
                  mutation_percent_genes=[40, 15],
                  parallel_processing=["thread", n_cores],
                  random_seed=seed,
                  on_generation=record_history)
    ga.run()

    ga_solution, ga_fitness, _ = ga.best_solution()
    print("\n🔍 GA finished.")
    print(f"GA best relative MSE: {-ga_fitness:.6e}")

    # local refinement ---------------------------------------------------------
    def r_mse(theta):
        X  = make_inputs(df, theta, in_cols=in_cols)
        Xn = torch.tensor(scX.transform(X)).double().to(device)
        with torch.no_grad():
            preds = net(Xn).cpu().numpy().ravel()
        preds_orig = scY.inverse_transform(preds.reshape(-1,1)).ravel()
        true = df["opt_price"].values
        rel_err = (preds_orig - true) / (true + EPS)
        return float(np.mean(rel_err**2))

    result = minimize(r_mse, ga_solution, method="L-BFGS-B",
                      bounds=bounds,
                      options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': True})

    solution = result.x
    final_rMSE = result.fun

    keys = ["a","b","c","d","kappa","r1","r2","r3","r4"]
    params = {k: round(v, 6) for k, v in zip(keys, solution)}
    print("\n✅ Final optimisation finished.")
    print("Final relative MSE =", final_rMSE)
    print("📌 Calibrated parameters:")
    for k, v in params.items():
        print(f"  {k:5s}= {v}")

    with open(PROJECT_DIR / "best_params_relative_MSE.txt", "w") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")

    # ── Progress plot ─────────────────────────────────────────────────────────
    plt.figure(figsize=(8,4))
    plt.plot(history, marker='o')
    plt.title("GA Calibration Progress (rMSE)")
    plt.xlabel("Generation"); plt.ylabel("Best rMSE")
    plt.grid(True); plt.tight_layout()
    plt.savefig(PROJECT_DIR / "progress_relative_MSE.png"); plt.close()

    # ── Diagnostics: market vs model ─────────────────────────────────────────
    X_final = make_inputs(df, solution, in_cols=in_cols)
    Xn_final = torch.tensor(scX.transform(X_final)).double().to(device)
    with torch.no_grad():
        preds_final_std = net(Xn_final).cpu().numpy().ravel()
    preds_final = scY.inverse_transform(preds_final_std.reshape(-1,1)).ravel()
    true_prices = df["opt_price"].values

    # (1) Scatter true vs predicted -------------------------------------------
    plt.figure(figsize=(7,6))
    plt.scatter(true_prices, preds_final, marker='x', s=60, linewidths=1.5,
                color="#1f77b4", label="Model (Predicted)")
    plt.scatter(true_prices, true_prices, marker='o', s=40, facecolors='none',
                edgecolors='gray', alpha=0.8, label="Market (Reference)")
    plt.xlabel("Market Prices"); plt.ylabel("Model Prices")
    plt.title("Market vs Model (rMSE-calibrated)")
    plt.grid(True, linestyle="--", alpha=0.5); plt.legend(); plt.tight_layout()
    plt.show()

    # (2) Price vs strike coloured by maturity --------------------------------
    plt.figure(figsize=(7,5))
    unique_T = np.sort(df["T"].unique())
    cmap_market, cmap_model = cm.get_cmap('Oranges'), cm.get_cmap('Blues')
    for i, t in enumerate(unique_T):
        mask = df["T"] == t
        label_T = f"T = {t:.3f}y"
        c_mkt = cmap_market(0.4 + 0.6 * i / max(1, len(unique_T) - 1))
        c_mod = cmap_model(0.4 + 0.6 * i / max(1, len(unique_T) - 1))
        plt.scatter(df.loc[mask, "k"], df.loc[mask, "opt_price"],
                    marker='o', facecolors='none', edgecolors=c_mkt,
                    label=f"Market {label_T}")
        plt.scatter(df.loc[mask, "k"], preds_final[mask.values],
                    marker='+', color=c_mod, label=f"Model {label_T}")
    plt.xlabel("Strike Price (k)"); plt.ylabel("Put Option Price")
    plt.title("Put Price vs Strike (by Maturity, Colored)(rMSE-calibrated)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=7, ncol=2); plt.tight_layout()
    plt.savefig(PROJECT_DIR / "price_vs_strike_by_T_colored_relative_MSE.png", dpi=150)
    plt.show()
    print("📈 Price vs Strike plot saved to 'price_vs_strike_by_T_colored_relative_MSE.png'.")

    elapsed = time.time() - start_time
    print(f"⏱️ Total calibration time: {elapsed:.2f} seconds")
    return params, ga, result

# ─── 6. Script Entry ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = (pd.read_excel(r"C:\Users\26876\Desktop\Math548\Project\02_jan_put.xlsx")
            .rename(columns={"Strike_Put":"k", "TTM_Put":"T", "mid_price_Put":"opt_price"}))
    df["opt_price"] /= 100
    df["T"] /= 252
    calibrate(df, device="cuda")
