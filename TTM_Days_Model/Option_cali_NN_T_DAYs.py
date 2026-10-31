'''


2-Stage calibration (GA + L-BFGS-B) using **relative-RMSE**
Records the GA loss history, refines the solution with L-BFGS-B, and saves
calibrated parameters, the loss curve, and a market-versus-model scatter plot.
'''

import os as _os
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _Path(_os.environ.get("GPU_NN_PROJECT_ROOT", str(_REPO_ROOT / "Data_Results"))).expanduser()

import pathlib, joblib, numpy as np, pandas as pd, torch
import torch.nn as nn, pygad, multiprocessing as mp
import matplotlib.pyplot as plt
from  scipy.optimize import minimize

# ── paths & consts ─────────────────────────────────────────────────────────
PROJECT_DIR = pathlib.Path(str(_PROJECT_ROOT))
MODEL_W     = PROJECT_DIR / "model_5000_days.pt"
SCALERS_PKL = PROJECT_DIR / "scalers_5000_days.pkl"
T_LIST      = np.array([258, 100, 48, 20], dtype=float)
EPS         = 1e-4                       # small number for stability
history     = []                         # GA progress tracker

# ── NN helper ──────────────────────────────────────────────────────────────

def get_act(name):
    return dict(linear=nn.Identity(), none=nn.Identity(),
                relu=nn.ReLU(),  elu=nn.ELU(),
                leaky=nn.LeakyReLU(0.01), tanh=nn.Tanh())[name]


class GenericNet(nn.Module):
    def __init__(self, d_in, d_out, hidden, layers, act="elu"):
        super().__init__()
        if isinstance(hidden, int):
            hidden = [hidden] * layers
        net, inp = [], d_in
        for h in hidden:
            net += [nn.Linear(inp, h), get_act(act)]
            inp = h
        net.append(nn.Linear(inp, d_out))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


# ── utilities ──────────────────────────────────────────────────────────────

def load_nn(device="cpu"):
    saved = joblib.load(SCALERS_PKL)
    net   = GenericNet(len(saved["in_cols"]), 1, [64, 32, 32], 3, "elu").double().to(device)
    net.load_state_dict(torch.load(MODEL_W, map_location=device))
    net.eval()
    return net, saved["scX"], saved["scY"]


def make_inputs(df, θ):
    """Stack constant parameters with option-specific T, k."""
    a, b, c, d, kappa, *r = θ
    # map each T to its r_i
    r_vec = np.choose(df["T"].map({t: i for i, t in enumerate(T_LIST)}).astype(int),
                      r, mode='clip')
    base = np.repeat(np.array([a, b, c, d, kappa])[None, :], len(df), 0)
    return np.column_stack([base,  np.array(r_vec).reshape(-1, 1), df[["T", "k"]].values])


# ── loss factory ───────────────────────────────────────────────────────────

def build_rel_rmse(df, net, scX, scY, device):
    y = df["opt_price"].values

    def rel_rmse(θ):
        Xn = torch.tensor(scX.transform(make_inputs(df, θ))).double().to(device)
        with torch.no_grad():
            p_std = net(Xn).cpu().numpy()
        p = scY.inverse_transform(p_std).ravel()
        rel = (p - y) / (y + EPS)
        return float(np.sqrt((rel ** 2).mean()))

    return rel_rmse


# ── calibrator ─────────────────────────────────────────────────

def calibrate(df: pd.DataFrame, device: str = "cuda", pop: int = 400,
              gens: int = 4000, seed=None):
    """GA + L-BFGS-B calibration using relative-RMSE as the objective."""

    # 1) prepare net & loss
    net, scX, scY = load_nn(device)
    loss          = build_rel_rmse(df, net, scX, scY, device)
    fit_func      = lambda ga, sol, idx: -loss(sol)

    # 2) search space
    lows  = np.array([0, 0, 1e-4, 0.5, 0, -.25, -.25, -.25, -.25])
    highs = np.array([5, 5, 0.9999, 0.999, 10, .25, .25, .25, .25])
    space = [{"low": lo, "high": hi} for lo, hi in zip(lows, highs)]

    # 3) (re)initialise GA-history container
    history.clear()

    # 4) run GA
    ga = pygad.GA(
        fitness_func           = fit_func,
        num_genes              = 9,
        sol_per_pop            = pop,
        num_generations        = gens,
        num_parents_mating     = pop // 2,
        parent_selection_type  = "rank",
        crossover_type         = "two_points",
        mutation_type          = "adaptive",
        mutation_percent_genes = [40, 15],
        gene_space             = space,
        parallel_processing    = ["thread", max(1, mp.cpu_count() - 2)],
        random_seed            = seed,
        on_generation          = lambda ga: (
            history.append(-ga.best_solution()[1]),
            print(f"Gen {ga.generations_completed:4d} | best rel-RMSE {history[-1]:.4e}")
        )
    )

    ga.run()
    θ_ga, fit_ga, _ = ga.best_solution()
    print("\n🔍 GA finished.")
    print(f"Best rel-RMSE from GA: { -fit_ga :.4e}")

    # 5) local refinement (L-BFGS-B)
    res = minimize(
        loss, θ_ga, method="L-BFGS-B",
        bounds=list(zip(lows, highs)),
        options=dict(maxiter=2000, ftol=1e-12, gtol=1e-12, disp=True)
    )

    θ_opt  = res.x
    rel_rmse_opt = res.fun

    print("\n✅ Final optimisation finished.")
    print(f"Final relative-RMSE = {rel_rmse_opt :.4e}")

    # 6) dump parameters
    keys = ["a", "b", "c", "d", "kappa", "r1", "r2", "r3", "r4"]
    params = {k: round(v, 6) for k, v in zip(keys, θ_opt)}
    print("📌 Calibrated parameters:")
    for k, v in params.items():
        print(f"  {k:5s}= {v}")
    with open(PROJECT_DIR / "best_params.txt", "w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")

    # 7) save GA curve
    plt.figure(figsize=(8, 4))
    plt.plot(history, marker='o')
    plt.title("GA Calibration Progress (relative-RMSE)")
    plt.xlabel("Generation")
    plt.ylabel("Best relative-RMSE")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    curve_path = PROJECT_DIR / "rel_rmse_curve.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"📈 loss curve saved to {curve_path}")

    # 8) scatter plot – Market vs Model after calibration
    X_final = make_inputs(df, θ_opt)
    Xn_final = torch.tensor(scX.transform(X_final)).double().to(device)
    with torch.no_grad():
        preds_std = net(Xn_final).cpu().numpy()
    preds = scY.inverse_transform(preds_std).ravel()
    true_prices = df["opt_price"].values

    plt.figure(figsize=(7, 6))
    plt.scatter(true_prices, preds, marker='x', s=60, linewidths=1.5,
                color="#1f77b4", label="Model (Predicted)")
    plt.scatter(true_prices, true_prices, marker='o', s=40,
                facecolors='none', edgecolors='gray', alpha=0.8,
                label="Market (Reference)")
    plt.xlabel("Market Prices")
    plt.ylabel("Model Prices")
    plt.title("Market vs Model Prices After Calibration")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    scatter_path = PROJECT_DIR / "fit_after_calibration.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"📈 Calibration fit plot saved to {scatter_path}")

    return params, ga, res


# ── run as script ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = (
        pd.read_excel(PROJECT_DIR / "02_jan_put.xlsx")
          .rename(columns={"Strike_Put": "k", "TTM_Put": "T", "mid_price_Put": "opt_price"})
    )
    df["opt_price"] /= 100  # make sure option prices are in the right scale
    calibrate(df, device="cuda")
