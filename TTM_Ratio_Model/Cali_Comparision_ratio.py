# ─────────── nn_price_plot.py ───────────
"""
After running nn_calibrator.py and obtaining the optimal parameter vector
`theta*  = [a,b,c,d,kappa,r1,r2,r3,r4]`,
this script:
1.  Loads the trained neural network and the scalers.
2.  Uses theta* to generate model prices for every row in the input Excel file.
3.  Plots model prices (+) against market prices (o) exactly like the sample figure.
"""

import pathlib, joblib, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import sys
sys.path.append(r"D:\BaiduSyncdisk\Course_project\Math548\Option_Pricing548")

# --------------------------------------------------------------------------- #
# 1.  paths & constants (adjust if yours differ)
# --------------------------------------------------------------------------- #
PROJECT_DIR = pathlib.Path(r"C:\Users\26876\Desktop\Math548\Project\TTM_Ratio_Results")
MODEL_W     = PROJECT_DIR / "model_5000_ratio.pt"
SCALERS_PKL = PROJECT_DIR / "scalers_5000_ratio.pkl"

T_LIST = np.array([258, 100, 48, 20], dtype=float)/252          # MUST match training

# --------------------------------------------------------------------------- #
# 2.  neural-net helper
# --------------------------------------------------------------------------- #
import torch.nn as nn

class GenericNet(nn.Module):
    def __init__(self, dim_in, dim_out, hidden, layers, act="elu"):
        super().__init__()
        get_act = dict(
            linear=nn.Identity(), none=nn.Identity(),
            relu=nn.ReLU(), elu=nn.ELU(),
            leaky=nn.LeakyReLU(0.01), tanh=nn.Tanh()
        ).__getitem__
        if isinstance(hidden, int):
            hidden = [hidden] * layers
        blocks, in_d = [], dim_in
        for h in hidden:
            blocks += [nn.Linear(in_d, h), get_act(act)]
            in_d = h
        blocks += [nn.Linear(in_d, dim_out)]               # output is linear
        self.net = nn.Sequential(*blocks)

    def forward(self, x): return self.net(x)

def load_nn(device="cpu"):
    saved      = joblib.load(SCALERS_PKL)
    scX, scY   = saved["scX"], saved["scY"]
    in_cols    = saved["in_cols"]
    hidden     = [64, 32, 32]                             # same as trainer
    net = (GenericNet(len(in_cols), 1, hidden, layers=3, act="elu")
           .double().to(device))
    net.load_state_dict(torch.load(MODEL_W, map_location=device))
    net.eval()
    return net, scX, scY

# --------------------------------------------------------------------------- #
# 3.  build NN input matrix  X(theta)
# --------------------------------------------------------------------------- #
def make_inputs(df: pd.DataFrame, theta):
    """theta = [a,b,c,d,kappa,r1,r2,r3,r4]  (same order returned by calibrator)"""
    a, b, c, d, kappa, *rates = theta
    # choose r according to the maturity in each row
    T_array = np.array(T_LIST)
    nearest_idx = np.abs(df["T"].values[:, None] - T_array[None, :]).argmin(axis=1)
    r_vec = np.take(rates, nearest_idx)
    consts = np.repeat(np.array([a, b, c, d, kappa])[None, :], len(df), axis=0)
    return np.column_stack([consts,
                            r_vec.reshape(-1, 1),
                            df[["T", "k"]].values]).astype(np.float64)


# --------------------------------------------------------------------------- #
# 4.  main routine – pricing + plotting
# --------------------------------------------------------------------------- #
def price_and_plot(excel_path: str, theta_opt, device="cpu"):
    # 4.1 load data & NN
    df = (pd.read_excel(excel_path)
            .rename(columns={"Strike_Put": "k",
                             "TTM_Put"  : "T",
                             "mid_price_Put": "opt_price"}))
    df["opt_price"] /= 100
    df["T"] /= 252
    net, scX, scY = load_nn(device)

    # 4.2 model prices
    X = make_inputs(df, theta_opt)
    Xn = torch.tensor(scX.transform(X)).double().to(device)
    with torch.no_grad():
        preds = net(Xn).cpu().numpy()
    model_prices = scY.inverse_transform(preds).ravel()

    # 4.3 attach to dataframe
    df["model_price"] = model_prices

    # 4.3.5 print Strike, Market, Model
    for i, row in df.iterrows():
        print(f"Strike: {row['k']:.2f}, Market Price: {row['opt_price']:.4f}, Model Price: {row['model_price']:.4f}")

    # 🆕 4.4 计算误差
    market = df["opt_price"].values
    model = df["model_price"].values

    mse  = np.mean((market - model)**2)
    mae  = np.mean(np.abs(market - model))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((market - model) / np.clip(market, 1e-8, None))) * 100  # 避免0除

    # 打印误差
    print("\n--- Error Metrics ---")
    print(f"MSE  = {mse:.6f}")
    print(f"MAE  = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"MAPE = {mape:.2f}%")

    # 🆕 4.5 保存误差到 txt 文件
    error_text = (
        f"Error Metrics:\n"
        f"MSE  = {mse:.6f}\n"
        f"MAE  = {mae:.6f}\n"
        f"RMSE = {rmse:.6f}\n"
        f"MAPE = {mape:.2f}%\n"
    )
    with open(PROJECT_DIR / 'put_price_errors.txt', 'w') as f:
        f.write(error_text)

    # 4.6 plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["k"], df["opt_price"], marker="o", facecolors='none',
               edgecolors='tab:orange', label="Market Price")
    ax.scatter(df["k"], df["model_price"], marker="+",
               color='tab:blue', label="Model Price")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Put Price")
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / 'put_price_fit.png', dpi=300)
    plt.show()

    # 🆕 4.7 可选：保存df到excel
    df.to_excel(PROJECT_DIR / 'put_price_fit.xlsx', index=False)

    return df


# --------------------------------------------------------------------------- #
# 5.  example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # theta_opt should be the array printed by nn_calibrator.calibrate(...)
    theta_opt = [
        0.05732,  # a
        4.57577,  # b
        0.707823,  # c
        0.926759,  # d
        3.914699,  # kappa
        0.046908,  # r1
        0.032521,  # r2
        0.014253,  # r3
        -0.009872,  # r4
    ]


    price_and_plot(
        excel_path=r"C:\Users\26876\Desktop\Math548\Project\02_jan_put.xlsx",
        theta_opt=theta_opt,
        device="cuda"                                                # or "cpu"
    )
