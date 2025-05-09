# =============================================
# Proposition‑2 GPU Calibration (fully vectorised)
# =============================================
# * φ(l) 预载一次 → 所有 strike 同时定价
# * mpmath 积分改为 GPU Simpson (无 Python 循环)
# * (T,r) 复用缓存
# ---------------------------------------------
# Tested on CUDA‑11 + CuPy‑12.x
# ---------------------------------------------

import os, sys, time, math
import numpy as np
import cupy as cp
import mpmath
from cupyx.scipy import special as cpsp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pygad
from multiprocessing import cpu_count

# ──────────────────────────────────────────────
# global const
# ──────────────────────────────────────────────
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
mpmath.mp.dps = 10
T_LIST = np.array([258, 100, 48, 20]) / 365
DTYPE  = cp.float32              # ← 若需更高精度可改 float64

# grids & Simpson weights
MS, XSZ, ELLZ = 8_001, 801, 5001
L_GRID = cp.linspace(1e-8, 5_000, 300, dtype=DTYPE)
_w_L   = cp.asarray([4 if i%2 else 2 for i in range(len(L_GRID))], dtype=DTYPE); _w_L[0] = _w_L[-1] = 1

_s_grid0  = cp.linspace(0, 1, MS,  dtype=DTYPE)
_w_s0     = cp.asarray([4 if i%2 else 2 for i in range(MS)], dtype=DTYPE); _w_s0[0]=_w_s0[-1]=1
_ds0      = float(_s_grid0[1]-_s_grid0[0])

_x_grid, _w_x = cp.linspace(-30, 30, XSZ, dtype=DTYPE), cp.asarray([4 if i%2 else 2 for i in range(XSZ)], dtype=DTYPE); _w_x[0]=_w_x[-1]=1
_dx       = float(_x_grid[1]-_x_grid[0])

_ell, _w_ell = cp.linspace(0, 30, ELLZ, dtype=DTYPE), cp.asarray([4 if i%2 else 2 for i in range(ELLZ)], dtype=DTYPE); _w_ell[0]=_w_ell[-1]=1
_dell     = float(_ell[1]-_ell[0])

# Simpson helper
SIMPS = lambda f, w, h: cp.sum(f*w, axis=-1) * h/3

# ──────────────────────────────────────────────
# φ(l)  —— GPU kernel (vectorised l)
# ──────────────────────────────────────────────

def phi_gpu(l_vec, a,b,c,d,kappa,r,t, spot, t0=0., Delta=0.083, alpha=1.78, zeta=0.01):
    # H(u)
    s = _s_grid0*(t-t0)+t0; ds=_ds0*(t-t0)
    u = t - s; tau0=(1-d)/(kappa+1e-8); u_plus=u+Delta
    gd1=cpsp.gamma(d+1); gdm1=cpsp.gamma(d-1)
    H=cp.zeros_like(u)
    m1, m2 = u_plus<tau0, (u<tau0)
    H[m1]=(u_plus[m1]**d-u[m1]**d)/(Delta*gd1)
    H[m2 & ~m1]=((tau0**d-u[m2 & ~m1]**d)/(Delta*gd1)+
                 (cp.exp(-kappa*u_plus[m2 & ~m1]+1-d)-1)/((kappa**d)*Delta*(1-d)**(2-d)*gdm1))
    H[~m1 & ~m2]=-(cp.exp(-kappa*u_plus[~m1 & ~m2]+1-d)*(cp.exp(kappa*Delta)-1))/((kappa**d)*Delta*(1-d)**(2-d)*gdm1)

    J = spot**2 - a*cpsp.gamma(1-c)/cp.power(b,c-1)*SIMPS(H*_w_s0,1.,ds) + r
    lH = H[:,None]*l_vec[None,:]
    log_phi=(a*cpsp.gamma(-c)*b**c)*(((b-1j*lH)/b)**c-1)
    log_int = cp.sum(log_phi * _w_s0[:, None], axis=0) * (ds / 3.0)

    phiZ = cp.exp(-cp.abs(_ell) ** alpha * (t - t0))  # (ELLZ,)
    integrand_g = cp.cos(_ell[:, None] * _x_grid[None, :]) \
                  * phiZ[:, None] * _w_ell[:, None]  # (ELLZ, XSZ)
    g = cp.sum(integrand_g, axis=0) * (_dell / 3.0)  # → (XSZ,)

    # ── 第二重积分（对 x） : 形状全部对齐为 (len(l_vec), XSZ)
    phi1 = float(cp.exp(-1))
    A = (phi1 ** Delta - 1) / math.log(phi1)
    psi = cp.exp(1j * l_vec[:, None] *
                 ((zeta / Delta * A) * cp.cos(_x_grid)[None, :] + zeta))  # (M, XSZ)

    integrand_x = psi * g[None, :] * _w_x[None, :]  # (M, XSZ)
    dbl = cp.sum(integrand_x, axis=1) * (_dx / 3.0)  # → (M,)

    return (1 / cp.pi) * cp.exp(1j * l_vec * J + log_int) * dbl

# ──────────────────────────────────────────────
# Batch‑pricing：一次性算完所有 strike   (GPU Simpson in l)
# ──────────────────────────────────────────────

def price_put_batch(k_vec, phi_vals):
    l, w = L_GRID, _w_L
    keys = [int(round(float(k) * SCALE)) for k in cp.asnumpy(k_vec)]
    gamma_mat = cp.stack([GAMMA_TAB[k] for k in keys])        # (N,M)      # (N,M) complex64

    exp_term = k_vec[:, None] * cp.exp(-1j * (k_vec**2)[:, None] * l[None, :])
    integrand = cp.real((exp_term + gamma_mat) *
                        (phi_vals[None, :] / (1j * l[None, :])))
    I = SIMPS(integrand, w[None, :], float(l[1] - l[0]))      # (N,)
    return k_vec / 2 - I / cp.pi


# γ(k) 预计算（在 CPU 上一次完成）
GAMMA_TAB = {}

SCALE = 100_000                      # 1e5 → 0.00001 精度

def precompute_gamma(k_unique):
    l_cpu = L_GRID.get()
    for k in k_unique:
        key = int(round(float(k) * SCALE))     # 0.135 → 13500
        if key in GAMMA_TAB:
            continue
        vals = [complex(gamma_term_factory(float(k))(float(x))) for x in l_cpu]
        GAMMA_TAB[key] = cp.asarray(vals, dtype=cp.complex64)


# γ factory（单点）

def gamma_term_factory(K):
    cache = {}
    def γ(l):
        if l not in cache:
            γ_u = mpmath.gammainc(1.5, 1j*K**2*l, mpmath.inf)
            cache[l]=(mpmath.sqrt(mpmath.pi)/2-γ_u)/mpmath.sqrt(1j*l)
        return cache[l]
    return γ

# ──────────────────────────────────────────────
# Loss (vectorised)
# ──────────────────────────────────────────────

def mse_loss(df, theta):
    a,b,c,d,kappa,r1,r2,r3,r4 = theta
    rates = np.array([r1,r2,r3,r4])
    spot  = df.spot_price.iloc[0]

    # group by (T,r)
    groups = {}
    for row in df.itertuples(index=False):
        T,k,p=row.T,row.k,row.opt_price
        r=rates[np.argmin(np.abs(T-T_LIST))]
        groups.setdefault((T,r),[]).append((k,p))

    errs=[]
    for (T,r),kp_list in groups.items():
        k_arr = cp.asarray([kp[0] for kp in kp_list], dtype=DTYPE)
        p_true= cp.asarray([kp[1] for kp in kp_list], dtype=DTYPE)

        phi_vals = phi_gpu(L_GRID, a,b,c,d,kappa,r,T, spot)
        p_pred = price_put_batch(k_arr, phi_vals)
        errs.append(cp.mean((p_pred-p_true)**2))

    return float(cp.mean(cp.stack(errs)))

# ──────────────────────────────────────────────
# GA fitness
# ──────────────────────────────────────────────

global_df=None

def fitness_func(ga, sol, idx):
    return -mse_loss(global_df, sol)

# ──────────────────────────────────────────────
# Calibration main
# ──────────────────────────────────────────────
def show_progress(ga_inst):
    """每一代结束后记录并打印最优 MSE"""
    best = -ga_inst.best_solution()[1]
    history.append(best)
    print(f"Gen {ga_inst.generations_completed:4d} | best MSE = {best:.6e}",
          flush=True)

import matplotlib.pyplot as plt      # 你的文件里已 import，忽略
import matplotlib.cm as cm

history = []  # 放在全局

def calibrate(df, pop_size=500, gens=800, seed=2025):
    global global_df
    global_df=df
    start_time = time.time()


    lows = np.array([0.0, 0.0 , 0.8575, 0.5, 0.0, -0.25, -0.25, -0.25, -0.25])
    highs = np.array([3.0, 4.0, 0.8575, 0.999, 10.0, 0.25, 0.25, 0.25, 0.25])
    gene_space=[{"low":lo,"high":hi} for lo,hi in zip(lows,highs)]

    ga = pygad.GA(
        num_generations=gens,
        sol_per_pop=pop_size,  # ← 总个体数
        num_parents_mating=pop_size // 2,  # ← 父代数
        num_genes=len(gene_space),
        gene_space=gene_space,
        fitness_func=fitness_func,
        random_seed=seed,
        parent_selection_type="tournament",
        crossover_type="two_points",
        mutation_type="adaptive",
        mutation_percent_genes=[60, 20] ,
        on_generation = show_progress,
        keep_parents = 2
    )

    ga.run()
    sol, fit, _ = ga.best_solution()  # fit 为负的 MSE
    best_ga = -fit  # ← 存下 GA 最优 MSE
    res = minimize(lambda x: mse_loss(df, x), sol,
                   method="L-BFGS-B",
                   bounds=list(zip(lows, highs)),
                   options={"maxiter": 3000})

    theta_fin = res.x
    best_fin = res.fun
    print("✅ Calibration Done,  final MSE (LBFGS) =", best_fin)

    # 绘制 GA 收敛曲线
    plt.figure(figsize=(6, 4))
    plt.plot(history, marker="o")
    plt.title("GA Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r"C:\Users\26876\Desktop\Math548\Project\Not_NN_Calibration\progress_cf.png")
    plt.show()
    plt.close()

    a, b, c, d, kappa, r1, r2, r3, r4 = theta_fin
    rates = np.array([r1, r2, r3, r4])
    spot = df.spot_price.iloc[0]

    # —— 先把 df 按 (T,r) 分组、批量算 φ(l) ——
    preds = np.empty(len(df), dtype=float)
    for (T, group) in df.groupby("T"):
        r = rates[np.argmin(np.abs(T - T_LIST))]  # 对应无风险利率
        phi_vals = phi_gpu(L_GRID, a, b, c, d, kappa, r, T, spot)  # φ(l) 只跟 (T,r) 有关

        k_vec = cp.asarray(group.k.values, dtype=DTYPE)
        preds[group.index] = cp.asnumpy(price_put_batch(k_vec, phi_vals))

    # ―― ①  Market-vs-Model ――
    true_prices = df.opt_price.values           # ⬅ 市场价格
    model_prices = preds                        # ⬅ 模型价格 (已按行对应)
    plt.figure(figsize=(6, 6))
    plt.scatter(true_prices, model_prices, marker="x", label="Model")

    plt.xlabel("Market Price")
    plt.ylabel("Model Price")
    plt.title("Market vs Model Prices")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"C:\Users\26876\Desktop\Math548\Project\Not_NN_Calibration\market_vs_model.png")  # 若只想显示改成 plt.show()
    plt.close()

    # ―― ②  Strike–Price 曲线按到期 T 上色 ――
    plt.figure(figsize=(7, 5))
    unique_T = np.sort(df["T"].unique())
    cmap_m, cmap_md = plt.cm.Oranges, plt.cm.Blues

    for i, T in enumerate(unique_T):
        msk = df["T"] == T
        c_m = cmap_m(0.3 + 0.6 * i / (len(unique_T) - 1))
        c_md = cmap_md(0.3 + 0.6 * i / (len(unique_T) - 1))

        # 市场
        plt.scatter(df.loc[msk, "k"], df.loc[msk, "opt_price"],
                    marker="o", facecolors="none",
                    edgecolors=c_m, label=f"Market T={T:.3f}")
        # 模型
        plt.scatter(df.loc[msk, "k"], model_prices[msk],
                    marker="+", color=c_md, label=f"Model  T={T:.3f}")

    plt.xlabel("Strike")
    plt.ylabel("Put Price")
    plt.title("Price vs Strike (by T)")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(r"C:\Users\26876\Desktop\Math548\Project\Not_NN_Calibration\strike_curves.png")  # 或者 plt.show()
    plt.show()
    plt.close()

    elapsed = time.time() - start_time  # ⏱️ 结束计时
    print(f"⏱️ Total calibration time: {elapsed:.2f} seconds")

    # -------- 返回 ----------
    return theta_fin, best_ga, best_fin

# ──────────────────────────────────────────────
# main demo
# ──────────────────────────────────────────────
if __name__=="__main__":
    df = pd.read_excel(r"C:\Users\26876\Desktop\Math548\Project\02_jan_put.xlsx")
    df=df.rename(columns={"Strike_Put":"k","TTM_Put":"T","mid_price_Put":"opt_price"})
    df["opt_price"]/=100; df["T"]/=365; df["spot_price"]=0.1793

    # 预计算 γ(l) 表（一次即可）
    precompute_gamma(df.k.unique())
    '''
    Test
    start=time.time()
    base_loss=mse_loss(df, np.array([1,2,0.5,0.5,1,0.01,0.01,0.01,0.01]))
    print("baseline loss", base_loss, "sec", time.time()-start)
     '''

    theta,best_ga,best_final=calibrate(df)
    param_names = ["a", "b", "c", "d", "kappa", "r1", "r2", "r3", "r4"]
    print("\n✅ Final Calibrated Parameters:")
    for name, val in zip(param_names, theta):
        print(f"  {name:<6s} = {val: .6f}")
    print(f"\nGA best MSE   = {best_ga:.6e}")
    print(f"L-BFGS MSE     = {best_final:.6e}")

