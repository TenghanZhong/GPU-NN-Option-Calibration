import os as _os
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _Path(_os.environ.get("GPU_NN_PROJECT_ROOT", str(_REPO_ROOT / "Data_Results"))).expanduser()

import numpy as np
import cupy as cp
import mpmath
from cupyx.scipy import special as cpsp
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pathlib
import pandas as pd
from scipy.optimize import minimize
from multiprocessing import cpu_count
import pygad
import sys

cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

# 设置 mpmath 精度
mpmath.mp.dps = 10  # 已调高精度以避免积分震荡的 nan 问题
T_LIST = np.array([258, 100, 48, 20], dtype=float) / 365


# ── Proposition 2 GPU 特征函数 φ ──

def simpson_weights(n, dtype=cp.float64):
    if (n - 1) % 2:
        raise ValueError("需要偶数个点")
    w = cp.ones(n, dtype)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return w

# ── 模块顶端预计算 ──
MS = 12001
XSZ = 801
ELLZ = 4001
L_GRID = np.linspace(1e-8, 5000, 300)

# 0–1 线性网格，后面根据 t 缩放到 [t0, t]
_s_grid0 = cp.linspace(0.0, 1.0, MS)  # 基准 s 网格
_w_s0 = simpson_weights(MS)
_ds0 = float(_s_grid0[1] - _s_grid0[0])

# x_grid、ell 网格也只算一次
_x_grid, _w_x = cp.linspace(-30, 30, XSZ), simpson_weights(XSZ)
_dx = float(_x_grid[1] - _x_grid[0])
_ell, _w_ell = cp.linspace(0, 30, ELLZ), simpson_weights(ELLZ)
_dell = float(_ell[1] - _ell[0])


def proposition2_cf_37_gpu_single_param(
    l_vec,
    a,
    b,
    c,
    d,
    kappa,
    r,
    t,
    t0=0.0,
    Delta=0.083,
    spot_price=0.1793,
    alpha=1.78,
    zeta=0.01,
):
    # —— 缩放基准 s 网格到 [t0, t] 区间 ——
    s_grid = _s_grid0 * (t - t0) + t0
    ds = _ds0 * (t - t0)
    w_s = _w_s0

    # —— 计算 H(u) ——
    u = t - s_grid
    tau0 = (1 - d) / (kappa + 1e-8)  # 防止除零
    u_plus = u + Delta
    gd1 = cpsp.gamma(d + 1)
    gdm1 = cpsp.gamma(d - 1)

    H = cp.zeros_like(u)
    m1 = u_plus < tau0
    m2 = (u < tau0) & ~m1
    m3 = ~m1 & ~m2

    H[m1] = (u_plus[m1] ** d - u[m1] ** d) / (Delta * gd1)
    H[m2] = (
        (tau0 ** d - u[m2] ** d) / (Delta * gd1)
        + (cp.exp(-kappa * u_plus[m2] + 1 - d) - 1)
        / ((kappa ** d) * Delta * (1 - d) ** (2 - d) * gdm1)
    )
    H[m3] = -(
        cp.exp(-kappa * u_plus[m3] + 1 - d)
        * (cp.exp(kappa * Delta) - 1)
    ) / ((kappa ** d) * Delta * (1 - d) ** (2 - d) * gdm1)

    # —— 计算 J ——
    xi1 = a * cpsp.gamma(1 - c) / cp.power(b, c - 1)
    J = spot_price ** 2 - xi1 * cp.sum(H * w_s) * (ds / 3.0) + r

    # —— φ 的第一个积分（对 u） ——
    lH = l_vec[None, :] * H[:, None]
    log_phi = (a * cpsp.gamma(-c) * b ** c) * (((b - 1j * lH) / b) ** c - 1)
    log_int = cp.sum(log_phi * w_s[:, None], axis=0) * (ds / 3.0)

    # —— 计算 g(x) ——
    phiZ = cp.exp(-cp.abs(_ell) ** alpha * (t - t0))
    g = (
        cp.sum(
            cp.cos(_ell[:, None] * _x_grid[None, :])
            * phiZ[:, None]
            * _w_ell[:, None],
            axis=0,
        )
        * (_dell / 3.0)
    )

    # —— φ 的第二个积分（对 x） ——
    phi1 = float(cp.exp(-1))
    A = (phi1 ** Delta - 1) / np.log(phi1)
    psi = cp.exp(
        1j
        * l_vec[:, None]
        * ((zeta / Delta * A) * cp.cos(_x_grid)[None, :] + zeta)
    )
    dbl = (
        cp.sum(psi * g[None, :] * _w_x[None, :], axis=1) * (_dx / 3.0)
    )

    # —— 最终 φ 值 ——
    return (1 / cp.pi) * cp.exp(1j * l_vec * J + log_int) * dbl


# ── γ 函数构造器 ──

def gamma_term_factory(K):
    cache = {}

    def γ(l):
        if l not in cache:
            γ_u = mpmath.gammainc(1.5, 1j * K ** 2 * l, mpmath.inf)
            cache[l] = (
                (mpmath.sqrt(mpmath.pi) / 2 - γ_u) / mpmath.sqrt(1j * l)
            )
        return cache[l]

    return γ


# ── GPU φ 缓存结构 ──

class PhiStore:
    """批量缓存 φ(l; T,r)，并一次性将计算结果从 GPU 转回 CPU"""

    def __init__(self, theta, spot_price, batch=64):
        self.a, self.b, self.c, self.d, self.kappa, self.r, self.t = theta
        self.spot_price = spot_price
        self.batch = batch
        self.buf = []  # 待计算的 l 值
        self.cache = {}

    def __call__(self, l):
        key = round(float(l), 10)
        if key not in self.cache:
            self.buf.append(key)

            # 情况 1：够批量 —— 一次性算
            if len(self.buf) >= self.batch:
                self._flush()
            # 情况 2：不够批量，但 caller 此刻就要用 —— 现在就算
            if key not in self.cache:  # 还没算出来就立即 flush
                self._flush()

        return self.cache[key]

    # Transfer the computed batch from GPU to CPU
    def _flush(self):
        if not self.buf:
            return
        l_arr = cp.asarray(self.buf, dtype=cp.float64)
        vals_gpu = proposition2_cf_37_gpu_single_param(
            l_arr,
            self.a,
            self.b,
            self.c,
            self.d,
            self.kappa,
            self.r,
            self.t,
            spot_price=self.spot_price,
        )
        vals_np = vals_gpu.get()  # ⬅️ 一次性从 GPU 拷贝到 CPU
        for k, φ in zip(self.buf, vals_np):
            self.cache[k] = complex(φ)  # 写入缓存
        self.buf.clear()

    flush = _flush  # 兼容旧接口
# ── 单个 Put 期权定价 ──
def price_put_single(K, gamma_func, phi_store, L_max=5000, rel_eps=1e-8):
    def integrand(l):
        φ = phi_store(l)
        exp_term = K * mpmath.exp(-1j * K**2 * l)
        return mpmath.re((exp_term + gamma_func(l)) * (φ / (1j * l)))

    price = float(K/2 - mpmath.quad(integrand, [1e-8, L_max], rel=rel_eps) / mpmath.pi)
    return price

# ── MSE 损失函数 ──
def mse_loss1(df, theta):
    a, b, c, d, kappa, r1, r2, r3, r4 = theta
    rates = np.array([r1, r2, r3, r4])
    spot  = df.spot_price.iloc[0]

    gamma_cache = {}

    # 创建 φ 批处理缓存器
    phi_store = PhiStore([a, b, c, d, kappa, None, None], spot, batch=128)

    # 第一步：批量收集所有 l 值（在定价中用到）
    L_vals_needed = set()
    meta = []

    for row in df.itertuples(index=False):
        T, k = row.T, row.k
        idx = np.abs(T - T_LIST).argmin()
        r = rates[idx]

        # 暂存每行需要的参数
        meta.append((T, r, k, row.opt_price))

        # 提前缓存 φ 所需的 l-grid（price_put_single 的内部积分）
        # 用一个典型的 l_grid 模拟：
        for l in L_GRID:
            L_vals_needed.add(round(float(l), 10))

    # 第二步：设置固定 T/r → 批量预先计算 φ
    for T, r in sorted(set((m[0], m[1]) for m in meta)):
        phi_store.r = r
        phi_store.t = T
        phi_store.buf = list(L_vals_needed)
        phi_store.flush()

    # 第三步：定价（逐个点，用已缓存 φ，加速）
    def gamma_cached(k):
        if k not in gamma_cache:
            gamma_cache[k] = gamma_term_factory(k)
        return gamma_cache[k]

    errs = []
    for T, r, k, opt_price in meta:
        phi_store.r = r
        phi_store.t = T
        γ = gamma_cached(k)
        price = price_put_single(k, γ, phi_store)
        errs.append(price - opt_price)

    return float(np.mean(np.square(errs)))

def mse_loss(df, theta):
    a, b, c, d, kappa, r1, r2, r3, r4 = theta
    rates = np.array([r1, r2, r3, r4])
    spot = df.spot_price.iloc[0]

    gamma_cache = {}
    meta = []

    # 收集 meta 信息和所有唯一 (T, r)
    unique_pairs = set()
    for row in df.itertuples(index=False):
        T, k, opt_price = row.T, row.k, row.opt_price
        idx = np.abs(T - T_LIST).argmin()
        r = rates[idx]
        meta.append((T, r, k, opt_price))
        unique_pairs.add((T, r))

    # 预先构造所有需要的 φ 存储器 + 批量缓存 φ
    L_vals = [round(float(l), 10) for l in L_GRID]
    phi_dict = {}

    for T, r in unique_pairs:
        store = PhiStore([a, b, c, d, kappa, r, T], spot, batch=len(L_vals))
        store.buf = L_vals.copy()
        store.flush()
        phi_dict[(T, r)] = store

    # γ 缓存函数
    def gamma_cached(k):
        if k not in gamma_cache:
            gamma_cache[k] = gamma_term_factory(k)
        return gamma_cache[k]

    # 计算误差（直接复用已缓存 φ）
    errs = []
    for T, r, k, opt_price in meta:
        φ_store = phi_dict[(T, r)]
        γ = gamma_cached(k)
        price = price_put_single(k, γ, φ_store)
        errs.append(price - opt_price)

    return float(np.mean(np.square(errs)))


def record_history(ga_instance):
    global history
    best_fitness = -ga_instance.best_solution()[1]
    history.append(best_fitness)
    print(f"Gen {ga_instance.generations_completed:4d} | Best MSE {best_fitness:.6e}", flush=True)
    sys.stdout.flush()  # 强制刷新输出缓存

# ── Fitness 工厂 ──
def make_fitness_function(df):
    def fitness_func(ga_instance, sol, idx):
        return -mse_loss(df, sol)
    return fitness_func
# 顶层全局变量
global_df = None

def fitness_func(ga_instance, sol, idx):
    print(f"Running fitness for solution {idx}")
    sys.stdout.flush()
    return -mse_loss(global_df, sol)
history = []
def on_gen(ga_inst):
    best = -ga_inst.best_solution()[1]
    history.append(best)
    print(f"Gen {ga_inst.generations_completed:4d} | Best MSE {best:.6e}", flush=True)
    sys.stdout.flush()

# ── 校准主函数 ──
def calibrate(df, pop_size=10, num_generations=10, seed=2025):
    global global_df
    global_df = df

    print("🔥 Calibration started...")
    lows  = np.array([0.0,0 ,0.8575,0.5,0.0,-0.25,-0.25,-0.25,-0.25])
    highs = np.array([2.0,4 ,0.8575,0.999,10.0,0.25,0.25,0.25,0.25])
    gene_space = [{"low": lo, "high": hi} for lo,hi in zip(lows, highs)]
    n_cores = max(1, cpu_count() - 4)

    ga = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=int(pop_size * 0.5),
        fitness_func=fitness_func,
        sol_per_pop=pop_size,
        num_genes=len(gene_space),
        gene_space=gene_space,
        parent_selection_type="rank",  # 保守设置sss steady state selection
        crossover_type="two_points",
        mutation_type="adaptive",
        mutation_percent_genes=[40, 15],
        parallel_processing= None,
        random_seed=seed,
        on_generation=record_history ,
    )
    ga.run()
    sol, fit, _ = ga.best_solution()
    print("\n🔍 GA Completed，Best MSE:", -fit)

    # L-BFGS-B 本地优化
    res = minimize(lambda x: mse_loss(df, x), sol,
                   method="L-BFGS-B", bounds=list(zip(lows, highs)),
                   options={"maxiter":5000, "xatol":1e-8, "fatol":1e-8, "disp":True})
    final_theta = res.x
    print("✅ Calibration Down，Final MSE:", res.fun)

    # 绘制 GA 收敛曲线
    plt.figure(figsize=(6,4))
    plt.plot(history, marker="o")
    plt.title("GA Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("progress_cf.png")
    plt.close()

    # Construct PhiStore for each row with the corresponding parameters
    a, b, c, d, kappa, r1, r2, r3, r4 = final_theta
    rates = np.array([r1, r2, r3, r4])
    spot = df.spot_price.iloc[0]

    strikes = []
    preds_final = []
    for row in df.itertuples(index=False):
        T = row.T
        # 找到最近的到期对应哪段利率
        idx = np.abs(T - T_LIST).argmin()
        r = rates[idx]

        # 用 [a,b,c,d,kappa,r,T] 构造 PhiStore
        phi_store = PhiStore([a, b, c, d, kappa, r, T], spot, batch=64)
        γ = gamma_term_factory(row.k)
        price = price_put_single(row.k, γ, phi_store)
        phi_store.flush()

        strikes.append(row.k)
        preds_final.append(price)

    strikes = np.array(strikes)
    preds_final = np.array(preds_final)
    true_prices = df.opt_price.values

    plt.figure(figsize=(6,6))
    plt.scatter(true_prices, preds_final, marker="x", label="Model")
    plt.scatter(true_prices, true_prices, marker="o", facecolors="none",
                edgecolors="gray", alpha=0.8, label="Market")
    plt.xlabel("Market Price")
    plt.ylabel("Model Price")
    plt.title("Market vs Model Prices")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Strike vs Price by T colored plot
    plt.figure(figsize=(7,5))
    unique_T = np.sort(df.T.unique())
    cmap_m = cm.get_cmap("Oranges")
    cmap_md = cm.get_cmap("Blues")
    for i, t in enumerate(unique_T):
        mask = df.T == t
        c_m = cmap_m(0.4 + 0.6*i/max(1,len(unique_T)-1))
        c_md = cmap_md(0.4 + 0.6*i/max(1,len(unique_T)-1))
        plt.scatter(df.k[mask], df.opt_price[mask], marker="o",
                    facecolors="none", edgecolors=c_m, label=f"Market T={t:.3f}")
        plt.scatter(df.k[mask], preds_final[mask], marker="+",
                    color=c_md, label=f"Model  T={t:.3f}")
    plt.xlabel("Strike")
    plt.ylabel("Put Price")
    plt.title("Price vs Strike (by T)")
    plt.grid(True)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()

    return final_theta, ga, res

# ── 脚本入口 ──
if __name__ == "__main__":
    df = pd.read_excel(str(_PROJECT_ROOT / "02_jan_put.xlsx"))
    df = df.rename(columns={
        "Strike_Put": "k", "TTM_Put": "T", "mid_price_Put": "opt_price"
    })
    df["opt_price"] /= 100
    df["T"] /= 365
    df["spot_price"] = 0.1793

    start = time.time()
    _ = mse_loss(df, [1.0, 2.0, 0.5, 0.5, 1.0, 0.01, 0.01, 0.01, 0.01])
    print("耗时：", time.time() - start)

    calibrate(df)
