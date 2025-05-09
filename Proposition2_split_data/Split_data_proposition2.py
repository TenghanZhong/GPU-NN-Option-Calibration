import numpy as np
import pandas as pd
from scipy.stats import qmc
import mpmath
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


mpmath.mp.dps = 10  # set desired precision

# Stable parameter for the jump process Z (phi_{Z1})
alpha = 1.78     # typically estimated empirically
# Instantaneous variance and long-term variance (VIX squared, etc.)
V_t0 = 0.04      # instantaneous variance at time t0 (from market data)
V_bar = 0.04     # long-term (mean-reversion) level for variance
# Jump scaling parameter for Z (small-scale jump adjustments)
zeta = 0.01       # typically a small number
# Initial jump value (often set to 0)
Z_t0 = 0.0
# Time settings (t = evaluation time, t0 = conditioning time, Delta = averaging window)
t = 27
t0 = 0
Delta = 0.083   # roughly 30 days expressed in years (e.g., for VIX)
I_t0_delta = 0.2667
# Your existing functions...
# -------------- 原 φ_Z1, φ_X1, kernel 等函数 --------------
def phi_Z1(l, alpha):
    return mpmath.exp(-abs(l) ** alpha)

def phi_X1(l, a, b, c):
    return mpmath.exp(a * mpmath.gamma(-c) * ((b - 1j * l) ** c - b ** c))

def H_kernel_III_delta(t, s, Delta, d, kappa):
    tau0 = (1 - d) / kappa
    u = t - s
    if u + Delta < tau0:
        return ((u + Delta)**d - u**d) / (Delta * mpmath.gamma(d + 1))
    elif u < tau0 <= u + Delta:
        term1 = (tau0**d - u**d) / (Delta * mpmath.gamma(d + 1))
        term2 = (mpmath.exp(-kappa * (u + Delta) + 1 - d) - 1) / (
                 (kappa**d) * Delta * (1 - d)**(2 - d) * mpmath.gamma(d - 1))
        return term1 + term2
    num = mpmath.exp(-kappa * (u + Delta) + 1 - d) * (mpmath.exp(kappa * Delta) - 1)
    den = (kappa**d) * Delta * (1 - d)**(2 - d) * mpmath.gamma(d - 1)
    return -num / den

# ---------- 预计算网格常量 ----------
x_min_global, x_max_global, N_X = -30.0, 30.0, 801   # 奇数节点→Simpson
x_grid = [mpmath.mpf(x_min_global + i*(x_max_global-x_min_global)/(N_X-1))
          for i in range(N_X)]
dx = x_grid[1] - x_grid[0]

L_MIN, L_MAX = -10_000.0, 10_000.0
N_L = 3_000
l_grid = np.linspace(L_MIN, L_MAX, N_L)
l_grid_mpf = [mpmath.mpf(l) for l in l_grid]

# ---------- 用于缓存的占位符，在主进程中真正赋值 ----------
g_vec   = None
dbl_vec = None

# ---------- 单次计算 g(x) 与 dbl(l) 的工具函数 ----------
@lru_cache(maxsize=None)
def _g_of_x(x, L_max=30):
    return mpmath.quad(lambda ell: mpmath.cos(ell * x) *
                       phi_Z1(ell, alpha) ** (t - t0), [0, L_max])

phi1 = phi_Z1(1, alpha)
A    = (phi1 ** Delta - 1) / mpmath.log(phi1)
C1   = zeta / Delta * A
C2   = zeta                            # real - imag + Δ = Δ

@lru_cache(maxsize=None)
def _psi(l, x):
    return mpmath.exp(1j * l * (C1 * mpmath.cos(x) + C2))

def _dbl_of_l(idx, g_vec_local):
    l = l_grid_mpf[idx]
    acc = mpmath.mpc(0)
    for j, x in enumerate(x_grid):
        coef = 1 if j in (0, N_X-1) else (4 if j % 2 else 2)
        acc += coef * _psi(l, x) * g_vec_local[j]
    return acc * dx / 3

# ---------- 其余原函数 ----------
def compute_J(t, t0, Delta, I_t0_delta, r, a, b, c, d, kappa):
    xi1 = a * mpmath.gamma(1 - c) / (b ** (c - 1))
    int1 = mpmath.quad(lambda s: H_kernel_III_delta(t, s, Delta, d, kappa), [t0, t])
    return I_t0_delta ** 2 - xi1 * int1 + r

def log_phi_X1_integral(l, t, t0, Delta, a, b, c, d, kappa):
    f = lambda s: mpmath.log(phi_X1(l * H_kernel_III_delta(t, s, Delta, d, kappa),
                                    a, b, c))
    return mpmath.quad(f, [t0, t])

# 以下两个缓存函数已无用，但保持符号
@lru_cache(maxsize=1024)
def psi_cached(l, x, zeta, Delta, alpha, Z_t0): ...
@lru_cache(maxsize=1024)
def inner_integral_cached(x, t, t0, alpha, L_max=30): ...

# ---------- dbl 查表 ----------
def double_integral(l, *args, **kwargs):
    idx = int(round((l - L_MIN) / (L_MAX - L_MIN) * (N_L - 1)))
    return dbl_vec[idx]

# ---------- 主 cf 函数 ----------
def proposition2_cf_37(
    l, t, t0, Delta, zeta, alpha, Z_t0, I_t0_delta,
    a, b, c, d, kappa, r, **kwargs
):
    J_val   = compute_J(t, t0, Delta, I_t0_delta, r, a, b, c, d, kappa)
    log_t   = log_phi_X1_integral(l, t, t0, Delta, a, b, c, d, kappa)
    dbl     = double_integral(l)
    return (1/mpmath.pi) * mpmath.exp(1j*l*J_val + log_t) * dbl

def phi_scalar(l, a, b, c, d, kappa, r):
    return proposition2_cf_37(
        l, t, t0, Delta, zeta, alpha, Z_t0=0,
        I_t0_delta=I_t0_delta, a=a, b=b, c=c, d=d, kappa=kappa, r=r)

# ---------- θ 采样 ----------
def sample_theta(n, lows, highs, seed):
    sampler = qmc.LatinHypercube(d=len(lows), seed=seed)
    return qmc.scale(sampler.random(n), lows, highs)

# ---------- worker 需要的初始化 ----------
def init_worker(g_shared, dbl_shared):
    global g_vec, dbl_vec
    g_vec   = g_shared
    dbl_vec = dbl_shared

# ---------- worker 具体计算 ----------
def process_theta(theta):
    a,b,c,d,kappa,r = theta
    J_val  = compute_J(t, t0, Delta, I_t0_delta, r, a, b, c, d, kappa)
    fac    = 1/mpmath.pi
    rows   = []
    log_vec = [log_phi_X1_integral(l, t, t0, Delta, a, b, c, d, kappa)
               for l in l_grid_mpf]
    for l, log_t, dbl in zip(l_grid_mpf, log_vec, dbl_vec):
        φ = fac * mpmath.exp(1j*l*J_val + log_t) * dbl
        rows.append((a,b,c,d,kappa,r, float(l), float(mpmath.re(φ))))
    return rows


# ======================= main =======================
if __name__ == "__main__":
    print("CPU cores available:", cpu_count())

    # ① 主进程预计算 g_vec, dbl_vec
    print("⏳ 预计算 g(x) ...")
    g_vec = [_g_of_x(x) for x in x_grid]
    print("✔ g(x) 完成")

    print("⏳ 预计算 dbl(l) ...")
    dbl_vec = [_dbl_of_l(i, g_vec) for i in range(N_L)]
    print("✔ dbl(l) 完成")

    # ② 生成 θ 样本
    lows  = np.array([0.0, 0.0, 1e-4, 0.5, 0.0, -5])
    highs = np.array([5.0, 5.0, 0.9999, 0.999, 5.0, 5])
    thetas = sample_theta(2_000, lows, highs, seed=2025)

    # ③ 多进程运算
    with Pool(processes=max(1, cpu_count()-4),
              initializer=init_worker,
              initargs=(g_vec, dbl_vec)) as pool:
        results = list(tqdm(pool.imap(process_theta, thetas),
                            total=len(thetas), desc="Processing θ"))

    # ④ 汇总并保存
    pairs = [row for sub in results for row in sub]

    df = pd.DataFrame(pairs,
                      columns=["a","b","c","d","kappa","r","l","phi"])
    df.head().to_parquet("test.parquet", engine="fastparquet")
    pd.read_parquet("test.parquet").head()
    df.to_parquet(r"C:\Users\26876\Desktop\Math548\Project\Proposition2_dataset.parquet",
                  index=False, engine="fastparquet")
    print("✓ 特征函数数据集已生成，共", len(df), "条记录")
