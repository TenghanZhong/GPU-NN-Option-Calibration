# ───── price_dataset_gpu_with_t.py ────────────────────────────────
from cupyx.scipy import special as cpsp
import cupy as cp
import numpy as np
import pandas as pd
from scipy.stats import qmc
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import mpmath
mpmath.mp.dps = 15

# ─── ① Simpson 工具 ───────────────────────────────────────────────
def simpson_weights(n, dtype=cp.float64):
    if (n-1) % 2: raise ValueError("需要奇数节点")
    w = cp.ones(n, dtype=dtype);  w[1:-1:2], w[2:-1:2] = 4, 2
    return w

def integrate_simpson(f_vals, w, dx):
    return cp.sum(w * f_vals, axis=-1) * (dx/3.0)

# ─── ② 上不完全 Γ 一次性缓存 ───────────────────────────────────────
def gammainc_upper_cpu(a, x_cp):
    x_np = x_cp.get()
    res = [complex(mpmath.gammainc(a, z, mpmath.inf)) for z in x_np.ravel()]
    return cp.asarray(res, dtype=cp.complex128).reshape(x_cp.shape)

# ─── ③ φ(l) 内核（把 t 作为参数传进来）────────────────────────────
def proposition2_cf_37_gpu_single_param(
        l_vec, a,b,c,d,kappa,r, t,
        t0   = 0.0, Delta=0.083, I_t0_delta=0.1793,
        alpha=1.78, zeta =0.01):

    # -------- Step-1  J(t) ---------
    M_s = 12_001
    s_grid = cp.linspace(t0, t, M_s)
    ds, w_s = float(s_grid[1]-s_grid[0]), simpson_weights(M_s)
    u, tau0, u_plus = t - s_grid, (1-d)/kappa, (t - s_grid) + Delta
    gd1, gdm1 = cpsp.gamma(d+1), cpsp.gamma(d-1)

    H = cp.zeros_like(u)
    m1 = u_plus < tau0
    m2 = (u < tau0) & ~m1
    m3 = ~m1 & ~m2
    H[m1] = (u_plus[m1]**d - u[m1]**d)/(Delta*gd1)
    H[m2] = ((tau0**d - u[m2]**d)/(Delta*gd1) +
             (cp.exp(-kappa*u_plus[m2] + 1-d)-1) /
             ((kappa**d)*Delta*(1-d)**(2-d)*gdm1))
    H[m3] = -(cp.exp(-kappa*u_plus[m3] + 1-d) *
              (cp.exp(kappa*Delta)-1)) / (
              (kappa**d)*Delta*(1-d)**(2-d)*gdm1)
    xi1 = a*cpsp.gamma(1-c)/cp.power(b, c-1)
    J   = I_t0_delta**2 - xi1*cp.sum(H*w_s)*(ds/3.0) + r

    # -------- Step-2  ∫log φ_X1 -------
    lH = l_vec[None,:]*H[:,None]
    log_phi = (a * cpsp.gamma(-c) * b**c) * (((b-1j*lH)/b)**c - 1)
    log_int = cp.sum(log_phi*w_s[:,None], axis=0)*(ds/3.0)

    # -------- Step-3  dbl(l) ----------
    x_grid = cp.linspace(-50, 50, 1601)
    w_x, dx = simpson_weights(x_grid.size), float(x_grid[1]-x_grid[0])
    ell = cp.linspace(0, 50, 10_001)
    w_ell, dell = simpson_weights(ell.size), float(ell[1]-ell[0])

    phiZ = cp.exp(-cp.abs(ell)**alpha * (t-t0))
    g = cp.sum(cp.cos(ell[:,None]*x_grid[None,:]) *
               phiZ[:,None]*w_ell[:,None], axis=0) * (dell/3.0)

    phi1 = float(cp.exp(-1))
    A = (phi1**Delta - 1)/np.log(phi1)
    psi = cp.exp(1j*l_vec[:,None] *
                 ( (zeta/Delta*A)*cp.cos(x_grid)[None,:] + zeta ))
    dbl = cp.sum(psi * g[None,:] * w_x[None,:], axis=1) * (dx/3.0)

    return (1/cp.pi)*cp.exp(1j*l_vec*J + log_int)*dbl

# ─── ④ 全局网格（l,K）+ Γ 缓存 ───────────────────────────────────
L_MAX, N_L = 1_000, 18_001
L_GRID = cp.linspace(1e-8, L_MAX, N_L)
DL, W_L = float(L_GRID[1]-L_GRID[0]), simpson_weights(N_L)

K_LIST = np.array([
    10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5,14.0,14.5,15.0,16.0,17.0,
    18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,
    15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,9.0])/100
K_CP = cp.asarray(K_LIST)
GAMMA_UP = gammainc_upper_cpu(1.5,
            1j*K_CP[:,None]**2 * L_GRID[None,:])

# ─── ⑤ 价格向量函数 ────────────────────────────────────────────────
def price_vec_given_phi(phi_vals):
    exp = K_CP[:,None]*cp.exp(-1j*K_CP[:,None]**2 * L_GRID)
    gamma = (cp.sqrt(cp.pi)/2 - GAMMA_UP)/cp.sqrt(1j*L_GRID)
    decay = cp.exp(-(L_GRID/600.0)**2)
    integ = cp.real((exp+gamma)*(phi_vals*decay)/(1j*L_GRID))*decay
    integral = integrate_simpson(integ, W_L, DL)
    return K_CP/2 - integral/cp.pi     # → (34,)

# ─── ⑥ worker：θ,t → 34 价格 ─────────────────────────────────────
def worker_theta_t(args):
    theta, t = args
    a,b,c,d,kappa,r = theta
    phi_vals = proposition2_cf_37_gpu_single_param(
        L_GRID, a,b,c,d,kappa,r, t)
    prices = price_vec_given_phi(phi_vals)
    base = np.repeat([[a,b,c,d,kappa,r,t]], K_LIST.size, axis=0)
    return np.column_stack([base, K_LIST, cp.asnumpy(prices)])

# ─── ⑦ 采样 θ + 生成 (θ,t) 任务表 ────────────────────────────────
def sample_theta(n, seed=2025):
    lows  = np.array([0.0,0.0,1e-4,0.5,0.0,-5.0])
    highs = np.array([5.0,5.0,0.9999,0.999,5.0, 5.0])
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    return qmc.scale(sampler.random(n), lows, highs)



# ─── ⑧ 主程序 ────────────────────────────────────────────────────
if __name__ == "__main__":
    N_THETA = 3_000
    thetas  = sample_theta(N_THETA)

    T_LIST = [258, 100, 48, 20]  # 需要遍历的 t
    # 生成 (θ,t) 笛卡儿积任务列表
    tasks = [(theta, t) for theta in thetas for t in T_LIST]

    n_proc = 1                        # 单卡 GPU 建议 1
    rows = []
    with Pool(processes=n_proc) as pool:
        for blk in tqdm(pool.imap(worker_theta_t, tasks),
                        total=len(tasks), desc="Pricing"):
            rows.append(blk)

    df = pd.DataFrame(np.vstack(rows),
          columns=["a","b","c","d","kappa","r","t","K","price_put"])
    out_path = r"C:\Users\26876\Desktop\Math548\Project\Option_put_price_dataset_gpu.parquet"
    df.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"✓ Saved {len(df)} rows to {out_path}")

'''
主进程（CPU）：
  ├─ 采样 θ 参数列表 thetas （共 N_THETA 条）
  ├─ 设定 T_LIST = [258, 100, 48, 20]
  ├─ 构建任务列表 tasks = [(θ, t) for θ in thetas for t in T_LIST]
  └─ Pool() 启动多个子进程（CPU）
         └─ 每个子进程：
             ├─ 接收一组参数 (θ, t)
             │     θ = (a, b, c, d, kappa, r)
             │     t 为某一到期时间
             └─ 在 GPU 上执行：
                   • proposition2_cf_37_gpu_single_param(l, θ, t)
                         ↳ 计算 φ(l)（特征函数）
                   • price_vec_given_phi(φ)
                         ↳ 计算所有执行价 K 对应的 Put 价格（共 34 个）
             → 输出 shape = (34, 9) 的结果表格：
                   [a, b, c, d, kappa, r, t, K, price_put]
'''