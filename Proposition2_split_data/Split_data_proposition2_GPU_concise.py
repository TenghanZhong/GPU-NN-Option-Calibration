#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Proposition‑2 characteristic‑function generator (GPU + multiprocessing, CuPy‑float64)

import cupy as cp
from cupyx.scipy import special as cpsp
import numpy as np
import pandas as pd
from scipy.stats import qmc
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm
import time
from pathlib import Path

# ───────────────────────── Utility ──────────────────────────
def simpson_weights(n: int, dtype=cp.float64) -> cp.ndarray:
    """Return Simpson‑1/3 weights for n nodes (n must be odd)."""
    if (n - 1) % 2:
        raise ValueError("Simpson 1/3 规则要求奇数节点")
    w = cp.ones(n, dtype=dtype)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return w

# ───────────────────────── Worker ───────────────────────────
def phi_worker_gpu(args):
    """GPU worker executed in subprocesses (§ same as original)."""
    (theta_idx, chunk_start, l_chunk,
     a_, b_, c_, d_, kappa_, r_, J_,
     term_col, dbl_chunk, w_s, ds) = args

    l_chunk   = cp.asarray(l_chunk,   dtype=cp.float64)
    term_col  = cp.asarray(term_col,  dtype=cp.float64)      # (M_s ,)
    dbl_chunk = cp.asarray(dbl_chunk, dtype=cp.complex128)   # (chunk_len,)
    w_s       = cp.asarray(w_s,       dtype=cp.float64)      # (M_s ,)

    gamma_neg_c = cpsp.gamma(cp.asarray(-c_, dtype=cp.complex128))
    b_pow_c     = cp.power(cp.asarray(b_, dtype=cp.complex128),
                           cp.asarray(c_, dtype=cp.complex128))

    lH      = l_chunk[None, :] * term_col[:, None]           # (M_s , chunk_len)
    comp_ratio_c = ( (cp.asarray(b_, dtype=cp.complex128) - 1j*lH)
                     / cp.asarray(b_, dtype=cp.complex128) )**c_ - 1.0

    log_phi = a_ * gamma_neg_c * b_pow_c * comp_ratio_c
    log_int = cp.sum(log_phi * w_s[:, None], axis=0) * (ds / 3.0)
    phi_c   = (1.0/cp.pi) * cp.exp(1j*l_chunk*J_) * cp.exp(log_int) * dbl_chunk
    return theta_idx, chunk_start, cp.asnumpy(cp.real(phi_c))

# ─────────────────────── Top‑level steps ────────────────────
def build_static_grids():
    """Pre‑compute all objects that do not depend on θ‑samples."""
    alpha, zeta = 1.78, 0.01
    t, t0, Delta = 27.0, 0.0, 0.083
    I_t0_delta = 0.2667

    # x‑grid
    x_grid = cp.linspace(-30, 30, 801, dtype=cp.float64)
    w_x    = simpson_weights(x_grid.size)
    dx     = float(x_grid[1] - x_grid[0])

    # l‑grid
    l_grid = cp.linspace(-1e4, 1e4, 3000, dtype=cp.float64)

    # g(x)
    ell_grid = cp.linspace(0, 30, 12001, dtype=cp.float64)
    w_ell    = simpson_weights(ell_grid.size)
    phiZ     = cp.exp(-cp.abs(ell_grid)**alpha * (t - t0))
    g_vec    = (cp.sum(cp.cos(ell_grid[:, None]*x_grid[None, :]) *
                       phiZ[:, None] * w_ell[:, None], axis=0)
                * ((ell_grid[1]-ell_grid[0])/3.0))

    # dbl(l)
    phi1 = float(cp.exp(-1.0))
    A    = (phi1**Delta - 1.0) / np.log(phi1)
    C1, C2 = zeta/Delta*A, zeta
    psi   = cp.exp(1j * l_grid[:, None] * (C1*cp.cos(x_grid)[None, :] + C2))
    dbl_vec = cp.tensordot(psi*w_x[None, :], g_vec, axes=([1], [0])) * (dx/3.0)

    return (t, t0, Delta, I_t0_delta,
            l_grid, dbl_vec,
            w_x, dx)

def sample_thetas(n_theta: int = 2000, seed: int = 2025):
    """Latin‑Hypercube sample θ‑vectors."""
    lows  = np.array([0.0, 0.0, 1e-4, 0.5, 0.0, -5.0])
    highs = np.array([5.0, 5.0, 0.9999, 0.999, 5.0, 5.0])
    thetas = qmc.scale(qmc.LatinHypercube(6, seed=seed).random(n_theta), lows, highs)
    return [cp.asarray(col, dtype=cp.float64) for col in thetas.T]   # a,b,c,d,kappa,r

def precompute_kernel(t, t0, Delta, d, kappa):
    """Tabulate H‑kernel terms for every θ (vectorised over θ)."""
    M_s   = 18001
    s_grid = cp.linspace(t0, t, M_s, dtype=cp.float64)
    w_s    = simpson_weights(M_s)
    ds     = float(s_grid[1] - s_grid[0])

    u      = t - s_grid[:, None]
    tau0   = (1.0 - d[None, :]) / kappa[None, :]
    u_plus = u + Delta

    case1 = u_plus < tau0
    case2 = (u < tau0) & (u_plus >= tau0)
    case3 = ~(case1 | case2)

    gd1, gdm1 = cpsp.gamma(d+1.0), cpsp.gamma(d-1.0)

    term  = case1.astype(cp.float64) * (
            (u_plus**d[None, :] - u**d[None, :]) / (Delta*gd1[None, :]))

    tmp1  = (tau0**d[None, :] - u**d[None, :]) / (Delta*gd1[None, :])
    tmp2  = (cp.exp(-kappa[None, :]*u_plus+1-d[None, :]) - 1) / (
            (kappa[None, :]**d[None, :])*Delta*(1-d[None, :])**(2-d[None, :])*gdm1[None, :])
    term += case2.astype(cp.float64) * (tmp1 + tmp2)

    num = cp.exp(-kappa[None, :]*u_plus+1-d[None, :]) * (cp.exp(kappa[None, :]*Delta) - 1)
    den = (kappa[None, :]**d[None, :])*Delta*(1-d[None, :])**(2-d[None, :])*gdm1[None, :]
    term -= case3.astype(cp.float64) * (num/den)

    return term, w_s, ds

def build_tasks(l_np, chunk_len, params, term_np, dbl_np, w_s_np, ds):
    """Create (θ, l‑chunk) task tuples for multiprocessing."""
    a,b,c,d,kappa,r,J_val = params
    l_chunks = np.array_split(l_np, len(l_np)//chunk_len)

    tasks = []
    for idx_chunk, l_chunk in enumerate(l_chunks):
        start = idx_chunk * chunk_len
        end   = start + len(l_chunk)
        for θ in range(len(a)):
            tasks.append((θ, start, l_chunk,
                          a[θ], b[θ], c[θ], d[θ],
                          kappa[θ], r[θ], J_val[θ],
                          term_np[:, θ], dbl_np[start:end],
                          w_s_np, ds))
    return tasks, len(l_chunks)

def compute_phi_parallel(tasks, n_theta, n_l, workers=None):
    """Run pool.imap_unordered and assemble φ‑matrix."""
    phi_real = np.empty((n_theta, n_l), dtype=np.float64)
    with Pool(workers or max(1, cpu_count() - 4)) as pool:
        for θ_idx, start, chunk_val in tqdm(pool.imap_unordered(phi_worker_gpu, tasks),
                                            total=len(tasks), desc="Computing φ"):
            phi_real[θ_idx, start:start+len(chunk_val)] = chunk_val
    return phi_real

def save_parquet(phi_real, l_np, params, out_path: Path):
    """Stack parameters + l + φ into a DataFrame and save."""
    a,b,c,d,kappa,r = params
    df = pd.DataFrame({
        k: np.repeat(v, len(l_np)) for k,v in zip(
            ["a","b","c","d","kappa","r"], [a,b,c,d,kappa,r])
    })
    df["l"]   = np.tile(l_np, len(a))
    df["phi"] = phi_real.ravel()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="fastparquet")
    return df

def compare_with_cpu(cpu_parquet: Path, gpu_parquet: Path,
                     tol_abs=1e-5, tol_rel=1e-8):
    """Load CPU & GPU parquet files and print error statistics."""
    cpu_df = pd.read_parquet(cpu_parquet)
    gpu_df = pd.read_parquet(gpu_parquet)

    key_cols = ['a', 'b', 'c', 'd', 'kappa', 'r', 'l']
    merged   = cpu_df.merge(gpu_df, on=key_cols, suffixes=('_cpu', '_gpu'))

    merged['abs_err'] = np.abs(merged['phi_cpu'] - merged['phi_gpu'])
    merged['rel_err'] = merged['abs_err'] / (np.abs(merged['phi_cpu']) + 1e-30)

    mask = np.isclose(merged['phi_cpu'], merged['phi_gpu'],
                      atol=tol_abs, rtol=tol_rel)

    print("✅ 全部通过容差" if mask.all() else "❌ 超出容差")
    print(f"max |Δφ|            : {merged['abs_err'].max():.3e}")
    print(f"mean |Δφ|           : {merged['abs_err'].mean():.3e}")
    print(f"max relative error  : {merged['rel_err'].max():.3e}")
    print(f"mean relative error : {merged['rel_err'].mean():.3e}")
    print(f"rows out of tolerance: {(~mask).sum()}")

# ───────────────────────── main() ───────────────────────────
def main():
    tic = time.time()

    # 1. Static objects
    (t, t0, Delta, I_t0_delta,
     l_grid, dbl_vec,
     _w_x, _dx) = build_static_grids()
    l_np, dbl_np = map(cp.asnumpy, (l_grid, dbl_vec))

    # 2. θ‑samples
    a,b,c,d,kappa,r = sample_thetas()
    # 3. Kernel & J(θ)
    term, w_s, ds = precompute_kernel(t, t0, Delta, d, kappa)
    xi1   = a * cpsp.gamma(1.0 - c) / cp.power(b, c - 1.0)
    int1  = cp.sum(term * w_s[:, None], axis=0) * (ds / 3.0)
    J_val = I_t0_delta**2 - xi1*int1 + r

    # 4. Build multiprocessing tasks
    params_np = [cp.asnumpy(x) for x in (a,b,c,d,kappa,r,J_val)]
    a_np,b_np,c_np,d_np,kappa_np,r_np,J_np = params_np
    term_np, w_s_np = map(cp.asnumpy, (term, w_s))

    tasks, _ = build_tasks(l_np,
                           chunk_len=300,
                           params=[a_np,b_np,c_np,d_np,kappa_np,r_np,J_np],
                           term_np=term_np, dbl_np=dbl_np,
                           w_s_np=w_s_np, ds=float(ds))

    # 5. Run GPU multiprocessing
    phi_real = compute_phi_parallel(tasks, n_theta=len(a_np), n_l=len(l_np))

    # 6. Save parquet
    gpu_parquet = Path(r"C:\Users\26876\Desktop\Math548\Project\proposition2\Proposition2_gpu_multi123.parquet")
    save_parquet(phi_real, l_np,
                 params=[a_np,b_np,c_np,d_np,kappa_np,r_np],
                 out_path=gpu_parquet)

    print(f"✓ GPU parquet saved → {gpu_parquet}")
    print(f"⏱ 总耗时 {time.time()-tic:.2f}s")

    # 7. Compare with CPU reference
    cpu_parquet = Path(r"C:\Users\26876\Desktop\Math548\Project\proposition2\Proposition2_dataset.parquet")
    compare_with_cpu(cpu_parquet, gpu_parquet)




# ───────────────────────── Runner ───────────────────────────
if __name__ == "__main__":          # <‑‑ One clear entry‑point
    freeze_support()                # needed on Windows
    main()
'''
def precompute_kernel(t, t0, Delta, d, kappa):
    """Tabulate H‑kernel terms for every θ (vectorised over θ)."""
    M_s   = 18001
⏱ 总耗时 330.17s
❌ 超出容差
max |Δφ|            : 6.577e-03
mean |Δφ|           : 3.954e-06
max relative error  : 5.942e+01
mean relative error : 3.153e-01
rows out of tolerance: 42
'''
