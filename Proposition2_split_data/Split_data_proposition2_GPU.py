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

# ------------------ 通用 -------------------------------------------------
def simpson_weights(n: int, dtype=cp.float64) -> cp.ndarray:
    if (n - 1) % 2:
        raise ValueError("Simpson 1/3 规则要求奇数节点")
    w = cp.ones(n, dtype=dtype)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return w

# ------------------ GPU‑worker ------------------------------------------
def phi_worker_gpu(args):
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

    #
    comp = cp.asarray(b_, dtype=cp.complex128) - 1j * lH
    comp_ratio = comp / cp.asarray(b_, dtype=cp.complex128)
    comp_ratio_c = cp.power(comp_ratio, cp.asarray(c_, dtype=cp.complex128)) - 1.0
    log_phi = a_ * gamma_neg_c * b_pow_c * comp_ratio_c
    '''
    comp    = cp.asarray(b_, dtype=cp.complex128) - 1j * lH
    comp_c  = cp.power(comp, cp.asarray(c_, dtype=cp.complex128))
    log_phi = a_ * gamma_neg_c * (comp_c - b_pow_c)          # (M_s , chunk_len)
    '''
    log_int = cp.sum(log_phi * w_s[:, None], axis=0) * (ds / 3.0)

    phi_c   = (1.0/cp.pi) * cp.exp(1j*l_chunk*J_) * cp.exp(log_int) * dbl_chunk
    return theta_idx, chunk_start, cp.asnumpy(cp.real(phi_c))

# ==================== 主流程 ============================================
if __name__ == "__main__":
    freeze_support()
    tic = time.time()

    # ===== 常量 =====
    alpha, zeta      = 1.78, 0.01
    t, t0, Delta     = 27.0, 0.0, 0.083
    I_t0_delta       = 0.2667
    # ===== x‑grid (801) =====
    x_grid = cp.linspace(-30, 30, 801, dtype=cp.float64)
    w_x    = simpson_weights(x_grid.size)
    dx     = float(x_grid[1] - x_grid[0])


    # ===== l‑grid (3000) =====
    l_grid = cp.linspace(-1e4, 1e4, 3000, dtype=cp.float64)
    # ===== g(x) =====
    ell_grid = cp.linspace(0, 30, 12001, dtype=cp.float64)
    w_ell    = simpson_weights(ell_grid.size)
    phiZ     = cp.exp(-cp.abs(ell_grid)**alpha * (t - t0))
    g_vec    = (cp.sum(cp.cos(ell_grid[:,None]*x_grid[None,:])*phiZ[:,None]*w_ell[:,None], axis=0)
                * ((ell_grid[1]-ell_grid[0])/3.0))                                    # (801,)

    # ===== dbl(l) =====
    phi1 = float(cp.exp(-1.0))
    A    = (phi1**Delta - 1.0)/np.log(phi1)
    C1, C2 = zeta/Delta*A, zeta
    psi   = cp.exp(1j * l_grid[:,None] * (C1*cp.cos(x_grid)[None,:] + C2))
    dbl_vec = cp.tensordot(psi*w_x[None,:], g_vec, axes=([1],[0])) * (dx/3.0)         # (3000,)

    # ===== LHS θ =====
    lows  = np.array([0.0, 0.0, 1e-4, 0.5, 0.0, -5.0])
    highs = np.array([5.0, 5.0, 0.9999, 0.999, 5.0, 5.0])
    thetas = qmc.scale(qmc.LatinHypercube(6, seed=2025).random(2000), lows, highs)
    a,b,c,d,kappa,r = cp.asarray(thetas, dtype=cp.float64).T

    # ===== H‑kernel pre‑tab =====
    M_s   = 6001
    s_grid = cp.linspace(t0, t, M_s, dtype=cp.float64)
    w_s    = simpson_weights(M_s)
    ds     = float(s_grid[1]-s_grid[0])
    u      = t - s_grid[:,None]
    tau0   = (1.0 - d[None,:]) / kappa[None,:]
    u_plus = u + Delta
    case1, case2 = u_plus < tau0, (u < tau0) & (u_plus >= tau0)
    case3 = ~(case1 | case2)
    gd1, gdm1 = cpsp.gamma(d+1.0), cpsp.gamma(d-1.0)
    term = (case1.astype(cp.float64) *
            (u_plus**d[None,:] - u**d[None,:]) / (Delta*gd1[None,:]))
    tmp1 = (tau0**d[None,:] - u**d[None,:]) / (Delta*gd1[None,:])
    tmp2 = (cp.exp(-kappa[None,:]*u_plus+1-d[None,:]) - 1) / (
           (kappa[None,:]**d[None,:])*Delta*(1-d[None,:])**(2-d[None,:])*gdm1[None,:])
    term =term+ case2.astype(cp.float64) * (tmp1+tmp2)
    num = cp.exp(-kappa[None,:]*u_plus+1-d[None,:]) * (cp.exp(kappa[None,:]*Delta) - 1)
    den = (kappa[None,:]**d[None,:])*Delta*(1-d[None,:])**(2-d[None,:])*gdm1[None,:]
    term =term- case3.astype(cp.float64) * (num/den)                                        # (M_s,2000)

    # ===== J(θ) =====
    xi1   = a*cpsp.gamma(1.0-c) / cp.power(b, c-1.0)
    int1  = cp.sum(term*w_s[:,None], axis=0) * (ds/3.0)
    J_val = I_t0_delta**2 - xi1*int1 + r                                                # (2000,)
    # ===== chunking =====
    l_np, dbl_np = map(cp.asnumpy, (l_grid, dbl_vec))
    w_s_np, term_np = map(cp.asnumpy, (w_s, term))
    arrs = [cp.asnumpy(x) for x in (a,b,c,d,kappa,r,J_val)]
    a_np,b_np,c_np,d_np,kappa_np,r_np,J_np = arrs
    chunk_len = 300
    l_chunks  = np.array_split(l_np, len(l_np)//chunk_len)

    # ===== task list =====
    tasks = []
    for idx_chunk, l_chunk in enumerate(l_chunks):
        start = idx_chunk*chunk_len
        end   = start + len(l_chunk)
        for θ in range(len(a_np)):
            tasks.append((θ, start, l_chunk,
                          a_np[θ], b_np[θ], c_np[θ], d_np[θ],
                          kappa_np[θ], r_np[θ], J_np[θ],
                          term_np[:,θ], dbl_np[start:end],
                          w_s_np, ds))
    # ===== 并行 =====
    phi_real = np.empty((len(a_np), len(l_np)), dtype=np.float64)
    with Pool(max(1, cpu_count()-4)) as pool:
        for θ_idx, start, chunk_val in tqdm(pool.imap_unordered(phi_worker_gpu, tasks),
                                            total=len(tasks), desc="Computing φ"):
            phi_real[θ_idx, start:start+len(chunk_val)] = chunk_val
    # ===== 保存 =====
    df = pd.DataFrame({
        k: np.repeat(v, len(l_np)) for k,v in zip(
            ["a","b","c","d","kappa","r"],
            [a_np,b_np,c_np,d_np,kappa_np,r_np])
    })
    df["l"]   = np.tile(l_np, len(a_np))
    df["phi"] = phi_real.ravel()
    out = r"C:\Users\26876\Desktop\Math548\Project\Proposition2_gpu_multi.parquet"
    df.to_parquet(out, index=False, engine="fastparquet")
    print(f"✓ 生成完毕：{len(df):,} 行 → {out}")
    print(f"⏱ 总耗时 {time.time()-tic:.2f}s")
    # ① 读取两个 parquet
    cpu_df = pd.read_parquet(r"C:\Users\26876\Desktop\Math548\Project\Proposition2_dataset.parquet")
    gpu_df = pd.read_parquet(r"C:\Users\26876\Desktop\Math548\Project\Proposition2_gpu_multi.parquet")

    # ② 按 6 个 θ 参数 + l 对齐
    key_cols = ['a', 'b', 'c', 'd', 'kappa', 'r', 'l']  # 这些列必须 dtype 完全一致
    merged = cpu_df.merge(gpu_df, on=key_cols, suffixes=('_cpu', '_gpu'))

    tol_abs, tol_rel = 1e-5, 1e-8
    for col in key_cols:
        cpu_df[col] = cpu_df[col].astype(np.float64)
        gpu_df[col] = gpu_df[col].astype(np.float64)

    merged = cpu_df.merge(gpu_df, on=key_cols, suffixes=('_cpu', '_gpu'))

    merged['abs_err'] = np.abs(merged['phi_cpu'] - merged['phi_gpu'])
    merged['rel_err'] = merged['abs_err'] / (np.abs(merged['phi_cpu']) + 1e-30)

    mask = np.isclose(merged['phi_cpu'], merged['phi_gpu'], atol=tol_abs, rtol=tol_rel)
    print("✅ 全部通过容差" if mask.all() else "❌ 超出容差")
    print(f"max |Δφ|  : {merged['abs_err'].max():.3e}")
    print(f"mean |Δφ| : {merged['abs_err'].mean():.3e}")
    print(f"max relative error  : {merged['rel_err'].max():.3e}")
    print(f"mean relative error : {merged['rel_err'].mean():.3e}")
    print(f"rows out of tolerance: {(~mask).sum()}")

    '''
    
    
    ⏱ 总耗时 113.44s
    ❌ 超出容差
    max |Δφ|  : 6.572e-03
    mean |Δφ| : 6.115e-06
    max relative error  : 7.666e+01
    mean relative error : 3.311e-01
    rows out of tolerance: 60
    
    
    
    
    
    
    
    、
    、
    
    '''


