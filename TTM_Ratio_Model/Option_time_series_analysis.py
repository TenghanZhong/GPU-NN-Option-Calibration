# vix_prop2_gpu.py  –––  Prop-2  GPU  forecast only
# ==================================================
import os, time, sys, math
import numpy as np
import pandas as pd
import cupy as cp
from cupyx.scipy import special as cpsp
import mpmath, matplotlib.pyplot as plt
# --------------------------------------------------
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
mpmath.mp.dps = 10
DTYPE  = cp.float32

# ─── 1. Constants & grids ─────────────────────────
T_LIST = np.array([258, 100, 48, 20]) / 365       # 用来选 r
MS, XSZ, ELLZ = 8_001, 801, 5_001                 # 控制显存
L_GRID  = cp.linspace(1e-8, 5_000, 300, dtype=DTYPE)
_w_L    = cp.asarray([4 if i%2 else 2 for i in range(len(L_GRID))], dtype=DTYPE); _w_L[[0,-1]]=1
_s0     = cp.linspace(0, 1, MS,   dtype=DTYPE);   _w_s0=_s0*0+2; _w_s0[[0,-1]]=1
_ds0    = float(_s0[1]-_s0[0])
_x_grid = cp.linspace(-30, 30, XSZ, dtype=DTYPE); _w_x=_x_grid*0+2; _w_x[[0,-1]]=1
_dx     = float(_x_grid[1]-_x_grid[0])
_ell    = cp.linspace(0, 30, ELLZ, dtype=DTYPE); _w_ell=_ell*0+2; _w_ell[[0,-1]]=1
_dell   = float(_ell[1]-_ell[0])
SIMPS   = lambda f,w,h: cp.sum(f*w, axis=-1)*h/3.

# ─── 2. φ(l) – GPU kernel ─────────────────────────
def phi_gpu(l_vec, a,b,c,d,kappa,r,t, spot,
            t0=0., Delta=0.083, alpha=1.78, zeta=0.01):
    s  = _s0*(t-t0)+t0; ds=_ds0*(t-t0)
    u  = t - s; tau0=(1-d)/(kappa+1e-8); u_plus=u+Delta
    gd1=cpsp.gamma(d+1); gdm1=cpsp.gamma(d-1)

    H=cp.zeros_like(u)
    m1=u_plus<tau0; m2=(u<tau0)&~m1; m3=~m1&~m2
    H[m1]=(u_plus[m1]**d-u[m1]**d)/(Delta*gd1)
    H[m2]=((tau0**d-u[m2]**d)/(Delta*gd1)+
           (cp.exp(-kappa*u_plus[m2]+1-d)-1)/((kappa**d)*Delta*(1-d)**(2-d)*gdm1))
    H[m3]=-(cp.exp(-kappa*u_plus[m3]+1-d)*(cp.exp(kappa*Delta)-1)) /((kappa**d)*Delta*(1-d)**(2-d)*gdm1)

    J = spot**2 - a*cpsp.gamma(1-c)/cp.power(b,c-1)*SIMPS(H*_w_s0,1.,ds)+r
    lH=H[:,None]*l_vec[None,:]
    log_phi=(a*cpsp.gamma(-c)*b**c)*(((b-1j*lH)/b)**c-1)
    log_int=cp.sum(log_phi*_w_s0[:,None], axis=0)*(ds/3.)

    phiZ=cp.exp(-cp.abs(_ell)**alpha*(t-t0))
    g   = cp.sum(cp.cos(_ell[:,None]*_x_grid[None,:])*phiZ[:,None]*_w_ell[:,None],axis=0)*(_dell/3.)

    phi1=float(cp.exp(-1)); A=(phi1**Delta-1)/math.log(phi1)
    psi=cp.exp(1j*l_vec[:,None]*((zeta/Delta*A)*cp.cos(_x_grid)[None,:]+zeta))
    dbl=cp.sum(psi*g[None,:]*_w_x[None,:],axis=1)*(_dx/3.)

    return (1/cp.pi)*cp.exp(1j*l_vec*J+log_int)*dbl

# ─── 3. Γ(k) cache  (CPU) ─────────────────────────
SCALE=100_000; GAMMA_TAB={}
def gamma_term_factory(K):
    cache={}
    def _γ(l):
        if l not in cache:
            γu=mpmath.gammainc(1.5,1j*K**2*l, mpmath.inf)
            cache[l]=(mpmath.sqrt(mpmath.pi)/2-γu)/mpmath.sqrt(1j*l)
        return cache[l]
    return _γ

def precompute_gamma(strikes):
    l_cpu=L_GRID.get()
    for k in strikes:
        key=int(round(float(k)*SCALE))
        if key in GAMMA_TAB: continue
        γ=gamma_term_factory(float(k))
        GAMMA_TAB[key]=cp.asarray([complex(γ(float(x))) for x in l_cpu],
                                  dtype=cp.complex64)

# ─── 4. Vectorised put-price (GPU) ─────────────────
def price_put_batch(k_vec, phi_vals):
    gamma_mat=cp.stack([GAMMA_TAB[int(round(float(k)*SCALE))] for k in cp.asnumpy(k_vec)])
    exp_term=k_vec[:,None]*cp.exp(-1j*(k_vec**2)[:,None]*L_GRID[None,:])
    integrand=cp.real((exp_term+gamma_mat)*(phi_vals[None,:]/(1j*L_GRID[None,:])))
    I=SIMPS(integrand, _w_L[None,:], float(L_GRID[1]-L_GRID[0]))
    return k_vec/2 - I/cp.pi

# ─── 5. *** Final calibrated parameters *** ───────
THETA = [0.049762, 0.849782, 0.857500,0.769302,7.798968
, 0.001980, -0.001292, -0.006008,  -0.012427]
a,b,c,d,kappa,r1,r2,r3,r4 = THETA
RATES = np.array([r1,r2,r3,r4])

# ─── 6. Main: time-series forecast  ───────────────
if __name__ == "__main__":
    # ─── 6. Paths ─────────────────────────────────────
    base = r"C:\Users\26876\Desktop\Math548\Project"  # ← 你的项目根目录
    outdir = os.path.join(base, "out");
    os.makedirs(outdir, exist_ok=True)

    # 6-1 γ-table  (一次就好) ---------------------------
    df_cal = pd.read_excel(os.path.join(base, "02_jan_put.xlsx"))
    precompute_gamma(df_cal["Strike_Put"].unique())

    # 6-2 load time-series -----------------------------
    df_ts = pd.read_excel(os.path.join(base,"48_k=0.2_modified.xlsx"))
    df_ts.rename(columns={"TTM_day":"TTM", "mid_price_Put":"opt_price"}, inplace=True)

    pred_p, obs_p, dates = [], [], []
    for row in df_ts.itertuples(index=False):
        k     = float(row.Strike)
        t     = float(row.TTM) / 365.0
        spot  = float(row.spot_price)
        nearest_idx = np.abs(t - T_LIST).argmin()
        r = RATES[nearest_idx]  # 使用 term-structure 对应的利率

        # --- GPU pricing ---
        phi   = phi_gpu(L_GRID, a,b,c,d,kappa,r,t, spot)
        price = float(price_put_batch(cp.asarray([k],dtype=DTYPE), phi).get()[0])

        pred_p.append(price)
        obs_p.append(float(row.opt_price))
        dates.append(row.TTM)

    # 6-3  error metrics & plots --------------------
    pred_p, obs_p = np.array(pred_p), np.array(obs_p)
    abs_err = np.abs(pred_p - obs_p)
    rel_err = abs_err / (obs_p + 1e-8)

    print("== Time-series summary ==")
    print("MAE :", abs_err.mean())
    print("RMSE:", np.sqrt((abs_err**2).mean()))
    print("MAPE:", (rel_err*100).mean(), "%")

    df_out = pd.DataFrame({"Date":dates, "Predicted":pred_p,
                           "Observed":obs_p, "AbsErr":abs_err,
                           "RelErr":rel_err})
    df_out.to_csv(os.path.join(outdir, "ts_errors.csv"), index=False)

    # --- plot ---
    # --- plot ---
    plt.figure(figsize=(9, 4))

    # 按 TTM 从大到小排序：第一天 48 → … → 最后一天 0
    order = np.argsort(dates)[::-1]  # 逆序
    ttm_days = np.array(dates)[order]  # x 轴：剩余到期天数
    mkt = obs_p[order]
    mdl = pred_p[order]

    plt.plot(ttm_days, mkt, "x-", label="Market")
    plt.plot(ttm_days, mdl, "o-", label="Model")

    plt.xlabel("Time to Maturity (days)")
    plt.ylabel("Option Price")
    plt.grid(alpha=.4)
    plt.legend()
    plt.tight_layout()

    # 让 x 轴从 48 递减到 0（可选）
    plt.gca().invert_xaxis()

    plt.savefig(os.path.join(outdir, "ts_curve.png"))
    plt.show()
    plt.close()
