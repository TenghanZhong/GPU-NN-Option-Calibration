# ───── price_dataset_gpu_with_t_mpmath_opt.py ────────────────────
import os, cupy as cp, numpy as np, pandas as pd, mpmath
from cupyx.scipy import special as cpsp
from scipy.stats import qmc
from multiprocessing import Pool
from tqdm import tqdm

mpmath.mp.dps = 10                       # 外层精度

# ═══════════ GPU工具 ═════════════════════════════════════════════
def simpson_weights(n, dtype=cp.float64):
    if (n-1) % 2: raise ValueError("奇数节点")
    w = cp.ones(n, dtype); w[1:-1:2], w[2:-1:2] = 4, 2
    return w

def proposition2_cf_37_gpu_single_param(l_vec, a,b,c,d,kappa,r,t,
                                        t0=0.0, Delta=0.083,
                                        I_t0_delta=0.1793,
                                        alpha=1.78, zeta=0.01):
    M_s = 12_001
    s_grid = cp.linspace(t0, t, M_s)
    ds,w_s = float(s_grid[1]-s_grid[0]), simpson_weights(M_s)
    u      = t - s_grid
    tau0   = (1-d)/kappa
    u_plus = u + Delta
    gd1,gdm1 = cpsp.gamma(d+1), cpsp.gamma(d-1)

    H = cp.zeros_like(u)
    m1 = u_plus < tau0
    m2 = (u < tau0) & ~m1
    m3 = ~m1 & ~m2
    H[m1] = (u_plus[m1]**d - u[m1]**d)/(Delta*gd1)
    H[m2] = ((tau0**d - u[m2]**d)/(Delta*gd1) +
             (cp.exp(-kappa*u_plus[m2]+1-d)-1) /
             ((kappa**d)*Delta*(1-d)**(2-d)*gdm1))
    H[m3] = -(cp.exp(-kappa*u_plus[m3]+1-d)*(cp.exp(kappa*Delta)-1)) / (
            (kappa**d)*Delta*(1-d)**(2-d)*gdm1)
    xi1 = a*cpsp.gamma(1-c)/cp.power(b,c-1)
    J   = I_t0_delta**2 - xi1*cp.sum(H*w_s)*(ds/3.0) + r

    lH = l_vec[None,:]*H[:,None]
    log_phi = (a*cpsp.gamma(-c)*b**c)*(((b-1j*lH)/b)**c - 1)
    log_int = cp.sum(log_phi*w_s[:,None], axis=0)*(ds/3.0)

    x_grid = cp.linspace(-30,30,801)
    w_x,dx = simpson_weights(x_grid.size),float(x_grid[1]-x_grid[0])
    ell    = cp.linspace(0,30,6001)
    w_ell,dell = simpson_weights(ell.size),float(ell[1]-ell[0])

    phiZ = cp.exp(-cp.abs(ell)**alpha*(t-t0))
    g    = cp.sum(cp.cos(ell[:,None]*x_grid[None,:])*phiZ[:,None]*w_ell[:,None],
                  axis=0)*(dell/3.0)

    phi1 = float(cp.exp(-1))
    A    = (phi1**Delta-1)/np.log(phi1)
    psi  = cp.exp(1j*l_vec[:,None]*((zeta/Delta*A)*cp.cos(x_grid)[None,:]+zeta))
    dbl  = cp.sum(psi*g[None,:]*w_x[None,:],axis=1)*(dx/3.0)
    return (1/cp.pi)*cp.exp(1j*l_vec*J + log_int)*dbl

# ═══════════ 价格计算结构 ════════════════════════════════════════
K_LIST = np.array([
    10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5,14.0,14.5,15.0,16.0,17.0,
    18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,
    15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,9.0
]) / 100


class PhiStore:                    # ❶ + ❷a
    def __init__(self, theta, batch=64):
        self.a,self.b,self.c,self.d,self.kappa,self.r,self.t = theta
        self.batch, self.buf, self.cache = batch, [], {}
    def __call__(self, l):
        key = round(float(l), 10)
        if key not in self.cache:
            self.buf.append(key)
            # 立即 flush，确保下一次同 key 已存在
            self._flush() if len(self.buf)>=self.batch or len(self.buf)==1 else None
        return self.cache[key]
    def _flush(self):
        if not self.buf: return
        l_arr = cp.asarray(self.buf, dtype=cp.float64)
        vals  = proposition2_cf_37_gpu_single_param(
                l_arr, self.a,self.b,self.c,self.d,self.kappa,self.r,self.t)
        for k,φ in zip(self.buf, cp.asnumpy(vals)):
            self.cache[k] = complex(φ)
        self.buf.clear()
    flush = _flush

def gamma_term_factory(K):         # ❸
    cache={}
    def γ(l):
        if l not in cache:
            γ_u = mpmath.gammainc(1.5,1j*K**2*l, mpmath.inf)
            cache[l]=(mpmath.sqrt(mpmath.pi)/2-γ_u)/mpmath.sqrt(1j*l)
        return cache[l]
    return γ

def price_put_mpmath(K, theta, φ_store, L_max=5000, rel_eps=1e-8):
    γ = gamma_term_factory(K)
    def f(l):
        φ  = φ_store(l)
        exp= K*mpmath.e**(-1j*K**2*l)
        return mpmath.re((exp+γ(l))*(φ/(1j*l)))
    return K/2 - mpmath.quad(f,[1e-8,L_max],rel=rel_eps)/mpmath.pi

def price_vec_given_phi_mpmath(theta, K_arr=K_LIST, L_max=5000):
    φ_store = PhiStore(theta)
    prices   = [price_put_mpmath(float(k), theta, φ_store, L_max) for k in K_arr]
    φ_store.flush()
    return np.asarray(prices, dtype=float)

# ═══════════ 任务分发 ════════════════════════════════════════════
def sample_theta(n, seed=2025):
    lows  = np.array([0,0,1e-4,0.5,0,-0.25])
    highs = np.array([5,5,0.9999,0.999,10,0.25])
    return qmc.scale(qmc.LatinHypercube(d=6,seed=seed).random(n), lows, highs)

def worker_theta_t(args):
    theta,t = args
    prices  = price_vec_given_phi_mpmath((*theta,t))
    base    = np.repeat([(*theta,t)], K_LIST.size, axis=0)
    return np.column_stack([base, K_LIST, prices])

# ═══════════ 主入口 ══════════════════════════════════════════════
if __name__ == "__main__":
    N_THETA = 5_000
    thetas  = sample_theta(N_THETA)
    T_LIST  = [258,100,48,20]
    tasks   = [(θ,t) for θ in thetas for t in T_LIST]

    proc = max(1, os.cpu_count()-4)
    rows=[]
    with Pool(proc) as pool:
        for blk in tqdm(pool.imap(worker_theta_t, tasks),
                        total=len(tasks),
                        desc=f"Pricing mpmath-opt ({proc} proc)"):
            rows.append(blk)

    df=pd.DataFrame(np.vstack(rows),
                    columns=["a","b","c","d","kappa","r","t","K","price_put"])
    out_path=r"C:\Users\26876\Desktop\Math548\Project\Option_put_price_dataset_mpmath_5000.parquet"
    df.to_parquet(out_path, index=False, engine="fastparquet")
    print("✓ Saved", len(df), "rows →", out_path)
