import numpy as np
import cupy as cp
import mpmath
from cupyx.scipy import special as cpsp
import time

mpmath.mp.dps = 10

# ── Proposition 2 GPU 特征函数 φ ──
def simpson_weights(n, dtype=cp.float64):
    if (n-1) % 2: raise ValueError("需要偶数个点")
    w = cp.ones(n, dtype)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return w

def proposition2_cf_37_gpu_single_param(l_vec, a,b,c,d,kappa,r,t,
                                        t0=0.0, Delta=0.083,
                                        spot_price=0.1793,
                                        alpha=1.78, zeta=0.01):
    M_s = 12001
    s_grid = cp.linspace(t0, t, M_s)
    ds, w_s = float(s_grid[1] - s_grid[0]), simpson_weights(M_s)
    u = t - s_grid
    tau0 = (1 - d) / kappa
    u_plus = u + Delta
    gd1, gdm1 = cpsp.gamma(d + 1), cpsp.gamma(d - 1)

    H = cp.zeros_like(u)
    m1 = u_plus < tau0
    m2 = (u < tau0) & ~m1
    m3 = ~m1 & ~m2
    H[m1] = (u_plus[m1] ** d - u[m1] ** d) / (Delta * gd1)
    H[m2] = ((tau0 ** d - u[m2] ** d) / (Delta * gd1) +
             (cp.exp(-kappa * u_plus[m2] + 1 - d) - 1) /
             ((kappa ** d) * Delta * (1 - d) ** (2 - d) * gdm1))
    H[m3] = -(cp.exp(-kappa * u_plus[m3] + 1 - d) * (cp.exp(kappa * Delta) - 1)) / (
            (kappa ** d) * Delta * (1 - d) ** (2 - d) * gdm1)
    xi1 = a * cpsp.gamma(1 - c) / cp.power(b, c - 1)
    J = spot_price ** 2 - xi1 * cp.sum(H * w_s) * (ds / 3.0) + r

    lH = l_vec[None, :] * H[:, None]
    log_phi = (a * cpsp.gamma(-c) * b ** c) * (((b - 1j * lH) / b) ** c - 1)
    log_int = cp.sum(log_phi * w_s[:, None], axis=0) * (ds / 3.0)

    x_grid = cp.linspace(-30, 30, 801)
    w_x, dx = simpson_weights(x_grid.size), float(x_grid[1] - x_grid[0])
    ell = cp.linspace(0, 30, 6001)
    w_ell, dell = simpson_weights(ell.size), float(ell[1] - ell[0])

    phiZ = cp.exp(-cp.abs(ell) ** alpha * (t - t0))
    g = cp.sum(cp.cos(ell[:, None] * x_grid[None, :]) * phiZ[:, None] * w_ell[:, None], axis=0) * (dell / 3.0)

    phi1 = float(cp.exp(-1))
    A = (phi1 ** Delta - 1) / np.log(phi1)
    psi = cp.exp(1j * l_vec[:, None] * ((zeta / Delta * A) * cp.cos(x_grid)[None, :] + zeta))
    dbl = cp.sum(psi * g[None, :] * w_x[None, :], axis=1) * (dx / 3.0)

    return (1 / cp.pi) * cp.exp(1j * l_vec * J + log_int) * dbl

# ── gamma项构造器 ──
def gamma_term_factory(K):
    cache = {}
    def γ(l):
        if l not in cache:
            γ_u = mpmath.gammainc(1.5, 1j * K ** 2 * l, mpmath.inf)
            cache[l] = (mpmath.sqrt(mpmath.pi) / 2 - γ_u) / mpmath.sqrt(1j * l)
        return cache[l]
    return γ

# ── GPU φ 缓存结构 ──
class PhiStore:
    def __init__(self, theta, spot_price, batch=64):
        self.a, self.b, self.c, self.d, self.kappa, self.r, self.t = theta
        self.spot_price = spot_price
        self.batch, self.buf, self.cache = batch, [], {}

    def __call__(self, l):
        key = round(float(l), 10)
        if key not in self.cache:
            self.buf.append(key)
            if len(self.buf) >= self.batch or len(self.buf) == 1:
                self._flush()
        return self.cache[key]

    def _flush(self):
        if not self.buf:
            return
        l_arr = cp.asarray(self.buf, dtype=cp.float64)
        vals = proposition2_cf_37_gpu_single_param(
            l_arr,
            self.a, self.b, self.c, self.d, self.kappa, self.r, self.t,
            t0=0.0, Delta=0.083,
            spot_price=self.spot_price,
            alpha=1.78, zeta=0.01)
        for k, φ in zip(self.buf, cp.asnumpy(vals)):
            self.cache[k] = complex(φ)
        self.buf.clear()

    flush = _flush

# ── 计算Put价格 ──
def price_put_single(K, theta, spot_price, L_max=5000, rel_eps=1e-8):
    φ_store = PhiStore(theta, spot_price)
    γ = gamma_term_factory(K)

    def f(l):
        φ = φ_store(l)
        exp = K * mpmath.exp(-1j * K ** 2 * l)
        return mpmath.re((exp + γ(l)) * (φ / (1j * l)))

    result = float(K / 2 - mpmath.quad(f, [1e-8, L_max], rel=rel_eps) / mpmath.pi)
    φ_store.flush()
    return result

# ── 示例调用 ──
if __name__ == "__main__":
    start_time = time.time()
    theta_opt = [
        0.020442,
        2.725817,
        0.087589,
        0.906155,
        0.945427,
        0.12889,    # 使用r4，对应T=20
        20
    ]
    strike_K = 0.2
    spot_price = 0.1793

    price = price_put_single(strike_K, theta_opt, spot_price)
    print(f"📌 Predicted Put Price (non-NN): {price:.6f}")

    elapsed = time.time() - start_time  # ⏱️ 结束计时
    print(f"⏱️ Total calibration time: {elapsed:.2f} seconds")
