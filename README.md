# GPU-NN-Option-Calibration
Calibration of power-type derivatives for rough volatility with jumps

##  Abstract

In this project, we develop an efficient calibration methodology for a recently proposed rough volatility model with jumps, tailored to the pricing of power-type volatility derivatives. Building on a framework that combines fractional Ornstein–Uhlenbeck variance dynamics driven by a tempered-stable Lévy subordinator and an independent sinusoidal jump component from Wang and Xia (2022) and motivated by high-frequency evidence of jump-dominated volatility dynamics, we focus on calibrating the model to market data for derivatives such as VIX  options. The mathematical model — a rough stochastic volatility process with pure-jump volatility components — serves as the foundation, and the core contribution of this paper is a novel calibration algorithm that leverages the model’s semi-analytical pricing capabilities. This procedure rapidly and robustly fits the model to observed option prices, significantly reducing computation time compared to traditional methods. We verify that a pure-jump rough volatility model adequately captures the VIX dynamics, consistent with prior empirical findings, and demonstrate that our calibration method achieves high accuracy and speed. The results underscore the practicality of rough volatility with jumps for volatility derivatives and the effectiveness of the proposed calibration approach in a real-market setting.

Keywords: Rough volatility, VIX options, Volatility jumps, Option Calibration, Neural network, Cupy

## 🔧 Calibration Pipeline Overview

This project accelerates option-pricing calibration by **training a neural‐network surrogate** to replace repeated calls to an expensive analytical model. The workflow is split into an **offline data-generation phase** and an **online calibration phase**:

| Stage                           | Goal                                                       | Key Steps                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Offline (pre-training)**      | Build a high-quality dataset <br>for surrogate training    | 1. **Parameter sampling:** 5 000 Latin-Hypercube samples in a 7-D space  $(a,b,c,d,\kappa,r,t/365)$. <br>2. **Pricing engine:** Proposition-2 formula evaluated with high-precision quadrature (CuPy + mpmath). <br>3. **Market grid:** 4 maturities $(20,48,100,258\text{ days})$ × 34 strikes. <br>4. **Storage:** Columnar Parquet files for fast PyTorch loading. |
| **Online (surrogate training)** | Learn $f_\theta:(\text{params},T,K)\mapsto P_{\text{put}}$ | • **Architecture:** 8-64-32-32-1 fully-connected, ELU activations. <br>• **Optimiser:** Adam on MSE, `ReduceLROnPlateau`, early-stop (patience = 25). <br>• **Split:** 90 % train / 5 % val / 5 % test, stratified by $(T,K)$.                                                                                                                                        |
| **Calibration**                 | Recover parameters from live quotes                        | 1. **Global search:** PyGAD GA (pop = 600, gen = 3 000) maximises –MSE. <br>2. **Local refine:** L-BFGS-B (bounded) on the same loss. <br>3. **Surrogate in loop:** NN substitutes the costly integrator, giving millisecond evaluations.                                                                                                                             |

---

## 🚀 Results

| Metric                   | Value                                                                                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Best offline RMSE (val)  | **3.1 × 10⁻⁴**                                                                                                                                      |
| Calibration (Jan 2 2025) | GA + L-BFGS-B converged in **< 20 min**                                                                                                               |
| Fitted parameters        | $a=0.0498$, $b=0.8498$, $c=0.8575$, $d=0.7693$, $\kappa=7.7990$,<br>$r_{20} = 0.0020$, $r_{48} = -0.0013$, $r_{100} = -0.0060$, $r_{258} = -0.0124$ |
| 31-day VIX put back-test | MAE = 0.0329, RMSE = 0.0330                                                                                                          |

*Observation:* The model tracks the trend but shows an upward price bias; extremely low premiums on some strikes inflate relative errors.

---

## ⚠️ Limitations & Next Steps

* **Maturity pooling:** Current calibration pools all expiries.
  → **TODO:** bucket by maturity and calibrate slices independently.
* **Data quality:** Snapshot quotes contain arbitrage violations & thin volume. Do not filter the data using rough arbitrage checking method of put-call parity. 
  → Clean data using restrictions or weight data before fitting.


---

### How to Reproduce

```bash
# 1. Generate synthetic data (≈30 min on GPU)
python src/generate_dataset.py --out data/option_prices.parquet

# 2. Train the FNN surrogate
python src/train_fnn.py --data data/option_prices.parquet --save models/fnn_surrogate.pt

# 3. Calibrate to market quotes
python src/calibrate_ga_lbfgsb.py --model models/fnn_surrogate.pt --market data/vix_quotes.csv
```

