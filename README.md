# GPU-NN-Option-Calibration

Pricing and calibration experiments for power-type derivatives under a rough-volatility model with jumps. The repository includes numerical pricing routines, feed-forward neural-network surrogates, and calibration scripts for VIX put quotes.

The model follows the fractional Ornstein–Uhlenbeck variance framework with a tempered-stable Lévy subordinator and an independent sinusoidal jump component described by Wang and Xia (2022). The scripts compare direct numerical pricing with surrogate-based calibration.

Keywords: rough volatility, VIX options, volatility jumps, option calibration, neural networks, CuPy.

Reference: Wang and W. Xia. Power-type derivatives for rough volatility with jumps. Journal of Futures Markets, 42(10):1369–1406, 2022.

## Repository layout

| Directory | Contents |
|---|---|
| `Proposition2_split_data/` | Characteristic-function calculations, CPU/GPU comparisons, and calibration variants. |
| `TTM_Days_Model/` | Pricing-data generation, surrogate training, and calibration with maturity expressed in days. |
| `TTM_Ratio_Model/` | Pricing-data generation, surrogate training, calibration, and time-series pricing with maturity ratios. |
| `Data_Results/` | Market-data snapshots, generated tables, checkpoints, scalers, plots, and saved run logs. |

The surrogate takes `(a, b, c, d, kappa, r, t, K)` as inputs and predicts a put price. The training configurations use ELU networks with hidden widths `[64, 32, 32]`, Adam, a learning-rate scheduler, and early stopping. Calibration variants combine a global search (GA or CMA-ES) with local refinement. The global-search settings and loss definition are specified in each script.

## Saved results

| Quantity | Saved or reported value |
|---|---|
| Surrogate validation RMSE reported for the project | 3.1 × 10⁻⁴ |
| Total calibration time in `Data_Results/Calibration_Result/summarization_.txt` | 24,203.71 seconds, approximately 6.72 hours |
| Final MSE in that calibration log | 2.1076569768694145 × 10⁻⁶ |
| Fitted model parameters from that log | `a=0.049762`, `b=0.849782`, `c=0.857500`, `d=0.769302`, `kappa=7.798968` |
| Maturity-specific rates in the code's `[258, 100, 48, 20]` day order | `r_258=0.001980`, `r_100=-0.001292`, `r_48=-0.006008`, `r_20=-0.012427` |
| Reported 31-observation VIX put time-series check | MAE = 0.0329, RMSE = 0.0330 |

The time-series prices track the trend with an upward bias. Very low premiums on some strikes produce large relative errors. Error and timing comparisons depend on the selected training and calibration configuration.

The neural-network trainers use a holdout split for early stopping and validation. Legacy output filenames containing `test` show predictions on this validation split. An independent held-out test metric is not reported by those trainers.

## Running the scripts

The entry points below exist in this repository. Input files, checkpoints, scalers, and outputs are resolved beneath `GPU_NN_PROJECT_ROOT`, which defaults to this repository's `Data_Results/` directory. Set that environment variable to an absolute directory to use another local workspace. Paths resolve from each script's location, so the defaults also work when a script is launched from its own subdirectory. Preserve the configured subdirectories and filenames when arranging the data.

These are individual research scripts. Review the selected configuration and required artifacts before running; the sequence below is not a tested one-command reproduction package.

| Task | Entry point |
|---|---|
| Generate pricing data with maturity in days | `TTM_Days_Model/Split_data_put_option_price_mpmath.py` |
| Train the corresponding surrogate | `TTM_Days_Model/NN-Train.py` |
| Calibrate with the day-input surrogate | `TTM_Days_Model/Option_cali_NN_T_DAYs.py` |
| Generate ratio-input pricing data | `TTM_Ratio_Model/Split_data_put_option_price_mpmath_ratio.py` |
| Train the ratio-input surrogate | `TTM_Ratio_Model/NN-Train_ratio.py` |
| Calibrate with the ratio-input surrogate | `TTM_Ratio_Model/Option_cali_NN_genetic_MSE_ratio.py` |
| Run direct GPU calibration for the saved parameter convention | `TTM_Ratio_Model/Final_Calibration.py` |
| Price the time-series quotes with the saved parameters | `TTM_Ratio_Model/Option_time_series_analysis.py` |

For example, after configuring the data and checkpoint locations:

```bash
python "TTM_Days_Model/Split_data_put_option_price_mpmath.py"
python "TTM_Days_Model/NN-Train.py"
python "TTM_Days_Model/Option_cali_NN_T_DAYs.py"
```

The ratio generator currently samples 6,000 six-dimensional parameter vectors. Several data/checkpoint filenames retain the historical `5000` stem. Maturity conversion also differs across variants: the ratio generator uses 252 trading days, while direct calibration and time-series scripts use 365 calendar days. Match the selected generator, trainer, calibrator, maturity convention, and checkpoint configuration; the variants are not interchangeable without checking those settings.

The data-preparation notebook is `Data_Results/data_validatin/Data_preparing.ipynb`. Start it in its data directory or set `VIX_DATA_DIR` to the directory containing `high_frequency.csv`, `jan_02.csv`, and `48_k=0.2_modified.csv`. Its saved numerical diagnostics and warnings are retained.

Dependencies used by the scripts include NumPy, pandas, SciPy, mpmath, CuPy, PyTorch, scikit-learn, joblib, Matplotlib, tqdm, PyGAD, CMA-ES, and the relevant Excel/Parquet readers. GPU scripts require a compatible CUDA/CuPy environment.

## Limitations and further work

- Calibration currently pools expiries. Independent maturity-slice calibration is a possible extension.
- Snapshot quotes have thin volume and apparent arbitrage violations. The notebook's simple parity and shape checks require the correct contract, maturity, and pricing conventions before they can be used as data filters. Data restrictions or weighting remain part of the calibration design.
- Low-premium options can have large relative errors even when their absolute errors are small. Both scales are useful when assessing the fit.
