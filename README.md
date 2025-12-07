# capillarity-supercell-sizing

## Overview

This repository contains a reproducible Python/MPI workflow that supports the paper:

> *Capillarity in Stationary Random Granular Media: Distribution-Aware Screening and Quantitative Supercell Sizing*

The main driver script

- builds 3D Boolean-sphere media with prescribed volume fraction and size distribution,
- evaluates a screened capillarity observable \(\beta_{\rm app}(Y)\),
- performs convergence and spectral checks of the variance law,
- quantifies self-interaction artifacts when \(L/a_{\max}\) is too small,
- and measures computational cost versus box size.

All production figures in the paper’s numerical validation section are generated from this script.

---

## Requirements

- Python 3.9+  
- `numpy`  
- `scipy`  
- `matplotlib`  
- `mpi4py`  
- An MPI implementation (e.g. OpenMPI, MPICH)

Example (conda) environment:

```bash
conda install numpy scipy matplotlib mpi4py
```

or with `pip` inside an MPI-enabled environment:

```bash
pip install numpy scipy matplotlib mpi4py
```

---

## Main script

The core script is:

- `capillarity_rve_demo_mpi.py`  
  (in your case renamed as `capillarity_pressure.py` with the same content)

It implements four experiments:

1. **Experiment 1 – Variance vs. volume (Fig. 1)**  
   - 10 box sizes \(L\).  
   - Computes \(\beta_{\rm app}(Y_L)\) for many realizations.  
   - Plots \(\mathrm{Var}[\beta_{\rm app}(Y)]\) vs \(1/V\) with error bars.  
   - Fits a power law \(\mathrm{Var} \sim (1/V)^\alpha\) in log–log coordinates and overlays a reference slope-1 line.

2. **Experiment 2 – Spectral low-\(k\) coverage (Fig. 2)**  
   - Same 10 box sizes as in Fig. 1.  
   - Computes radially averaged spectral density \(\widehat{C}(k)\) of the phase indicator.  
   - Top panel: full \(k\)-range, log–log, markers only, vertical dashed lines at \(k_{\min} = 2\pi/L\).  
   - Bottom panel: zoom for \(k \in [10^{0}, k_{\max}]\), \(y \in [10^{-2},10^{2}]\), same marker style.  
   - Shows collapse of spectra at large \(k\) for \(L > 10\).

3. **Experiment 3 – Self-interaction and bias (Fig. 3)**  
   - Uses a high quantile \(a_{\max} = Q_{1-\delta}(D)\) of the grain diameters.  
   - Samples 10 values of \(L/a_{\max}\) in \([1, 5]\).  
   - For each ratio, computes the mean and variance of \(\beta_{\rm app}(Y_L)\) with error bars.  
   - Fits exponential models for both mean and variance vs \(L/a_{\max}\) to quantify decay of self-interaction artifacts.  
   - Stores representative 2D slices of the microstructure for visualization.

4. **Experiment 4 – Timing / cost vs. \(L\) (Fig. 4)**  
   - Measures wall-clock time per realization as a function of \(L\) for:
     - microstructure generation,  
     - spectral FFT and radial averaging,  
     - observable evaluation (toy, via \(\phi\)).  
   - Plots each component and the total cost vs \(L\) (log-scale on the \(y\)-axis).

---

## Running the code

Typical MPI run (adjust `-np` to your machine):

```bash
mpirun -np 8 python capillarity_rve_demo_mpi.py
```

or, if you renamed it:

```bash
mpirun -np 8 python capillarity_pressure.py
```

Rank 0 (only) handles:

- creation of the `FIGS/` folder,
- saving figures,
- writing CSV files with statistics and timing.

---

## Output

After a successful run, you should see:

- **Figures** in `FIGS/`:
  - `var_beta_vs_invV.png` – variance vs \(1/V\) (Fig. 1).  
  - `spectral_Ck_with_kmin.png` – spectral density with \(k_{\min}\) markers, two-panel layout (Fig. 2).  
  - `self_interaction_beta_vs_L_over_amax.png` – mean and variance vs \(L/a_{\max}\) with exponential fits (Fig. 3).  
  - `timing_vs_L.png` – time per realization vs \(L\) (Fig. 4).  
  - `self_interaction_slice_L*.png` – representative 2D slices illustrating self-interaction.

- **CSV files**:
  - `var_beta_vs_volume.csv` – statistics for Fig. 1.  
  - `spectrum_L*.csv` – spectral data for each \(L\) (used in Fig. 2).  
  - `self_interaction_beta_stats.csv` – statistics vs \(L/a_{\max}\) for Fig. 3.  
  - `timing_vs_L.csv` – timing data for Fig. 4.

These files allow you to remake or further post-process each figure.

---

## Repository structure (suggested)

```text
capillarity-supercell-sizing/
├─ capillarity_rve_demo_mpi.py      # Main MPI script
├─ README.md                        # This file
├─ requirements.txt                 # Optional: list Python deps
├─ FIGS/                            # Created automatically; stores PNG figures
└─ data/                            # (Optional) extra processed data or examples
```

---

## License

> This code is released under the MIT License. See `LICENSE` for details.

---

## Citation

If you use this code in academic work, please cite:

> C. Tantardini *et al.*, “Capillarity in Stationary Random Granular Media: Distribution-Aware Screening and Quantitative Supercell Sizing”, submitted to *Physical Review E* (2025).
