#!/usr/bin/env python3
"""
capillarity_pressure.py

Boolean-sphere medium + screened capillarity observable beta_app(Y):
- Experiment 1 (Fig. 1): Var[beta_app] vs V
    * 10 box sizes L
    * Error bars on Var estimates
    * Power-law fit Var ~ (1/V)^alpha in log-log
    * Reference slope-1 line
- Experiment 2 (Fig. 2): Ĉ(k) vs k, mark k_min = 2π / L
    * Same 10 L values as Fig. 1
    * Emphasize curves for L >= 10
    * Open markers only + inset zoom 10^{-4}–10^{-2}
- Experiment 3 (Fig. 3): self-interaction / bias
    * 10 values of L / a_max from ~1 to 5
    * Error bars on mean and variance
    * Exponential fits for mean and variance
- Experiment 4 (Timing): simple per-step wall-clock timing vs L

MPI/mpi4py version:
- Parallelizes over realizations (and over L in the spectral and self-interaction experiments)
- Rank 0 handles all file and figure output

Run e.g.:
    mpirun -np 8 python capillarity_rve_demo_mpi.py

Requirements:
    numpy, scipy, matplotlib, mpi4py
"""

import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -------------------------------------------------------------------
# MPI communicator:
#   - rank 0 is the master process (does all I/O and plotting)
#   - ranks 1..size-1 compute subsets of realizations / box sizes
# -------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------------
# Global plotting style (PR-E-like)
# -------------------------

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.top": True,
    "ytick.right": True,
})

# -------------------------
# Global config parameters
# -------------------------

@dataclass
class Config:
    """
    Container for all tunable parameters in the demo:
      - microstructure statistics (Boolean spheres)
      - physical parameters for the capillarity observable
      - numerical grid resolution
      - experiment-specific lists of L and sampling sizes
      - random seed for reproducibility
    """
    # Boolean-sphere parameters
    target_phi: float = 0.4          # desired solid volume fraction (approx)
    mu_logD: float = 0.0             # ln D mean (in arbitrary length units)
    sigma_logD: float = 0.4          # ln D std (controls polydispersity)
    # Capillarity / contrast
    K_g: float = 1.0                 # "grain" conductivity
    K_m: float = 0.1                 # "matrix" conductivity
    lambda_phys: float = 3.0         # capillary length (same units as D and L)
    # Grid / discretization
    points_per_unit: int = 8         # grid points per unit length
    # Variance vs volume experiment (Fig. 1) – 10 L values
    L_list: Tuple[float, ...] = (
        5.0, 6.0, 7.0, 8.0, 9.0,
        10.0, 11.0, 12.0, 13.0, 14.0
    )
    n_realizations_var: int = 50
    # Spectral experiment (Fig. 2) – same 10 L values
    L_list_spectral: Tuple[float, ...] = (
        5.0, 6.0, 7.0, 8.0, 9.0,
        10.0, 11.0, 12.0, 13.0, 14.0
    )
    n_bins_spectrum: int = 30
    # Self-interaction experiment (Fig. 3) – 10 points in L/a_max up to 5
    delta_quantile: float = 1e-3     # for a_max = Q_{1-delta}(D)
    L_factors_self: Tuple[float, ...] = (
        1.0, 1.25, 1.5, 1.75,
        2.0, 2.5, 3.0, 3.5,
        4.0, 5.0
    )
    n_realizations_self: int = 30
    # Timing experiment
    n_realizations_timing: int = 5
    # Random seeds
    base_seed: int = 1234


cfg = Config()

# Create output folder once at startup (fix for FileNotFoundError)
# (only rank 0 touches the filesystem)
if rank == 0:
    os.makedirs("FIGS", exist_ok=True)

# -------------------------
# Helper: lognormal diameters and Poisson intensity
# -------------------------

def lognormal_moments(mu: float, sigma: float) -> Tuple[float, float]:
    """
    Return E[D] and E[D^3] for D ~ lognormal(mu, sigma^2).

    Used to:
      - compute the mean volume of a grain,
      - infer the Poisson intensity for a prescribed volume fraction.
    """
    E_D = np.exp(mu + 0.5 * sigma**2)
    E_D3 = np.exp(3.0 * mu + 4.5 * sigma**2)
    return E_D, E_D3


def poisson_intensity_for_target_phi(phi_target: float,
                                     mu_logD: float,
                                     sigma_logD: float) -> float:
    """
    For a Poisson Boolean model of spheres with diameters D (lognormal),
    void fraction = exp(-nu * E[volume]),
    volume = pi D^3 / 6,
    so solid fraction = 1 - exp(-nu * E[volume]).
    Solve for nu given target phi.

    This gives a consistent Boolean-sphere medium independent of L.
    """
    _, E_D3 = lognormal_moments(mu_logD, sigma_logD)
    m3 = (np.pi / 6.0) * E_D3
    nu = -np.log(1.0 - phi_target) / m3
    return nu


def sample_diameters(rng: np.random.Generator,
                     n: int,
                     mu_logD: float,
                     sigma_logD: float) -> np.ndarray:
    """
    Sample n diameters from lognormal distribution.

    All spheres in a realization share the same lognormal law.
    """
    return np.exp(rng.normal(mu_logD, sigma_logD, size=n))


# -------------------------
# Boolean-sphere microstructure on a grid (periodic)
# -------------------------

def generate_boolean_spheres_grid(
    L: float,
    points_per_unit: int,
    nu: float,
    mu_logD: float,
    sigma_logD: float,
    seed: int
):
    """
    Generate a 3D Boolean-sphere medium on a cubic grid with edge L.

    Periodic boundary conditions are enforced via the minimum-image
    convention so that the microstructure is compatible with FFTs.

    Returns:
        chi: (N,N,N) boolean array, True in grains
        phi: solid volume fraction (mean of chi)
        diameters: diameters used for this realization
    """
    rng = np.random.default_rng(seed)
    V = L**3
    N_spheres = rng.poisson(nu * V)

    centers = rng.uniform(0.0, L, size=(N_spheres, 3))
    diameters = sample_diameters(rng, N_spheres, mu_logD, sigma_logD)
    radii = diameters / 2.0

    N = int(L * points_per_unit)
    ax = np.linspace(0.0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    chi = np.zeros((N, N, N), dtype=bool)

    # Loop over spheres and paint them into the indicator field
    for c, r in zip(centers, radii):
        dx = X - c[0]
        dy = Y - c[1]
        dz = Z - c[2]
        # minimum-image convention (periodic)
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        dz -= L * np.round(dz / L)
        mask = dx*dx + dy*dy + dz*dz <= r*r
        chi |= mask

    phi = chi.mean()
    return chi, phi, diameters


# -------------------------
# Screened capillarity observable beta_app(Y)
# -------------------------

def beta_app_from_phi(phi: float,
                      K_g: float,
                      K_m: float,
                      lambda_phys: float) -> float:
    """
    First-order homogenization result (exact for the linear reaction):
        beta_app(Y) = <K(x)/lambda^2>_Y
                    = (K_g * phi + K_m * (1 - phi)) / lambda^2

    In this demo we bypass a PDE solve and use this expression directly.
    """
    return (K_g * phi + K_m * (1.0 - phi)) / (lambda_phys**2)


# -------------------------
# Spectral density and radial average
# -------------------------

def radial_average_spectrum(chi: np.ndarray,
                            L: float,
                            nbins: int):
    """
    Compute radially averaged spectral density of the centered indicator
    X = chi - phi, using FFT.

    Steps:
      1. FFT of the zero-mean field X.
      2. Power spectrum S(k) = |FFT(X)|^2 / N^3.
      3. Bin in shells |k| in [0, k_max] and average.

    Returns:
      k_centers: array of radial wavenumber bin centers
      S_radial:  array of averaged spectral density values for each bin
    """
    N = chi.shape[0]
    assert chi.ndim == 3 and chi.shape[1] == N and chi.shape[2] == N
    phi = chi.mean()

    X = chi.astype(float) - phi

    F = np.fft.fftn(X)
    S = (F * np.conj(F)).real / (N**3)  # power spectrum
    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)

    k_flat = kmag.ravel()
    S_flat = S.ravel()

    kmax = k1d.max()
    bins = np.linspace(0.0, kmax, nbins + 1)
    idx = np.digitize(k_flat, bins) - 1

    k_centers = 0.5 * (bins[:-1] + bins[1:])
    S_radial = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=int)

    for i, val in enumerate(S_flat):
        b = idx[i]
        if 0 <= b < nbins:
            S_radial[b] += val
            counts[b] += 1

    S_radial = np.where(counts > 0, S_radial / counts, 0.0)
    return k_centers, S_radial


# -------------------------
# Experiment 1: Var[beta_app] vs V (MPI)
# -------------------------

def experiment_variance_vs_volume(cfg: Config):
    """
    For each L in cfg.L_list:
      - distribute cfg.n_realizations_var over MPI ranks
      - compute beta_app(Y_L) for each local realization
      - gather on rank 0, compute mean, variance, CV, stderr(Var)

    Rank 0 saves:
      - var_beta_vs_volume.csv
      - FIGS/var_beta_vs_invV.png (log–log with error bars and fits)
    """
    nu = poisson_intensity_for_target_phi(cfg.target_phi,
                                          cfg.mu_logD,
                                          cfg.sigma_logD)

    records = []

    # Loop over RVE sizes L
    for iL, L in enumerate(cfg.L_list):
        V = L**3
        betas_local = []

        # Distribute realizations across ranks by modulo
        for j in range(cfg.n_realizations_var):
            if j % size != rank:
                continue
            seed = cfg.base_seed + 1000 * iL + j
            chi, phi, _ = generate_boolean_spheres_grid(
                L=L,
                points_per_unit=cfg.points_per_unit,
                nu=nu,
                mu_logD=cfg.mu_logD,
                sigma_logD=cfg.sigma_logD,
                seed=seed
            )
            beta = beta_app_from_phi(phi, cfg.K_g, cfg.K_m, cfg.lambda_phys)
            betas_local.append(beta)

        # Gather all beta_app from every rank
        all_betas = comm.gather(betas_local, root=0)

        if rank == 0:
            betas = np.array([b for sub in all_betas for b in sub])
            assert len(betas) == cfg.n_realizations_var

            n = len(betas)
            mean_beta = betas.mean()
            var_beta = betas.var(ddof=1)
            cv_beta = np.sqrt(var_beta) / abs(mean_beta)
            # std error of sample variance for Gaussian: std(S^2) ≈ S^2 * sqrt(2/(n-1))
            stderr_var = var_beta * np.sqrt(2.0 / (n - 1))

            records.append((L, V, n, mean_beta, var_beta, cv_beta, stderr_var))

            print(f"[Var vs V] L={L:.3f}, V={V:.3f}, n={n}, "
                  f"mean(beta)={mean_beta:.4e}, Var={var_beta:.4e}, "
                  f"CV={cv_beta:.3f}, stderr(Var)={stderr_var:.4e}")

    if rank == 0:
        out = np.array(records,
                       dtype=[("L", float),
                              ("V", float),
                              ("n", int),
                              ("mean_beta", float),
                              ("var_beta", float),
                              ("cv_beta", float),
                              ("stderr_var", float)])
        np.savetxt(
            "var_beta_vs_volume.csv",
            np.column_stack([out["L"], out["V"], out["n"],
                             out["mean_beta"], out["var_beta"],
                             out["cv_beta"], out["stderr_var"]]),
            header="L, V, n, mean_beta, var_beta, cv_beta, stderr_var",
            delimiter=","
        )

        invV = 1.0 / out["V"]
        var_beta = out["var_beta"]
        stderr_var = out["stderr_var"]

        # Power-law fit: Var ~ C * (1/V)^alpha in log-log
        logx = np.log(invV)
        logy = np.log(var_beta)
        alpha, logC = np.polyfit(logx, logy, 1)
        C = np.exp(logC)
        print(f"[Var vs V] Power-law fit Var ≈ C (1/V)^alpha with "
              f"alpha = {alpha:.3f}, C = {C:.3e}")

        x_fit = np.linspace(invV.min(), invV.max(), 200)
        y_fit = C * x_fit**alpha

        # Reference slope-1 line through the largest-volume point
        i_ref = np.argmax(out["V"])  # largest box => smallest 1/V
        x_ref = invV[i_ref]
        y_ref = var_beta[i_ref]
        y_slope1 = y_ref * (x_fit / x_ref)**1.0

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.errorbar(invV, var_beta, yerr=stderr_var,
                    fmt="o", mfc="none", mec="k", ecolor="k",
                    capsize=3, label="data")
        ax.plot(x_fit, y_fit, "--",
                label=fr"power-law fit, $\alpha \approx {alpha:.2f}$",
                color="C0")
        ax.plot(x_fit, y_slope1, ":",
                label=r"reference slope $1$",
                color="C1")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$1/V$")
        ax.set_ylabel(r"$\mathrm{Var}[\beta_{\mathrm{app}}(Y)]$")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig("FIGS/var_beta_vs_invV.png", dpi=300)
        plt.close(fig)

        print("Saved var_beta_vs_volume.csv and FIGS/var_beta_vs_invV.png")


# -------------------------
# Experiment 2: spectral low-k coverage vs k_min (MPI over L)
# -------------------------

def experiment_spectral_lowk(cfg: Config):
    """
    For each L in cfg.L_list_spectral:
      - distribute L values across MPI ranks (iL % size == rank)
      - compute one realization and its radially averaged spectrum
    Gather on rank 0, then rank 0 saves CSVs and the figure.

    Figure 2:
      - top panel: full k-range, log–log, markers only (no lines),
        vertical dashed lines at k_min = 2π/L
      - bottom panel: zoom in 1 <= k <= k_max (same markers only),
        y-range [1e-2, 1e2]
    """
    nu = poisson_intensity_for_target_phi(cfg.target_phi,
                                          cfg.mu_logD,
                                          cfg.sigma_logD)

    spectra_local = []

    # Each rank computes spectra for its subset of L
    for iL, L in enumerate(cfg.L_list_spectral):
        if iL % size != rank:
            continue

        seed = cfg.base_seed + 5000 * iL
        chi, phi, _ = generate_boolean_spheres_grid(
            L=L,
            points_per_unit=cfg.points_per_unit,
            nu=nu,
            mu_logD=cfg.mu_logD,
            sigma_logD=cfg.sigma_logD,
            seed=seed
        )
        k_centers, S_rad = radial_average_spectrum(chi, L, cfg.n_bins_spectrum)
        k_min = 2.0 * np.pi / L
        spectra_local.append((L, k_centers, S_rad, k_min))

    # gather spectra from all ranks
    all_spectra = comm.gather(spectra_local, root=0)

    if rank == 0:
        spectra_data = [item for sub in all_spectra for item in sub]
        # sort by L so curves appear in progressive order
        spectra_data.sort(key=lambda x: x[0])

        # global k_max and zoom range: lower panel shows the decay region
        kmax_global = max(kc.max() for (_, kc, _, _) in spectra_data)
        kmin_zoom = 1.0           # 10^0
        kmax_zoom = kmax_global * 1.1  # small margin so markers don't touch the border

        # Markers: x (cross), empty square, empty triangle, repeating
        marker_cycle = ["x", "s", "^"]

        fig, (ax_full, ax_zoom) = plt.subplots(
            2, 1, figsize=(6, 7), sharex=False
        )

        for idx, (L, k_centers, S_rad, k_min) in enumerate(spectra_data):
            color = f"C{idx % 10}"
            marker = marker_cycle[idx % len(marker_cycle)]

            # style: markers only, empty markers for squares/triangles
            if marker == "x":
                marker_kwargs = dict(
                    marker=marker,
                    linestyle="none",
                    markersize=6.0,
                    markeredgewidth=1.4,
                    color=color,
                )
            else:
                marker_kwargs = dict(
                    marker=marker,
                    linestyle="none",
                    markersize=6.0,
                    markeredgewidth=1.4,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    color=color,
                )

            label = fr"$L = {L:.1f}$"

            # --- full range panel ---
            ax_full.loglog(k_centers, S_rad, **marker_kwargs, label=label)
            ax_full.axvline(
                k_min,
                linestyle="--",
                linewidth=1.0,
                color=color,
                alpha=0.5,
            )

            # --- zoom panel (10^0 to kmax_global) ---
            mask = (k_centers >= kmin_zoom) & (k_centers <= kmax_zoom)
            if np.any(mask):
                ax_zoom.loglog(
                    k_centers[mask],
                    S_rad[mask],
                    **marker_kwargs,
                )
                if kmin_zoom <= k_min <= kmax_zoom:
                    ax_zoom.axvline(
                        k_min,
                        linestyle="--",
                        linewidth=1.0,
                        color=color,
                        alpha=0.5,
                    )

        # ---- axes formatting (Phys. Rev style, no titles) ----
        # Top panel
        ax_full.set_xscale("log")
        ax_full.set_yscale("log")
        ax_full.set_xlabel(r"$k$")
        ax_full.set_ylabel(r"$\widehat{C}(k)$ (arb. units)")
        ax_full.legend(loc="best", ncol=2, frameon=False)

        # Bottom (zoom) panel: focus on the power-law tail
        ax_zoom.set_xscale("log")
        ax_zoom.set_yscale("log")
        ax_zoom.set_xlim(kmin_zoom, kmax_zoom)
        ax_zoom.set_ylim(1e-2, 1e2)  # 10^{-2} to 10^{2}
        ax_zoom.set_xlabel(r"$k$")
        ax_zoom.set_ylabel(r"$\widehat{C}(k)$ (arb. units)")

        fig.tight_layout()
        fig.savefig("FIGS/spectral_Ck_with_kmin.png", dpi=300)
        plt.close(fig)

        # save each spectrum as CSV (unchanged)
        for (L, k_centers, S_rad, k_min) in spectra_data:
            fname = f"spectrum_L{L:.2f}".replace(".", "p") + ".csv"
            np.savetxt(
                fname,
                np.column_stack([k_centers, S_rad]),
                header=f"k, C_hat(k)   (L={L}, k_min={k_min})",
                delimiter=","
            )
        print("Saved spectra CSVs (spectrum_L*.csv) and FIGS/spectral_Ck_with_kmin.png")

# -------------------------
# Experiment 3: self-interaction / bias (L < L⋆) – MPI
# -------------------------

def approximate_a_max_from_distribution(cfg: Config) -> float:
    """
    Approximate a_max = Q_{1-delta}(D) for lognormal diameters
    by Monte Carlo sampling.

    This is done once (on rank 0) and then broadcast to all ranks.
    """
    rng = np.random.default_rng(cfg.base_seed + 999)
    nsample = 200_000
    Ds = sample_diameters(rng, nsample, cfg.mu_logD, cfg.sigma_logD)
    a_max = np.quantile(Ds, 1.0 - cfg.delta_quantile)
    return a_max


def experiment_self_interaction(cfg: Config):
    """
    Evaluate how self-interaction and bias appear when L is too small relative
    to a high grain-size quantile a_max.

    For each factor in cfg.L_factors_self:
      - set L ≈ factor * a_max
      - distribute cfg.n_realizations_self over MPI ranks
      - compute mean and variance of beta_app
      - rank 0 stores and plots results; also saves slices for visualization.

    Final plot:
      - mean and variance vs L/a_max with error bars
      - exponential fits to both mean and variance
    """
    nu = poisson_intensity_for_target_phi(cfg.target_phi,
                                          cfg.mu_logD,
                                          cfg.sigma_logD)

    # a_max computed on rank 0 and broadcast
    if rank == 0:
        a_max = approximate_a_max_from_distribution(cfg)
        print(f"Estimated a_max (Q_1-delta) ~ {a_max:.3f}")
    else:
        a_max = None
    a_max = comm.bcast(a_max, root=0)

    records = []

    # Sweep over L/a_max factors to visualize self-interaction
    for iF, factor in enumerate(cfg.L_factors_self):
        L = factor * a_max
        betas_local = []

        if rank == 0:
            print(f"[Self-interaction] Factor={factor:.2f}, L≈{L:.3f}")

        # Distribute realizations across ranks
        for j in range(cfg.n_realizations_self):
            if j % size != rank:
                continue
            seed = cfg.base_seed + 8000 * iF + j
            chi, phi, _ = generate_boolean_spheres_grid(
                L=L,
                points_per_unit=cfg.points_per_unit,
                nu=nu,
                mu_logD=cfg.mu_logD,
                sigma_logD=cfg.sigma_logD,
                seed=seed
            )
            beta = beta_app_from_phi(phi, cfg.K_g, cfg.K_m, cfg.lambda_phys)
            betas_local.append(beta)

        all_betas = comm.gather(betas_local, root=0)

        if rank == 0:
            betas = np.array([b for sub in all_betas for b in sub])
            assert len(betas) == cfg.n_realizations_self

            n = len(betas)
            mean_beta = betas.mean()
            var_beta = betas.var(ddof=1)
            cv_beta = np.sqrt(var_beta) / abs(mean_beta)
            stderr_mean = np.sqrt(var_beta / n)
            stderr_var = var_beta * np.sqrt(2.0 / (n - 1))

            records.append((factor, L, a_max, n,
                            mean_beta, var_beta, cv_beta,
                            stderr_mean, stderr_var))

            print(f"  n={n}, mean(beta)={mean_beta:.4e}, "
                  f"Var={var_beta:.4e}, CV={cv_beta:.3f}")

            # Representative slice for this L (deterministic extra realization)
            seed_slice = cfg.base_seed + 8000 * iF + 0
            chi_slice, _, _ = generate_boolean_spheres_grid(
                L=L,
                points_per_unit=cfg.points_per_unit,
                nu=nu,
                mu_logD=cfg.mu_logD,
                sigma_logD=cfg.sigma_logD,
                seed=seed_slice
            )
            mid_z = chi_slice.shape[2] // 2
            plt.figure(figsize=(4, 4))
            plt.imshow(chi_slice[:, :, mid_z].T,
                       origin="lower",
                       cmap="gray")
            # No title; manuscript caption will describe panels
            plt.axis("off")
            fname = f"FIGS/self_interaction_slice_L{L:.2f}.png".replace(".", "p")
            plt.tight_layout()
            plt.savefig(fname, dpi=300)
            plt.close()

    if rank == 0:
        out = np.array(records,
                       dtype=[("factor", float),
                              ("L", float),
                              ("a_max", float),
                              ("n", int),
                              ("mean_beta", float),
                              ("var_beta", float),
                              ("cv_beta", float),
                              ("stderr_mean", float),
                              ("stderr_var", float)])
        np.savetxt(
            "self_interaction_beta_stats.csv",
            np.column_stack([out["factor"], out["L"], out["a_max"], out["n"],
                             out["mean_beta"], out["var_beta"], out["cv_beta"],
                             out["stderr_mean"], out["stderr_var"]]),
            header=("factor, L, a_max, n, mean_beta, var_beta, cv_beta, "
                    "stderr_mean, stderr_var"),
            delimiter=","
        )

        ratio = out["L"] / out["a_max"]
        idx = np.argsort(ratio)
        ratio_sorted = ratio[idx]
        mean_beta_sorted = out["mean_beta"][idx]
        var_beta_sorted = out["var_beta"][idx]
        stderr_mean_sorted = out["stderr_mean"][idx]
        stderr_var_sorted = out["stderr_var"][idx]

        # Exponential fit for mean: beta(L/a_max) ≈ beta_inf + A exp(-r / Lc)
        def exp_model_mean(r, beta_inf, A, Lc):
            return beta_inf + A * np.exp(-r / Lc)

        p0_mean = (mean_beta_sorted[-1],
                   mean_beta_sorted[0] - mean_beta_sorted[-1],
                   1.0)
        popt_mean, _ = curve_fit(exp_model_mean,
                                 ratio_sorted,
                                 mean_beta_sorted,
                                 p0=p0_mean,
                                 maxfev=10000)
        beta_inf, A_mean, Lc_mean = popt_mean
        print(f"[Self-interaction] Mean fit: beta_inf={beta_inf:.4e}, "
              f"A={A_mean:.4e}, Lc={Lc_mean:.3f}")

        # Exponential fit for variance: Var(L/a_max) ≈ Var_inf + B exp(-r / Lc_var)
        def exp_model_var(r, var_inf, B, Lc_var):
            return var_inf + B * np.exp(-r / Lc_var)

        p0_var = (var_beta_sorted[-1],
                  var_beta_sorted[0] - var_beta_sorted[-1],
                  1.0)
        popt_var, _ = curve_fit(exp_model_var,
                                ratio_sorted,
                                var_beta_sorted,
                                p0=p0_var,
                                maxfev=10000)
        var_inf, B_var, Lc_var = popt_var
        print(f"[Self-interaction] Var fit: var_inf={var_inf:.4e}, "
              f"B={B_var:.4e}, Lc_var={Lc_var:.3f}")

        ratio_fit = np.linspace(ratio_sorted.min(),
                                ratio_sorted.max(), 200)
        mean_fit = exp_model_mean(ratio_fit, *popt_mean)
        var_fit = exp_model_var(ratio_fit, *popt_var)

        fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        # Mean with error bars + exponential fit (open circles)
        axs[0].errorbar(ratio_sorted, mean_beta_sorted,
                        yerr=stderr_mean_sorted,
                        fmt="o", mfc="none", mec="k",
                        ecolor="k", capsize=3,
                        label=r"data")
        axs[0].plot(ratio_fit, mean_fit, "--", color="C0",
                    label=r"exp fit")
        axs[0].set_ylabel(r"$\overline{\beta_{\rm app}}$")
        axs[0].legend(frameon=False)

        # Variance with error bars + exponential fit (open squares)
        axs[1].errorbar(ratio_sorted, var_beta_sorted,
                        yerr=stderr_var_sorted,
                        fmt="s", mfc="none", mec="k",
                        ecolor="k", capsize=3,
                        label=r"data")
        axs[1].plot(ratio_fit, var_fit, "--", color="C1",
                    label=r"exp fit")
        axs[1].set_xlabel(r"$L/a_{\max}$")
        axs[1].set_ylabel(r"$\mathrm{Var}[\beta_{\rm app}]$")
        axs[1].legend(frameon=False)

        fig.tight_layout()
        fig.savefig("FIGS/self_interaction_beta_vs_L_over_amax.png", dpi=300)
        plt.close(fig)

        print("Saved self_interaction_beta_stats.csv, slices in FIGS/, "
              "and FIGS/self_interaction_beta_vs_L_over_amax.png.")


# -------------------------
# Experiment 4: simple timing / cost vs L (rank 0 only)
# -------------------------

def experiment_timing(cfg: Config):
    """
    Simple timing experiment (rank 0 only) to estimate the cost per realization
    as a function of box size L for:
      - microstructure generation
      - spectral FFT / radial average
      - observable evaluation (beta_app from phi; toy, no PDE)

    Results:
      - timing_vs_L.csv
      - FIGS/timing_vs_L.png
    """
    if rank != 0:
        return

    nu = poisson_intensity_for_target_phi(cfg.target_phi,
                                          cfg.mu_logD,
                                          cfg.sigma_logD)

    L_list_timing = cfg.L_list
    nrep = cfg.n_realizations_timing

    records = []

    print("\n[Timing] Measuring cost per realization vs L")
    for L in L_list_timing:
        N = int(L * cfg.points_per_unit)
        print(f"[Timing] L={L:.2f}, N={N}, nrep={nrep}")

        t_micro = []
        t_spec = []
        t_obs = []

        for j in range(nrep):
            seed = cfg.base_seed + 9000 + j

            # Microstructure generation
            t0 = time.perf_counter()
            chi, phi, _ = generate_boolean_spheres_grid(
                L=L,
                points_per_unit=cfg.points_per_unit,
                nu=nu,
                mu_logD=cfg.mu_logD,
                sigma_logD=cfg.sigma_logD,
                seed=seed
            )
            t1 = time.perf_counter()
            t_micro.append(t1 - t0)

            # Spectral FFT
            t0 = time.perf_counter()
            _k_centers, _S_rad = radial_average_spectrum(
                chi, L, cfg.n_bins_spectrum
            )
            t1 = time.perf_counter()
            t_spec.append(t1 - t0)

            # Observable evaluation (toy)
            t0 = time.perf_counter()
            _beta = beta_app_from_phi(phi, cfg.K_g, cfg.K_m, cfg.lambda_phys)
            t1 = time.perf_counter()
            t_obs.append(t1 - t0)

        micro_mean = float(np.mean(t_micro))
        micro_std = float(np.std(t_micro, ddof=1))
        spec_mean = float(np.mean(t_spec))
        spec_std = float(np.std(t_spec, ddof=1))
        obs_mean = float(np.mean(t_obs))
        obs_std = float(np.std(t_obs, ddof=1))
        total_mean = micro_mean + spec_mean + obs_mean

        records.append((L, N, nrep,
                        micro_mean, micro_std,
                        spec_mean, spec_std,
                        obs_mean, obs_std,
                        total_mean))

        print(f"  micro: {micro_mean:.3e} s, "
              f"spectrum: {spec_mean:.3e} s, "
              f"observable: {obs_mean:.3e} s, "
              f"total: {total_mean:.3e} s")

    out = np.array(records,
                   dtype=[("L", float),
                          ("N", int),
                          ("nrep", int),
                          ("micro_mean", float),
                          ("micro_std", float),
                          ("spec_mean", float),
                          ("spec_std", float),
                          ("obs_mean", float),
                          ("obs_std", float),
                          ("total_mean", float)])
    np.savetxt(
        "timing_vs_L.csv",
        np.column_stack([out["L"], out["N"], out["nrep"],
                         out["micro_mean"], out["micro_std"],
                         out["spec_mean"], out["spec_std"],
                         out["obs_mean"], out["obs_std"],
                         out["total_mean"]]),
        header=("L, N, nrep, micro_mean, micro_std, "
                "spec_mean, spec_std, obs_mean, obs_std, total_mean"),
        delimiter=","
    )

    # Plot (no title, for manuscript use), open markers
    L_arr = out["L"]
    micro_mean = out["micro_mean"]
    spec_mean = out["spec_mean"]
    obs_mean = out["obs_mean"]
    total_mean = out["total_mean"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(L_arr, micro_mean, "o-", mfc="none", mec="C0", label="microstructure")
    ax.plot(L_arr, spec_mean, "s-", mfc="none", mec="C1", label="spectrum (FFT)")
    ax.plot(L_arr, obs_mean, "d-", mfc="none", mec="C2", label="observable (toy)")
    ax.plot(L_arr, total_mean, "^-", mfc="none", mec="C3", label="total")

    ax.set_xlabel(r"$L$")
    ax.set_ylabel("time per realization [s]")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("FIGS/timing_vs_L.png", dpi=300)
    plt.close(fig)

    print("Saved timing_vs_L.csv and FIGS/timing_vs_L.png.")


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    # The three physics/geometry experiments are run in sequence,
    # each separated by an MPI barrier. Rank 0 prints progress messages
    # and writes all CSV/PNG output to the FIGS/ folder.
    if rank == 0:
        print("Running convergence study Var[beta_app] vs V (MPI)...")
    experiment_variance_vs_volume(cfg)

    comm.Barrier()
    if rank == 0:
        print("\nRunning spectral low-k coverage experiment (MPI over L)...")
    experiment_spectral_lowk(cfg)

    comm.Barrier()
    if rank == 0:
        print("\nRunning self-interaction / bias experiment (MPI)...")
    experiment_self_interaction(cfg)

    comm.Barrier()
    if rank == 0:
        print("\nRunning timing experiment (rank 0 only)...")
    experiment_timing(cfg)

    comm.Barrier()
    if rank == 0:
        print("\nDone.")