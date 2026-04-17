"""
Nature-Quality Figures for PINN Reef Thermal Paper
===================================================
Generates 6 publication figures for Environmental Modelling and Software.

  Fig 1: Study area map (single column, 90mm)
  Fig 2: Performance overview — heatmap + scatter + win rate (double column, 190mm)
  Fig 3: Sparsity robustness — Davies 5.0m + 9.1m (single column, 90mm)
  Fig 4: Holdout time series — 2×2 grid (double column, 190mm) [placeholder]
  Fig 5: Depth-resolved thermal stress — DHD profile (double column, 190mm)
  Fig 6: Learned physical parameters — κ + Kd (single column, 90mm)

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --skip-map   # skip Fig 1 (needs cartopy)
    python scripts/make_figures.py --fig4       # run Fig 4 (needs GPU, ~16 min)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════
# Global paths
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "figures")
EXP14_PATH = os.path.join(BASE_DIR, "results", "exp14_results.json")
EXP15_PATH = os.path.join(BASE_DIR, "results", "exp15_results.json")
EXP16_PATH = os.path.join(BASE_DIR, "results", "exp16_results.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Nature / RSE style configuration
# ═══════════════════════════════════════════════════════════════════════
MM_TO_INCH = 1 / 25.4
SINGLE_COL = 90 * MM_TO_INCH   # 90 mm
DOUBLE_COL = 190 * MM_TO_INCH  # 190 mm

def setup_nature_style():
    """Set rcParams for Nature-standard figures."""
    plt.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # PDF text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Lines and axes
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        # Layout
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Grid off by default
        "axes.grid": False,
        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
    })

# ═══════════════════════════════════════════════════════════════════════
# Okabe-Ito colorblind-safe palette
# ═══════════════════════════════════════════════════════════════════════
REEF_COLORS = {
    "davies_reef":   "#E69F00",  # amber
    "rib_reef":      "#56B4E9",  # sky blue
    "myrmidon_reef": "#009E73",  # teal
    "kelso_reef":    "#0072B2",  # blue
}
REEF_SHORT = {
    "davies_reef": "Davies",
    "rib_reef": "Rib",
    "myrmidon_reef": "Myrmidon",
    "kelso_reef": "Kelso",
}

METHOD_COLORS = {
    "GP":        "#009E73",  # teal
    "IDW":       "#56B4E9",  # sky blue
    "NN":        "#CC79A7",  # pink
    "RF":        "#0072B2",  # blue
    "FD":        "#882255",  # wine/plum
    "Satellite": "#999999",  # gray
    "PINN":      "#D55E00",  # vermillion
}
METHOD_ORDER = ["GP", "IDW", "NN", "RF", "FD", "PINN"]

# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════
def load_exp14():
    with open(EXP14_PATH) as f:
        return json.load(f)

def load_exp15():
    with open(EXP15_PATH) as f:
        return json.load(f)

def load_exp16():
    with open(EXP16_PATH) as f:
        return json.load(f)

def get_full_depth_runs(results):
    """Return one row per (reef, holdout_depth) with max n_training_depths."""
    grouped = {}
    for r in results:
        key = (r["reef"], r["holdout_depth"])
        if key not in grouped or r["n_training_depths"] > grouped[key]["n_training_depths"]:
            grouped[key] = r
    return sorted(grouped.values(), key=lambda r: (r["reef"], r["holdout_depth"]))

def add_panel_label(ax, label, x=-0.12, y=1.06):
    """Add bold lowercase panel label (a, b, c, ...)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

def savefig(fig, name):
    """Save PNG + PDF."""
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")

def best_method(row):
    """Return name of method with lowest RMSE."""
    rmses = {m: row.get(f"{m}_rmse") for m in METHOD_ORDER}
    rmses = {k: v for k, v in rmses.items() if v is not None}
    return min(rmses, key=rmses.get) if rmses else None


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Study Area Map (single column, 90mm)
# ═══════════════════════════════════════════════════════════════════════
def fig1_study_area():
    """Publication-quality GBR study area map with Australia inset."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from pinn_reef_thermal import REEF_COORDS

    reef_markers = {
        "davies_reef":   "o",
        "rib_reef":      "s",
        "myrmidon_reef": "^",
        "kelso_reef":    "D",
    }
    # Offsets to avoid label overlap (lon, lat)
    reef_offsets = {
        "davies_reef":   ( 0.08,  0.06),
        "rib_reef":      ( 0.08,  0.06),
        "myrmidon_reef": ( 0.08,  0.06),
        "kelso_reef":    ( 0.08, -0.12),
    }

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(SINGLE_COL, SINGLE_COL * 1.1))

    # Main panel
    ax = fig.add_axes([0.02, 0.02, 0.70, 0.96], projection=proj)
    ax.set_extent([145.8, 148.2, -19.5, -17.5], crs=proj)

    ax.add_feature(cfeature.LAND, facecolor="#f0e6d2", edgecolor="#666666", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="#d6eaf8")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="#333333")

    # Plot reefs
    for reef, (lat, lon) in REEF_COORDS.items():
        marker = reef_markers[reef]
        color = REEF_COLORS[reef]
        label = REEF_SHORT[reef]
        ax.plot(lon, lat, marker=marker, color=color, markersize=7,
                markeredgecolor="black", markeredgewidth=0.8,
                transform=proj, zorder=10, label=label)
        dx, dy = reef_offsets[reef]
        ax.text(lon + dx, lat + dy, label, fontsize=7, fontweight="bold",
                transform=proj, zorder=11,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="0.7", alpha=0.85, linewidth=0.4))

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    # Scale bar
    scale_lon, scale_lat = 147.5, -19.35
    scale_km = 50
    scale_deg = scale_km / 111.32
    ax.plot([scale_lon, scale_lon + scale_deg], [scale_lat, scale_lat],
            "k-", linewidth=1.5, transform=proj)
    ax.text(scale_lon + scale_deg / 2, scale_lat - 0.06, f"{scale_km} km",
            ha="center", fontsize=6, transform=proj)

    ax.legend(loc="lower right", fontsize=7, framealpha=0.9,
              edgecolor="0.7", markerscale=0.9, handletextpad=0.3)

    # Inset: Australia
    ax_in = fig.add_axes([0.65, 0.58, 0.33, 0.38], projection=proj)
    ax_in.set_extent([110, 160, -45, -8], crs=proj)
    ax_in.add_feature(cfeature.LAND, facecolor="#f0e6d2", edgecolor="#666666", linewidth=0.3)
    ax_in.add_feature(cfeature.OCEAN, facecolor="#d6eaf8")
    ax_in.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#333333")

    from matplotlib.patches import Rectangle
    rect = Rectangle((145.8, -19.5), 2.4, 2.0,
                      linewidth=1.2, edgecolor="red", facecolor="red",
                      alpha=0.25, transform=proj)
    ax_in.add_patch(rect)

    savefig(fig, "fig1_study_area")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Performance Overview (double column, 190mm)
#   (a) RMSE heatmap  (b) PINN advantage scatter  (c) Win rate bar
# ═══════════════════════════════════════════════════════════════════════
def fig2_performance(results):
    """Three-panel performance overview."""
    full_runs = get_full_depth_runs(results)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38),
        gridspec_kw={"width_ratios": [3, 2, 1.3], "wspace": 0.55}
    )

    # ── Panel (a): RMSE heatmap ──────────────────────────────────────
    methods = METHOD_ORDER  # GP, IDW, NN, RF, PINN
    n_holdouts = len(full_runs)
    n_methods = len(methods)
    rmse_matrix = np.full((n_holdouts, n_methods), np.nan)
    row_labels = []

    for i, r in enumerate(full_runs):
        reef_short = REEF_SHORT[r["reef"]]
        d = r['holdout_depth']
        depth_str = f"{d:.0f}" if d == int(d) else f"{d:.1f}"
        row_labels.append(f"{reef_short} {depth_str}m")
        for j, m in enumerate(methods):
            val = r.get(f"{m}_rmse")
            if val is not None:
                rmse_matrix[i, j] = val

    im = ax_a.imshow(rmse_matrix, aspect="auto", cmap="YlOrRd",
                     vmin=0, vmax=np.nanmax(rmse_matrix) * 0.85)

    ax_a.set_xticks(range(n_methods))
    ax_a.set_xticklabels(methods, rotation=45, ha="right")
    ax_a.set_yticks(range(n_holdouts))
    ax_a.set_yticklabels(row_labels)

    # Annotate cells: bold best method per row
    for i in range(n_holdouts):
        row_vals = rmse_matrix[i]
        best_j = np.nanargmin(row_vals) if not np.all(np.isnan(row_vals)) else -1
        for j in range(n_methods):
            val = rmse_matrix[i, j]
            if not np.isnan(val):
                is_best = (j == best_j)
                color = "white" if val > np.nanmedian(rmse_matrix) else "black"
                weight = "bold" if is_best else "normal"
                txt = f"{val:.2f}"
                if is_best:
                    txt = f"{val:.2f}*"
                ax_a.text(j, i, txt, ha="center", va="center",
                         fontsize=5.5, color=color, fontweight=weight)

    cb = fig.colorbar(im, ax=ax_a, orientation="horizontal",
                      fraction=0.04, pad=0.18, aspect=25, shrink=0.6)
    cb.set_label("RMSE (\u00b0C)", fontsize=6)
    cb.ax.tick_params(labelsize=5, length=2)
    add_panel_label(ax_a, "a")

    # ── Panel (b): PINN advantage scatter ────────────────────────────
    for r in full_runs:
        pinn_rmse = r.get("PINN_rmse")
        if pinn_rmse is None:
            continue
        baseline_rmses = [r.get(f"{m}_rmse") for m in ["GP", "IDW", "NN", "RF"]
                          if r.get(f"{m}_rmse") is not None]
        if not baseline_rmses:
            continue
        best_baseline = min(baseline_rmses)
        reef = r["reef"]
        ax_b.scatter(best_baseline, pinn_rmse,
                     c=REEF_COLORS[reef], s=20 + r["n_training_depths"] * 2,
                     edgecolors="black", linewidth=0.4, alpha=0.85,
                     zorder=5)

    # Parity line
    lim_max = max(ax_b.get_xlim()[1], ax_b.get_ylim()[1])
    lim = [0, lim_max * 1.05]
    ax_b.plot(lim, lim, "--", color="0.5", linewidth=0.6, zorder=1)
    ax_b.fill_between(lim, lim, [lim[1], lim[1]], alpha=0.04, color="red",
                      zorder=0, label="Baseline wins")
    ax_b.fill_between(lim, [0, 0], lim, alpha=0.04, color="green",
                      zorder=0, label="PINN wins")
    ax_b.set_xlim(lim)
    ax_b.set_ylim(lim)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("Best baseline RMSE (\u00b0C)")
    ax_b.set_ylabel("PINN RMSE (\u00b0C)")

    # Legend: reef colors
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=REEF_COLORS[r], markeredgecolor="black",
                       markeredgewidth=0.4, markersize=5,
                       label=REEF_SHORT[r])
               for r in REEF_COLORS]
    ax_b.legend(handles=handles, loc="lower right", fontsize=5.5, handletextpad=0.2,
                borderpad=0.3, labelspacing=0.25, framealpha=0.7)
    add_panel_label(ax_b, "b")

    # ── Panel (c): Win rate bar chart ────────────────────────────────
    # Count wins across ALL runs (not just full-depth)
    reef_order = ["davies_reef", "myrmidon_reef", "rib_reef", "kelso_reef"]
    reef_method_wins = {reef: defaultdict(int) for reef in reef_order}
    reef_totals = defaultdict(int)

    for r in results:
        reef = r["reef"]
        bm = best_method(r)
        if bm:
            reef_method_wins[reef][bm] += 1
            reef_totals[reef] += 1

    x = np.arange(len(reef_order))
    bottom = np.zeros(len(reef_order))

    # Draw methods in order, PINN last so it's on top visually
    draw_order = ["GP", "IDW", "NN", "RF", "PINN"]
    for m in draw_order:
        heights = []
        for reef in reef_order:
            total = reef_totals[reef]
            if total > 0:
                heights.append(reef_method_wins[reef][m] / total)
            else:
                heights.append(0)
        heights = np.array(heights)
        edge = "black" if m == "PINN" else "none"
        lw = 0.6 if m == "PINN" else 0
        ax_c.bar(x, heights, bottom=bottom, width=0.65,
                 color=METHOD_COLORS[m], label=m,
                 edgecolor=edge, linewidth=lw)
        bottom += heights

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([REEF_SHORT[r] for r in reef_order], rotation=45, ha="right")
    ax_c.set_ylabel("Win fraction")
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(fontsize=5, loc="upper right", ncol=1, handletextpad=0.2,
                borderpad=0.2, labelspacing=0.15, framealpha=0.7,
                handlelength=1.0)
    add_panel_label(ax_c, "c")

    savefig(fig, "fig2_performance")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Sparsity Robustness (single column, 90mm)
#   (a) Davies 5.0m holdout  (b) Davies 9.1m holdout
# ═══════════════════════════════════════════════════════════════════════
def fig3_sparsity(results):
    """Two-panel sparsity analysis for Davies Reef."""
    # Group by (reef, holdout_depth)
    groups = defaultdict(list)
    for r in results:
        groups[(r["reef"], r["holdout_depth"])].append(r)

    targets = [
        ("davies_reef", 5.0,  "Davies 5.0 m holdout"),
        ("davies_reef", 9.1,  "Davies 9.1 m holdout"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL, SINGLE_COL * 1.15),
                              sharex=False)

    for idx, (reef, depth, title) in enumerate(targets):
        ax = axes[idx]
        rows = sorted(groups[(reef, depth)], key=lambda r: r["n_training_depths"])
        n_depths_arr = [r["n_training_depths"] for r in rows]

        for m in METHOD_ORDER:
            rmses = [r.get(f"{m}_rmse", np.nan) for r in rows]
            if all(np.isnan(v) for v in rmses):
                continue
            is_pinn = (m == "PINN")
            is_fd = (m == "FD")
            if is_fd:
                # FD is constant across sparsity — draw horizontal line
                fd_val = rmses[0]
                if not np.isnan(fd_val):
                    ax.axhline(fd_val, color=METHOD_COLORS["FD"],
                               linestyle=":", linewidth=1.2, alpha=0.8,
                               zorder=6, label="FD")
            else:
                ax.plot(n_depths_arr, rmses,
                        marker="o" if is_pinn else "s",
                        linestyle="-" if is_pinn else "--",
                        color=METHOD_COLORS[m],
                        linewidth=1.8 if is_pinn else 0.8,
                        markersize=5 if is_pinn else 3,
                        markeredgecolor="black" if is_pinn else "none",
                        markeredgewidth=0.4 if is_pinn else 0,
                        zorder=10 if is_pinn else 5,
                        label=m, alpha=1.0 if is_pinn else 0.7)

        ax.set_ylabel("RMSE (\u00b0C)")
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks(n_depths_arr)
        if idx == 0:
            ax.legend(fontsize=5.5, ncol=3, loc="upper center",
                      handletextpad=0.2, columnspacing=0.5, borderpad=0.2)
        add_panel_label(ax, chr(ord("a") + idx))

    axes[-1].set_xlabel("Number of training depths")
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=1.2)
    savefig(fig, "fig3_sparsity")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Holdout Time Series (double column, 190mm)
#   Requires PINN re-training (~16 min). Placeholder by default.
# ═══════════════════════════════════════════════════════════════════════
def fig4_timeseries(run_training=False):
    """2x2 holdout time series. Needs GPU + ~16 min if run_training=True."""
    if not run_training:
        # Generate placeholder figure
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
        selections = [
            ("Davies", "18.5 m"),
            ("Myrmidon", "14.7 m"),
            ("Rib", "9.0 m"),
            ("Kelso", "19.0 m"),
        ]
        for ax, (reef, depth) in zip(axes.flat, selections):
            ax.text(0.5, 0.5, f"{reef} Reef\n{depth} holdout\n\n(run with --fig4)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="0.4", style="italic")
            ax.set_xlabel("Date")
            ax.set_ylabel("T (\u00b0C)")
            add_panel_label(ax, chr(ord("a") + list(axes.flat).index(ax)))
        fig.tight_layout()
        savefig(fig, "fig4_timeseries")
        print("  (placeholder — rerun with --fig4 for real data)")
        return

    # Full training path — import heavy deps
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    import jax
    import jax.numpy as jnp
    import pandas as pd
    import time as time_module
    import gc
    import matplotlib.dates as mdates
    from pinn_reef_thermal import (
        load_reef_subdaily,
        prepare_pinn_data,
        generate_stratified_collocation_points,
        ReefThermalPhysics,
        ReefPINN,
    )

    print(f"  GPU: {jax.devices()}")

    HOLDOUT_SELECTIONS = [
        {"reef": "davies_reef", "date_from": "2011-03-01", "date_to": "2014-10-31",
         "holdout_depth": 18.5, "label": "Davies \u2014 18.5 m"},
        {"reef": "myrmidon_reef", "date_from": "2020-06-01", "date_to": "2024-10-01",
         "holdout_depth": 14.7, "label": "Myrmidon \u2014 14.7 m"},
        {"reef": "rib_reef", "date_from": "2018-03-01", "date_to": "2023-07-01",
         "holdout_depth": 9.0, "label": "Rib \u2014 9.0 m"},
        {"reef": "kelso_reef", "date_from": "1998-04-18", "date_to": "2001-08-07",
         "holdout_depth": 19.0, "label": "Kelso \u2014 19.0 m"},
    ]

    N_EPOCHS = 15000
    N_PDE = 2500
    PDE_CHUNK = 2500
    N_FOURIER = 8
    BATCH_SIZE = 5000
    W_PDE_SCHEDULE = [(0, 1.0), (3000, 0.5), (6000, 0.2), (10000, 0.05)]

    all_results = []
    for sel in HOLDOUT_SELECTIONS:
        print(f"\n  Training: {sel['label']}")
        reef_name = sel["reef"]

        df = load_reef_subdaily(reef_name, date_from=sel["date_from"],
                                 date_to=sel["date_to"], verbose=False)
        data = prepare_pinn_data(df, sst_df=None, use_hourly=True)
        meta = data["metadata"]
        t_start = pd.Timestamp(meta["t_start"])

        # Load CRW SST
        sst_path = os.path.join(BASE_DIR, "data", "ALZ", f"crw_sst_{reef_name}.csv")
        sst_df = pd.read_csv(sst_path)
        sst_df["time"] = pd.to_datetime(sst_df["time"], utc=True)
        sst_df["t_seconds"] = (sst_df["time"] - t_start).dt.total_seconds()
        sst_df = sst_df[(sst_df["t_seconds"] >= 0) & (sst_df["t_seconds"] <= meta["t_max"])]
        t_sst = sst_df["t_seconds"].values.astype(np.float32)
        T_sst = sst_df["CRW_SST"].values.astype(np.float32)

        # Split holdout
        holdout_depth = sel["holdout_depth"]
        z_d = data["z_data"]
        tol = 0.3
        holdout_mask = np.abs(z_d - holdout_depth) < tol

        train_z = z_d[~holdout_mask]
        train_t = data["t_data"][~holdout_mask]
        train_T = data["T_data"][~holdout_mask]
        holdout_t = data["t_data"][holdout_mask]
        holdout_T = data["T_data"][holdout_mask]
        sort_idx = np.argsort(holdout_t)
        holdout_t = holdout_t[sort_idx]
        holdout_T = holdout_T[sort_idx]

        train_depths = {k: v for k, v in data["depths"].items()
                        if abs(v - holdout_depth) >= tol}

        physics = ReefThermalPhysics(
            kappa=5e-4, Kd=0.25, Q_max=350.0, rho_cp=4.1e6,
            T_mean=meta["T_mean"], T_amp=1.2,
            z_max=meta["z_max"], t_days=meta["t_max"] / 86400.0, utc_offset=10.0)

        z_pde, t_pde = generate_stratified_collocation_points(
            meta["z_max"], meta["t_max"], N_PDE,
            logger_depths=list(train_depths.values()))

        T_range = meta.get("T_range", (20.0, 30.0))
        T_scale = max(1.0, (T_range[1] - T_range[0]) * 0.5)

        pinn = ReefPINN(
            physics=physics, learn_kappa=True, learn_Kd=True,
            use_hard_bc=True, n_fourier=N_FOURIER, seed=42,
            init_kappa=1e-3, init_Kd=0.2,
            pde_chunk_size=PDE_CHUNK, pde_depth_scale=20.0,
            w_bc_bottom=0.1, kappa_mode='log_linear',
            t_sst=t_sst, T_sst=T_sst,
            use_modified_mlp=True, hidden_dim=128, n_hidden=5,
            t_start_utc_hour=meta.get("t_start_utc_hour", 0.0),
            start_day_of_year=meta.get("start_day_of_year", 0.0),
            T_scale=T_scale, activation='tanh', grad_clip=1.0)

        t0 = time_module.time()
        pinn.train(
            z_pde=jnp.array(z_pde), t_pde=jnp.array(t_pde),
            z_data=jnp.array(train_z), t_data=jnp.array(train_t),
            T_data=jnp.array(train_T),
            z_bc=jnp.zeros(1), t_bc=jnp.zeros(1), T_bc=jnp.array([meta["T_mean"]]),
            n_epochs=N_EPOCHS, lr=1e-3, w_pde=1.0, w_data=10.0, w_bc=0.0,
            print_every=5000, resample_pde_every=2000, n_pde_points=N_PDE,
            lr_physics=1e-3, w_pde_schedule=W_PDE_SCHEDULE,
            batch_size=BATCH_SIZE, lr_schedule='cosine')
        print(f"    Trained in {time_module.time() - t0:.0f}s")

        # Predict
        n_plot = min(10000, len(holdout_t))
        plot_idx = np.linspace(0, len(holdout_t) - 1, n_plot, dtype=int)
        pred_fn = jax.jit(lambda z_, t_: pinn._predict_scalar(pinn.trained_params, z_, t_))
        T_pred_list = []
        for ci in range(0, n_plot, 5000):
            idx_chunk = plot_idx[ci:ci + 5000]
            T_pred_list.append(np.array(jax.vmap(pred_fn)(
                jnp.full(len(idx_chunk), holdout_depth, dtype=jnp.float32),
                jnp.array(holdout_t[idx_chunk]))))
        T_pred = np.concatenate(T_pred_list)

        dates_h = t_start + pd.to_timedelta(holdout_t[plot_idx], unit='s')
        dates_sst = t_start + pd.to_timedelta(t_sst, unit='s')
        rmse = float(np.sqrt(np.mean((T_pred - holdout_T[plot_idx]) ** 2)))

        all_results.append({
            "dates": dates_h, "T_obs": holdout_T[plot_idx], "T_pred": T_pred,
            "dates_sst": dates_sst, "T_sst": T_sst, "rmse": rmse,
        })
        print(f"    RMSE: {rmse:.4f}\u00b0C")
        del pinn; gc.collect()

    # Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))

    for i, (res, sel) in enumerate(zip(all_results, HOLDOUT_SELECTIONS)):
        ax = axes.flat[i]
        obs_step = max(1, len(res["dates"]) // 2000)

        # SST
        ax.plot(res["dates_sst"], res["T_sst"], color="#999999",
                alpha=0.5, linewidth=0.6, label="CRW SST", zorder=1)
        # Logger
        ax.scatter(res["dates"][::obs_step], res["T_obs"][::obs_step],
                   s=0.5, alpha=0.3, color="#0072B2", label="Logger", zorder=2)
        # PINN
        ax.plot(res["dates"][::obs_step], res["T_pred"][::obs_step],
                color=METHOD_COLORS["PINN"], linewidth=0.8, alpha=0.9,
                label="PINN", zorder=3)

        ax.set_ylabel("T (\u00b0C)")
        ax.set_title(f"{sel['label']}  (RMSE = {res['rmse']:.3f}\u00b0C)",
                     fontsize=7, pad=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=6)

        if i == 0:
            ax.legend(fontsize=5.5, loc="upper right", ncol=3,
                      markerscale=5, handletextpad=0.3)
        add_panel_label(ax, chr(ord("a") + i))

    for ax in axes[1]:
        ax.set_xlabel("Date")

    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    savefig(fig, "fig4_timeseries")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Depth-Resolved Thermal Stress (double column, 190mm)
#   (a) DHD vs depth profile — Davies Reef
#   (b) Mean temperature vs depth profile — all reefs with DHD variation
# ═══════════════════════════════════════════════════════════════════════
def _plot_dhd_panel(ax, reef_data, title, legend_loc="lower left",
                    show_legend=True):
    """Plot a single DHD depth profile panel with matched-window validation.

    Shows: PINN full-window curve, satellite line, and logger validation pairs
    (logger DHD + matched PINN DHD over the same time window, connected by lines).
    """
    depths_all = sorted([float(d) for d in reef_data["depth_dhd"].keys()])
    pinn_dhd = []
    for d in depths_all:
        key = str(d) if str(d) in reef_data["depth_dhd"] else f"{d:.1f}"
        pinn_dhd.append(reef_data["depth_dhd"][key]["pinn_dhd_final"])
    sat_dhd = reef_data["depth_dhd"][
        str(depths_all[0]) if str(depths_all[0]) in reef_data["depth_dhd"]
        else f"{depths_all[0]:.1f}"]["satellite_dhd_final"]

    # Extract matched-window logger validation pairs
    logger_depths, logger_dhds, pinn_matched = [], [], []
    for d in depths_all:
        key = str(d) if str(d) in reef_data["depth_dhd"] else f"{d:.1f}"
        entry = reef_data["depth_dhd"].get(key, {})
        if "logger_dhd_final" in entry and "pinn_dhd_matched" in entry:
            logger_depths.append(d)
            logger_dhds.append(entry["logger_dhd_final"])
            pinn_matched.append(entry["pinn_dhd_matched"])

    # Satellite (constant vertical line)
    ax.axvline(sat_dhd, color="#999999", linestyle="--", linewidth=0.8,
               label=f"Satellite ({sat_dhd:.1f})", zorder=1)

    # Fill between satellite and PINN (stress correction zone)
    ax.fill_betweenx(depths_all, pinn_dhd, [sat_dhd] * len(depths_all),
                     alpha=0.12, color="#D55E00", zorder=0)

    # PINN DHD curve (full-window — the scientific output)
    ax.plot(pinn_dhd, depths_all, "-o", color=METHOD_COLORS["PINN"],
            linewidth=1.5, markersize=3, markeredgecolor="black",
            markeredgewidth=0.3, label="PINN (full window)", zorder=5)

    # Matched-window validation pairs: logger diamond + PINN-matched square
    # connected by thin lines to show the comparison
    if logger_depths:
        # Connecting lines (logger <-> PINN matched)
        for ld, lg_v, pm_v in zip(logger_depths, logger_dhds, pinn_matched):
            ax.plot([lg_v, pm_v], [ld, ld], "-", color="0.5",
                    linewidth=0.4, zorder=4)
        # Logger truth
        ax.scatter(logger_dhds, logger_depths, marker="D", s=20,
                   facecolors="none", edgecolors="black", linewidth=0.6,
                   label="Logger (matched window)", zorder=7)
        # PINN matched
        ax.scatter(pinn_matched, logger_depths, marker="s", s=14,
                   facecolors=METHOD_COLORS["PINN"], edgecolors="black",
                   linewidth=0.3, alpha=0.7,
                   label="PINN (matched window)", zorder=6)

    ax.set_xlabel("DHD (\u00b0C\u00b7days)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8, pad=3)
    if show_legend:
        ax.legend(fontsize=5.5, loc=legend_loc)


def fig5_thermal_stress(exp15):
    """DHD depth profile for reefs with thermal stress variation."""
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.48),
        gridspec_kw={"wspace": 0.35}
    )

    # ── Panel (a): Davies DHD vs depth ───────────────────────────────
    _plot_dhd_panel(ax_a, exp15["davies_reef"],
                    "Davies Reef: thermal stress profile",
                    show_legend=False)
    ax_a.set_xlim(left=-0.02)
    add_panel_label(ax_a, "a")

    # ── Panel (b): Kelso Reef DHD (cleaner signal than Rib) ──────────
    panel_b_reef = "kelso_reef" if "kelso_reef" in exp15 else "rib_reef"
    panel_b_name = {"kelso_reef": "Kelso", "rib_reef": "Rib"}[panel_b_reef]
    if panel_b_reef in exp15:
        _plot_dhd_panel(ax_b, exp15[panel_b_reef],
                        f"{panel_b_name} Reef: thermal stress profile",
                        show_legend=False)
        ax_b.set_xlim(left=-0.05)
        add_panel_label(ax_b, "b")
    else:
        ax_b.set_visible(False)

    # Shared legend below both panels
    handles, labels = ax_a.get_legend_handles_labels()
    # Replace satellite-specific label with generic one
    labels = [l if not l.startswith("Satellite") else "Satellite DHD" for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               frameon=True, framealpha=0.8, edgecolor="0.8",
               handletextpad=0.4, columnspacing=1.2)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.90, wspace=0.35)
    savefig(fig, "fig5_thermal_stress")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: Learned Physical Parameters (single column, 90mm)
#   (a) Thermal diffusivity κ   (b) Light attenuation Kd
# ═══════════════════════════════════════════════════════════════════════
def fig6_parameters(results):
    """Dot plot of learned κ and Kd grouped by reef."""
    rows_with_params = [r for r in results
                        if r.get("PINN_params") or r.get("PINN_params_all")]
    if not rows_with_params:
        print("  No params data, skipping fig6")
        return

    reef_order = ["davies_reef", "myrmidon_reef", "rib_reef", "kelso_reef"]
    reef_x = {r: i for i, r in enumerate(reef_order)}

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(SINGLE_COL, SINGLE_COL * 1.15))

    # Collect data per reef for jitter
    # Supports both exp14 (single PINN_params) and exp16 (PINN_params_all list)
    reef_kappas = defaultdict(list)
    reef_kds = defaultdict(list)
    for r in rows_with_params:
        reef = r["reef"]
        params_list = r.get("PINN_params_all", None)
        if params_list is None:
            # exp14 format: single params dict
            params_list = [r.get("PINN_params")]
        for p in params_list:
            if p:
                reef_kappas[reef].append(p.get("kappa", np.nan) * 1e4)  # scale to 10^-4
                reef_kds[reef].append(p.get("Kd", np.nan))

    # ── Panel (a): κ ─────────────────────────────────────────────────
    for reef in reef_order:
        vals = reef_kappas.get(reef, [])
        if not vals:
            continue
        x_base = reef_x[reef]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_a.scatter(x_base + jitter, vals, c=REEF_COLORS[reef],
                     s=20, edgecolors="black", linewidth=0.3, alpha=0.8,
                     zorder=5, label=REEF_SHORT[reef])

    # Literature range: effective eddy diffusivity 1e-4 to 5e-4 m²/s → 1.0 to 5.0 in 10^-4
    ax_a.axhspan(1.0, 5.0, alpha=0.08, color="0.5", zorder=0)
    ax_a.text(0.97, 0.05, "literature range", fontsize=5.5, color="0.4",
              ha="right", va="bottom", style="italic", transform=ax_a.transAxes)

    ax_a.set_xticks(range(len(reef_order)))
    ax_a.set_xticklabels([REEF_SHORT[r] for r in reef_order])
    ax_a.set_ylabel("\u03ba (\u00d710\u207b\u2074 m\u00b2/s)")
    ax_a.set_title("Learned thermal diffusivity", fontsize=8, pad=3)
    ax_a.legend(fontsize=5.5, ncol=4, loc="upper center",
                handletextpad=0.2, columnspacing=0.5, borderpad=0.2,
                bbox_to_anchor=(0.5, 1.0))
    add_panel_label(ax_a, "a")

    # ── Panel (b): Kd ────────────────────────────────────────────────
    for reef in reef_order:
        vals = reef_kds.get(reef, [])
        if not vals:
            continue
        x_base = reef_x[reef]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_b.scatter(x_base + jitter, vals, c=REEF_COLORS[reef],
                     s=20, edgecolors="black", linewidth=0.3, alpha=0.8,
                     zorder=5)

    # Literature Kd for clear coral reef water: 0.04–0.15 m⁻¹
    ax_b.axhspan(0.04, 0.15, alpha=0.08, color="0.5", zorder=0)
    ax_b.text(0.97, 0.95, "clear reef water", fontsize=5.5, color="0.4",
              ha="right", va="top", style="italic", transform=ax_b.transAxes)

    ax_b.set_xticks(range(len(reef_order)))
    ax_b.set_xticklabels([REEF_SHORT[r] for r in reef_order])
    ax_b.set_ylabel("K\u1d48 (m\u207b\u00b9)")
    ax_b.set_title("Learned light attenuation", fontsize=8, pad=3)
    add_panel_label(ax_b, "b")

    fig.align_ylabels([ax_a, ax_b])
    fig.tight_layout(h_pad=1.2)
    savefig(fig, "fig6_parameters")


# ═══════════════════════════════════════════════════════════════════════
# Summary table (printed to console)
# ═══════════════════════════════════════════════════════════════════════
def print_summary(results):
    """Print PINN win/loss stats."""
    wins = sum(1 for r in results if best_method(r) == "PINN")
    total = len(results)
    print(f"\n  PINN wins: {wins}/{total} ({100 * wins / total:.0f}%)")

    # Per-reef breakdown
    reef_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    for r in results:
        reef = r["reef"]
        reef_stats[reef]["total"] += 1
        if best_method(r) == "PINN":
            reef_stats[reef]["wins"] += 1

    for reef in sorted(reef_stats):
        s = reef_stats[reef]
        pct = 100 * s["wins"] / s["total"] if s["total"] else 0
        print(f"    {REEF_SHORT.get(reef, reef):10s}: {s['wins']}/{s['total']} ({pct:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Nature-quality figures")
    parser.add_argument("--skip-map", action="store_true",
                        help="Skip Fig 1 (requires cartopy)")
    parser.add_argument("--fig4", action="store_true",
                        help="Run Fig 4 with PINN training (~16 min, needs GPU)")
    args = parser.parse_args()

    setup_nature_style()

    print("Generating Nature-quality figures...")
    print(f"  Output: {OUT_DIR}/")

    # Load data — use exp16 (multi-seed + FD) for performance/sparsity/params
    exp16 = load_exp16()
    print(f"  Loaded {len(exp16)} exp16 results (multi-seed + FD baseline)")

    exp15 = load_exp15()
    print(f"  Loaded exp15 results ({len(exp15)} reefs)")

    # Fig 1: Study area map
    if not args.skip_map:
        print("\nFig 1: Study area map")
        try:
            fig1_study_area()
        except ImportError:
            print("  Skipped (cartopy not installed)")
    else:
        print("\nFig 1: Skipped (--skip-map)")

    # Fig 2: Performance overview (exp16 — includes FD baseline)
    print("\nFig 2: Performance overview")
    fig2_performance(exp16)

    # Fig 3: Sparsity robustness (exp16 — FD as horizontal line)
    print("\nFig 3: Sparsity robustness")
    fig3_sparsity(exp16)

    # Fig 4: Holdout time series
    print("\nFig 4: Holdout time series")
    fig4_timeseries(run_training=args.fig4)

    # Fig 5: Thermal stress
    print("\nFig 5: Depth-resolved thermal stress")
    fig5_thermal_stress(exp15)

    # Fig 6: Learned parameters (exp16 — 5 seeds per config)
    print("\nFig 6: Learned physical parameters")
    fig6_parameters(exp16)

    # Summary
    print_summary(exp16)

    print(f"\nAll figures saved to {OUT_DIR}/")
