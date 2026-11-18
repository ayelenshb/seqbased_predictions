"""Plots of a run. Each function saves the figure and closes it."""
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLOR = "#1f4e79"
CONTROL_COLOR = "#9a9a9a"


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def observed_vs_predicted(replicates, controls, label, title, path):
    """All predictions of the method (all replicates), their fitted line, and the control lines."""
    obs = np.concatenate([r.observed for r in replicates])
    pred = np.concatenate([r.predicted for r in replicates])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    lo, hi = float(min(obs.min(), pred.min())), float(max(obs.max(), pred.max()))
    xs = np.array([lo, hi])
    for c in controls:
        if np.isfinite(c.slope):
            ax.plot(xs, c.slope * xs + c.intercept, color=CONTROL_COLOR, lw=0.6, alpha=0.25, zorder=1)
    ax.scatter(obs, pred, s=14, color=METHOD_COLOR, alpha=0.6, zorder=3, label="method (all replicates)")
    if len(obs) > 1 and np.ptp(obs) > 0:
        m, b = np.polyfit(obs, pred, 1)
        ax.plot(xs, m * xs + b, color=METHOD_COLOR, lw=2, zorder=4)
    ax.plot(xs, xs, ls="--", color="k", lw=1, zorder=2)
    if controls:
        ax.plot([], [], color=CONTROL_COLOR, lw=1, label=f"random controls (n={len(controls)})")
    ax.set_xlabel(f"Observed {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, path)


def cumulative_rho(sel, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x = np.arange(1, len(sel.cumulative_rho) + 1)
    ax.plot(x, sel.cumulative_rho, color=METHOD_COLOR, lw=1.5)
    ax.axvline(len(sel.idx), color="k", ls="--", lw=1, label=f"selection: {len(sel.idx)} ({sel.rho:.3f})")
    ax.set_xlabel("positions (pairs) added in order of individual ρ")
    ax.set_ylabel("cumulative ρ")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    _save(fig, path)


def individual_rho(sel, data, title, path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    chosen = set(int(k) for k in sel.idx)
    colors = [METHOD_COLOR if k in chosen else CONTROL_COLOR for k in range(data.npos)]
    ax.scatter(data.col_list, sel.individual_rho, s=12, c=colors)
    ax.set_xlabel("alignment position")
    ax.set_ylabel("individual ρ")
    ax.set_title(title + "  (blue: selected)", fontsize=10)
    _save(fig, path)
