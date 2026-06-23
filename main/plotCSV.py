import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ==========================================
# CONFIG
# ==========================================

# Which signal column each error type should be shaded over. An error whose
# type isn't listed here still gets drawn on the "summary" band at the bottom.
ERROR_TO_SIGNAL = {
    "EYES_OPEN":      "avg_ar",
    "HANDS_OFF_HIPS": "l_wrist_hip",
    "FOOT_LIFT":      "l_foot_y",
    "HIP_ABDUCTION":  "l_hip_angle",
    "STUMBLE_SWAY":   "mid_shoulder_x",
}

# Stable color per error type so they read the same across every figure.
ERROR_COLORS = {
    "EYES_OPEN":      "tab:red",
    "HANDS_OFF_HIPS": "tab:orange",
    "FOOT_LIFT":      "tab:green",
    "HIP_ABDUCTION":  "tab:purple",
    "STUMBLE_SWAY":   "tab:blue",
}

# Signals to plot, in stacking order (top -> bottom). Only those actually
# present in the CSV are drawn, so this list can stay broad.
SIGNAL_ORDER = [
    "avg_blink",
    "avg_ar",
    "l_wrist_hip", "r_wrist_hip",
    "l_foot_y", "r_foot_y",
    "l_hip_angle", "r_hip_angle",
    "mid_shoulder_x",
    "l_ankle_x", "r_ankle_x",
]

# Human-readable y-axis labels.
SIGNAL_LABELS = {
    "avg_ar":         "eye aspect\nratio",
    "l_wrist_hip":    "L wrist-hip",
    "r_wrist_hip":    "R wrist-hip",
    "l_foot_y":       "L foot y",
    "r_foot_y":       "R foot y",
    "l_hip_angle":    "L hip angle",
    "r_hip_angle":    "R hip angle",
    "mid_shoulder_x": "mid shoulder x",
    "l_ankle_x":      "L ankle x",
    "r_ankle_x":      "R ankle x",
}


# ==========================================
# FILE SELECTION
# ==========================================

def pick_files():
    """Return (signals_path, errors_path, out_path) from argv or file dialogs."""
    args = sys.argv[1:]
    if len(args) >= 2:
        out = args[2] if len(args) >= 3 else None
        return args[0], args[1], out

    # No args: fall back to file dialogs.
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    signals_path = filedialog.askopenfilename(
        title="Select *_signals.csv",
        filetypes=[("CSV files", "*.csv")])
    if not signals_path:
        raise RuntimeError("No signals CSV selected.")
    errors_path = filedialog.askopenfilename(
        title="Select *_errors.csv",
        filetypes=[("CSV files", "*.csv")])
    if not errors_path:
        raise RuntimeError("No errors CSV selected.")
    return signals_path, errors_path, None


# ==========================================
# PLOTTING
# ==========================================

def load_errors(errors_path):
    """Load committed errors, dropping the placeholder NONE row if present."""
    err = pd.read_csv(errors_path)
    err = err[err["error_type"] != "NONE"].copy()
    return err


def plot_bess(signals_path, errors_path, out_path=None):
    sig = pd.read_csv(signals_path)
    err = load_errors(errors_path)

    if "t_trial_s" not in sig.columns:
        raise RuntimeError("signals CSV missing 't_trial_s' column.")
    t = sig["t_trial_s"].values

    # Keep only the signals that actually exist in this CSV.
    signals = [s for s in SIGNAL_ORDER if s in sig.columns]
    if not signals:
        raise RuntimeError("No known signal columns found in signals CSV.")

    # One row per signal, plus a summary band at the bottom.
    n = len(signals) + 1
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    title = os.path.basename(signals_path).replace("_signals.csv", "")
    fig.suptitle(f"BESS signals — {title}", fontsize=13, y=0.995)

    # Draw each signal, then shade any error mapped to it.
    for ax, name in zip(axes[:-1], signals):
        ax.plot(t, sig[name].values, color="black", linewidth=0.9)
        ax.set_ylabel(SIGNAL_LABELS.get(name, name), fontsize=8, rotation=0,
                      ha="right", va="center", labelpad=28)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

        for _, e in err.iterrows():
            if ERROR_TO_SIGNAL.get(e["error_type"]) == name:
                ax.axvspan(e["start_s"], e["start_s"] + e["duration_s"],
                           color=ERROR_COLORS.get(e["error_type"], "gray"),
                           alpha=0.25)

    # Summary band: every committed error as a shaded span, all on one row.
    summ = axes[-1]
    summ.set_ylim(0, 1)
    summ.set_yticks([])
    summ.set_ylabel("all errors", fontsize=8, rotation=0,
                    ha="right", va="center", labelpad=28)
    summ.margins(x=0)
    summ.set_xlim(t.min(), t.max())
    summ.set_xticks(np.arange(t.min(), t.max()+0.5, 0.5))
    for _, e in err.iterrows():
        summ.axvspan(e["start_s"], e["start_s"] + e["duration_s"],
                     color=ERROR_COLORS.get(e["error_type"], "gray"), alpha=0.5)
    summ.set_xlabel("trial time (s)", fontsize=10)

    # Legend of error types that actually appear.
    present = [et for et in ERROR_COLORS if (err["error_type"] == et).any()]
    if present:
        handles = [mpatches.Patch(color=ERROR_COLORS[et], alpha=0.5, label=et)
                   for et in present]
        fig.legend(handles=handles, loc="upper right", fontsize=8,
                   ncol=len(present), frameon=False)
    else:
        summ.text(0.5, 0.5, "no committed errors", ha="center", va="center",
                  transform=summ.transAxes, fontsize=9, color="gray")

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Default: save a PNG next to the signals CSV, named after the trial.
    if out_path is None:
        folder = os.path.dirname(os.path.abspath(signals_path))
        base = os.path.basename(signals_path).replace("_signals.csv", "")
        out_path = os.path.join(folder, base + "_plot.png")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    plt.show()


if __name__ == "__main__":
    sp, ep, op = pick_files()
    plot_bess(sp, ep, op)