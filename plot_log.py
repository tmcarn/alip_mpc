'''
Plot a run log saved by run.py.

    python plot_log.py logs/run.npz            # show
    python plot_log.py logs/run.npz --save     # also write PNGs next to the log
'''
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

# fixed categorical order: right = blue, left = orange, planned/reference = dashed
C_RIGHT, C_LEFT, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, GRID = "#52514e", "#e6e5e1"

plt.rcParams.update({
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "legend.frameon": False, "font.size": 9,
})


def stance_shading(ax, t, stance_is_left):
    '''Light gray band while the LEFT foot is stance, so steps are visible on every plot.'''
    on = stance_is_left > 0.5
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts = [0] + list(edges + 1)
    for s, e in zip(starts, list(edges) + [len(t) - 1]):
        if on[s]:
            ax.axvspan(t[s], t[e], color="#f0efec", lw=0, zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="logs/run.npz")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    d = np.load(args.log)
    t = d["t"]
    st = d["stance_is_left"]
    figs = {}

    # --- 1. foot forces (vertical + tangential) ---
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    fig.suptitle("Foot ground reaction forces (measured, on the robot)")
    for ax in axs:
        stance_shading(ax, t, st)
    axs[0].plot(t, d["foot_force_right"][:, 2], color=C_RIGHT, label="right foot")
    axs[0].plot(t, d["foot_force_left"][:, 2], color=C_LEFT, label="left foot")
    axs[0].plot(t, d["wbc_stance_force"][:, 2], "--", color=INK, lw=1, label="WBC planned (stance)")
    axs[0].set_ylabel("F_z (N)")
    axs[0].legend(loc="upper right", ncol=3)
    for f, c, name in ((d["foot_force_right"], C_RIGHT, "right"), (d["foot_force_left"], C_LEFT, "left")):
        axs[1].plot(t, f[:, 0], color=c, label=f"{name} F_x")
        axs[1].plot(t, f[:, 1], color=c, ls=":", label=f"{name} F_y")
    axs[1].set_ylabel("tangential (N)")
    axs[1].set_xlabel("time (s)")
    axs[1].legend(loc="upper right", ncol=4)
    figs["foot_forces"] = fig

    # --- 2. hand forces: measured normal vs planned, plus full 3D ---
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    fig.suptitle("Hand contact forces (shaded = left stance)")
    for ax in axs:
        stance_shading(ax, t, st)
    axs[0].plot(t, d["hand_fn_right"], color=C_RIGHT, label="right measured")
    axs[0].plot(t, d["hand_fn_planned_right"], "--", color=C_RIGHT, lw=1, label="right planned")
    axs[0].plot(t, d["hand_fn_left"], color=C_LEFT, label="left measured")
    axs[0].plot(t, d["hand_fn_planned_left"], "--", color=C_LEFT, lw=1, label="left planned")
    axs[0].set_ylabel("normal force (N)")
    axs[0].legend(loc="upper right", ncol=4)
    axs[1].plot(t, np.linalg.norm(d["hand_force_right"], axis=1), color=C_RIGHT, label="right |F|")
    axs[1].plot(t, np.linalg.norm(d["hand_force_left"], axis=1), color=C_LEFT, label="left |F|")
    axs[1].set_ylabel("|F| (N)")
    axs[1].set_xlabel("time (s)")
    axs[1].legend(loc="upper right", ncol=2)
    figs["hand_forces"] = fig

    # --- 3. box velocity ---
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    fig.suptitle("Box velocity (world frame)")
    v, w = d["box_vel"], d["box_omega"]
    axs[0].plot(t, v[:, 0], color=C_RIGHT, label="v_x")
    axs[0].plot(t, v[:, 1], color=C_LEFT, label="v_y")
    axs[0].plot(t, np.linalg.norm(v[:, :2], axis=1), color=C_3, label="speed")
    axs[0].set_ylabel("m/s")
    axs[0].legend(loc="upper right", ncol=3)
    axs[1].plot(t, w[:, 2], color=C_RIGHT, label="yaw rate ω_z")
    axs[1].set_ylabel("rad/s")
    axs[1].set_xlabel("time (s)")
    axs[1].legend(loc="upper right")
    figs["box_velocity"] = fig

    # --- 4. box tracking error ---
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 6.5))
    fig.suptitle("Box tracking error (goal − box)")
    e = d["box_pos_err"]
    axs[0].plot(t, e[:, 0], color=C_RIGHT, label="x")
    axs[0].plot(t, e[:, 1], color=C_LEFT, label="y")
    axs[0].plot(t, d["box_pos_err_norm"], color=C_3, label="distance")
    axs[0].axhline(0.05, color=INK, lw=0.8, ls=":")       # paper's 5 cm success threshold
    axs[0].set_ylabel("position (m)")
    axs[0].legend(loc="upper right", ncol=3)
    axs[1].plot(t, np.degrees(d["box_yaw_err"]), color=C_RIGHT)
    axs[1].set_ylabel("yaw (deg)")
    axs[2].plot(t, d["box_vel_err"], color=C_RIGHT)
    axs[2].set_ylabel("speed along push (m/s)")
    axs[2].set_xlabel("time (s)")
    figs["box_tracking_error"] = fig

    for fig in figs.values():
        fig.tight_layout()
    if args.save:
        out = os.path.splitext(args.log)[0]
        for name, fig in figs.items():
            fig.savefig(f"{out}_{name}.png", dpi=150)
            print(f"wrote {out}_{name}.png")
    plt.show()


if __name__ == "__main__":
    main()
