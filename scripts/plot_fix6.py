from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

SEEDS = [2001, 2002, 2003]
COLOURS = {2001: "#1f77b4", 2002: "#ff7f0e", 2003: "#2ca02c"}
ROLL = 100

OUT = Path("docs/figures/fix6")
OUT.mkdir(parents=True, exist_ok=True)


def phase_df(seed, phase):
    sub = "B/seeker" if phase == "B" else "C/simultaneous"
    return pd.read_csv(f"runs/fix6_train_s{seed}/{sub}_s{seed}/episodes.csv")


def updates_df(seed):
    return pd.read_csv(f"runs/fix6_train_s{seed}/C/simultaneous_s{seed}/updates.csv")


def eval_df(seed):
    return pd.read_csv(f"runs/fix6_train_s{seed}/eval/episodes.csv")


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _rolling_plot(ax, series_by_seed, ylabel, title, roll=ROLL):
    for seed, series in series_by_seed.items():
        smooth = series.rolling(roll, min_periods=10).mean()
        ax.plot(series.index, smooth, color=COLOURS[seed], label=f"seed {seed}", linewidth=1.6)
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_phase_b():
    for metric, ylabel, title, out in [
        ("visible_pct", "Visible % (rolling 100-ep mean)",
         "Phase B - Seeker vs frozen PPO Evader - visible %", "B_visible.png"),
        ("longest_hidden_s", "Longest hidden stretch (s)",
         "Phase B - longest hidden stretch per episode", "B_longest_hidden.png"),
        ("reacquisition_mean_s", "Mean reacquisition time (s)",
         "Phase B - mean reacquisition time per episode", "B_reacq.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        _rolling_plot(ax, {s: phase_df(s, "B")[metric] for s in SEEDS}, ylabel, title)
        _save(fig, out)


def plot_phase_c_curves():
    fig, ax = plt.subplots(figsize=(9, 5))
    _rolling_plot(ax, {s: phase_df(s, "C")["visible_pct"] for s in SEEDS},
                  "Visible % (rolling 100-ep mean)",
                  "Phase C - MADDPG self-play - visible % over training")
    ax.axhspan(40, 60, alpha=0.12, color="green", label="healthy band (40-60%)")
    ax.legend(loc="best", fontsize=9)
    _save(fig, "C_visible.png")


def plot_phase_c_losses():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for seed in SEEDS:
        df = updates_df(seed).copy()
        df["seeker_smooth"] = df.seeker_actor_loss.rolling(500, min_periods=50).mean()
        df["evader_smooth"] = df.evader_actor_loss.rolling(500, min_periods=50).mean()
        ax1.plot(df.step, df.seeker_smooth, color=COLOURS[seed], label=f"seed {seed}", linewidth=1.2)
        ax2.plot(df.step, df.evader_smooth, color=COLOURS[seed], label=f"seed {seed}", linewidth=1.2)
    for ax, who in [(ax1, "Seeker"), (ax2, "Evader")]:
        ax.set_xlabel("training step")
        ax.set_ylabel("actor loss (rolling 500-update mean)")
        ax.set_title(f"Phase C - {who} actor loss")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    _save(fig, "C_losses.png")


def plot_phase_c_boundary():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for seed in SEEDS:
        df = phase_df(seed, "C")
        ax1.plot(df.index, df.evader_boundary_steps.rolling(ROLL, min_periods=10).mean(),
                 color=COLOURS[seed], label=f"seed {seed}", linewidth=1.6)
        ax2.plot(df.index, df.seeker_boundary_steps.rolling(ROLL, min_periods=10).mean(),
                 color=COLOURS[seed], label=f"seed {seed}", linewidth=1.6)
    for ax, who in [(ax1, "Evader"), (ax2, "Seeker")]:
        ax.set_xlabel("episode")
        ax.set_ylabel(f"{who} boundary steps / 600 (rolling 100-ep)")
        ax.set_title(f"Phase C - {who} wall-contact time (lower = less wall-hugging)")
        ax.axhline(600, color="red", linestyle="--", alpha=0.4, linewidth=0.8, label="max (full episode on wall)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    _save(fig, "C_boundary.png")


def plot_eval_bar():
    rows = []
    for seed in SEEDS:
        df = eval_df(seed)
        for exp in ("learned_seeker_ppo", "joint_seeker_ppo", "stage2_joint"):
            sub = df[df.experiment == exp]
            rows.append({"seed": seed, "experiment": exp,
                         "mean": sub.visible_pct.mean(), "std": sub.visible_pct.std()})
    import numpy as np
    data = pd.DataFrame(rows)
    exps = ["learned_seeker_ppo", "joint_seeker_ppo", "stage2_joint"]
    labels = ["Phase B\n(Seeker vs PPO)", "Joint Seeker\nvs PPO Evader", "stage2_joint\n(pure self-play)"]
    x = np.arange(len(exps))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, seed in enumerate(SEEDS):
        vals = [data[(data.seed == seed) & (data.experiment == e)]["mean"].iloc[0] for e in exps]
        ax.bar(x + (i - 1) * width, vals, width, color=COLOURS[seed], label=f"seed {seed}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Visible % (60 paired eval episodes)")
    ax.set_title("Fix6 - final evaluation across 3 seeds (unshaped rewards)")
    ax.axhspan(40, 60, alpha=0.12, color="green", label="healthy band (40-60%)")
    ax.axhline(50, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)
    _save(fig, "C_eval_bar.png")


def main():
    plot_phase_b()
    plot_phase_c_curves()
    plot_phase_c_losses()
    plot_phase_c_boundary()
    plot_eval_bar()


if __name__ == "__main__":
    main()
