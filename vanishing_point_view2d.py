"""Polished 2D viewer: top-down map + side elevation + stats sidebar -> GIF / PNG.

    python vanishing_point_view2d.py --obstacles city --matchup joint \
        --joint-checkpoint runs/fixed_long_s2004/simultaneous_s2004/final.pt --out runs/joint.gif

Needs no PyBullet. Stage 2 matchups (all learned): joint, seeker_ppo.
Stage 1 reference only (scripted Seeker): scripted_random, scripted_ppo, scripted_joint.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle
from vanishing_point_core import (ARENA_X, ARENA_Y, ARENA_Z, DT, segment_blocked)
from vanishing_point_parallel import VanishingPointParallelEnv

BG, PANEL, FLOOR, GRID = "#12161c", "#1a1f27", "#222a33", "#2b343f"
TEXT, DIM = "#e8ecf1", "#8b95a5"
SEEKER, EVADER, AMBER, GREEN = "#e5484d", "#4cc38a", "#f0b429", "#4cc38a"
SOLID = {"low": "#586274", "mid": "#6c7788", "tall": "#8590a5"}


def _tone(height):
    return SOLID["low"] if height <= 4.5 else SOLID["mid"] if height <= 7 else SOLID["tall"]


def _view_polygon(sim, rays=40):
    """Top-down footprint of the Seeker's view cone, cut off by obstacles (shadows)."""
    forward = sim.camera_forward
    yaw0 = math.atan2(forward[1], forward[0])
    half = math.radians(sim.fov_deg / 2)
    horizontal = max(math.hypot(forward[0], forward[1]), 1e-6)
    points = [sim.seeker_pos[:2]]
    for offset in np.linspace(-half, half, rays):
        yaw = yaw0 + offset
        direction = np.array([math.cos(yaw) * horizontal, math.sin(yaw) * horizontal, forward[2]])
        direction /= np.linalg.norm(direction)
        reach = sim.detection_range
        if segment_blocked(sim.seeker_pos, sim.seeker_pos + direction * reach, sim.obstacles):
            low, high = 0.0, reach
            for _ in range(14):
                mid = (low + high) / 2
                if segment_blocked(sim.seeker_pos, sim.seeker_pos + direction * mid, sim.obstacles):
                    high = mid
                else:
                    low = mid
            reach = low
        points.append((sim.seeker_pos + direction * reach)[:2])
    return np.array(points)


def _drone(ax, xy, heading, colour, altitude, label):
    ax.add_patch(Circle(xy + np.array([0.18, -0.18]) * (1 + altitude / 10), 0.5, color="black", alpha=0.35, lw=0))
    ax.add_patch(Circle(xy, 0.5, facecolor=colour, edgecolor="white", lw=1.4, zorder=6))
    tip = xy + 0.95 * np.array([math.cos(heading), math.sin(heading)])
    left = xy + 0.45 * np.array([math.cos(heading + 2.2), math.sin(heading + 2.2)])
    right = xy + 0.45 * np.array([math.cos(heading - 2.2), math.sin(heading - 2.2)])
    ax.add_patch(Polygon([tip, left, right], closed=True, facecolor="white", edgecolor="none", zorder=7))
    ax.text(xy[0], xy[1] + 1.0, f"{label} z={altitude:.1f}m", color=colour, fontsize=7.5,
            ha="center", va="bottom", zorder=8, fontweight="bold")


def draw_frame(fig, env, state):
    sim = env.sim
    fig.clear()
    fig.patch.set_facecolor(BG)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.62], height_ratios=[1.0, 0.42],
                            left=0.02, right=0.98, top=0.97, bottom=0.04, wspace=0.05, hspace=0.12)
    top, side, info = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[0, 1])

    # ---------------- top-down map ----------------
    top.set_facecolor(FLOOR)
    top.set_xlim(0, ARENA_X)
    top.set_ylim(0, ARENA_Y)
    top.set_aspect("equal")
    top.set_xticks(range(0, int(ARENA_X) + 1, 2))
    top.set_yticks(range(0, int(ARENA_Y) + 1, 2))
    top.grid(color=GRID, lw=0.8)
    top.tick_params(colors=DIM, labelsize=7, length=0)
    for spine in top.spines.values():
        spine.set_color(GRID)
    for lo, hi in sim.obstacles:
        top.add_patch(FancyBboxPatch((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                                     boxstyle="round,pad=0,rounding_size=0.25", facecolor=_tone(hi[2]),
                                     edgecolor="#aab3c5", lw=1.2, zorder=3))
        top.text((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, f"{hi[2]:.0f} m", color="white", fontsize=8,
                 ha="center", va="center", alpha=0.85, zorder=4)
    cone = _view_polygon(sim)
    cone_colour = SEEKER if sim.visible else AMBER
    top.add_patch(Polygon(cone, closed=True, facecolor=cone_colour, alpha=0.20, edgecolor=cone_colour,
                          lw=1.0, zorder=2))
    for name, colour in (("seeker", SEEKER), ("evader", EVADER)):  # fading trails
        trail = np.array(state["trail"][name][-40:])
        for i in range(1, len(trail)):
            top.plot(trail[i - 1:i + 1, 0], trail[i - 1:i + 1, 1], color=colour, alpha=0.08 + 0.5 * i / len(trail),
                     lw=2, zorder=5, solid_capstyle="round")
    blocked = (not sim.visible) and sim.distance <= sim.detection_range and sim.angle <= sim.fov_deg / 2
    if sim.visible:
        top.plot(*zip(sim.seeker_pos[:2], sim.evader_pos[:2]), color=GREEN, lw=1.8, zorder=5)
    elif blocked:
        top.plot(*zip(sim.seeker_pos[:2], sim.evader_pos[:2]), color=AMBER, lw=1.6, ls=(0, (3, 3)), zorder=5)
    seeker_heading = math.atan2(sim.camera_forward[1], sim.camera_forward[0])
    velocity = sim.evader_velocity
    if np.hypot(velocity[0], velocity[1]) > 0.1:
        state["evader_heading"] = math.atan2(velocity[1], velocity[0])
    _drone(top, sim.seeker_pos[:2], seeker_heading, SEEKER, sim.seeker_pos[2], "Seeker")
    _drone(top, sim.evader_pos[:2], state["evader_heading"], EVADER, sim.evader_pos[2], "Evader")
    top.set_title("TOP-DOWN", color=DIM, fontsize=8, loc="left", pad=4)

    # ---------------- side elevation (x-z) ----------------
    side.set_facecolor(FLOOR)
    side.set_xlim(0, ARENA_X)
    side.set_ylim(0, ARENA_Z)
    side.set_xticks(range(0, int(ARENA_X) + 1, 4))
    side.set_yticks(range(0, int(ARENA_Z) + 1, 2))
    side.grid(color=GRID, lw=0.8)
    side.tick_params(colors=DIM, labelsize=7, length=0)
    for spine in side.spines.values():
        spine.set_color(GRID)
    for lo, hi in sim.obstacles:
        side.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], facecolor=_tone(hi[2]),
                                 edgecolor="#aab3c5", lw=1.0, alpha=0.55, zorder=3))
    side.plot(*zip(sim.seeker_pos[[0, 2]], sim.evader_pos[[0, 2]]), color=GREEN if sim.visible else AMBER,
              lw=1.4, ls="-" if sim.visible else (0, (3, 3)), zorder=4, alpha=0.9 if (sim.visible or blocked) else 0.2)
    side.scatter(*sim.seeker_pos[[0, 2]], s=70, color=SEEKER, edgecolor="white", zorder=6)
    side.scatter(*sim.evader_pos[[0, 2]], s=70, color=EVADER, edgecolor="white", zorder=6)
    side.set_title("SIDE VIEW  (x vs altitude)", color=DIM, fontsize=8, loc="left", pad=4)

    # ---------------- sidebar ----------------
    info.set_facecolor(BG)
    info.axis("off")
    info.set_xlim(0, 1)
    info.set_ylim(0, 1)
    info.text(0.0, 0.97, "VANISHING POINT", color=TEXT, fontsize=19, fontweight="bold", va="top")
    info.text(0.0, 0.855, "pursuit-evasion  /  " + state["subtitle"], color=DIM, fontsize=8.5, va="top")
    info.text(0.0, 0.76, f"Step  {sim.step_count} / {sim.episode_steps}", color=TEXT, fontsize=12, va="top")
    status = "VISIBLE" if sim.visible else ("BLOCKED" if blocked else "HIDDEN")
    colour = SEEKER if sim.visible else (AMBER if blocked else GREEN)
    info.text(0.0, 0.715, "Evader status", color=DIM, fontsize=9, va="top")
    info.text(0.0, 0.655, status, color=colour, fontsize=17, fontweight="bold", va="top")
    # exposure bar: share of the last 100 steps spent visible
    recent = state["visible_history"][-100:]
    exposure = float(np.mean(recent)) if recent else 0.0
    segments = 4
    for i in range(segments):
        lit = exposure >= (i + 0.5) / segments
        info.add_patch(FancyBboxPatch((i * 0.245, 0.40), 0.225, 0.07, boxstyle="round,pad=0,rounding_size=0.015",
                                      facecolor=(SEEKER if exposure > 0.5 else AMBER) if lit else "#2a313c",
                                      edgecolor="none", transform=info.transAxes))
    info.text(0.0, 0.485, f"Exposure (last 10 s)  {exposure * 100:.0f}%", color=DIM, fontsize=9, va="bottom")
    rows = [("Distance", f"{sim.distance:.1f} m"), ("Seeker mode", getattr(env, "seeker_mode", "-")),
            ("Hidden so far", f"{100 * sim.undetected_steps / max(sim.step_count, 1):.0f} %"),
            ("Evader contacts", f"{sim.wall_touches}"), ("Seeker contacts", f"{sim.seeker_wall_touches}")]
    for i, (name, value) in enumerate(rows):
        y = 0.32 - i * 0.058
        info.text(0.0, y, name, color=DIM, fontsize=9, va="top")
        info.text(1.0, y, value, color=TEXT, fontsize=9, va="top", ha="right")
    for x, colour_, name in ((0.0, SEEKER, "seeker"), (0.38, EVADER, "evader"), (0.76, "#aab3c5", "obstacle")):
        info.add_patch(Circle((x + 0.02, 0.0), 0.016, color=colour_, transform=info.transAxes, clip_on=False))
        info.text(x + 0.06, 0.0, name, color=DIM, fontsize=8, va="center")
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def build_policies(matchup, joint_path, seeker_path, ppo_path, seed):
    from policy_io import load_policy
    from stage2_opponents import PPOEvader, RandomEvader
    joint = load_policy(joint_path) if joint_path else None
    if matchup in ("scripted_joint", "joint") and joint is None:
        raise SystemExit("--joint-checkpoint is required for this matchup")
    if matchup == "scripted_random":
        return None, RandomEvader(seed).act, "scripted Seeker vs random Evader"
    if matchup == "scripted_ppo":
        return None, PPOEvader(ppo_path).act, "scripted Seeker vs PPO Evader"
    if matchup == "scripted_joint":
        return None, (lambda o: joint.act("evader", o)), "scripted Seeker vs MADDPG Evader"
    if matchup == "joint":
        return (lambda o: joint.act("seeker", o)), (lambda o: joint.act("evader", o)), "MADDPG vs MADDPG"
    seeker_net = load_policy(seeker_path) if seeker_path else joint
    if seeker_net is None:
        raise SystemExit("--seeker-checkpoint or --joint-checkpoint required")
    return (lambda o: seeker_net.act("seeker", o)), PPOEvader(ppo_path).act, "MADDPG Seeker vs PPO Evader"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matchup", default="joint",
                        choices=["scripted_random", "scripted_ppo", "scripted_joint", "joint", "seeker_ppo"])
    parser.add_argument("--obstacles", default="city")
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--seeker-checkpoint", type=Path)
    parser.add_argument("--ppo-model", type=Path, default=Path("models/ppo_500k.zip"))
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2, help="render every Nth step")
    parser.add_argument("--out", type=Path, default=Path("runs/demo.gif"))
    args = parser.parse_args()

    options = {"obstacles": None if args.obstacles == "none" else args.obstacles}
    if args.joint_checkpoint:
        from policy_io import load_policy
        options = {**load_policy(args.joint_checkpoint).config["environment"], **options}
    env = VanishingPointParallelEnv(**options)
    seeker, evader, subtitle = build_policies(args.matchup, args.joint_checkpoint, args.seeker_checkpoint,
                                              args.ppo_model, args.seed)
    obs, _ = env.reset(seed=args.seed)
    state = {"trail": {"seeker": [], "evader": []}, "visible_history": [], "evader_heading": 0.0,
             "subtitle": subtitle + ("  /  " + args.obstacles + " obstacles" if args.obstacles != "none" else "")}
    fig = plt.figure(figsize=(12, 6.8), dpi=100)
    frames = []

    def record():
        state["trail"]["seeker"].append(env.sim.seeker_pos[:2].copy())
        state["trail"]["evader"].append(env.sim.evader_pos[:2].copy())
        state["visible_history"].append(float(env.sim.visible))

    record()
    frames.append(draw_frame(fig, env, state))
    for step in range(1, args.steps + 1):
        if not env.agents:
            break
        evader_action = evader(obs["evader"])
        if seeker is None:
            obs, *_ = env.step_scripted(evader_action)
        else:
            obs, *_ = env.step({"seeker": seeker(obs["seeker"]), "evader": evader_action})
        record()
        if step % args.stride == 0:
            frames.append(draw_frame(fig, env, state))
    from PIL import Image
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    if args.out.suffix != ".png":  # small GIFs: 64-colour adaptive palette
        images = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in images]
    if args.out.suffix == ".png":
        images[len(images) // 2].save(args.out)
    else:
        images[0].save(args.out, save_all=True, append_images=images[1:],
                       duration=int(1000 * DT * args.stride / 2), loop=0, optimize=True)
    print(f"wrote {args.out} ({len(images)} frames)")


if __name__ == "__main__":
    main()
