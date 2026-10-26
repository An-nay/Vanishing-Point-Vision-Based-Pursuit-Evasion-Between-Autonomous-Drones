import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from vp.core import (ARENA_X, ARENA_Y, ARENA_Z, DT, LAMPS, TREES, ray_hits)
from vp.parallel import DroneEnv
from vp.paths import MODELS, RUNS

INK = "#000000"
WALL = "#2121de"
WALL_EDGE = "#b8c0ff"
PELLET = "#4e493c"
PELLET_LIT = "#a89872"
TEXT, DIM = "#f4f6ff", "#8d93c7"
SEEKER, EVADER, AMBER = "#ff3b30", "#3dff9a", "#ffb000"
LAMP, CANOPY, TRUNK = "#ffe08a", "#1c7a48", "#6a5040"


def box_color(height):
    lift = float(np.clip(height / ARENA_Z, 0, 1))
    dark = np.array([0.08, 0.08, 0.55])
    light = np.array([0.45, 0.55, 1.0])
    rgb = dark + (light - dark) * lift
    return tuple(rgb)


def draw_roads(ax):
    road, cross = "#1c1c2c", "#34344a"
    for y in (5.2, 13.8):
        ax.plot([0.3, ARENA_X - 0.3], [y, y], color=road, lw=1.15, zorder=0, solid_capstyle="butt")
    for x in (5.6, 14.2):
        ax.plot([x, x], [0.3, ARENA_Y - 0.3], color=road, lw=1.15, zorder=0, solid_capstyle="butt")
    for x in (5.6, 14.2):
        for y in (5.2, 13.8):
            ax.plot([x - 0.85, x + 0.85], [y, y], color=cross, lw=2.4, zorder=0, solid_capstyle="butt")
            ax.plot([x, x], [y - 0.85, y + 0.85], color=cross, lw=2.4, zorder=0, solid_capstyle="butt")


def tower_tops(obstacles):
    return [((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, float(hi[2]))
            for lo, hi in obstacles if hi[2] >= 9.5]


def draw_bits(ax, obstacles):
    for x, y in LAMPS:
        ax.plot([x, x], [y - 0.32, y + 0.32], color=LAMP, lw=0.8, zorder=3, solid_capstyle="round")
        ax.add_patch(Circle((x, y), 0.16, facecolor=LAMP, edgecolor="none", lw=0, zorder=4))
    for x, y in TREES:
        ax.add_patch(Circle((x, y), 0.58, facecolor=CANOPY, edgecolor="#0e3d24", lw=0.5, zorder=3))
        ax.add_patch(Circle((x, y), 0.11, facecolor=TRUNK, edgecolor="none", lw=0, zorder=4))
    for x, y, _height in tower_tops(obstacles):
        ax.plot([x, x], [y - 0.15, y + 0.55], color="#d5d8ee", lw=0.9, zorder=6, solid_capstyle="round")


def draw_side_bits(ax, obstacles):
    for x, _y in TREES:
        ax.plot([x, x], [0.0, 1.15], color=TRUNK, lw=1.3, zorder=3, solid_capstyle="round")
        ax.add_patch(Circle((x, 1.45), 0.42, facecolor=CANOPY, edgecolor="none", lw=0, zorder=3))
    for x, _y in LAMPS:
        ax.plot([x, x], [0.0, 1.05], color="#8a8478", lw=0.8, zorder=3, solid_capstyle="round")
        ax.add_patch(Circle((x, 1.14), 0.12, facecolor=LAMP, edgecolor="none", lw=0, zorder=4))
    for x, _y, height in tower_tops(obstacles):
        ax.plot([x, x], [height, min(height + 0.75, ARENA_Z + 0.55)], color="#d5d8ee", lw=0.9, zorder=4)


def dots(obstacles):
    spots = []
    for x in np.arange(1.0, ARENA_X, 1.0):
        for y in np.arange(1.0, ARENA_Y, 1.0):
            point = np.array([x, y, 1.0])
            if len(obstacles) and np.any(np.all((point >= obstacles[:, 0] - 0.2) & (point <= obstacles[:, 1] + 0.2), axis=1)):
                continue
            spots.append((float(x), float(y)))
    return spots


def ray_len(origin, direction, reach):
    limits = [reach]
    for axis, bound in ((0, 0.0), (0, ARENA_X), (1, 0.0), (1, ARENA_Y)):
        if abs(direction[axis]) < 1e-8:
            continue
        distance = (bound - origin[axis]) / direction[axis]
        if distance > 0.05:
            limits.append(distance)
    return min(limits)


def cone(sim, rays=28):
    forward = sim.camera_forward
    yaw0 = math.atan2(forward[1], forward[0])
    half = math.radians(sim.fov_deg / 2)
    flat = np.array(sim.obstacles, dtype=float, copy=True)
    if len(flat):
        flat[:, 0, 2] = -1.0
        flat[:, 1, 2] = ARENA_Z + 1.0
    origin = sim.seeker_pos[:2]
    points = [origin]
    for offset in np.linspace(-half, half, rays):
        yaw = yaw0 + offset
        direction = np.array([math.cos(yaw), math.sin(yaw)])
        reach = ray_len(origin, direction, sim.detection_range)
        end = np.array([origin[0], origin[1], 0.0]) + np.array([direction[0], direction[1], 0.0]) * reach
        start = np.array([origin[0], origin[1], 0.0])
        if ray_hits(start, end, flat):
            low, high = 0.0, reach
            for _ in range(14):
                mid = (low + high) / 2
                if ray_hits(start, start + np.array([direction[0], direction[1], 0.0]) * mid, flat):
                    high = mid
                else:
                    low = mid
            reach = low
        points.append(origin + direction * reach)
    return np.array(points)


def stop_at_box(sim):
    start, end = sim.seeker_pos, sim.evader_pos
    if not ray_hits(start, end, sim.obstacles):
        return end[:2]
    low, high = 0.0, 1.0
    for _ in range(16):
        mid = (low + high) / 2
        if ray_hits(start, start + (end - start) * mid, sim.obstacles):
            high = mid
        else:
            low = mid
    return (start + (end - start) * low)[:2]


def draw_drone(ax, xy, heading, colour, locked=False):
    forward = np.array([math.cos(heading), math.sin(heading)])
    right = np.array([-forward[1], forward[0]])
    reach = 0.62
    for sign in (1, -1):
        a = xy + reach * (forward + sign * right) * 0.65
        b = xy - reach * (forward + sign * right) * 0.65
        ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=2.4, zorder=8, solid_capstyle="round")
    ax.add_patch(Circle(xy, 0.30, facecolor=colour, edgecolor="white", lw=1.2 if locked else 0.7, zorder=9))
    nose = xy + forward * 0.48
    ax.plot([xy[0], nose[0]], [xy[1], nose[1]], color="white", lw=1.8, zorder=10, solid_capstyle="round")


def put_num(ax, x, value, label):
    ax.text(x, 0.34, value, color=TEXT, fontsize=11, ha="center", va="center", fontfamily="monospace")
    ax.text(x, 0.12, label, color=DIM, fontsize=6.5, ha="center", va="center", fontfamily="monospace")


def draw_bar(ax, sim, env, state, blocked):
    ax.set_facecolor(INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    status = "VISIBLE" if sim.visible else ("BLOCKED" if blocked else "HIDDEN")
    colour = SEEKER if sim.visible else (AMBER if blocked else EVADER)
    mode = env.seeker_policy.mode if state["scripted"] else getattr(env, "seeker_mode", "LEARNED")
    score = sim.undetected_steps * DT
    remaining = (sim.episode_steps - sim.step_count) * DT
    hidden_pct = 100.0 * sim.undetected_steps / max(sim.step_count, 1)
    recent = state["visible_history"][-100:]
    exposure = 100.0 * float(np.mean(recent)) if recent else 0.0
    ax.text(0.0, 0.84, status, color=colour, fontsize=18, fontweight="bold", va="center", fontfamily="monospace")
    ax.text(0.34, 0.84, f"{score:.0f}s", color=TEXT, fontsize=15, ha="center", va="center", fontfamily="monospace")
    ax.text(0.34, 0.66, "HIDDEN", color=DIM, fontsize=6.5, ha="center", va="center", fontfamily="monospace")
    ax.text(0.50, 0.84, f"{remaining:.0f}s", color=TEXT, fontsize=15, ha="center", va="center", fontfamily="monospace")
    ax.text(0.50, 0.66, "LEFT", color=DIM, fontsize=6.5, ha="center", va="center", fontfamily="monospace")
    ax.text(0.70, 0.84, mode, color=TEXT, fontsize=15, ha="center", va="center", fontfamily="monospace")
    ax.text(0.70, 0.66, "MODE", color=DIM, fontsize=6.5, ha="center", va="center", fontfamily="monospace")
    ax.plot([0.0, 1.0], [0.54, 0.54], color="#1c1c40", lw=0.6)
    columns = [
        (f"{sim.distance:.1f}m", "DIST"),
        (f"{sim.angle:.0f}°", "ANGLE"),
        (f"{sim.seeker_pos[2]:.1f}m", "SEEKER"),
        (f"{sim.evader_pos[2]:.1f}m", "EVADER"),
        (f"{hidden_pct:.0f}%", "HIDDEN"),
        (f"{exposure:.0f}%", "EXPOSED"),
        (f"{sim.wall_touches}/{sim.seeker_wall_touches}", "TOUCH"),
        (f"{sim.step_count}", "STEP"),
    ]
    for index, (value, label) in enumerate(columns):
        put_num(ax, (index + 0.5) / len(columns), value, label)


def draw_side(ax, sim):
    ax.set_facecolor(INK)
    ax.set_xlim(-0.4, ARENA_X + 0.4)
    ax.set_ylim(-0.5, ARENA_Z + 0.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([0, 5, 10])
    ax.tick_params(colors=DIM, labelsize=7, length=0)
    for spine in ax.spines.values():
        spine.set_color(WALL)
    ax.axhline(0.06, color=WALL, lw=1.0)
    for lo, hi in sorted(sim.obstacles, key=lambda box: float(box[1][2]), reverse=True):
        ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2],
                               facecolor=box_color(hi[2]), edgecolor=WALL_EDGE, lw=0.6, zorder=2))
    colour = EVADER if sim.visible else DIM
    ax.plot([sim.seeker_pos[0], sim.evader_pos[0]], [sim.seeker_pos[2], sim.evader_pos[2]],
            color=colour, lw=1.2, zorder=3)
    ax.scatter([sim.seeker_pos[0]], [sim.seeker_pos[2]], s=36, c=SEEKER, edgecolors="white", linewidths=0.5, zorder=4)
    ax.scatter([sim.evader_pos[0]], [sim.evader_pos[2]], s=36, c=EVADER, edgecolors="white", linewidths=0.5, zorder=4)
    draw_side_bits(ax, sim.obstacles)
    ax.text(0.3, ARENA_Z - 0.2, "SIDE", color=DIM, fontsize=8, fontfamily="monospace", va="top")


def draw_frame(fig, env, state):
    sim = env.sim
    fig.clear()
    fig.patch.set_facecolor(INK)
    strip = fig.add_axes([0.06, 0.905, 0.88, 0.082])
    maze = fig.add_axes([0.08, 0.34, 0.84, 0.55])
    side = fig.add_axes([0.08, 0.035, 0.84, 0.27])

    maze.set_facecolor(INK)
    maze.set_xlim(-0.6, ARENA_X + 0.6)
    maze.set_ylim(-0.6, ARENA_Y + 0.6)
    maze.set_aspect("equal")
    maze.axis("off")
    if state.get("lost"):
        maze.add_patch(Rectangle((-0.15, -0.15), ARENA_X + 0.3, ARENA_Y + 0.3, fill=False, edgecolor="white", lw=2.4, zorder=12))
    maze.add_patch(Rectangle((0, 0), ARENA_X, ARENA_Y, fill=False, edgecolor=WALL, lw=2.6, zorder=4))
    maze.text(0.35, ARENA_Y - 0.25, "TOP", color=DIM, fontsize=8, fontfamily="monospace", va="top")
    draw_roads(maze)

    cone = cone(sim)
    cone_colour = SEEKER if sim.visible else AMBER
    maze.plot([cone[0, 0], cone[1, 0]], [cone[0, 1], cone[1, 1]], color=cone_colour, lw=1.25, zorder=1, solid_capstyle="round")
    maze.plot([cone[0, 0], cone[-1, 0]], [cone[0, 1], cone[-1, 1]], color=cone_colour, lw=1.25, zorder=1, solid_capstyle="round")
    inside = MplPath(cone).contains_points(state["pellets"]) if len(state["pellets"]) else np.array([], dtype=bool)
    for index, (x, y) in enumerate(state["pellets"]):
        maze.add_patch(Circle((x, y), 0.10 if inside[index] else 0.065,
                              facecolor=PELLET_LIT if inside[index] else PELLET, lw=0, zorder=2))

    for lo, hi in sim.obstacles:
        width, depth = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        maze.add_patch(FancyBboxPatch((lo[0], lo[1]), width, depth,
                                     boxstyle="round,pad=0,rounding_size=0.15", facecolor="#14145c",
                                     edgecolor="#9aa6ff", lw=1.8, zorder=4))
        pad = min(0.22, width * 0.18, depth * 0.18)
        maze.add_patch(Rectangle((lo[0] + pad, lo[1] + pad), width - 2 * pad, depth - 2 * pad,
                                 facecolor=box_color(hi[2]), lw=0, zorder=5))
    draw_bits(maze, sim.obstacles)

    for name, colour in (("seeker", SEEKER), ("evader", EVADER)):
        trail = np.array(state["trail"][name][-16:])
        for i in range(1, len(trail)):
            maze.plot(trail[i - 1:i + 1, 0], trail[i - 1:i + 1, 1], color=colour,
                     alpha=0.12 + 0.35 * i / len(trail), lw=1.2, zorder=5, solid_capstyle="round")

    blocked = (not sim.visible) and sim.distance <= sim.detection_range and sim.angle <= sim.fov_deg / 2
    if sim.visible:
        maze.plot(*zip(sim.seeker_pos[:2], sim.evader_pos[:2]), color=EVADER, lw=1.6, zorder=6)
    elif blocked:
        stop = stop_at_box(sim)
        maze.plot(*zip(sim.seeker_pos[:2], stop), color=AMBER, lw=1.5, ls=(0, (3, 2)), zorder=6)

    seeker_heading = math.atan2(sim.camera_forward[1], sim.camera_forward[0])
    velocity = sim.evader_velocity
    if np.hypot(velocity[0], velocity[1]) > 0.1:
        state["evader_heading"] = math.atan2(velocity[1], velocity[0])
    draw_drone(maze, sim.seeker_pos[:2], seeker_heading, SEEKER, locked=sim.visible)
    draw_drone(maze, sim.evader_pos[:2], state["evader_heading"], EVADER)
    draw_bar(strip, sim, env, state, blocked)
    draw_side(side, sim)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def build_policies(matchup, joint_path, seeker_path, ppo_path, seed):
    from vp.policy_io import load_net
    from vp.opponents import PPOOpp, RandEvader
    joint = load_net(joint_path) if joint_path else None
    if matchup in ("scripted_joint", "joint") and joint is None:
        raise SystemExit("--joint-checkpoint is required for this matchup")
    if matchup == "scripted_random":
        return None, RandEvader(seed).act, "scripted Seeker vs random Evader"
    if matchup == "scripted_ppo":
        return None, PPOOpp(ppo_path).act, "scripted Seeker vs PPO Evader"
    if matchup == "scripted_joint":
        return None, (lambda o: joint.act("evader", o)), "scripted Seeker vs MADDPG Evader"
    if matchup == "joint":
        return (lambda o: joint.act("seeker", o)), (lambda o: joint.act("evader", o)), "MADDPG vs MADDPG"
    seeker_net = load_net(seeker_path) if seeker_path else joint
    if seeker_net is None:
        raise SystemExit("--seeker-checkpoint or --joint-checkpoint required")
    return (lambda o: seeker_net.act("seeker", o)), PPOOpp(ppo_path).act, "MADDPG Seeker vs PPO Evader"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matchup", default="joint",
                        choices=["scripted_random", "scripted_ppo", "scripted_joint", "joint", "seeker_ppo"])
    parser.add_argument("--obstacles", default="maze")
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--seeker-checkpoint", type=Path)
    parser.add_argument("--ppo-model", type=Path, default=MODELS / "ppo_500k.zip")
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2, help="render every Nth step")
    parser.add_argument("--out", type=Path, default=RUNS / "demo.gif")
    args = parser.parse_args()

    options = {"obstacles": None if args.obstacles == "none" else args.obstacles}
    if args.joint_checkpoint:
        from vp.policy_io import load_net
        options = {**load_net(args.joint_checkpoint).config["environment"], **options}
    env = DroneEnv(**options)
    seeker, evader, _subtitle = build_policies(args.matchup, args.joint_checkpoint, args.seeker_checkpoint,
                                              args.ppo_model, args.seed)
    obs, _ = env.reset(seed=args.seed)
    pellets = np.asarray(dots(env.sim.obstacles), dtype=float)
    if len(pellets) == 0:
        pellets = np.zeros((0, 2))
    state = {"trail": {"seeker": [], "evader": []}, "evader_heading": 0.0,
             "scripted": seeker is None, "pellets": pellets,
             "seen": bool(env.sim.visible), "lost": False, "visible_history": []}
    fig = plt.figure(figsize=(7.6, 10.6), dpi=110)
    frames = []

    def record():
        state["lost"] = state["seen"] and not env.sim.visible
        state["seen"] = bool(env.sim.visible)
        state["visible_history"].append(1.0 if env.sim.visible else 0.0)
        state["trail"]["seeker"].append(env.sim.seeker_pos[:2].copy())
        state["trail"]["evader"].append(env.sim.evader_pos[:2].copy())

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
    if args.out.suffix != ".png":
        images = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in images]
    if args.out.suffix == ".png":
        images[len(images) // 2].save(args.out)
    else:
        images[0].save(args.out, save_all=True, append_images=images[1:],
                       duration=int(1000 * DT * args.stride / 2), loop=0, optimize=True)
    print(f"wrote {args.out} ({len(images)} frames)")


if __name__ == "__main__":
    main()
