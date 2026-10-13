"""Build docs/Vanishing_Point_Full_Overview.pdf (Stage 1 report extended with Stage 2).

Needs reportlab (not part of requirements.txt; install it only to rebuild the report):
    pip install reportlab pillow
    python docs/build_report.py

Every number below comes from the evaluation outputs described in docs/STAGE2_RESULTS.md and docs/COMPARISONS.md.
The original Stage 1 report is kept as docs/Vanishing_Point_Full_Overview_Stage1_original.pdf.
"""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

DOCS = Path(__file__).resolve().parent
FIG = DOCS / "figures"
OUT = DOCS / "Vanishing_Point_Full_Overview.pdf"
WIDTH = letter[0] - 2 * inch

base = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=base["Normal"], fontName="Helvetica", fontSize=9.6, leading=13.2, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=14, leading=17, spaceBefore=10, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=11.2, leading=14, spaceBefore=8, spaceAfter=4, keepWithNext=1)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=21, leading=25, alignment=1, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=10.5, alignment=1, textColor=colors.HexColor("#444444"), spaceAfter=10)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.2, leading=10.2, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.4, leading=10.5, textColor=colors.HexColor("#444444"), spaceAfter=8)
CODE = ParagraphStyle("code", parent=base["Code"], fontName="Courier", fontSize=7.2, leading=9, leftIndent=6,
                      backColor=colors.HexColor("#f4f4f4"), spaceAfter=6)
NOTE = ParagraphStyle("note", parent=BODY, fontSize=8.6, leading=11.4, textColor=colors.HexColor("#333333"),
                      leftIndent=8, borderPadding=3, backColor=colors.HexColor("#fff7e0"), spaceAfter=7)

story = []


def P(text, style=BODY):
    story.append(Paragraph(text, style))


def H(text, level=1):
    story.append(Paragraph(text, H1 if level == 1 else H2))


def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(t, BODY), leftIndent=14) for t in items],
                              bulletType="bullet", start="•", leftIndent=14, bulletFontSize=8))
    story.append(Spacer(1, 3))


def table(rows, widths, header=True, bold_first_col=False, note=None):
    data = []
    for r, row in enumerate(rows):
        data.append([Paragraph(str(c), CELLB if (header and r == 0) or (bold_first_col and i == 0) else CELL)
                     for i, c in enumerate(row)])
    total = sum(widths)
    t = Table(data, colWidths=[w / total * WIDTH for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#999999")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e9ef")))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 4))
    if note:
        P(note, CAP)


def figure(path, caption, width=WIDTH, heading=None):
    """Heading (optional) is kept on the same page as the figure."""
    image = Image(str(path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth, image.drawHeight = width, width * ratio
    parts = [Paragraph(heading, H2)] if heading else []
    story.append(KeepTogether(parts + [image, Paragraph(caption, CAP)]))


def side_by_side(left, right, caption):
    half = WIDTH / 2 - 6
    cells = []
    for path in (left, right):
        im = Image(str(path))
        im.drawWidth, im.drawHeight = half, half * im.imageHeight / im.imageWidth
        cells.append(im)
    t = Table([cells], colWidths=[WIDTH / 2, WIDTH / 2])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(KeepTogether([t, Paragraph(caption, CAP)]))


def code(text):
    story.append(Preformatted(text.strip("\n"), CODE))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(inch, 0.55 * inch, "Vanishing Point - Project Overview (Phases 1-2 and Stage 2)")
    canvas.drawRightString(letter[0] - inch, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


# =====================================================================================================
P("Vanishing Point", TITLE)
P("Project Overview: Phase 1 (Simulator and Seeker), Phase 2 (PPO Evader) and Stage 2 (learning both drones)", SUB)

H("1. Project in one paragraph")
P("Vanishing Point trains an Evader drone to stay out of a Seeker drone's camera view in a bounded 3D arena "
  "(20 x 20 x 10 m, 0.1 s timestep, 600-step episodes). Phase 1 built the simulator, a scripted Seeker, and a "
  "random-Evader baseline. Phase 2 trained the Evader with PPO, first plainly and then with a difficulty curriculum, and "
  "validated the results across multiple seeds. Stage 2 then replaced the scripted Seeker with a learning Seeker: both "
  "drones are trained with model-free, policy-gradient actor-critic methods (MADDPG, with MAPPO as a comparison) in a "
  "world that now includes solid obstacles that block line of sight. Stage 2 is complete in simulation, trained and "
  "evaluated on 5 independent seeds with held-out opponents; the physical-drone phase of the original proposal is not "
  "done and is documented as future work (Section 7).")
P("Rules the project follows: no value iteration, no policy iteration and no planning formulation; policy-gradient, "
  "model-free learning only. The scripted Seeker belongs to Stage 1; <b>no part of Stage 2 training or evaluation uses it</b>.", NOTE)

# ------------------------------------------------------------------------------------------------ Phase 1
H("2. Phase 1: building the simulator and the scripted Seeker")
H("2.1 First code version", 2)
P("The first simulator version was a clean single file with correct FOV geometry and simple velocity-controlled drones, "
  "but it had several problems that made its results unreliable: a wall penalty of -10 per step that dwarfed the +/-1 "
  "visibility reward, a fixed random seed (7) that made every run identical, and a Seeker that simply gave up and span "
  "in place after losing the Evader, with a scan pattern that left a blind spot above and below.")
table([["Metric", "Value (seed 3167782036)"], ["Total reward", "426.0"], ["Undetected time", "86.2%"],
       ["Time to first detection", "0.0 s (uninformative metric)"], ["Wall touches", "4"]], [3, 3])
P("With this version, the Seeker locked onto the Evader early (by step 80), lost it around step 100 when the Evader "
  "drifted 64 degrees off-angle, and then never found it again for the rest of the episode, about 517 of 600 steps hidden. "
  "That is a misleading baseline: the high undetected percentage reflects a Seeker that gave up, not an Evader that is "
  "hard to catch.")
H("2.2 Fixing the Seeker", 2)
P("The Evader was left random and the Seeker was tuned into a fair opponent before any PPO training began:")
table([["Change", "Before", "After", "Why"],
       ["Wall penalty", "-10 every step against a wall", "-2 once on first contact", "Keeps hiding/being-seen as the main signal"],
       ["Random seed", "Fixed at 7", "New each run, logged", "Shows real variation, allows replay"],
       ["Seeker modes", "Follow, then spin forever", "TRACK, SCAN (4 s timeout), SEARCH", "Recovers after losing the Evader"],
       ["Scan pattern", "Left/right only", "Spin + 45 degree tilt", "Removes blind spot above/below"],
       ["Start behaviour", "Slow forward patrol", "Scan first, then search", "Avoids getting stuck at a wall"],
       ["Metrics", "Time to first detection", "Time until first loss, longest hidden stretch, wall touches, search restarts",
        "Describes how an episode unfolds"]], [1.3, 1.8, 2.2, 2.4],
      note="Note added after code review: the wall penalty fires at the start of each wall-contact streak, so leaving a wall and "
           "touching it again fires it again. It is not charged for every step spent against a wall, which later turned out to matter (Section 4.5).")
H("2.3 Results after tuning", 2)
table([["Metric", "First version", "Tuned version (seed 3171674840)"], ["Total reward", "426.0", "-256.0"],
       ["Undetected time", "86.2%", "30.3%"], ["Time until first loss of sight", "~8 s (inferred)", "8.6 s"],
       ["Longest undetected stretch", "~50 s (inferred)", "8.3 s"], ["Wall touches", "4", "10"],
       ["Seeker search restarts", "n/a", "1"]], [3, 2, 2.6])
P("The reward sign flipped because the fixed Seeker now sees the random Evader about 70% of the time, giving PPO a real "
  "benchmark to beat instead of a Seeker that could be escaped by accident. A 20-seed headless test confirmed the "
  "improvement generalises rather than being a one-off:")
table([["Metric (20-seed average)", "Old Seeker", "New Seeker"], ["Undetected time", "56.0%", "37.7%"],
       ["Longest hidden stretch", "22.2 s", "13.4 s"], ["Worst single stretch", "50.0 s", "48.9 s"]], [3, 2, 2])
P("The worst single hidden stretch barely moved (50.0 s to 48.9 s), showing the Seeker's random search waypoints still leave "
  "it blind on unlucky seeds, a known weakness carried into Phase 2.")
H("2.4 Phase 1 pros and cons", 2)
bullets(["<b>Pros:</b> recovers after losing the Evader via SCAN/SEARCH, scans the full 3D space, reproducible seeds, a meaningful "
         "opponent and baseline for PPO.",
         "<b>Cons:</b> random search waypoints still leave long worst-case hidden stretches; the Seeker (1.8 m/s) is slower than the "
         "Evader (2.2 m/s) and does not predict movement; drones can overlap with no penalty."])

# ------------------------------------------------------------------------------------------------ Phase 2
H("3. Phase 2: training the Evader with PPO")
P("Three controllers were compared against the same tuned scripted Seeker: acting randomly, PPO trained the plain way for "
  "500,000 steps, and PPO trained the same length with a curriculum that starts against a weak Seeker and strengthens it "
  "across four stages. Each setup was run across 3 training seeds (1 original + 2 more added later) and evaluated on the "
  "same 20 episodes at full difficulty.")
H("3.1 Results per seed", 2)
table([["Setup", "Seed", "Mean return", "Undetected", "Wall touches"],
       ["Random baseline", "12345", "173.3", "66.0%", "9.25"],
       ["Plain PPO", "577448602 (original)", "466.5", "89.6%", "4.20"], ["Plain PPO", "2001", "470.5", "90.0%", "4.55"],
       ["Plain PPO", "2002", "472.7", "91.0%", "9.50"],
       ["Curriculum PPO", "577448602 (original)", "458.8", "88.7%", "2.60"], ["Curriculum PPO", "2001", "452.6", "88.2%", "3.00"],
       ["Curriculum PPO", "2002", "447.0", "87.9%", "3.80"]], [2.2, 2.2, 1.4, 1.4, 1.4])
H("3.2 Averaged comparison", 2)
table([["Metric", "Random baseline", "Plain PPO (avg of 3 seeds)", "Curriculum PPO (avg of 3 seeds)"],
       ["Mean return", "173.3", "469.9", "452.8"], ["Undetected time", "66.0%", "90.2%", "88.3%"],
       ["Time until first loss", "3.5 s", "1.7 s", "1.8 s"], ["Longest hidden stretch", "10.7 s", "33.3 s", "27.4 s"],
       ["Wall touches (mean)", "9.25", "6.08", "3.13"], ["Seeker search restarts", "1.30", "5.55", "4.78"]], [2.4, 1.8, 2.2, 2.4])
P("PPO is a large jump over random behaviour on every metric: undetected time rises from 66.0% to about 90%, sight is lost "
  "almost immediately (1.7-1.8 s versus 3.5 s), and wall touches fall. Between the two PPO variants, the pattern held across "
  "all three seeds without exception: plain PPO scored higher every time, and curriculum PPO touched walls less often every "
  "time. Averaged out, plain PPO wins on return (469.9 vs 452.8) and undetected time (90.2% vs 88.3%), while curriculum PPO "
  "roughly halves wall touches (6.08 vs 3.13). This is a repeatable trade-off rather than single-seed noise.")
P("<b>A plausible explanation (no ablation was run to confirm it).</b> Reward only measures undetected time, and the wall "
  "penalty fires once per contact rather than scaling with how close or how often the drone grazes a wall, so a policy that "
  "clips corners to break line of sight a step or two earlier is not punished much for the risk it takes. Training the whole "
  "run at full Seeker speed and field of view may let plain PPO find and keep that riskier shortcut, while curriculum training "
  "spends its early stages against a slow, narrow-FOV Seeker, where survival depends on clean navigation. The training logs "
  "are consistent with this: under curriculum, ep_rew_mean recovers within a few thousand timesteps after each difficulty "
  "increase, and wall-touch variance across seeds stays in a tight band (2.60-3.80) while plain PPO ranges far wider "
  "(4.20-9.50), including one seed (2002) that settled on a distinctly wall-heavy strategy.")
figure(FIG / "convergence.png", "Figure 1. Stage 1 (open arena) reward convergence, plain vs curriculum PPO, one seed each.", WIDTH * 0.82,
       heading="3.3 Reward convergence")
P("Curriculum training starts far higher since its early stages face an easy, slow, narrow-sighted Seeker, so an early lead "
  "there is expected, not meaningful. Both curves reach a similar plateau by the end, and curriculum PPO dips slightly below "
  "plain PPO in the final stretch, which lines up with curriculum stage 4 (full difficulty) arriving late.")
H("3.4 Independent tests", 2)
P("Automated tests were added to check behaviour that training runs alone could not verify: the FOV boundary, Seeker mode "
  "transitions, the wall penalty, valid resets, correct episode length, and the absence of NaN/infinite values. All eleven "
  "Stage 1 test cases still pass; the full project suite is now 74 tests (Section 4.9).")
table([["Test area", "Result"], ["FOV boundary (visible/not visible at edges and range)", "Pass"],
       ["Seeker mode transitions (visible to TRACK, timeout to SEARCH)", "Pass"],
       ["Wall penalty fires once per contact streak, not once per step in contact", "Pass"],
       ["Reset returns a valid, bounded observation", "Pass"], ["Episode truncates at exactly 600 steps", "Pass"],
       ["No NaN or infinite values across random episodes", "Pass"]], [5.5, 1])
H("3.5 Phase 2 pros and cons", 2)
bullets(["<b>Pros:</b> PPO massively outperforms random behaviour; the Evader loses the Seeker almost immediately and touches walls "
         "far less; curriculum training is confirmed (3 seeds) to trade a small amount of score for fewer wall collisions.",
         "<b>Cons:</b> the Seeker is still slower than the Evader, so part of the score may reflect out-running rather than pure "
         "evasive skill; the Evader's observation excludes the Seeker's current mode; only 3 seeds were tested, and only at full "
         "difficulty for the final comparison."])

# ------------------------------------------------------------------------------------------------ Stage 2
story.append(PageBreak())
H("4. Stage 2: learning both drones")
H("4.1 Goal and design rules", 2)
P("Stage 2 asks whether the Seeker can learn its own search behaviour while the Evader learns to evade it. Both agents are "
  "neural policies trained model-free with actor-critic policy-gradient updates. Nothing in Stage 2 uses a scripted "
  "controller, a model of the environment, value/policy iteration or planning. The only fixed opponents during training are "
  "a frozen PPO Evader (also a policy-gradient method) and a frozen, previously trained learned Seeker.")
H("4.2 The Stage 2 environment", 2)
bullets(["<b>Shared simulator.</b> One NumPy world drives both the Gymnasium (PPO) and PettingZoo (multi-agent) interfaces. "
         "A regression check against the original first-commit code (random and fixed starts, three difficulty levels, 10,800 "
         "transitions) found a maximum difference of exactly 0, so Stage 1 behaviour is preserved.",
         "<b>Actions.</b> Evader: 3D velocity (capped at 2.2 m/s). Seeker: 3D velocity (capped at 1.8 m/s) plus yaw and pitch "
         "rates (100 and 90 degrees per second, pitch limited to 89 degrees). Unlike the scripted Seeker, a learned Seeker's camera "
         "turns at a limited rate.",
         "<b>Observations.</b> Evader: 21 values (as in Stage 1, so saved PPO models stay compatible). Seeker: 21 values with only "
         "its own state, its camera, and the last-seen position of the Evader, never the Evader's live position when hidden. The "
         "Seeker also receives 4 past observations spaced 5 steps apart (the observation history).",
         "<b>Centralised critics (training only).</b> Each agent's critic sees the 26-value global state and both agents' actions; "
         "execution uses only local observations.",
         "<b>Rewards.</b> Evader +1 hidden, -1 visible; the Seeker gets the opposite. Wall contact costs -2 at the start of a "
         "contact streak. For training, an extra per-step penalty (2.0) is charged while a drone stays in contact with a wall or "
         "obstacle; every evaluation uses the original unshaped rewards."])
H("4.3 The obstacle world", 2)
P("To make the task less plain, the arena can contain solid boxes (preset <i>city</i>: a 6 m central wall, two full-height "
  "pillars, two low blocks and a low canopy). Obstacles block movement (collisions are scored like wall contacts) and block "
  "line of sight: a drone inside the view cone and range is still hidden if a box lies between the two drones. Each actor "
  "also receives the vectors to its three nearest obstacles (9 extra values). With obstacles off, every Stage 1 behaviour is "
  "unchanged.")
side_by_side(FIG / "frame_joint.png", FIG / "frame_seeker_ppo.png",
             "Figure 2. The 2D viewer on the obstacle world (top-down map, side elevation, statistics sidebar). Left: the two jointly "
             "trained agents. Right: the Seeker against the held-out PPO Evader. The shaded wedge is the Seeker's view cone, cut off "
             "behind obstacles; the line shows a visible (green) sight line.")
H("4.4 Method and final training recipe", 2)
P("<b>MADDPG</b> (Lowe et al., 2017): one actor per agent (two 128-unit layers, tanh), one centralised critic per agent, target "
  "networks with soft updates (tau 0.01), a replay buffer of 100,000 transitions, Gaussian exploration noise decaying from 0.3 "
  "to 0.05, discount 0.99. Time-limit endings bootstrap from the final state instead of the reset state.")
table([["Component", "Setting in the final recipe"],
       ["Phase B (Seeker pretraining)", "Seeker alone against a frozen PPO Evader, 400k steps"],
       ["Phase C (joint training)", "600k steps, warm start from Phase B, one learner at a time in blocks of 20k steps"],
       ["Opponent mixing (per episode)", "50% frozen PPO Evader, 25% frozen Phase B Seeker, 25% joint game"],
       ["Seeker observation history", "4 past observations, 5 steps apart"],
       ["Wall/obstacle penalty (training only)", "2.0 per step in contact"],
       ["Action regulariser (training only)", "0.05 on actor pre-tanh outputs, prevents saturated actions"],
       ["Evaluation opponents", "PPO Evader of a different seed (2002) and learned Seekers of other training runs"]], [2.6, 4.2])
H("4.5 What went wrong first, and what fixed it", 2)
P("The original Stage 2 pipeline ran but produced a degenerate result, which is worth recording because it shaped the final "
  "design. In joint training the Seeker parked in a corner with its camera pointing at the floor, and both drones sat against "
  "walls for about 500 of 600 steps. The reward function explains it: the wall penalty only fires once per contact, so once "
  "a drone touches a wall it can stay there for free. (The Stage 1 PPO Evader also spends about 126 of 600 steps on walls, so "
  "the behaviour was latent from the start.)")
table([["Change", "Effect (development runs, one seed, 200k steps)"],
       ["Per-step wall penalty at least as large as the +1 hidden reward", "Removes corner parking. 0.5 was too small and collapsed on another seed; 3 and 5 hurt the Seeker"],
       ["Seeker observation history (4 x 5 steps)", "Largest single gain: Seeker vs PPO 45% to 83% visible (3 seeds, dedicated training)"],
       ["Opponent mixing and Seeker pretraining", "Stopped joint training from erasing the Seeker's skill; essential (Section 4.8)"],
       ["More steps, lower learning rates, more updates per step, regulariser or warm start alone", "No reliable improvement"],
       ["Fix: frozen opponent's next action may enter the OTHER agent's critic target, never its own", "A mistake of this kind made the Evader worse until corrected"]], [3.2, 4.2])
P("Two evaluation problems were also found and fixed in a review: training seeds overlapped across runs (episode seeds now "
  "come from separate blocks per seed), and the old evaluation reported shaped rewards and a wrong seed count (it now scores "
  "unshaped rewards, reports 95% bootstrap confidence intervals, and uses paired starts).")
H("4.6 Baselines in the obstacle world", 2)
P("PPO was retrained inside the obstacle world (it needs the 9 extra observation values), 3 seeds each for plain and curriculum, "
  "and evaluated on 50 episodes against the scripted Seeker. The PPO Evader hides about 88% of the time, the same quality as in "
  "the open arena.")
table([["Setup", "Seed 2001", "Seed 2002", "Seed 2003", "Mean"],
       ["Random (return / undetected)", "-", "-", "-", "185.5 / 67.4%"],
       ["Plain PPO return", "444.0", "439.5", "447.7", "443.7"], ["Plain PPO undetected", "88.3%", "87.7%", "88.3%", "88.1%"],
       ["Plain PPO wall/obstacle touches", "8.00", "6.16", "5.88", "6.68"],
       ["Curriculum PPO return", "443.8", "444.0", "452.0", "446.6"], ["Curriculum PPO undetected", "88.3%", "87.9%", "88.3%", "88.2%"],
       ["Curriculum PPO wall/obstacle touches", "8.18", "5.42", "3.98", "5.86"]], [3, 1.3, 1.3, 1.3, 1.8])
figure(FIG / "ppo_city_convergence.png", "Figure 3. PPO reward convergence in the obstacle world (3 seeds, mean and min-max band). Both variants plateau by about 250k "
       "steps, so the extra steps up to 500k add almost nothing.", WIDTH * 0.8)
H("4.7 Stage 2 results", 2)
P("All numbers are the percentage of the 60 s episode in which the Evader is visible to the Seeker (lower is better for the "
  "Evader), on 60 paired held-out starts per matchup per training seed, with original unshaped rewards. The scripted-Seeker "
  "reference (Stage 1) is 11.4% [95% CI 10.4, 12.4] visible against the same PPO Evader.")
table([["Matchup", "Mean", "Seed 2001", "2002", "2003", "2004", "2005"],
       ["Phase B Seeker vs held-out PPO Evader", "64.1", "96.4", "17.9", "67.6", "87.1", "51.7"],
       ["Joint-trained Seeker vs held-out PPO Evader", "52.8", "80.8", "34.7", "58.1", "11.8", "78.7"],
       ["Joint Seeker vs joint Evader (co-trained pair)", "20.2", "33.3", "7.3", "11.6", "20.2", "28.8"],
       ["Joint Evader vs a held-out learned Seeker", "19.6", "5.2", "4.9", "19.5", "41.7", "26.7"],
       ["PPO Evader vs the same held-out Seekers", "64.1", "17.9", "67.6", "87.1", "51.7", "96.4"]], [3.6, 0.9, 1, 0.8, 0.8, 0.8, 0.8])
bullets(["A learned Seeker keeps a PPO Evader in view 53-64% of the time on average against 11% for the scripted Seeker, but the "
         "spread between seeds is large (12% to 96%).",
         "The jointly trained Evader is seen far less (19.6%) than the PPO Evader (64.1%) by the same held-out learned Seekers in all "
         "five pairings. This comparison is tilted against PPO, because those Seekers were trained against PPO-style Evaders.",
         "In the co-trained game neither side dominates consistently (7% to 33% visible)."])
H("Proposal metrics: time to first detection and violations", 2)
P("Time to first detection is 0 s by definition when the Seeker already sees the Evader at the start, so a second evaluation "
  "points the camera away at step 0 (hidden start). Means over 5 seeds x 60 starts = 300 episodes per row:")
table([["Matchup", "Visible %, visible start", "Visible %, hidden start", "First detection, hidden start (mean / median, episodes detected)"],
       ["Phase B Seeker vs PPO Evader", "64.1", "60.0", "4.2 s / 2.0 s (290 of 300)"],
       ["Joint Seeker vs PPO Evader", "52.8", "47.1", "6.8 s / 2.1 s (280 of 300)"],
       ["Joint Seeker vs joint Evader", "20.2", "18.4", "6.1 s / 2.2 s (265 of 300)"],
       ["Held-out learned Seeker vs joint Evader", "19.6", "14.6", "6.2 s / 2.0 s (254 of 300)"]], [3, 1.4, 1.4, 3.2])
P("Episodes where the Seeker never finds the Evader within 60 s are counted separately and not averaged in: against the joint "
  "Evader that is 12-15% of hidden-start episodes.")
table([["Violations per 60 s episode (visible start)", "Evader contacts", "Seeker contacts", "Evader steps in contact", "Seeker steps in contact", "Drone-drone collision steps"],
       ["Phase B Seeker vs PPO Evader", "10.3", "8.4", "170", "68", "2.4"],
       ["Joint Seeker vs PPO Evader", "13.0", "8.5", "196", "125", "1.8"],
       ["Joint Seeker vs joint Evader", "13.5", "14.8", "262", "155", "0.2"],
       ["Held-out learned Seeker vs joint Evader", "13.9", "19.1", "293", "127", "0.2"]], [3.2, 1, 1, 1.1, 1.1, 1.3],
      note="Contacts count new contact streaks with a wall or obstacle; steps in contact count every step touching. Drone-drone "
           "collisions (centres closer than 0.7 m) are measured only, not penalised in training; they are rare.")
figure(FIG / "final2_joint" / "learning_curves.png",
       "Figure 4. MADDPG joint training, 5 seeds (thin lines) and seed mean (bold). Competing learners keep adapting to each other, "
       "so visible time oscillates (individual seeds swing between roughly 5% and 80%) instead of converging; the seed mean ends "
       "around 25%.", heading="Reward convergence and stability")
side_by_side(FIG / "crossplay_nopool.png", FIG / "crossplay_pool.png",
             "Figure 5. Cross-play: Evader visible % when every Seeker (rows) meets every Evader (columns) from independent training "
             "runs. The diagonal is each run's own co-trained pair. Left: main recipe. Right: with an opponent pool (Section 4.8).")
H("4.8 Comparisons: ablations, MAPPO and an opponent pool", 2)
P("These use 3 training seeds (pool: 5), the same budgets, and the same held-out opponents. With n = 3 the seed-to-seed "
  "standard deviation (5-40 points) is as large as many differences, so only large effects (more than about 20 points) are clearly real.")
P("<b>Ablations</b> remove one component from the final recipe. Columns: Seeker vs held-out PPO (higher is a stronger Seeker), "
  "Evader vs held-out learned Seekers (lower is a stronger Evader), joint game, and steps in contact with walls/obstacles.")
table([["Variant", "Seeker vs PPO", "Evader vs held-out", "Joint game", "Contact steps (Evader / Seeker)"],
       ["Final recipe", "57.9 +/- 18.8", "25.4 +/- 9.0", "17.4 +/- 11.4", "222 / 167"],
       ["No Seeker observation history", "33.0 +/- 16.8", "24.7 +/- 9.0", "35.7 +/- 15.8", "214 / 131"],
       ["No wall/obstacle penalty", "50.6 +/- 22.8", "18.5 +/- 3.6", "12.3 +/- 5.5", "564 / 408"],
       ["No opponent mixing", "24.9 +/- 6.5", "34.4 +/- 14.7", "20.3 +/- 13.2", "150 / 150"],
       ["No alternating learner blocks", "47.7 +/- 19.7", "23.5 +/- 5.0", "24.6 +/- 11.5", "203 / 187"],
       ["No Seeker pretraining (cold start)", "28.4 +/- 15.1", "37.2 +/- 9.6", "13.2 +/- 3.4", "130 / 136"]], [3.1, 1.5, 1.6, 1.4, 1.8])
bullets(["<b>Clearly matter:</b> opponent mixing, Seeker pretraining and the Seeker's observation history.",
         "<b>Wall penalty:</b> removing it does not hurt the visibility score, but wall and obstacle contact more than doubles "
         "(564 / 408 steps): both drones park against walls. Its job is to stop degenerate wall-hugging.",
         "<b>Alternating blocks:</b> the difference is inside the noise."])
P("<b>MAPPO.</b> A comparison method (on-policy clipped PPO, per-agent Gaussian actors, a centralised value critic per agent, GAE) "
  "was implemented on the same environment, opponents and budgets.")
table([["", "Seeker vs PPO", "Evader vs held-out", "Joint game", "Contact steps (Evader / Seeker)"],
       ["MADDPG (3 seeds)", "57.9 +/- 18.8", "25.4 +/- 9.0", "17.4 +/- 11.4", "222 / 167"],
       ["MAPPO (same 3 seeds)", "36.5 +/- 41.1", "30.5 +/- 11.0", "22.4 +/- 21.2", "36 / 33"]], [2.4, 1.6, 1.6, 1.4, 1.8])
P("MAPPO's Seekers overfit the one PPO Evader they trained against: they see it 96%, 95% and 72% of the time, but against a different "
  "PPO Evader only seed 2002 holds up (94%); the other two fall to 6-8%. Stochastic and deterministic actions give the same numbers, so "
  "this is real over-specialisation, not an evaluation artefact. MAPPO's drones almost never touch walls (about 35 steps versus "
  "170-220 for MADDPG). On this task MAPPO was not better overall: weaker and far more variable Seeker skill, slightly weaker Evader, "
  "much cleaner movement.")
P("<b>Opponent pool and stability.</b> Every 30k steps a snapshot of both learners is stored (20 kept); 15% of episodes play the "
  "learning Evader against a sampled past Seeker and 10% the learning Seeker against a sampled past Evader (5 seeds).")
table([["Measure", "Main recipe", "With pool"],
       ["Training oscillation: std of smoothed visible % over the second half (points)", "14.7", "10.8"],
       ["Joint-game visible %: mean (std across seeds)", "20.2 (9.9)", "16.9 (5.4)"],
       ["Joint Seeker vs held-out PPO: mean (std across seeds)", "52.8 (26.4)", "49.8 (20.3)"],
       ["Cross-play, strangers (off-diagonal) mean", "25.2", "25.4"],
       ["Cross-play gap: strangers minus own pair", "8.6", "10.7"]], [4.6, 1.5, 1.5])
P("The pool gives smoother training and about half the seed-to-seed spread in the joint game, but no gain in average strength and "
  "no gain in cross-play robustness. With 5 seeds none of these differences is statistically solid.")
H("4.9 Tests and validation", 2)
P("The suite now has 74 tests in the tests/ folder. Besides the 11 Stage 1 tests: Stage 2 environment (joint dynamics, speed "
  "caps, FOV geometry, camera updates, hidden-target non-leakage, wall rewards, truncation, the PettingZoo contract); MADDPG "
  "(learning, replay alignment, targets, frozen-opponent and opponent-pool handling, checkpoints, reproducibility); obstacles "
  "(line of sight, collisions, sliding, observation sizes, start states, scripted-Seeker freezing, observation history, "
  "first-detection time, drone collisions); MAPPO (GAE, masking, checkpoints, training smoke runs); evaluation (paired starts, "
  "confidence intervals, seed counts, cross-play). A viewer bug (the status label showing a tuple when the Evader was visible) "
  "was found while preparing this report and fixed.")
H("4.10 Stage 2 pros and cons", 2)
bullets(["<b>Pros:</b> rule-compliant (policy-gradient, model-free, no scripted controller); multi-seed with held-out opponents, "
         "confidence intervals, ablations and a MAPPO comparison; a learned Seeker clearly beats the scripted one against PPO; every "
         "proposal metric is measured.",
         "<b>Cons:</b> large seed-to-seed variance (a Seeker anywhere from 12% to 96% against PPO); co-training oscillates rather than "
         "converges; the game is asymmetric (the Evader sees the Seeker's true state and is faster, 2.2 versus 1.8 m/s); the learned "
         "Seeker's camera is rate-limited while the scripted one turns instantly, which confounds Stage 1 versus Stage 2 comparisons; "
         "wall and obstacle contact stays common for MADDPG drones; most ablations use only 3 seeds; the simulator is point-mass."])

# ------------------------------------------------------------------------------------------------ status
H("5. Which model to use, and current status")
P("For the open arena, plain PPO (seed 577448602, saved as models/ppo_500k) remains the headline Stage 1 result. In the obstacle world "
  "the reference opponents are the PPO models models/ppo_city_plain_s2001 (used in training) and s2002 (held out for evaluation). For "
  "Stage 2 the recommended system is the final MADDPG recipe (Section 4.4; configs/maddpg_config_final_phaseB.json then configs/maddpg_config_final.json), "
  "reported as a mean with its seed spread rather than as one best seed. Seed 2001 is used for the figures in this report. More training "
  "steps are not expected to help: PPO plateaus by about 250k steps and the joint MADDPG curves oscillate without trend. "
  "Phases 1, 2 and 3 (simulator, PPO with curriculum, simultaneous MADDPG learning) are complete; Phase 4 (physical drones) is not.")

H("6. Alignment with the project proposal")
table([["Proposal item", "Status", "Where"],
       ["Lightweight arena, point-mass drones, geometric FOV cone", "Done", "Sections 2, 4.2"],
       ["Evader trained with PPO against a rule-based Seeker, with a curriculum", "Done (3 seeds, open arena and obstacle world)", "Sections 3, 4.6. The curriculum raises Seeker speed and widens the FOV from 60 to 90 degrees; the proposal text says it narrows it"],
       ["Both drones trained simultaneously (MADDPG), repeated runs", "Done: 5 seeds, held-out opponents. Implemented as co-training with opponent mixing", "Sections 4.4-4.8"],
       ["Proportion of time undetected", "Done", "Section 4.7"],
       ["Average time to first detection", "Done with a hidden-start evaluation", "Section 4.7"],
       ["Boundary and collision violation rate", "Done: walls, obstacles and drone-drone collisions (the last measured, not penalised)", "Section 4.7"],
       ["Reward convergence over training", "Done for PPO and MADDPG; MADDPG oscillates", "Figures 3, 4"],
       ["Deployment on two physical drones", "Not done: future work", "Section 7"],
       ["Beyond the proposal", "Obstacle world, MAPPO comparison, opponent pool, cross-play, ablations, viewers", "Sections 4.3, 4.8"]], [3.2, 3, 2.4])

# ------------------------------------------------------------------------------------------------ future
H("7. Future work: physical drones and next steps")
P("Phase 4 of the proposal (hardware) was not attempted. The proposal makes it conditional on reliable simulation behaviour, and "
  "the seed-to-seed variance above means the policies are not yet reliable enough; it also needs two drones, a tracking system and "
  "a netted arena. No hardware result is claimed anywhere in this project.")
table([["Simulator assumption", "Real-drone difference"],
       ["Point-mass velocity commands applied instantly", "Quadrotor attitude dynamics, motor limits, a velocity controller with lag"],
       ["0.1 s step, no sensing or actuation delay", "Latency from tracking, inference and radio"],
       ["The Evader sees the Seeker's exact state", "Needs motion capture or onboard estimation, both noisy"],
       ["Visibility is a perfect geometric test", "A real camera and detector: misses, false positives, motion blur, lighting"],
       ["Free yaw and pitch rates for the camera", "Gimbal or body-yaw limits and lag"]], [3, 4])
P("Proposed plan: (1) wrap both policies in a velocity-command interface and keep the observation definitions; (2) identify the "
  "drone's velocity response and latency and add them to the simulator; (3) retrain with randomised lag, delay, noise and detection "
  "misses and re-run the same evaluation; (4) add safety (a geofence smaller than the arena, a hard minimum separation, speed caps, "
  "a kill switch); (5) test in stages (one drone against a replayed Seeker camera, then the learned Seeker against a human-flown "
  "Evader, then both learned policies); (6) report the sim-to-real gap honestly. Other next steps: recurrent policies instead of "
  "fixed observation history, symmetric information, a larger opponent pool, and more seeds (for example 10) to firm up the ablations.")

# ------------------------------------------------------------------------------------------------ appendix
H("8. Appendix: how to reproduce")
P("All commands assume the project's Python environment (pip install -r requirements.txt; on macOS pybullet may need "
  "<font face='Courier'>CFLAGS=\"-Dfdopen=fdopen\"</font>) and that the terminal is in the project folder. Outputs go to runs/ "
  "(git-ignored).")
H("8.1 Stage 1 (open arena)", 2)
code("""
python -m vp.ppo random --episodes 20
python -m vp.ppo train --timesteps 500000 --seed 577448602 \\
    --model models/ppo_500k --tensorboard-log logs/
python -m vp.ppo train --curriculum --timesteps 500000 --seed 577448602 \\
    --model models/ppo_curr --tensorboard-log logs/
python -m vp.ppo eval --model models/ppo_500k --episodes 20
python -m scripts.plot_convergence --plain logs/PPO_1 --curriculum logs/PPO_3 --out docs/figures/convergence.png
pytest -q
""")
P("Give every training run its own --tensorboard-log folder if runs are started at the same time; otherwise they race for the "
  "PPO_N folder names and interleave their logs.")
H("8.2 Stage 2 (obstacle world)", 2)
code("""
# opponents inside the obstacle world
python -m vp.ppo train --obstacles city --timesteps 500000 --seed 2001 \\
    --model models/ppo_city_plain_s2001 --tensorboard-log logs_city/curve_plain_2001
# Phase B: Seeker alone against a frozen PPO Evader
python -m scripts.train_maddpg --phase seeker --config configs/maddpg_config_final_phaseB.json \\
    --opponent ppo --opponent-model models/ppo_city_plain_s2001.zip \\
    --output runs/B --seeds 2001
# Phase C: joint training with opponent mixing
python -m scripts.train_maddpg --phase simultaneous --config configs/maddpg_config_final.json \\
    --initialize runs/B/seeker_s2001/final.pt --output runs/C --seeds 2001
# evaluation: 60 paired starts, unshaped rewards, bootstrap CIs
# (add --hidden-start to measure time to first detection)
python -m scripts.evaluate_stage2 --ppo-model models/ppo_city_plain_s2002.zip \\
    --seeker-checkpoint runs/B/seeker_s2001/final.pt \\
    --joint-checkpoint runs/C/simultaneous_s2001/final.pt --output runs/eval
# stability, ablations and the MAPPO comparison
python -m scripts.crossplay_stage2 --checkpoints <runs...> --labels <names...> --output runs/cross
python -m scripts.ablation_report A0_final A1_nohist ...     # reads runs/ev_<variant>_s<seed>
python -m scripts.train_mappo --phase seeker --config configs/mappo_config_phaseB.json \\
    --opponent-model models/ppo_city_plain_s2001.zip --output runs/mappo_B --seeds 2001
# watch: PyBullet 3D window and 2D GIF
python -m scripts.replay_stage2 --joint-checkpoint runs/C/simultaneous_s2001/final.pt --seed 10003
python -m scripts.view2d --matchup joint --obstacles city \\
    --joint-checkpoint runs/C/simultaneous_s2001/final.pt --out joint.gif
""")

H("References")
for ref in [
    "Gerkey, B. P., Thrun, S., and Gordon, G. (2006). Visibility-based pursuit-evasion with limited field of view. The International Journal of Robotics Research, 25(4), 299-315.",
    "Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., and Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. NeurIPS 30, 6379-6390.",
    "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.",
    "Wang, Y., Cao, Y., Chiun, J., Koley, S., Pham, M., and Sartoretti, G. (2025). ViPER: Visibility-based pursuit-evasion via reinforcement learning. PMLR 270, 4188-4200.",
    "Xiao, J., and Feroskhan, M. (2024). Learning multi-pursuit evasion for safe targeted navigation of drones. arXiv:2304.03443.",
    "De Souza, C., Newbury, R., Cosgun, A., Castillo, P., Vidolov, B., and Kulic, D. (2021). Decentralized multi-agent pursuit using deep reinforcement learning. IEEE Robotics and Automation Letters, 6(3), 4552-4559.",
]:
    P(ref, CELL)

doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.85 * inch,
                        bottomMargin=0.9 * inch, title="Vanishing Point: Project Overview",
                        author="Team Little Wing")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
