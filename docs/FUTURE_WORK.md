# Future work: Phase 4, deployment on two physical drones

**Status: not done.** The project proposal lists hardware deployment as the final phase, to be attempted only once
the simulation "consistently produces reliable evasive behaviour". Everything in this repository is simulation only.
No hardware results are claimed anywhere. This document records why the phase was not attempted and what it would take.

## Why it was not attempted

- The simulation results are not yet reliable enough to justify hardware time: joint training oscillates and the
  seed-to-seed spread is large (see [STAGE2_RESULTS.md](STAGE2_RESULTS.md)). The proposal's own gate for Phase 4 is
  reliable behaviour, so deploying now would test the sim-to-real gap and the training instability at the same time.
- It needs equipment and a safe flight space that were not available for this assignment (two drones, a tracking
  system, a netted arena).

## What the simulator assumes (the sim-to-real gap)

| Simulator | Real drones |
|---|---|
| Point-mass velocity commands, applied instantly (speeds capped at 1.8 and 2.2 m/s) | Quadrotor attitude dynamics, motor limits, a low-level velocity controller with lag |
| 0.1 s step, no sensing or actuation delay | Latency from tracking, inference and radio |
| The Evader observes the Seeker's exact position, velocity and camera direction | Needs motion capture or onboard estimation; both are noisy |
| Visibility is a perfect geometric test (90 degree cone, 10 m, box occlusion) | A real camera plus a detector: missed detections, false positives, motion blur, lighting |
| Camera direction is a free yaw/pitch rate | A gimbal or whole-body yaw; limits and lag differ from the 100 / 90 degree-per-second assumption |
| Static box obstacles with exact geometry | Mapped or tracked obstacles |

## Proposed plan

1. **Match the interface.** Run both policies through a velocity-command wrapper (policy output scaled to the cap,
   then a standard velocity controller such as PX4 offboard mode or a Crazyflie high-level commander). Keep the
   observation definitions identical so no retraining is needed for the first test.
2. **System identification.** Measure the closed-loop velocity response and latency of the chosen drone, then add a
   matching first-order lag and delay to the simulator.
3. **Robustness training.** Retrain with domain randomisation over lag, delay, speed caps, observation noise and
   detection failure (for example a random detection-miss probability), and re-run the same evaluation
   (`scripts/evaluate_stage2.py`, hidden and visible starts, cross-play) to confirm the policies survive the perturbations
   in simulation first.
4. **Safety.** Geofence smaller than the arena, a hard minimum drone separation (the simulator only measures drone
   collisions today, it does not penalise them), speed caps, a manual kill switch, and flights with props guarded.
5. **Staged hardware tests.** (a) One drone against a ground-truth Seeker camera replayed from the simulator,
   (b) the learned Seeker against a human-flown or scripted Evader, (c) both learned policies against each other
   in a netted arena. Log the same metrics as the simulation: undetected time, time to first detection, boundary and
   collision violations.
6. **Report the gap.** Compare hardware metrics with simulation on matched starting positions and state the drop
   honestly, including any run that was aborted for safety.

## Risks

- Competing learned policies are brittle to opponent changes (the cross-play results quantify this in simulation),
  so a real opponent with different dynamics may behave very differently from a simulated one.
- Detection in the real system is the biggest uncertainty: the Seeker's learned search assumes noise-free sightings.
- Battery life and arena size limit the number of 60 s episodes, so confidence intervals will be much wider than in
  simulation.

## Other next steps

Recurrent policies (a GRU instead of fixed observation history), symmetric information (give the Evader only what a
camera-based sensor could provide), a larger opponent pool, and more seeds.
