# Metric definitions and interpretation

The primary unit of analysis is **one recorded episode in one fixed scenario with one controller/checkpoint**. 8 scenarios x 4 controllers = 32 episodes, all frozen Seed 41. Individual timesteps are **serially correlated** and may not be counted as independent experimental replicates.

- **Goal success**: `summary.goal` (1 if reached).
- **Threshold-safe goal**: `summary.safe_goal` (1 if reached and no logged sub-0.25-m center-distance violations). Neither this variable nor a zero logged violation count certifies absence of physical contact.
- **Minimum center distance**: logged `summary.min_clearance_m` (`center_to_obstacle_m`); this is NOT necessarily free surface-to-surface clearance.
- **Threshold violation steps**: `summary.violation_steps`; count of recorded steps below scene's 0.25-m threshold; time samples uneven and not duration estimates.
- **Simulation duration**: last `sim_time_s` minus first; distinguish from wall-clock runtime and from nominal `steps * Ts`.
- **Mean executed speed**: per-episode arithmetic mean of `abs(v_odom)` and `abs(w_odom)`; not aggregated across episodes as independent data here.
- **Velocity intervention**: logged `v_nom - v_cmd`, with magnitude `|v_nom-v_cmd|`. It is a command-level intervention indicator and **not** an independently verified physical CBF guarantee.
- **Gate/slack**: recorded diagnostic `gate` and QP slack; without full controller-source review do not assert that `gate > 0.5` precisely denotes an active CBF inequality.
- **Adaptive alpha and bias**: raw logged `alpha` and `b`. Bias units/normalization require verification in SAC action-mapping source before labeling in rad/s.
- **Conditional parameter heatmaps**: mean of logged values in predefined (center distance, goal heading-error) bins, with bins containing fewer than 8 logged steps masked. These are exploratory conditional **descriptions of on-policy visited states**, not independent measurements or causal treatment effects. Different controllers follow different state distributions; do not directly compare heatmap colors across policies as a causal effect.
- **Dynamic scenarios**: obstacle outlines show nominal centers at **initial/reference geometry**. The archive has no synchronized complete actuator/obstacle trajectories packed for a time-resolved occupancy animation. Do not call these frames replays of moving obstacles.
- **Model selection**: per `READ_ME_FIRST`, three trained checkpoints each for alpha-only, bias-only, full, seeds 41/42/43, selected as latest after 300000 training steps, **not** best-on-test. Frozen evaluation includes only seed 41.
- **R04**: use actual `FINAL_RESULTS_SEED41/R04/protocol/R04_offset_gate_clearance_v2.{json,world}`. The completed-scenarios R04 is older and **not** corresponding evaluation geometry.
- **Generalization**: layouts include post-inspection modifications, cannot be described as blind held-out; dynamic R08 remains excluded.
