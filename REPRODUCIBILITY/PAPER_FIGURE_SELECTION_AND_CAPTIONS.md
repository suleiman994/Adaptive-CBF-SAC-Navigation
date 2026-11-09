# Publication figure selection and defensible captions

These plots were reproduced **exclusively** from the frozen archive of eight inspected scenarios, four controllers, and one checkpoint seed 41. They are not a new independent evaluation. Unless stated otherwise, all plotted trajectories are from recorded robot odometry in ROS 2/Gazebo.

## Main-text selection (avoid repeating scenario-by-scenario 7-panel figures)

1. **FIG01_scenario_geometry_atlas.pdf** — *Planar geometries of the eight frozen ROS 2/Gazebo test scenarios. Start and goal positions are marked; obstacle outlines show their configured geometries. Dashed circles depict nominal initial locations of scripted moving obstacles, rather than synchronized obstacle positions at a specific episode time. Body-frame arrows indicate robot orientation at the start.*
2. **FIG02_ablation_outcomes.pdf** — *Goal-reaching and logged threshold-safe-goal counts for four methods across eight fixed layouts (one evaluation per method and layout, checkpoint seed 41). Threshold-safe goal is not equivalent to collision-free physical navigation.*
3. **FIG03_outcome_matrix.pdf** — *Scenario-by-scenario status (D: deadlock; G: goal reached with logged threshold violation; S: goal reached without a logged threshold violation). The layouts were refined during development and therefore do not constitute a blind held-out test set.*
4. **FIG04_adaptive_gain_and_bias_maps.pdf** — *Empirical state-conditioned means of logged adaptive barrier gain and steering bias during Full CBF–SAC control, binned by measured center-to-obstacle distance and goal heading error. All eight scenarios are pooled; bins with fewer than eight logged samples are masked. This visualizes associations within visited states, not independent replicates, generalization, or causality. The three-panel version including the gate diagnostic (`FIG04_conditional_adaptive_state_maps.pdf`) and the sample-coverage heatmap are supporting diagnostics.*
5. **FIG05_translational_filter_intervention.pdf** — *Example of logged nominal-to-filtered forward-command modification in the static spiral passage and the fraction of steps showing nonzero modification across the frozen scenarios. The measured command difference does not establish a formal forward-invariance guarantee.*
6. **FIG06_R07_threshold_violation_case.pdf** — *Full controller in the mixed static–dynamic clutter: logged trajectory and measured center distance with six sub-threshold samples highlighted. The goal was reached, but the archived safe-goal condition failed. Physical contact was not independently measured.*

## Supplemental / per-scenario

Use `01_SCENARIOS/<descriptive_name>/01b_ablation_small_multiples.pdf` for a detailed four-controller trajectory comparison without hiding early deadlocks. The original overlaid-trajectory plot is present for reference; small multiples are more honest for cases with overlapping initial paths.

`02_spatial_adaptive_parameter_maps.pdf` and `02b_spatial_ablation_parameters.pdf` show the **logged** alpha and bias values at robot-visited positions, not an environment-wide spatial policy field. Both mode-specific training and occupancy distributions differ; do not attribute causal improvements to a single heatmap.

`03_full_controller_profiles.pdf` and `06_executed_velocity_ablation.pdf` contrast nominal commands, filtered commands, and odometry velocities. Interpret simulation time from the recorded `sim_time_s`; do not replace it with `N * 0.10 s` when sampling is irregular.

`04_clearance_and_intervention.pdf` and `05_state_response_scatter.pdf` provide per-scenario diagnostics, not repeated independent tests.

## Reviewer claims these figures cannot answer

- R1 C5 / R3 C3: one-step predicted vs subsequently measured LiDAR clearance residuals need aligned logged predictions and observed values.
- R1 C9: reward-weight sensitivity needs controlled reward ablations/retraining.
- R1 C13 / R2 C2: fair contemporary baselines and multi-seed evaluation not established by the current archive.
- R1 C17 / R2 C1: physical contact was **NOT MEASURED**, thus the plots cannot substantiate collision-free navigation.
- R1 C18 / R3 C5: multiple real-robot trials are not included.
- R1 C21: use these six main evidence figures, move per-scenario detailed plots to supplement.
- R4 C4: checkpoint selection follows latest at 300k training steps; document remaining SAC hyperparameters from training source, not inferred figures.

## Caution on new 'novelty' claim

The frozen ablation results support a descriptive **deadlock reduction associated with the learned steering bias**. They do **not** show that gain-only adaptation improves the goal-reaching rate in these eight layouts, nor establish that a relaxed, single-beam first-order CBF constraint guarantees global safety. The incremental effect of combined gain + bias versus bias-only is 2/8 successful layouts in seed 41, without a sufficient independent replication base to claim a generalizable statistical effect.
