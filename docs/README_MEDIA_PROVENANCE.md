# README visualization provenance

These figure files are visual explanations, not additional experiments.

- `docs/media/scenario_atlas.png`: archived nine-geometry figure with D05 explicitly separate.
- `docs/media/alternating_baffle_comparison.png`: archived comparative trajectory figure; includes separately adapted exploratory baseline and must not be interpreted as a parity comparison.
- `docs/media/pedestrian_ablation.png`: archived four-mode trajectory comparison.
- `docs/media/dynamic_intersection.png`: archived D05 exploratory trajectory figure; selected Joint-adaptive retry.
- `docs/media/mixed_clearance_response.png`: archived Joint-adaptive recorded clearance and commanded velocity response in mixed static/dynamic clutter.
- `docs/media/state_conditioned_adaptation.png`: observational visited-state heatmaps.
- `docs/media/*.gif`: visualizations reconstructed from frozen episode CSVs and scenario JSON within `docs/animation_sources/`, built using `scripts/regenerate_readme_gifs.py`. They are not Gazebo video capture.

The source `docs/animation_sources/baffle_*` corresponds to the fixed R03 archived trajectories for four modes. `pedestrian_*` comes from R06 Joint-adaptive and same-episode actor tracking. `mixed_*` comes from R07 Joint-adaptive and same-episode actor tracking.

Within GIF visualizations, dynamic positions are displayed **within their matched episode**; four-controller baffle plots show **separate** episodes synchronized by elapsed time only. This distinction is material for interpreting the experiments.
