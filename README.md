# Adaptive CBF-SAC navigation V4 — research artifacts (draft)

**Status: PRE-PUBLICATION DRAFT.** This release package is drawn from a frozen controller source snapshot and the eight-scenario Seed41 Gazebo evidence. It is not the manuscript's final multi-seed benchmark. All files need the authors' release approval.

## Included evidence

- `src/`: frozen-source SHA256 verified Python control and Gazebo transfer source from research master.
- `checkpoints/`: 9 SAC final checkpoints, 300k training steps each (Alpha-only/Bias-only/Full × seeds 41/42/43), plus training logs. Checkpoint **training** does not mean every seed was independently evaluated on the eight final scenarios.
- `scenarios/`: eight locked source Gazebo world + JSON pairs (R01–R07,R09), hashes verified against `SCENARIO_LOCK.csv`.
- `REPRODUCIBILITY/data`: source dataset summaries, per-controller episode traces, scene geometry and metrics.
- `REPRODUCIBILITY/scripts/generate_selected_figures.py`: regenerate core publication charts from stored data using pandas/numpy/matplotlib.
- `PAPER_MAIN`: vector figures, LaTeX captions and comparison tables for the frozen Seed41 evidence.

## Limitations that MUST remain in a public README

1. 32 archived episodes over eight development-inspected layouts are not 32 independent training seeds, nor a strictly blind held-out test.
2. Safety statistics reflect a **0.25 m robot-center--to--obstacle-surface measurement threshold**, not a collision/contact sensor or a universal safe-navigation certificate.
3. Full succeeded at the goal in 8/8 but threshold-safe-goal occurred in 7/8; R07 registered six threshold violations.
4. An adapted Mohammad--Bezzo 2025 benchmark is **not included as a validated final baseline**. The upstream authors' source has not been republished here.
5. The 9th uploaded screenshot D05 comes from a separate exploratory run series and is **not** part of the frozen eight-scenario evaluation.
6. Frozen-version controller source is published as an audit record. Running ROS/Gazebo requires platform dependencies and a workspace setup; a fully verified one-command fresh simulation reproduction has NOT been established here.

## Regenerate figures

```
python3 -m pip install numpy pandas matplotlib
python3 REPRODUCIBILITY/scripts/verify_evidence.py
python3 REPRODUCIBILITY/scripts/generate_selected_figures.py
```

### Before pushing publicly

- Confirm authorship and consent of all coauthors.
- Check license, and any upstream source code reuse; **do not add the original authors' unlicensed files**.
- Scan all files and logs for local filesystem names, personal addresses, tokens, secrets and identifying details. Edit **copies** if necessary, not frozen source. If edited, document that the published source differs from the hash-frozen source.
- Add ROS2 Humble / Gazebo Classic versions, tested dependency versions and a verified launch recipe, after a clean-environment run.
- Do not push large derived logs unless the repository and dataset license/size budget are suitable; consider GitHub Releases or Zenodo after author approval.
- Tag the final approved release and record its commit SHA, files and checksums in the revised manuscript.
