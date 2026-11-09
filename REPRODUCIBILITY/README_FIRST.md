# Mechatronics V4 — frozen-source publication graphics

**Input:** `/mnt/data/00_V4_FROZEN_RESULTS.zip`  
**Scope:** frozen Seed 41 only, 8 inspected scenarios and four modes; **R08 excluded**.  
**Output:** selected final figures as vector PDF and high-resolution PNG, each from actual odometry/command logs; detailed per-scenario trace exports; source audit and reviewer-mapping document.

## Directory structure

- `00_MAIN_TEXT_FIGURES`: a **nonredundant recommended main-text set**, small enough to satisfy Reviewer 1's concern about Figures 6–11.
- `01_SCENARIOS`: eight **human-readable named scene directories**; each has eight figure types, `DATA` CSV for each controller, geometry metadata and episode metrics. `Rxx` IDs remain in metadata for traceability.
- `02_DATA_TABLES`: episode-level metrics, primary ablation table and LaTeX source.
- `03_METHODS_NOTES`: precise metric interpretations, audit source and evidence/gap matrix.
- `04_GAZEBO_VIEWER`: optional original frozen preview helper; the images are generated locally through Gazebo; not archived physics replay.

## Scientific limitations

1. Do not claim 32 independent training runs; this is **8 scene layouts × 4 modes × one training checkpoint seed**.
2. A `safe_goal=1` means only the recorded distance-threshold test passed. Contact measurements were **NOT recorded**.
3. R07 Full reached goal with **6 logged violation steps**, so the safe-goal rate is lower than goal rate.
4. Heatmap cells represent within-policy **logged samples at visited states**. They are non-independent and cannot establish causal novel mechanism by themselves. Report episode-level ablation separately.
5. **R04** geometry was loaded from the real V2 protocol, not the older `completed_scenes` world.
6. No physical robot figures, independent validation seeds 42–43, new RL-CBF baselines, reward sensitivity experiments, or one-step prediction error may be inferred from this frozen archive.

## Using the figures in Overleaf

Prefer `PDF` versions for vector text and mathematical labels (or `PNG` for software compatibility). Example:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/FIG04_adaptive_gain_and_bias_maps.pdf}
  \caption{State-conditioned maps of the logged adaptive CBF gain and steering bias for the Full controller across eight frozen scenarios (seed 41). Only bins with at least eight logged samples are shown; these temporally correlated observations do not represent independent experiments.}
  \label{fig:adaptive_maps}
\end{figure*}
```

Captions and filenames deliberately avoid a large title inside the axes. Panel labels, quantities and units are minimal and consistent. Side-by-side same-observation comparisons do NOT imply randomized treatment effects.
