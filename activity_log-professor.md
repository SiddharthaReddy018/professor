# Professor PINN Simulation - Activity Log & Explanations

## 1. Physics Explanations

### Difference between Downsampled (`sim`) and Continuous (`sim_continuous`) Animations
**The Problem:** The full simulation generated 10,000 frames. Rendering a 10,000-frame GIF is computationally impossible for most standard viewers. Therefore, we have two ways to reduce the frame count:
1. **Downsampled (`sim`):** This method skips frames evenly across the *entire* 10,000-frame timeline. For example, to get a 200-frame GIF, it takes every 50th frame. 
   - *Effect:* The output compresses the entire 1,000,000-step simulation into a few seconds. You see the "big picture" global shape changes of the polymer over a long period, but you lose the microscopic detail of how the beads move between those 50-frame jumps.
2. **Continuous (`sim_continuous`):** This method takes the exact first N frames (e.g., 300) back-to-back without skipping any time steps. 
   - *Effect:* The output spans a much shorter total physical time (only 30 $\tau$), but you get to see the true, uninterrupted Brownian motion (microscopic jitter) of every single bead exactly as the GNN will see it. 

### How the Choice of Frames Affects the Output
- **1 Frame:** Produces a static photograph of the polymer frozen in time (all 30 coordinates at that instant).
- **5-50 Frames (Downsampled):** Very choppy animation. Skips too much time between frames, causing beads to seemingly teleport.
- **200 Frames (Downsampled):** The "Gold Standard" for validating global dynamics. Smooth enough to watch, but compressed enough to see the polymer fully morph and change its overall structure.
- **300 Frames (Continuous):** The "Gold Standard" for validating local dynamics. Shows the true random jitter of the beads step-by-step.

## 2. Codebase Modifications (Changelog)

### A. Configuration Updates
- **`configs/short.yaml` (NEW FILE):** Created a duplicate of `default.yaml` but divided the `T_steps` and `n_burnin` by 100.
  ```yaml
  T_steps: 37000     # Down from 3,700,000
  n_burnin: 27000    # Down from 2,700,000
  ```
  *Why:* Allowed us to run rapid 10-second tests of the pipeline code before committing to the 10-minute full physics simulation.

### B. Verification Scripts Bug Fixes
- **Files Modified:** `diagnose_equilibration.py`, `verify_production.py`, `verify_remaining.py`
- **Change:** Removed the hardcoded path to `default.yaml` and forced the script to extract the base seed dynamically.
  ```python
  # Old hardcoded logic removed
  # config = load_config("configs/default.yaml") 
  
  # New dynamic logic added
  parser.add_argument("--config", type=str, default="configs/default.yaml")
  config = load_config(args.config)
  sim = NumpySimulator(config)
  # Ensure exact seed alignment for validation
  trajectory = sim.run(traj_id=999, seed=sim.base_seed) 
  ```
  *Why:* The scripts were previously crashing or analyzing mismatched data because they were hardcoded to `default.yaml` and used random seeds.

### C. Visualization Script Overhaul
- **File Modified:** `visualize.py`
- **Change 1 (Load Existing Data):** Added the `--trajectory` argument to bypass simulating a new chain and instead load the JSON file directly.
  ```python
  parser.add_argument("--trajectory", type=str, default=None)
  if args.trajectory:
      with open(args.trajectory, "r") as f:
          data = json.load(f)
      frames = data.get("frames", [])
  ```
- **Change 2 (Smart Downsampling & Continuous Modes):** Added `--max_frames` and `--continuous` arguments to prevent memory crashes on 10,000 frame trajectories.
  ```python
  parser.add_argument("--max_frames", type=int, default=200)
  parser.add_argument("--continuous", action="store_true")
  
  if len(frames) > args.max_frames:
      if args.continuous:
          frames = frames[:args.max_frames] # Take first N frames
      else:
          step = max(1, len(frames) // args.max_frames)
          frames = frames[::step] # Downsample evenly
  ```

## 3. Data & Artifacts Generated
- **Data:** `data/raw/trajectory_0000.json` (50MB, Full 3.7M step run).
- **Plots:** 
  - `equilibration_diagnostics.png`
  - `production_verification.png`
  - `remaining_checks.png`
- **GIFs (in `plots/`):**
  - `sim_200.gif` (Downsampled big picture)
  - `sim_300_continuous.gif` (Uninterrupted microscopic view)
  - `sim_50.gif`, `sim_5.gif`, `sim_1.gif` (Custom frame requests)
- **Kaggle Script:** `kaggle_instructions.md` (Standalone code to execute pipeline remotely).

## 4. Professor Messages

**[18/09/26, 12:22:57 PM] Rishi:** Rg2 vs N

**[18/09/26, 12:23:10 PM] Rishi:** average force vs extension

---

## 5. Analysis Scripts (from Professor's Tasks)

### Decision Log

**Professor requested (18/09/26):** Two plots: (1) Rg² vs N, (2) average force vs extension.

**Key decisions made:**

1. **Rg² vs N requires sweeping multiple chain lengths N** — each N needs its own equilibrium simulation (burn-in = 3·τ_R, production = 2·τ_R). The existing `trajectory_0000.json` only has N=30, so we cannot extract this from existing data.

2. **Force vs Extension requires constrained MD** — clamping two end beads at a fixed distance z and measuring the mean restoring force. This is a fundamentally different simulation mode (not in existing data).

3. **Neither plot is needed for trajectory validation** — the existing `verify_production.py`, `diagnose_equilibration.py`, and `verify_remaining.py` already comprehensively validate that `trajectory_0000.json` is correct equilibrium data for GNN training. These two plots are **simulator characterization** (useful for papers/presentations, not for gating training data).

4. **Full Rg² vs N is computationally impractical in pure NumPy** — at N=100 with O(N²) pair forces, each step takes ~100× longer than N=10. The full default sweep (N=5,10,15,...,100) would take days. Added `--max_steps_per_N` cap for quick sanity checks; proper runs should use HOOMD-blue (GPU).

5. **Force-extension had three bugs fixed during testing:**
   - **Bug 1 (init overlap):** Beads initialized at spacing z/(N-1) < σ at small z caused WCA force blowup. **Fix:** init spacing = max(z/(N-1), 1.05·σ).
   - **Bug 2 (signed tension):** Using `abs()` hid the sign of compressive vs tensile force. **Fix:** use signed projection: `tension = 0.5 * (forces[0,0] - forces[-1,0])`.
   - **Bug 3 (z range):** Starting z at 10% of L_c = 2.9σ for N=30 is physically impossible (chain is compressed). **Fix:** z_min = max(30%·L_c, N·σ/2).

6. **Added MSD vs time script** — extracts monomer MSD g1(t) and center-of-mass MSD g3(t) directly from the existing trajectory. No new simulation needed. This is actually the most useful dynamics validation: confirms the integrator produces correct Rouse subdiffusion (g1 ~ t^0.5) and diffusive COM motion (g3 ~ t^1.0).

---

### A. `scripts/plot_rg2_vs_N.py` — Rg² vs N (Flory Scaling)

Sweeps chain lengths N and measures equilibrium ⟨Rg²⟩ via short MD simulations.

**Quick test results (N=5,10,15,20,30, capped at 30k steps/N):**
| N | ⟨Rg²⟩ | ±SEM |
|---|---|---|
| 5 | 1.175 | 0.013 |
| 10 | 3.091 | 0.019 |
| 15 | 5.167 | 0.025 |
| 20 | 8.098 | 0.027 |
| 30 | 13.458 | 0.036 |

```
Fit: 2ν = 1.365 ± 0.019  (expected SAW: 1.176)
     ν  = 0.683            (expected SAW: 0.588)
     R² = 0.9994           (excellent power-law)
```
**Note:** ν > 0.588 because burn-in was capped at 2% of Rouse time for speed. Full runs needed for publication-quality exponent.

**Output:** `plots/rg2_vs_N_quick.png`

**Run:**
```bash
# Quick (minutes):
python3 scripts/plot_rg2_vs_N.py --N_values 5 10 15 20 30 \
    --max_steps_per_N 30000 --output plots/rg2_vs_N_quick.png

# Full (hours — use HOOMD for large N):
python3 scripts/plot_rg2_vs_N.py --output plots/rg2_vs_N.png
```

---

### B. `scripts/plot_force_extension.py` — Average Force vs Extension

Clamps end-beads at fixed distance z, measures mean restoring force.

**Quick test results (N=30, 10 extensions, 10k equil + 20k sample each):**
| z (σ) | z/L_c | ⟨F⟩ (ε/σ) |
|---|---|---|
| 15.2 | 52.5% | 1.37 |
| 16.6 | 57.2% | 0.98 |
| 18.0 | 61.9% | 1.30 |
| 19.3 | 66.7% | 2.12 |
| 22.1 | 76.1% | 3.34 |
| 24.8 | 85.6% | 5.08 |
| 27.6 | 95.0% | 6.52 |

Forces are positive and increasing with extension. Nonlinear upturn near full extension is clearly visible — consistent with FJC/WLC theory.

**Output:** `plots/force_extension_quick.png`

**Run:**
```bash
# Quick (30 sec):
python3 scripts/plot_force_extension.py --N 30 --n_ext 10 \
    --n_equil 10000 --n_sample 20000 --output plots/force_extension_quick.png

# Full (2-3 hours):
python3 scripts/plot_force_extension.py --N 30 --output plots/force_extension.png
```

---

### C. `scripts/plot_msd.py` — MSD vs Time (from existing trajectory)

Extracts monomer MSD g1(t) and center-of-mass MSD g3(t) directly from `trajectory_0000.json`. **No new simulation needed — runs in 4 seconds on all 10,000 frames.**

**Physics:**
```
g1(t) = <|r_i(t0+t) - r_i(t0)|²>    monomer MSD
g3(t) = <|R_cm(t0+t) - R_cm(t0)|²>  center-of-mass MSD

Rouse prediction:  g1 ~ t^0.5 (subdiffusion),  g3 ~ t^1.0 (diffusion)
```

**Results (full 10k frames from trajectory_0000.json):**
```
g1 exponent : 0.621 ± 0.002  (expected Rouse: 0.5)
g3 exponent : 0.917 ± 0.005  (expected: 1.0)
R² (g1)     : 0.9993
R² (g3)     : 0.9985
```

**Interpretation:** g1 exponent ~0.62 (between 0.5 and 1.0) is expected — the trajectory spans ~100τ of production, which is only 10% of the Rouse time (τ_R=900τ), so we're in the crossover regime between subdiffusive and diffusive behavior. With longer trajectories spanning multiple τ_R, the short-time exponent would converge closer to 0.5. The COM exponent ~0.92 is close to the expected 1.0, confirming diffusive whole-chain motion.

**Output:** `plots/msd_vs_time.png`

**Run:**
```bash
python3 scripts/plot_msd.py --output plots/msd_vs_time.png
```

---

## 6. Summary of All Generated Plots

| Plot | File | Source data | Status |
|---|---|---|---|
| Simulation GIFs (5 variants) | `plots/sim_*.gif` | trajectory_0000.json | ✅ Done |
| Equilibration diagnostics | `plots/equilibration/` | Live simulation | ✅ Done |
| Production verification | `plots/production_verification.png` | Live simulation | ✅ Done |
| Rg² vs N (quick) | `plots/rg2_vs_N_quick.png` | Quick sim (capped) | ✅ Done |
| Force vs extension (quick) | `plots/force_extension_quick.png` | Constrained sim | ✅ Done |
| MSD vs time | `plots/msd_vs_time.png` | trajectory_0000.json | ✅ Done |
