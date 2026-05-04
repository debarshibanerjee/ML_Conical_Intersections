# ML_Conical_Intersections

Small collection of scripts and notebooks for analyzing molecular-trajectory data
in the context of conical intersections. The current repository content is centered
on the `Molecular_Motors/` case study.

## Contents

`Molecular_Motors/` currently contains:

- `1_KLD.py`: computes KL/JS-divergence-based matrices from ground-state and excited-state trajectories, and writes Coulomb-matrix outputs such as `pos_cm-kld.txt`.
- `calculate_internal_coords.py`: converts trajectory frames plus a z-matrix definition into internal-coordinate tables and a `ZMAT_INFO.dat` mapping file.
- `2_PCA.ipynb`: essential-dynamics / PCA analysis for S0 and S1 trajectories, including PC projections and S0 vs S1 comparison.
- `3_IIB.ipynb`: information-imbalance analysis focused on high-KLD atoms / Coulomb-matrix elements against a `ΔE`/FOSC target.
- `4_DIIB.ipynb`: DII analysis on Coulomb-matrix atom pairs, again tied to high-KLD regions and `ΔE`/FOSC ranking. Same as `3_IIB.ipynb` but with the more modern DII framework instead.
- `5_dii-internal_vs_kld-hotspots.py`: compares internal coordinates against KLD hotspot pairs using differentiable information imbalance (DII).
- `6_DIIB-feature-selection.ipynb`: selects internal coordinates from DII-hotspot outputs and runs DII-based feature selection against a target observable.
- `7_DIIB-modes.ipynb`: same as `6_DIIB-feature-selection.ipynb` but with the pre-selected modes for Molecular Motors (Dihedral/Double Bond/Pyramidalization).
- `8_DIIB-feature-selection-FOSC.ipynb`: feature-selection workflow analogous to notebook 6, but using FOSC as the target.
- `8_DIIB-feature-selection-FOSC-TDM.ipynb`: feature-selection workflow analogous to notebook 6, but using FOSC/ΔE (i.e., only the transition dipole moment (TDM) contribution to FOSC) as the target.
- `9_PCA-DII.ipynb`: DII feature selection for `ΔE`/FOSC using the principal components (PCs)
- `9_PCA-DII_FOSC-TDM.ipynb`: DII feature selection for FOSC(TDM contribution only) using the principal components (PCs)

## Workflow:

Run the scripts in order. Ensure you have [DADApy](https://github.com/sissa-data-science/DADApy) installed.
Use [version 0.3.4](https://github.com/sissa-data-science/DADApy/releases/tag/v0.3.4) to reproduce the results here.
For installation instructions, in particular using Jax to speed up the DII calculations refer to the DADApy repository.

1. compute descriptor divergence matrices with `1_KLD.py` for 2 electronic states, say, S1 and S0
2. derive internal coordinates with `calculate_internal_coords.py` for all frames and also derive all the principal components using `2_PCA.ipynb`
3. using `3_IIB.ipynb` or `4_DIIB.ipynb` **(preferred)** to calculate the coulomb/descriptor matrix hotspots from the output of step 1
4. compare which internal coordinates best correlate to the hotspots with `5_dii-internal_vs_kld-hotspots.py`
5. use `6_DIIB-feature-selection.ipynb` or `8_DIIB-feature-selection-FOSC.ipynb`  or `9_PCA-DII.ipynb` to calculate which internal coordinates/PCs best correspond to `ΔE`/FOSC (this uses some filtered internal coordinates from the previous step)

## Citation:

