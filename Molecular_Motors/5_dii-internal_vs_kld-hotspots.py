#!/usr/bin/env python3
"""
Compute differentiable information imbalance (DII) between internal
coordinates and KLD hotspots from Coulomb matrix time series.
"""

import argparse
import gzip
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax

jax.config.update("jax_platform_name", "gpu")

try:
    from dadapy import DiffImbalance
except ImportError as e:
    raise SystemExit("dadapy is required for DiffImbalance. Install it and re-run.") from e


def _all_numeric(tokens):
    """Return True if every token can be parsed as float, else False."""
    try:
        for t in tokens:
            float(t)
        return True
    except ValueError:
        return False


def load_internal_coordinates(path: Path):
    """
    Load internal-coordinate table and header from a text file.

    Supports both:
    - Headered files with a first row like '#Frame\\tBond_1...'
    - Headerless numeric files (synthetic headers are generated)
    """
    with path.open("r") as f:
        first_line = f.readline().strip()
    if first_line.startswith("#"):
        first_line = first_line[1:].lstrip()
    tokens = first_line.split()
    if tokens and _all_numeric(tokens):
        # No header; load full file as data
        data = np.loadtxt(path, delimiter="\t")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        header = ["Frame"] + [f"Coord_{i+1}" for i in range(data.shape[1] - 1)]
        return header, data

    header = first_line.split("\t")
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if len(header) != data.shape[1]:
        raise ValueError(f"Header/data column mismatch: {len(header)} vs {data.shape[1]}")
    return header, data


def group_columns(header):
    """
    Group column indices into bonds/angles/dihedrals from header names.

    If no known naming pattern is found, falls back to a single 'all' group
    containing all non-frame columns.
    """
    groups = {"bonds": [], "angles": [], "dihedrals": []}
    for idx, name in enumerate(header):
        if name.lstrip("#").strip().lower() == "frame":
            continue
        if name.startswith("Bond_") or name.startswith("Extra_Bond_"):
            groups["bonds"].append(idx)
        elif name.startswith("Angle_") or name.startswith("Extra_Angle_"):
            groups["angles"].append(idx)
        elif name.startswith("Dihedral_") or name.startswith("Extra_Dihedral_"):
            groups["dihedrals"].append(idx)
    if not groups["bonds"] and not groups["angles"] and not groups["dihedrals"]:
        groups = {"all": [i for i, n in enumerate(header) if n.lstrip("#").strip().lower() != "frame"]}
    return groups


def load_kld_matrix(path: Path):
    """Load a square KLD matrix from text and validate shape."""
    kld = np.loadtxt(path)
    if kld.ndim != 2 or kld.shape[0] != kld.shape[1]:
        raise ValueError(f"KLD matrix must be square; got {kld.shape}")
    return kld


def find_hotspots(kld, cutoff=0.2):
    """
    Select hotspot pairs (i, j) from upper triangle where KLD[i, j] > cutoff.

    This matches the notebook-style hotspot logic using unique pairs i < j.
    """
    upper_triangle_mask = np.triu(np.ones_like(kld, dtype=bool), k=1)
    value_mask = kld > cutoff
    combined_mask = upper_triangle_mask & value_mask
    rows, cols = np.where(combined_mask)
    idx = list(zip(rows.tolist(), cols.tolist()))
    return idx, cutoff


def parse_user_hotspot_pairs(pair_args, matrix_size):
    """
    Parse and validate user-provided hotspot pairs from CLI arguments.

    Input shape:
    - pair_args: list of (i, j)

    Rules:
    - each pair is canonicalized to (min(i, j), max(i, j))
    - self-pairs are rejected
    - duplicates are removed while preserving first-seen order
    - indices must satisfy 0 <= i,j < matrix_size
    """
    hotspots = []
    seen = set()
    for raw_i, raw_j in pair_args:
        i = int(raw_i)
        j = int(raw_j)
        if i == j:
            raise ValueError(f"Invalid hotspot pair ({i}, {j}): self-pairs are not allowed")
        if i < 0 or j < 0 or i >= matrix_size or j >= matrix_size:
            raise ValueError(
                f"Invalid hotspot pair ({i}, {j}): indices must be within [0, {matrix_size - 1}]"
            )
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        seen.add(pair)
        hotspots.append(pair)
    if not hotspots:
        raise ValueError("No valid hotspot pairs were provided.")
    return hotspots


def load_coulomb_matrix(path: Path):
    """Load compressed Coulomb trajectory matrix and validate 3D square layout."""
    with gzip.GzipFile(path, "rb") as f:
        cm = np.load(f)
    if cm.ndim != 3:
        raise ValueError(f"Coulomb matrix must be 3D; got shape {cm.shape}")
    if cm.shape[1] != cm.shape[2]:
        raise ValueError(f"Coulomb matrix must be square per frame; got {cm.shape}")
    return cm


def parse_bond_pairs_from_zmat_info(path: Path):
    """
    Return a set of unique bond atom pairs parsed from `ZMAT_INFO.dat`.

    Parsing rules:
    - Only lines starting with `Bond_` or `Extra_Bond_` are considered.
    - Atom indices are taken from the `Atoms i-j` field.
    - Each pair is stored in canonical sorted form `(min(i, j), max(i, j))`.
    - Self-pairs `(i, i)` are ignored.

    Why this function exists:
    - Fast membership checks such as "is hotspot pair (i, j) also a bond?".
    - Used for counting overlap and for bond-specific filtering logic.

    Example:
    - Input line: `Bond_13 | Atoms 13-12 | ...`
    - Stored pair: `(12, 13)`
    """
    bond_pairs = set()
    pattern = re.compile(r"Atoms\s+(\d+)\s*-\s*(\d+)")
    with path.open("r") as f:
        for line in f:
            if line.startswith("Bond_") or line.startswith("Extra_Bond_"):
                m = pattern.search(line)
                if m:
                    i = int(m.group(1))
                    j = int(m.group(2))
                    if i != j:
                        bond_pairs.add((min(i, j), max(i, j)))
    return bond_pairs


def parse_bond_label_pair_map_from_zmat_info(path: Path):
    """
    Build a mapping from bond coordinate labels to canonical atom pairs.

    Output shape:
    - dict[str, tuple[int, int]]
      e.g. `{"Bond_13": (12, 13), "Extra_Bond_2": (1, 33)}`

    Parsing rules:
    - Only `Bond_*` and `Extra_Bond_*` lines are included.
    - Label text is normalized to no spaces in numbering:
      `Bond_ 13` -> `Bond_13`.
    - Atom pairs are canonicalized to sorted order `(min, max)`.
    - Self-pairs are skipped.

    Why this function exists:
    - Matrix rows are keyed by coordinate label (`Bond_13`), while hotspots are
      keyed by atom pair (`(12, 13)`); this mapping bridges row labels to pairs.
    """
    label_to_pair = {}
    id_pattern = re.compile(r"^(Bond_|Extra_Bond_)\s*(\d+)")
    atom_pattern = re.compile(r"Atoms\s+(\d+)\s*-\s*(\d+)")
    with path.open("r") as f:
        for line in f:
            id_match = id_pattern.match(line)
            if not id_match:
                continue
            atom_match = atom_pattern.search(line)
            if not atom_match:
                continue
            prefix, num = id_match.groups()
            i = int(atom_match.group(1))
            j = int(atom_match.group(2))
            if i == j:
                continue
            label = f"{prefix}{int(num)}"
            label_to_pair[label] = (min(i, j), max(i, j))
    return label_to_pair


def parse_coord_label_to_atoms_from_zmat_info(path: Path):
    """
    Map coordinate labels to display-friendly atom-index strings.

    Included labels:
    - Bond_*, Angle_*, Dihedral_*
    - Extra_Bond_*, Extra_Angle_*, Extra_Dihedral_*

    Output shape:
    - dict[str, str]
      e.g.:
      - `Bond_13 -> "13-12"`
      - `Angle_13 -> "14-12-9"`
      - `Dihedral_13 -> "15-14-12-9"`

    Why this function exists:
    - Heatmap row labels should show atom-index geometry directly, not only
      abstract coordinate IDs like `Dihedral_13`.
    """
    label_to_atoms = {}
    id_pattern = re.compile(r"^(Bond_|Angle_|Dihedral_|Extra_Bond_|Extra_Angle_|Extra_Dihedral_)\s*(\d+)")
    atom_pattern = re.compile(r"Atoms\s+(\d+(?:\s*-\s*\d+){1,3})")
    with path.open("r") as f:
        for line in f:
            id_match = id_pattern.match(line)
            if not id_match:
                continue
            atom_match = atom_pattern.search(line)
            if not atom_match:
                continue
            prefix, num = id_match.groups()
            atoms = "-".join(part.strip() for part in atom_match.group(1).split("-"))
            label_to_atoms[f"{prefix}{int(num)}"] = atoms
    return label_to_atoms


def degrees_to_wrapped_radians(X_deg: np.ndarray) -> np.ndarray:
    """Convert degree-valued columns to radians wrapped to [0, 2*pi)."""
    two_pi = 2.0 * np.pi
    return np.mod(np.deg2rad(X_deg), two_pi)


def compute_pair_dii(
    x_series: np.ndarray,
    y_series: np.ndarray,
    periods_A,
    seed: int,
    num_epochs: int,
    k_init: int,
    k_final: int,
    learning_rate: float,
    learning_rate_decay,
    num_points_rows,
):
    """
    Train one DiffImbalance model for a single (feature, hotspot) pair.

    data_A is a single internal-coordinate time series (N, 1).
    data_B is a single hotspot Coulomb time series (N, 1).
    Returns final scalar DII value after training.
    """
    data_A = x_series.reshape(-1, 1)
    data_B = y_series.reshape(-1, 1)

    dii = DiffImbalance(
        data_A=data_A,
        data_B=data_B,
        periods_A=periods_A,
        periods_B=None,
        seed=seed,
        num_epochs=num_epochs,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=True,
        k_init=k_init,
        k_final=k_final,
        lambda_factor=0.1,
        params_init=None,
        optimizer_name="adam",
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        num_points_rows=num_points_rows,
    )
    dii.train()
    dii_value, _ = dii.return_final_dii(
        compute_error=False,
        ratio_rows_columns=None,
        seed=seed,
        discard_close_ind=0,
    )
    return float(dii_value)


def save_matrix_with_labels(path: Path, matrix, row_labels, col_labels):
    """Write a labeled 2D matrix to a tab-delimited text file."""
    header = "\t".join([""] + col_labels)
    with path.open("w") as f:
        f.write(header + "\n")
        for label, row in zip(row_labels, matrix):
            f.write(label + "\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")


def save_heatmap(path: Path, matrix, row_labels, col_labels, title, y_axis_label):
    """Render and save a heatmap image for a labeled 2D matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_title(title)
    ax.set_xlabel("Hotspots (i-j)")
    ax.set_ylabel(y_axis_label)

    if len(col_labels) <= 50:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    else:
        ax.set_xticks([])

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("DII (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    """
    End-to-end DII workflow:
    1) load data and hotspot definitions,
    2) align/subsample/normalize trajectories,
    3) compute DII matrices per coordinate group,
    4) save text, numpy, and heatmap outputs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", default="./traj_data/internal_coordinates_s1.dat")
    parser.add_argument("--kld", default="./computed_data/pos_cm-kld.txt")
    parser.add_argument("--coulomb", default="./computed_data/pos_coul_matrix_S1.npy.gz")
    parser.add_argument("--zmat-info", default="./traj_data/ZMAT_INFO.dat")
    parser.add_argument("--cutoff", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--k-init", type=int, default=60)
    parser.add_argument("--k-final", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--learning-rate-decay", default="cos", choices=["none", "cos", "exp"])
    parser.add_argument("--num-points-rows", type=int, default=None)
    parser.add_argument("--outdir", default="./computed_data/dii_outputs")
    args = parser.parse_args()
    if args.frame_start < 0:
        raise ValueError("--frame-start must be >= 0")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    internal_path = Path(args.internal)
    kld_path = Path(args.kld)
    coulomb_path = Path(args.coulomb)
    zmat_info_path = Path(args.zmat_info)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    header, data = load_internal_coordinates(internal_path)
    groups = group_columns(header)

    kld = load_kld_matrix(kld_path)
    # User-editable hotspot list.
    # Set to None to use automatic cutoff-based hotspot detection from KLD.
    hotspots = [(8, 22), (8, 27), (8, 28), (8, 44), (16, 32), (17, 32)] 
    if hotspots is not None:
        hotspots = parse_user_hotspot_pairs(hotspots, matrix_size=kld.shape[0])
        cutoff = None
        print(f"Using {len(hotspots)} user-specified hotspots from main():")
    else:
        hotspots, cutoff = find_hotspots(kld, cutoff=args.cutoff)
        print(f"KLD max={np.max(kld):.6f}, cutoff={cutoff:.6f}")
        print(f"Found {len(hotspots)} hotspots (ii<jj):")
    if not hotspots:
        raise ValueError("No hotspots found with the given cutoff.")
    for i, j in hotspots:
        print(f"  hotspot {i}-{j} KLD={kld[i, j]:.6f}")

    cm = load_coulomb_matrix(coulomb_path)
    nframes_cm, natoms, _ = cm.shape
    nframes_ic = data.shape[0]
    if nframes_cm != nframes_ic:
        raise ValueError(
            f"Frame mismatch: internal_coordinates has {nframes_ic}, " f"coulomb matrix has {nframes_cm}"
        )
    if kld.shape[0] != natoms:
        raise ValueError(f"KLD matrix size {kld.shape[0]} does not match natoms {natoms}")
    print(f"Loaded Coulomb matrix: nframes={nframes_cm}, natoms={natoms}")

    hotspot_labels = [f"{i}-{j}" for i, j in hotspots]
    hotspot_pairs = {(min(i, j), max(i, j)) for i, j in hotspots}
    bond_pairs = parse_bond_pairs_from_zmat_info(zmat_info_path)
    bond_label_to_pair = parse_bond_label_pair_map_from_zmat_info(zmat_info_path)
    coord_label_to_atoms = parse_coord_label_to_atoms_from_zmat_info(zmat_info_path)
    hotspot_is_bond = [((min(i, j), max(i, j)) in bond_pairs) for i, j in hotspots]
    bond_hotspots = sum(1 for v in hotspot_is_bond if v)
    print(f"Hotspots that are bonds (per ZMAT_INFO): {bond_hotspots}/{len(hotspots)}")

    frame_col = None
    for i, name in enumerate(header):
        if name.lstrip("#").strip().lower() == "frame":
            frame_col = i
            break
    if frame_col is None:
        print("Warning: 'Frame' column not found; using row indices for frame alignment")
        frame_values = np.arange(data.shape[0], dtype=int)
        X_all = data
    else:
        frame_values_raw = data[:, frame_col]
        frame_values = np.rint(frame_values_raw).astype(int)
        if not np.allclose(frame_values_raw, frame_values, atol=1e-6):
            raise ValueError("Frame column contains non-integer values; cannot align to Coulomb frames")
        if np.any(frame_values < 0) or np.any(frame_values >= nframes_cm):
            raise ValueError(
                "Frame column contains indices outside Coulomb frame range " f"[0, {nframes_cm - 1}]"
            )
        X_all = np.delete(data, frame_col, axis=1)

    # Subsample both descriptor and Coulomb trajectories consistently.
    frame_idx = np.arange(args.frame_start, X_all.shape[0], args.frame_stride)
    if frame_idx.size == 0:
        raise ValueError(
            f"No frames selected with frame-start={args.frame_start} " f"and frame-stride={args.frame_stride}"
        )
    X_all = X_all[frame_idx]
    cm = cm[frame_values[frame_idx]]
    # Notebook-style normalization over the selected frame subset.
    max_per_feature = np.max(cm, axis=0, keepdims=True)
    max_per_feature[np.abs(max_per_feature) < 1e-14] = 1.0
    cm = cm / max_per_feature
    print(
        f"Using frames start={args.frame_start}, stride={args.frame_stride}: "
        f"{X_all.shape[0]} frames retained"
    )

    lr_decay = None if args.learning_rate_decay == "none" else args.learning_rate_decay

    for group_name, col_indices in groups.items():
        if not col_indices:
            continue
        # Align group indices to X_all (Frame removed)
        if frame_col is None:
            col_indices_no_frame = col_indices
        else:
            col_indices_no_frame = [(idx - 1) if idx > frame_col else idx for idx in col_indices]
        group_labels = [header[i] for i in col_indices]
        heatmap_row_labels = [coord_label_to_atoms.get(lbl, lbl) for lbl in group_labels]
        y_axis_label = (
            "Internal Coordinates (Bonds)"
            if group_name == "bonds"
            else (
                "Internal Coordinates (Angles)"
                if group_name == "angles"
                else (
                    "Internal Coordinates (Dihedrals)"
                    if group_name == "dihedrals"
                    else "Internal Coordinates"
                )
            )
        )
        X = X_all[:, col_indices_no_frame]
        periods_A = None
        if group_name in {"angles", "dihedrals"}:
            X = degrees_to_wrapped_radians(X)
            periods_A = np.array([2.0 * np.pi], dtype=float)
            print(
                f"Converted group '{group_name}' from degrees to radians and "
                f"wrapped to [0, 2*pi); using periods_A=[2*pi]"
            )

        dii_matrix = np.zeros((X.shape[1], len(hotspots)))
        for h_idx, (i, j) in enumerate(hotspots):
            y = cm[:, i, j]
            print(f"Computing DII for group '{group_name}' hotspot {h_idx + 1}/{len(hotspots)} " f"({i}-{j})")
            for feat_idx in range(X.shape[1]):
                dii_matrix[feat_idx, h_idx] = compute_pair_dii(
                    x_series=X[:, feat_idx],
                    y_series=y,
                    periods_A=periods_A,
                    seed=args.seed,
                    num_epochs=args.num_epochs,
                    k_init=args.k_init,
                    k_final=args.k_final,
                    learning_rate=args.learning_rate,
                    learning_rate_decay=lr_decay,
                    num_points_rows=args.num_points_rows,
                )

        save_matrix_with_labels(
            outdir / f"dii_{group_name}_vs_hotspots.txt",
            dii_matrix,
            group_labels,
            hotspot_labels,
        )
        np.save(outdir / f"dii_{group_name}_vs_hotspots.npy", dii_matrix)
        save_heatmap(
            outdir / f"dii_{group_name}_vs_hotspots.png",
            dii_matrix,
            heatmap_row_labels,
            hotspot_labels,
            title=f"DII: {group_name} vs hotspots",
            y_axis_label=y_axis_label,
        )
        print(f"Wrote {group_name} DII matrix: {dii_matrix.shape}")

        if group_name == "bonds":
            keep_rows = []
            filtered_row_labels = []
            removed_count = 0
            for row_idx, label in enumerate(group_labels):
                pair = bond_label_to_pair.get(label)
                if pair is not None and pair in hotspot_pairs:
                    removed_count += 1
                    continue
                keep_rows.append(row_idx)
                filtered_row_labels.append(label)

            filtered = dii_matrix[keep_rows, :]
            filtered_heatmap_rows = [coord_label_to_atoms.get(lbl, lbl) for lbl in filtered_row_labels]
            save_matrix_with_labels(
                outdir / "dii_bonds_vs_hotspots_no_bond_pairs.txt",
                filtered,
                filtered_row_labels,
                hotspot_labels,
            )
            np.save(
                outdir / "dii_bonds_vs_hotspots_no_bond_pairs.npy",
                filtered,
            )
            if filtered.shape[0] > 0:
                save_heatmap(
                    outdir / "dii_bonds_vs_hotspots_no_bond_pairs.png",
                    filtered,
                    filtered_heatmap_rows,
                    hotspot_labels,
                    title="DII: bonds vs hotspots (bond-coordinate rows removed)",
                    y_axis_label="Internal Coordinates (Bonds)",
                )
            print(
                f"Wrote bonds DII matrix with hotspot columns unchanged and "
                f"{removed_count} bond-coordinate rows removed: {filtered.shape}"
            )

    # Save hotspot list for reference
    with (outdir / "hotspots.txt").open("w") as f:
        if cutoff is None:
            f.write("source=user-specified\n")
        else:
            f.write(f"cutoff={cutoff:.6f}\n")
        for i, j in hotspots:
            f.write(f"{i}\t{j}\t{kld[i, j]:.6f}\n")


if __name__ == "__main__":
    main()
