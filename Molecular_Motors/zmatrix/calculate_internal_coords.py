#!/usr/bin/env python3
"""
Calculate internal coordinates from z-matrix definition for trajectory frames.
"""

import numpy as np
import sys
from typing import List, Tuple, Dict, Optional


class InternalCoordinates:
    def __init__(self):
        self.zmat_data = []
        self.atom_names = []
        self.extra_bonds = []
        self.extra_angles = []
        self.extra_dihedrals = []

    def set_extra_coords(
        self,
        bonds: List[Tuple[int, int]],
        angles: List[Tuple[int, int, int]],
        dihedrals: List[Tuple[int, int, int, int]],
    ):
        """Set additional internal coordinates (0-based indexing)."""
        self.extra_bonds = bonds
        self.extra_angles = angles
        self.extra_dihedrals = dihedrals

    def read_zmatrix(self, filename: str):
        """Read z-matrix definition from file."""
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    atom = parts[0]
                    conn1 = int(parts[1]) if parts[1] != "0" else 0
                    conn2 = int(parts[2]) if parts[2] != "0" else 0
                    conn3 = int(parts[3]) if parts[3] != "0" else 0
                    self.zmat_data.append(
                        {
                            "atom": atom,
                            "conn1": conn1,  # bond connection
                            "conn2": conn2,  # angle connection
                            "conn3": conn3,  # dihedral connection
                        }
                    )
                    self.atom_names.append(atom)

    def read_xyz_frame(self, f) -> Optional[np.ndarray]:
        """Read a single XYZ frame from a stream and return coordinates or None at EOF."""
        # Skip blank lines between frames
        while True:
            line = f.readline()
            if not line:
                return None
            if line.strip():
                break

        try:
            n_atoms = int(line.strip())
        except ValueError as e:
            raise ValueError(f"Invalid atom count line: {line.strip()}") from e

        # Read and ignore comment line (may be blank)
        comment = f.readline()
        if comment == "":
            raise ValueError("Unexpected EOF after atom count line")

        coords = np.zeros((n_atoms, 3))
        i = 0
        while i < n_atoms:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading atom coordinates")
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid coordinate line: {line.strip()}")
            coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
            i += 1

        return coords

    def calculate_distance(self, coords: np.ndarray, i: int, j: int) -> float:
        """Calculate distance between atoms i and j."""
        return np.linalg.norm(coords[i] - coords[j])

    def calculate_angle(self, coords: np.ndarray, i: int, j: int, k: int) -> float:
        """Calculate angle between atoms i-j-k (j is center)."""
        vec1 = coords[i] - coords[j]
        vec2 = coords[k] - coords[j]

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-12 or norm2 < 1e-12:
            raise ValueError(f"Zero-length vector in angle calculation for atoms {i}-{j}-{k}")

        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calculate_dihedral(self, coords: np.ndarray, i: int, j: int, k: int, l: int) -> float:
        """Calculate dihedral angle between atoms i-j-k-l."""
        vec1 = coords[j] - coords[i]
        vec2 = coords[k] - coords[j]
        vec3 = coords[l] - coords[k]

        # Calculate normal vectors to planes
        n1 = np.cross(vec1, vec2)
        n2 = np.cross(vec2, vec3)

        # Normalize
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm < 1e-12 or n2_norm < 1e-12:
            raise ValueError(f"Zero-length normal in dihedral calculation for atoms {i}-{j}-{k}-{l}")
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        # Calculate dihedral angle
        cos_dihedral = np.dot(n1, n2)
        cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)
        dihedral = np.degrees(np.arccos(cos_dihedral))

        # Determine sign
        if np.dot(np.cross(n1, n2), vec2) < 0:
            dihedral = -dihedral

        return dihedral

    def write_zmat_info(self, info_filename: str):
        """Write information about which atoms correspond to each internal coordinate."""
        with open(info_filename, "w") as f:
            f.write("# Z-Matrix Internal Coordinate Information\n")
            f.write("# This file maps each internal coordinate to the atoms involved\n")
            f.write("# Format: CoordinateType_Number | Atoms (0-based indexing) | Description\n")
            f.write("#\n")

            bond_count = 1
            angle_count = 1
            dihedral_count = 1

            for i, zmat_entry in enumerate(self.zmat_data):
                atom_idx = i  # 0-based indexing for output
                atom_name = zmat_entry["atom"]

                # Bond information
                if zmat_entry["conn1"] > 0:
                    conn1_idx = zmat_entry["conn1"] - 1  # Convert to 0-based
                    conn1_atom = self.atom_names[conn1_idx]
                    f.write(
                        f"Bond_{bond_count:2d}     | Atoms {atom_idx:2d}-{conn1_idx:2d} | {atom_name}{atom_idx}-{conn1_atom}{conn1_idx} distance\n"
                    )
                    bond_count += 1

                # Angle information
                if zmat_entry["conn2"] > 0:
                    conn1_idx = zmat_entry["conn1"] - 1  # Convert to 0-based
                    conn2_idx = zmat_entry["conn2"] - 1  # Convert to 0-based
                    conn1_atom = self.atom_names[conn1_idx]
                    conn2_atom = self.atom_names[conn2_idx]
                    f.write(
                        f"Angle_{angle_count:2d}    | Atoms {atom_idx:2d}-{conn1_idx:2d}-{conn2_idx:2d} | {atom_name}{atom_idx}-{conn1_atom}{conn1_idx}-{conn2_atom}{conn2_idx} angle\n"
                    )
                    angle_count += 1

                # Dihedral information
                if zmat_entry["conn3"] > 0:
                    conn1_idx = zmat_entry["conn1"] - 1  # Convert to 0-based
                    conn2_idx = zmat_entry["conn2"] - 1  # Convert to 0-based
                    conn3_idx = zmat_entry["conn3"] - 1  # Convert to 0-based
                    conn1_atom = self.atom_names[conn1_idx]
                    conn2_atom = self.atom_names[conn2_idx]
                    conn3_atom = self.atom_names[conn3_idx]
                    f.write(
                        f"Dihedral_{dihedral_count:2d} | Atoms {atom_idx:2d}-{conn1_idx:2d}-{conn2_idx:2d}-{conn3_idx:2d} | {atom_name}{atom_idx}-{conn1_atom}{conn1_idx}-{conn2_atom}{conn2_idx}-{conn3_atom}{conn3_idx} dihedral\n"
                    )
                    dihedral_count += 1

            if self.extra_bonds or self.extra_angles or self.extra_dihedrals:
                f.write("#\n")
                f.write("# Extra internal coordinates (0-based indexing)\n")

            extra_bond_count = 1
            for i, j in self.extra_bonds:
                i_atom = self.atom_names[i] if 0 <= i < len(self.atom_names) else "?"
                j_atom = self.atom_names[j] if 0 <= j < len(self.atom_names) else "?"
                f.write(
                    f"Extra_Bond_{extra_bond_count:2d}     | Atoms {i:2d}-{j:2d} | "
                    f"{i_atom}{i}-{j_atom}{j} distance\n"
                )
                extra_bond_count += 1

            extra_angle_count = 1
            for i, j, k in self.extra_angles:
                i_atom = self.atom_names[i] if 0 <= i < len(self.atom_names) else "?"
                j_atom = self.atom_names[j] if 0 <= j < len(self.atom_names) else "?"
                k_atom = self.atom_names[k] if 0 <= k < len(self.atom_names) else "?"
                f.write(
                    f"Extra_Angle_{extra_angle_count:2d}    | Atoms {i:2d}-{j:2d}-{k:2d} | "
                    f"{i_atom}{i}-{j_atom}{j}-{k_atom}{k} angle\n"
                )
                extra_angle_count += 1

            extra_dihedral_count = 1
            for i, j, k, l in self.extra_dihedrals:
                i_atom = self.atom_names[i] if 0 <= i < len(self.atom_names) else "?"
                j_atom = self.atom_names[j] if 0 <= j < len(self.atom_names) else "?"
                k_atom = self.atom_names[k] if 0 <= k < len(self.atom_names) else "?"
                l_atom = self.atom_names[l] if 0 <= l < len(self.atom_names) else "?"
                f.write(
                    f"Extra_Dihedral_{extra_dihedral_count:2d} | Atoms {i:2d}-{j:2d}-{k:2d}-{l:2d} | "
                    f"{i_atom}{i}-{j_atom}{j}-{k_atom}{k}-{l_atom}{l} dihedral\n"
                )
                extra_dihedral_count += 1

    def calculate_internal_coords_frame(self, coords: np.ndarray) -> Dict[str, List[float]]:
        """Calculate all internal coordinates for a single frame."""
        bonds = []
        angles = []
        dihedrals = []
        extra_bonds = []
        extra_angles = []
        extra_dihedrals = []

        for i, zmat_entry in enumerate(self.zmat_data):
            atom_idx = i

            # Bond length (if conn1 > 0)
            if zmat_entry["conn1"] > 0:
                j = zmat_entry["conn1"] - 1  # Convert to 0-based indexing
                bond = self.calculate_distance(coords, atom_idx, j)
                bonds.append(bond)

            # Bond angle (if conn2 > 0)
            if zmat_entry["conn2"] > 0:
                j = zmat_entry["conn1"] - 1  # center atom
                k = zmat_entry["conn2"] - 1
                angle = self.calculate_angle(coords, atom_idx, j, k)
                angles.append(angle)

            # Dihedral angle (if conn3 > 0)
            if zmat_entry["conn3"] > 0:
                j = zmat_entry["conn1"] - 1
                k = zmat_entry["conn2"] - 1
                l = zmat_entry["conn3"] - 1
                dihedral = self.calculate_dihedral(coords, atom_idx, j, k, l)
                dihedrals.append(dihedral)

        for i, j in self.extra_bonds:
            extra_bonds.append(self.calculate_distance(coords, i, j))
        for i, j, k in self.extra_angles:
            extra_angles.append(self.calculate_angle(coords, i, j, k))
        for i, j, k, l in self.extra_dihedrals:
            extra_dihedrals.append(self.calculate_dihedral(coords, i, j, k, l))

        return {
            "bonds": bonds,
            "angles": angles,
            "dihedrals": dihedrals,
            "extra_bonds": extra_bonds,
            "extra_angles": extra_angles,
            "extra_dihedrals": extra_dihedrals,
        }

    def process_trajectory(self, xyz_filename: str, output_filename: str):
        """Process entire trajectory and write internal coordinates."""
        frame_idx = 0

        with open(xyz_filename, "r") as f, open(output_filename, "w") as outfile:
            # Write header
            outfile.write("#Frame")

            # Count expected coordinates for header
            n_bonds = sum(1 for entry in self.zmat_data if entry["conn1"] > 0)
            n_angles = sum(1 for entry in self.zmat_data if entry["conn2"] > 0)
            n_dihedrals = sum(1 for entry in self.zmat_data if entry["conn3"] > 0)

            for i in range(n_bonds):
                outfile.write(f"\tBond_{i+1}")
            for i in range(len(self.extra_bonds)):
                outfile.write(f"\tExtra_Bond_{i+1}")
            for i in range(n_angles):
                outfile.write(f"\tAngle_{i+1}")
            for i in range(len(self.extra_angles)):
                outfile.write(f"\tExtra_Angle_{i+1}")
            for i in range(n_dihedrals):
                outfile.write(f"\tDihedral_{i+1}")
            for i in range(len(self.extra_dihedrals)):
                outfile.write(f"\tExtra_Dihedral_{i+1}")
            outfile.write("\n")

            # Process each frame
            while True:
                try:
                    coords = self.read_xyz_frame(f)
                    if coords is None:
                        break
                    if len(coords) != len(self.zmat_data):
                        raise ValueError(
                            f"Atom count mismatch: z-matrix has {len(self.zmat_data)} atoms, "
                            f"frame has {len(coords)} atoms"
                        )
                    internal_coords = self.calculate_internal_coords_frame(coords)

                    # Write frame data
                    outfile.write(f"{frame_idx}")
                    for bond in internal_coords["bonds"]:
                        outfile.write(f"\t{bond:.6f}")
                    for bond in internal_coords["extra_bonds"]:
                        outfile.write(f"\t{bond:.6f}")
                    for angle in internal_coords["angles"]:
                        outfile.write(f"\t{angle:.6f}")
                    for angle in internal_coords["extra_angles"]:
                        outfile.write(f"\t{angle:.6f}")
                    for dihedral in internal_coords["dihedrals"]:
                        outfile.write(f"\t{dihedral:.6f}")
                    for dihedral in internal_coords["extra_dihedrals"]:
                        outfile.write(f"\t{dihedral:.6f}")
                    outfile.write("\n")

                    frame_idx += 1

                    if frame_idx % 1000 == 0:
                        print(f"Processed {frame_idx} frames...")

                except (IndexError, ValueError) as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    break

        print(f"Completed processing {frame_idx} frames")
        print(f"Internal coordinates written to: {output_filename}")


def main():
    if len(sys.argv) > 1:
        zmat_file = sys.argv[1]
        xyz_file = sys.argv[2] if len(sys.argv) > 2 else "rmsfit_traj_S1.xyz"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "internal_coordinates.dat"
    else:
        zmat_file = "zmat.dat"
        xyz_file = "rmsfit_traj_S1.xyz"
        output_file = "internal_coordinates_s1.dat"

    print(f"Reading z-matrix from: {zmat_file}")
    print(f"Reading trajectory from: {xyz_file}")
    print(f"Writing output to: {output_file}")

    calculator = InternalCoordinates()
    calculator.read_zmatrix(zmat_file)

    print(f"Loaded z-matrix with {len(calculator.zmat_data)} atoms")

    # Optional: additional internal coordinates (0-based indexing).
    # Set enable_extra_coords to True to include them.
    enable_extra_coords = True
    extra_bonds = [(3, 22), (3, 23)]
    extra_angles = [(3, 21, 22)]
    extra_dihedrals = [(22, 21, 19, 3), (19, 22, 23, 21)]
    if enable_extra_coords:
        calculator.set_extra_coords(extra_bonds, extra_angles, extra_dihedrals)

    # Write ZMAT_INFO.dat file first
    info_file = "ZMAT_INFO.dat"
    calculator.write_zmat_info(info_file)
    print(f"Z-matrix coordinate mapping written to: {info_file}")

    calculator.process_trajectory(xyz_file, output_file)


if __name__ == "__main__":
    main()
