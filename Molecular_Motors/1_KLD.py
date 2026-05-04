import gzip
import argparse
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
from ase import Atom, Atoms
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from dscribe.descriptors import CoulombMatrix

#!#------------------------------------------------------------------------------------------------------------------#!#
#!#------------------------------------ FUNCTIONS TO CALCULATE KL/JS DIVERGENCE -------------------------------------#!#
#!#------------------------------------------------------------------------------------------------------------------#!#


## Calculate Kullback–Leibler(KL)/Jensen–Shannon(JS) divergence
## It is a measure of how one probability distribution P is different from a second, reference probability distribution Q
## JS is the symmetric (& always finite) version of KL divergence
def kl_divergence(p, q, eps=0.00001, JS=False):
    ## the 'epsilon' is added to ensure finite values of the KL/JS divergence
    pp = np.array(p, dtype=np.double) + eps
    qq = np.array(q, dtype=np.double) + eps
    if JS:
        mm = 0.5 * (pp + qq)
        div_P = np.sum(pp * np.log(pp / mm))
        div_Q = np.sum(qq * np.log(qq / mm))
        divergence = 0.5 * (div_P + div_Q)
    else:
        divergence = np.sum(pp * np.log(pp / qq))

    return divergence


## This returns the KL/JS Divergence after normalizing 'p' & 'q', i.e., they should EACH sum to 1.
def kl_divergence_scipy(p, q, eps=0.00001, JS=False):
    pp = np.array(p, dtype=np.double) + eps
    qq = np.array(q, dtype=np.double) + eps
    if JS:
        mm = 0.5 * (pp + qq)
        divergence = 0.5 * (entropy(pp, mm) + entropy(qq, mm))
    else:
        divergence = entropy(pp, qq)

    return divergence


#!#------------------------------------------------------------------------------------------------------------------#!#
#!#------------------------------------------------------------------------------------------------------------------#!#
#!#------------------------------------------------------------------------------------------------------------------#!#


#!#------------------------------------------------------------------------------------------------------------------#!#
#!#--------------------- COMPUTE KL/JS DIVERGENCE BETWEEN CERTAIN RELEVANT DESCRIPTORS IN GS & ES  ------------------#!#
#!#------------------------------------------------------------------------------------------------------------------#!#


class KLdiverg:
    def __init__(self, args):
        self.filename_ES = args.excited
        self.filename_GS = args.ground
        self.num_procs = args.processors

    #!#--------------------------------------------------------------------------------------------------------------#!#
    #!#------------------------------------------ LOAD XYZ FILE AS TRAJECTORY ---------------------------------------#!#
    #!#--------------------------------------------------------------------------------------------------------------#!#

    def _load_trajectory(self, filename, save=False):
        univ = mda.Universe(filename)
        traj = univ.trajectory.trajectory
        if save:
            self.atom_names = univ.atoms.names  # type: ignore
            self.num_atoms = univ.atoms.n_atoms  # type: ignore
        return traj

    #!#--------------------------------------------------------------------------------------------------------------#!#
    #!#--------------------- CALCULATE DISTANCE MATRIX FOR ALL FRAMES AND RETURN A LIST OF THEM ---------------------#!#
    #!#--------------------------------------------------------------------------------------------------------------#!#

    def _calculate_Distance_Matrix(self, traj, save_array=False, array_name="S0"):
        dist_matrix_list = []

        for frame in traj:
            xyz = frame.positions
            ## create a distance matrix for all atom positions
            tmp_dist_matrix = cdist(xyz, xyz, metric="euclidean")
            ## store each distance matrix for each time step in a list
            dist_matrix_list.append(tmp_dist_matrix)

        if save_array:
            ## save Distance Matrix in a file
            print("Saving Distance Matrix:", np.array(dist_matrix_list).shape)
            with gzip.GzipFile(
                f"computed_data/pos_dist_matrix_{array_name}.npy.gz", "wb", compresslevel=9
            ) as f:
                np.save(file=f, arr=np.array(dist_matrix_list))

        return dist_matrix_list

    #!#--------------------------------------------------------------------------------------------------------------#!#
    #!#--------------------- CALCULATE COULOMB MATRIX FOR ALL FRAMES AND RETURN A LIST OF THEM ----------------------#!#
    #!#--------------------------------------------------------------------------------------------------------------#!#

    def _calculate_Coulomb_Matrix(self, traj, save_array=False, array_name="S0"):
        local_atom_names = self.atom_names
        local_n_atoms = self.num_atoms
        ## initiate an instance of CoulombMatrix from Dscribe
        coul_matrix = CoulombMatrix(n_atoms_max=local_n_atoms, permutation="none", sigma=None, seed=None)
        coul_matrix_list = []

        for frame in traj:
            xyz = frame.positions
            atoms = []
            ## for each atom, create a list of 'Atom' instances suitable for ASE to process
            for ii in range(local_n_atoms):
                symbol = local_atom_names[ii]
                pos = xyz[ii].tolist()
                atoms.append(Atom(symbol, pos))
            mol = Atoms(atoms)
            ## create the coulomb matrix and reshape it as (natoms, natoms)
            tmp_coul_matrix = coul_matrix.create(mol, n_jobs=self.num_procs).reshape(  # type: ignore
                local_n_atoms, local_n_atoms
            )
            ## store each coulomb matrix for each time step in a list
            coul_matrix_list.append(tmp_coul_matrix)

        if save_array:
            ## save Coulomb Matrix in a file
            print("Saving Coulomb Matrix:", np.array(coul_matrix_list).shape)
            with gzip.GzipFile(
                f"computed_data/pos_coul_matrix_{array_name}.npy.gz", "wb", compresslevel=9
            ) as f:
                np.save(file=f, arr=np.array(coul_matrix_list))

        return coul_matrix_list

    #!#--------------------------------------------------------------------------------------------------------------#!#
    #!#------ CALCULATE PDF (HISTOGRAM) FOR VALUES OF A DESCRIPTOR (IN BOTH GS & ES SEPARATELY) FOR ALL FRAMES ------#!#
    #!#--------------------------------------------------------------------------------------------------------------#!#

    def _get_PDF(self, descriptor_GS, descriptor_ES, shift=0.1, nbins=100, hist_density=False):
        data = {"GS": {}, "ES": {}, "bin": {}}
        for iatom in range(self.num_atoms):
            for jatom in range(iatom):
                ## We put the descriptor value between the i and j atom into an array for the ground state
                pdf_GS = []
                for iframe in range(len(descriptor_GS)):
                    descriptor = descriptor_GS[iframe]
                    pdf_GS.append(descriptor[iatom, jatom])

                ## We put the descriptor value between the i and j atom into an array for the excited state
                pdf_ES = []
                for iframe in range(len(descriptor_ES)):
                    descriptor = descriptor_ES[iframe]
                    pdf_ES.append(descriptor[iatom, jatom])

                ## First we get extreme values for the distribution: we need this a-priori for the histogram
                max_GS, min_GS = max(pdf_GS), min(pdf_GS)
                max_ES, min_ES = max(pdf_ES), min(pdf_ES)
                max_val = max_GS if max_GS > max_ES else max_ES
                min_val = min_GS if min_GS < min_ES else min_ES
                max_val += shift
                min_val -= shift

                ## Construct the bins
                step = (max_val - min_val) / float(nbins)
                bins = np.arange(min_val, max_val, step)

                ## Perform the distribution
                hist_GS, _ = np.histogram(pdf_GS, bins=bins, density=hist_density)
                hist_ES, _ = np.histogram(pdf_ES, bins=bins, density=hist_density)

                ## Collect data:
                ## for each (iatom, jatom) pair the histograms/PDFs/bins are stored as a "tuple"
                ## for the corresponding dictionary entry (GS/ES/bin)
                data["GS"][(iatom, jatom)] = hist_GS
                data["ES"][(iatom, jatom)] = hist_ES
                data["bin"][(iatom, jatom)] = bins[:-1]

        return data

    #!#--------------------------------------------------------------------------------------------------------------#!#
    #!#----- KL/JS DIVERGENCE BETWEEN PDF OF A DESCRIPTOR IN GS & ES, FOR ALL PAIRS OF ATOMS (RETURNS A MATRIX) -----#!#
    #!#--------------------------------------------------------------------------------------------------------------#!#

    def _get_KLD_Matrix(self, pdf_GS, pdf_ES, symmetry=True):
        ## pdf_GS/pdf_ES: these are the tuples that store the (iatom, jatom) PDFs for GS & ES respectively
        ## pdf_GS[(7,5)] will give us the PDF/histogram for the descriptor between atoms 7 & 5
        ## OUTPUT: where each value is the KL divergence between the descriptor PDFs in GS v/s ES, for (iatom, jatom)
        kld_matrix = np.zeros((self.num_atoms, self.num_atoms))
        for key in pdf_GS.keys():
            iatom = key[0]
            jatom = key[1]
            if symmetry:
                # val = kl_divergence(pdf_GS[key], pdf_ES[key], JS=True)
                val = kl_divergence_scipy(pdf_GS[key], pdf_ES[key], JS=True)
                kld_matrix[iatom, jatom] = val
                kld_matrix[jatom, iatom] = val
            else:
                # val_PQ = kl_divergence(pdf_GS[key], pdf_ES[key], JS=False)
                # val_QP = kl_divergence(pdf_ES[key], pdf_GS[key], JS=False)
                val_PQ = kl_divergence_scipy(pdf_GS[key], pdf_ES[key], JS=False)
                val_QP = kl_divergence_scipy(pdf_ES[key], pdf_GS[key], JS=False)
                kld_matrix[iatom, jatom] = val_PQ
                kld_matrix[jatom, iatom] = val_QP

        return kld_matrix

    #!#--------------------------------------------------------------------------------------------------------------#!#
    #!#--------------------------------------------- EXECUTE THE ANALYSIS -------------------------------------------#!#
    #!#--------------------------------------------------------------------------------------------------------------#!#

    def run(self):
        ## Load the trajectories of the Ground and Excited states
        self.trj_GS = self._load_trajectory(self.filename_GS, True)
        self.trj_ES = self._load_trajectory(self.filename_ES, True)

        #!#---------------------------------------------- DISTANCE MATRIX -------------------------------------------#!#
        ## Calculate the Distance Matrix for the Ground and Excited states
        # self.Dist_Mat_GS = self._calculate_Distance_Matrix(
        #     self.trj_GS, save_array=True, array_name="S0"
        # )
        # self.Dist_Mat_ES = self._calculate_Distance_Matrix(
        #     self.trj_ES, save_array=True, array_name="S1"
        # )

        # ## Compute the distribution all the distance between 2 pairs of atoms
        # self.data_Dist_Mat = self._get_PDF(
        #     self.Dist_Mat_GS, self.Dist_Mat_ES, hist_density=True
        # )

        # ## Calculate KL divergence matrix for Distance Matrix descriptor
        # KLD_Dist_Mat = self._get_KLD_Matrix(
        #     self.data_Dist_Mat["GS"], self.data_Dist_Mat["ES"], symmetry=True
        # )

        # plt.matshow(KLD_Dist_Mat)
        # plt.savefig("computed_data/pos_dist_mat-kld-full.png", dpi=300)
        # plt.close()

        # np.savetxt("computed_data/pos_dist-mat-kld.txt", KLD_Dist_Mat)

        #!#----------------------------------------------------------------------------------------------------------#!#

        #!#---------------------------------------------- COULOMB MATRIX --------------------------------------------#!#
        ## Calculate the Coulomb Matrix for the Ground and Excited states
        self.Coul_Mat_GS = self._calculate_Coulomb_Matrix(self.trj_GS, save_array=True, array_name="S0")
        self.Coul_Mat_ES = self._calculate_Coulomb_Matrix(self.trj_ES, save_array=True, array_name="S1")

        ## Compute the distribution all the Coulomb Matrix interactions between 2 pairs of atoms
        self.data_CM = self._get_PDF(self.Coul_Mat_GS, self.Coul_Mat_ES, hist_density=True)

        ## Calculate KL divergence matrix for Coulomb Matrix descriptor
        KLD_Coul_Mat = self._get_KLD_Matrix(self.data_CM["GS"], self.data_CM["ES"], symmetry=True)

        plt.matshow(KLD_Coul_Mat)
        plt.savefig("computed_data/pos_cm-kld-full.png", dpi=300)
        plt.close()

        # plt.matshow(KLD_Coul_Mat)
        # plt.xlim([30, 35])
        # plt.ylim([35, 30])
        # plt.savefig("computed_data/cm-kld-zoom.png", dpi=300)
        # plt.close()

        np.savetxt("computed_data/pos_cm-kld.txt", KLD_Coul_Mat)
        #!#----------------------------------------------------------------------------------------------------------#!#


#!#------------------------------------------------------------------------------------------------------------------#!#
#!#------------------------------------------------------------------------------------------------------------------#!#
#!#------------------------------------------------------------------------------------------------------------------#!#


#!#------------------------------------------------------------------------------------------------------------------#!#
#!#-------------------------------------------- MAIN(): WHERE THE MAGIC HAPPENS -------------------------------------#!#
#!#------------------------------------------------------------------------------------------------------------------#!#


def main():
    parser = argparse.ArgumentParser(
        prog="1_KLD.py",
        description="Compute KL/JS divergence for a descriptor between GS & ES.",
    )
    parser.add_argument(
        "-s0",
        "--ground",
        type=str,
        help="filename of the ground state trajectory",
        required=True,
    )
    parser.add_argument(
        "-s1",
        "--excited",
        type=str,
        help="filename of the excited state trajectory",
        required=True,
    )
    parser.add_argument("-n", "--processors", type=int, default=1)
    args = parser.parse_args()
    KL = KLdiverg(args)
    KL.run()


#!#------------------------------------------------------------------------------------------------------------------#!#
#!#------------------------------------------------------------------------------------------------------------------#!#
#!#------------------------------------------------------------------------------------------------------------------#!#

if __name__ == "__main__":
    main()
