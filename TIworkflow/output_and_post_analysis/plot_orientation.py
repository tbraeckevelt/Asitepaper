from lib.utils import from_h5_to_atoms_traj
from lib.DefineCV import Get_MAmol_lst, MA_calc_unit_vector_CN, Get_FAmol_lst, FA_calc_unit_vector_N
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

path_cwd = Path.cwd()

for mat in ['FAPbI3', 'MAPbI3']:
    Path_mat = path_cwd / mat

    for phase in ['gamma', 'Csdelta', 'FAdelta']:
        Path_phase = Path_mat / phase
        for NVT_T in [150, 234, 350]:#, 501]:
            path_nvt = Path_phase / str("NVT_T" + str(NVT_T))
            for Tem in [150, 234, 350]:#, 501]:
                path_sim = path_nvt / "REX_NVT"
                traj = []
                for nr in range(5):
                    h5_file = path_sim / str("traj_T" + str(Tem) + "_" + str(nr) + "_trans.h5")
                    traj += from_h5_to_atoms_traj(h5_file, calib_step = 1000, samp_freq=5)

                if mat == 'FAPbI3':
                    mol_lst = Get_FAmol_lst(traj[0])
                else:
                    mol_lst = Get_MAmol_lst(traj[0])

                vector_lst = []
                plt.figure(figsize=(6, 4))
                
                Cdist = np.zeros(8)
                for i in range(8):
                    Cdist[i] = traj[0].get_distance(mol_lst[0]['C'], mol_lst[i]['C'], mic=True)

                sub_mol_lst = []
                for i in [0, np.argmin(Cdist[1:]) + 1, np.argmax(Cdist)]:
                    sub_mol_lst.append(mol_lst[i])

                for i, mol_dct in enumerate(sub_mol_lst):
                    #Cdistvec = np.round(traj[0].get_distance(mol_lst[0]['C'], mol_lst[i]['C'], mic=True, vector=True), 0)
                    #Cdistvec = np.where(np.abs(Cdistvec) > 3, 1, 0)
                    #Cdist.append(Cdistvec)
                    

                    vector_lst.append([])
                    for atoms in traj:
                        #for MAmol in MAmol_lst:
                        if mat == 'FAPbI3':
                            vector_lst[i].append(FA_calc_unit_vector_N(atoms.get_positions(), atoms.get_masses(), mol_dct))
                        else:
                            vector_lst[i].append(MA_calc_unit_vector_CN(atoms.get_positions(), atoms.get_masses(), mol_dct))
                    # vector_lst.append(MA_calc_unit_vector_CN(atoms.get_positions(), atoms.get_masses(), MAmol_lst[0]))

                    # Convert vector list to numpy array for easier manipulation
                    vector_array = np.array(vector_lst[i])

                    # Extract x and y components of the vectors
                    x = vector_array[:, 0]
                    y = vector_array[:, 1]
                    z = vector_array[:, 2]

                    if i == 0:
                        # Determine the dominant direction for each vector
                        # Calculate the fraction of each direction
                        total_magnitude = np.linalg.norm(vector_array, axis=1)
                        fraction_x = np.abs(x) / total_magnitude
                        fraction_y = np.abs(y) / total_magnitude
                        fraction_z = np.abs(z) / total_magnitude

                        # Create a color map based on the fraction of each direction
                        colors = np.zeros(np.shape(vector_array))
                        colors[:, 0] = fraction_x  # Red for x
                        colors[:, 1] = fraction_y  # Green for y
                        colors[:, 2] = fraction_z  # Blue for z

                    # Create a 2D plot with smaller markers and color based on index in array
                    plt.scatter(0.9*x + 2.0*i, 0.9*y, s=1, c=colors)
                    plt.scatter(0.9*x + 2.0*i, 0.9*z+2.0, s=1, c=colors)
                #plt.xlim(-1, 2.0*i+1)
                #plt.ylim(-1, 3)
                #plt.xlabel('X component')
                #plt.ylabel('Y component')
                #plt.xticks(ticks=np.arange(0, 2.0*i+1, 2.0), labels=Cdist)
                #plt.title('2D Plot of Vectors')
                #plt.grid()
                plt.axis('off')
                plt.savefig(Path_mat / "Plots" / "orient" / str("a_" +phase + "_C" + str(NVT_T) +"_T"+str(Tem)+".png"), bbox_inches='tight')
                plt.clf()