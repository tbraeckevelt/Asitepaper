from lib.utils import from_h5_to_atoms_traj
from lib.DefineCV import Get_MAmol_lst, MA_calc_unit_vector_CN, Get_FAmol_lst, FA_calc_unit_vector_N
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

mat = 'MAPbI3'
phase = 'gamma'
NVT_T = 234
Tem = 150
nr = 0

path_cwd = Path.cwd()

for mat in ['FAPbI3']: #, 'MAPbI3']:
    Path_mat = path_cwd / mat

    for phase in ['Csdelta']: #, 'Csdelta', 'FAdelta']:
        Path_phase = Path_mat / phase
        for NVT_T in [501]: #, 234, 350, 501]:
            path_nvt = Path_phase / str("NVT_T" + str(NVT_T))
            for Tem in [150]: #, 234, 350, 501]:
                path_sim = path_nvt / "REX_NVT"
                traj = []
                for nr in range(5):
                    h5_file = path_sim / str("traj_T" + str(Tem) + "_" + str(nr) + "_trans.h5")
                    traj += from_h5_to_atoms_traj(h5_file, calib_step = 1000, samp_freq=1)

                if mat == 'FAPbI3':
                    mol_lst = Get_FAmol_lst(traj[0])
                else:
                    mol_lst = Get_MAmol_lst(traj[0])

                vector_lst = []
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                Cdist = []
                for i in range(1):
                    Cdistvec = np.round(traj[0].get_distance(mol_lst[0]['C'], mol_lst[i]['C'], mic=True, vector=True), 0)
                    Cdistvec = np.where(np.abs(Cdistvec) > 3, 1, 0)
                    Cdist.append(Cdistvec)

                    vector_lst.append([])
                    for atoms in traj:
                        #for MAmol in MAmol_lst:
                        if mat == 'FAPbI3':
                            vector_lst[i].append(FA_calc_unit_vector_N(atoms.get_positions(), atoms.get_masses(), mol_lst[i]))
                        else:
                            vector_lst[i].append(MA_calc_unit_vector_CN(atoms.get_positions(), atoms.get_masses(), mol_lst[i]))
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

                    # Create a 3D scatter plot
                    j= i%2
                    if i in [2, 3, 6, 7]:
                        k = 1
                    else:
                        k = 0
                    if i > 3:
                        l = 1
                    else:
                        l = 0
                    ax.scatter(0.9*x + 2.0*j, 0.9*y+ 2.0*k, 0.9*z+ 2.0*l, c=colors, s=1)
                plt.show()

                plt.title('3D Plot of Vectors')
                plt.grid()
                plt.savefig(Path_mat / "Plots" / "orient" / str(phase + "_C" + str(NVT_T) +"_T"+str(Tem)+"_3D.png"), bbox_inches='tight')
                plt.close(fig)