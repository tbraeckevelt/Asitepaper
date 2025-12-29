import numpy as np
from pathlib import Path
from ase.io import read
import csv


path_cwd = Path.cwd()

min_atoms_dct = {}
min_atoms_dct_sup = {}
for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:
    Path_mat = path_cwd / mat

    min_atoms_dct[mat] = {}
    min_atoms_dct_sup[mat] = {}

    T_lst = [150, 187, 234, 293, 350, 419, 501, 600]
    if mat == 'CsPbI3':
        T_lst_sup = [150, 350, 600]
    else:
        T_lst_sup = [234, 350, 501]

    for phase in ['gamma', 'Csdelta', 'FAdelta']:
        Path_phase = Path_mat / phase

        min_atoms_dct[mat][phase] = {}
        min_atoms_dct[mat][phase]["NPT"] = read(Path_phase / "NPT_opt" / "min_struc.xyz")
        for T_NVT in T_lst:
            Path_NVT = Path_phase / str("NVT_T" + str(T_NVT))
            min_atoms_dct[mat][phase][T_NVT] = read(Path_NVT / "NVT_opt" / "min_struc.xyz")

        min_atoms_dct_sup[mat][phase] = {}
        min_atoms_dct_sup[mat][phase]["NPT"] = read(Path_phase / "NPT_opt_sup_2" / "min_struc.xyz")
        for T_NVT in T_lst_sup:
            Path_NVT = Path_phase / str("NVT_T" + str(T_NVT))
            min_atoms_dct_sup[mat][phase][T_NVT] = read(Path_NVT / "NVT_opt_sup_2" / "min_struc.xyz")


vol_dct = {}
for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:
    Path_mat = path_cwd / mat
    vol_dct[mat] = {}

    cell_lst = [["phase", "Average cell at temperature", "|$\\vec{a}$| [$\\text{\\AA}$]", "|$\\vec{b}$| [$\\text{\\AA}$]", "|$\\vec{c}$| [$\\text{\\AA}$]", "$\\alpha$ [$^\\circ$]", "$\\beta$ [$^\\circ$]", "$\\gamma$ [$^\\circ$]", "Volume pfu [$\\text{\\AA}^3$]"]]

    for phase in ['gamma', 'Csdelta', 'FAdelta']:
        Path_phase = Path_mat / phase
        vol_dct[mat][phase] = []
        if phase == "gamma":
            phaselatex = "$\\gamma$"
        elif phase == "Csdelta":
            phaselatex = "$\\delta_\\text{Cs}$"
        elif phase == "FAdelta":
            phaselatex = "$\\delta_\\text{FA}$"

        
        for i, key in enumerate(min_atoms_dct[mat][phase]):
            if key == "NPT":
                cell_row = [phaselatex, 0]
            else:
                cell_row = [phaselatex, key]
            for ln in min_atoms_dct[mat][phase][key].cell.lengths():
                cell_row.append(np.round(ln,2))
            for ang in min_atoms_dct[mat][phase][key].cell.angles():
                cell_row.append(np.round(ang,2))
            cell_row.append(np.round(min_atoms_dct[mat][phase][key].cell.volume/8,0))
            vol_dct[mat][phase].append(min_atoms_dct[mat][phase][key].cell.volume/8)
            cell_lst.append(cell_row)

    # File path to save the CSV
    file_path = Path_mat / "Plots" / "opt" / "cell_parameters.csv"

    # Write nested list to CSV
    #with open(file_path, mode='w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerows(cell_lst)


# Plotting volumes
tem_lst = [0, 150, 187, 234, 293, 350, 419, 501, 600]
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
linestyles = {'gamma': '-', 'Csdelta': '--', 'FAdelta': ':'}
colors = {'CsPbI3': 'tab:blue', 'FAPbI3': 'tab:orange', 'MAPbI3': 'tab:green'}
markers = {'gamma': 'o', 'Csdelta': '^', 'FAdelta': 's'}
for mat in vol_dct:
    for phase in vol_dct[mat]:
        plt.plot(tem_lst, vol_dct[mat][phase], label=f'{mat} {phase}', linestyle=linestyles[phase], color=colors[mat], marker=markers[phase])

plt.xlabel('Temperature (K)')
plt.ylabel('Volume per formula unit ($\\text{\\AA}^3$)')
plt.title('Volume per formula unit vs Temperature')
plt.legend()
#plt.grid(True)

# Save the plot
plot_path = path_cwd / "volume_vs_temperature.pdf"
plt.savefig(plot_path)
plt.show()
