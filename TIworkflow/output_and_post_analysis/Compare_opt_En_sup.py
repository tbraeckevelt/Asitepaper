import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ase.io import read
import molmod.units


path_cwd = Path.cwd()
width = 0.1
fig, ax = plt.subplots()
x_pos = width/2.0
x_ticks = [width*6, width*20, width*32]

T_lst_all = [150, 187, 234, 293, 350, 419, 501, 600]
colors_blue = ["#001F3F", "#003F7F", "#005FBF", "#007FFF", "#339FFF", "#66BFFF", "#99DFFF", "#CCEFFF"]
colors_orange = ["#FF4500", "#FF6522", "#FF8544", "#FFA566", "#FFBD80", "#FFD199", "#FFE0B3", "#FFEBD4"]
colors_green = [ "#004F1F", "#007F3F", "#009F5F", "#00BF7F", "#33DF9F", "#66EFBF", "#99F7DF", "#CCFFEF"]


for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:
    Path_mat = path_cwd / mat

    if mat == 'CsPbI3':
        T_lst_sup = [150, 350, 600]
    else:
        T_lst_sup = [234, 350, 501]

    for phase in ['gamma', 'Csdelta', 'FAdelta']:
        Path_phase = Path_mat / phase
        if phase == 'Csdelta':
            colors = colors_blue
        elif phase == 'FAdelta':
            colors = colors_orange
        else:
            colors = colors_green

        for T_NVT in T_lst_sup:
            Path_NVT = Path_phase / str("NVT_T" + str(T_NVT))
            min_atoms = read(Path_NVT / "NVT_opt_sup_1" / "min_struc.xyz")
            energies = min_atoms.get_potential_energy() * molmod.units.electronvolt / (molmod.units.kjmol*64)
            min_atoms_sup = read(Path_NVT / "NVT_opt_sup_2" / "min_struc.xyz")
            energies_sup = min_atoms_sup.get_potential_energy() * molmod.units.electronvolt / (molmod.units.kjmol*64)
            print(mat, phase, T_NVT, energies_sup, energies)

            if mat == 'CsPbI3' or mat == 'FAPbI3':
                ax.bar(x_pos, energies_sup - energies, width, color=colors[-1-T_lst_all.index(T_NVT)], label=phase + str(T_NVT))
            else:
                ax.bar(x_pos, energies_sup - energies, width, color=colors[-1-T_lst_all.index(T_NVT)])
            x_pos += width
        x_pos += width
    x_pos += 2*width

ax.set_xlabel('simulation')
ax.set_ylabel('energy [kJ/mol] pfu')

ax.set_ylim(-2, 2)
plt.gcf().set_size_inches(7.5, 7.5)
plt.axhline(0.0, color='k', linestyle=":")
ax.set_title("ground state energy difference")

ax.set_xticks(x_ticks)
ax.set_xticklabels(['CsPbI3', 'FAPbI3', 'MAPbI3'])
ax.legend()

plt.savefig(path_cwd / "contrib_plots" / str('gse_supeffect_reopt.pdf'), bbox_inches='tight')
plt.clf()