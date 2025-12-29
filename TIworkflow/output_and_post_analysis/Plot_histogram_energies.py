import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ase.io import read
import molmod.units

def get_energies_list(Path_txtfile):
    energies = []
    with open(Path_txtfile, 'r') as file:
        for i, line in enumerate(file):
            values = line.split()
            if i == 0:
                min_energy = float(values[7])
            elif len(values) == 7:
                energy = float(values[5]) + min_energy
                energies.append(energy)
    return energies

def get_hist_and_bin(energies, bin_size):
    min_val = np.floor(np.min(energies)/bin_size)*bin_size
    max_val = np.ceil(np.max(energies)/bin_size)*bin_size
    bin = np.arange(min_val - bin_size, max_val + bin_size, bin_size)
    hist, bins = np.histogram(energies, bins= bin)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    return hist, bin_centers


tem_min = 150.0
tem_max = 600.0
num_tem = 32
tem_np = tem_min*np.exp(np.linspace(0.0, np.log(tem_max/tem_min), num_tem))
bin_size = 0.2
path_cwd = Path.cwd()
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

energy_range_dct = {
    'CsPbI3': [-1501.5, -1481.5],
    'FAPbI3': [-5525.5, -5505.5],
    'MAPbI3': [-5060, -5040]
}

offset_dct = {}
bin_dct = {}
hist_dct = {}
min_atoms_dct = {}
bin_dct_sup = {}
hist_dct_sup = {}
min_atoms_dct_sup = {}
for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:
    Path_mat = path_cwd / mat

    bin_dct[mat] = {}
    hist_dct[mat] = {}
    min_atoms_dct[mat] = {}
    bin_dct_sup[mat] = {}
    hist_dct_sup[mat] = {}
    min_atoms_dct_sup[mat] = {}

    T_lst = [150, 187, 234, 293, 350, 419, 501, 600]
    if mat == 'CsPbI3':
        T_lst_sup = [150, 350, 600]
    else:
        T_lst_sup = [234, 350, 501]

    offset_dct[mat] = {}
    max_count={}
    max_count["NPT"] = 0
    for T_NVT in T_lst:
        max_count[T_NVT] = 0


    for phase in ['gamma', 'Csdelta', 'FAdelta']:
        Path_phase = Path_mat / phase

        bin_dct[mat][phase] = {}
        hist_dct[mat][phase] = {}
        min_atoms_dct[mat][phase] = {}

        min_atoms_dct[mat][phase]["NPT"] = read(Path_phase / "NPT_opt" / "min_struc.xyz")
        Path_txtfile = Path_phase / "NPT_opt" / "Compare_minima.txt"
        energies = get_energies_list(Path_txtfile)
        hist_dct[mat][phase]["NPT"], bin_dct[mat][phase]["NPT"] = get_hist_and_bin(energies, bin_size)
        max_count["NPT"] = max(max_count["NPT"], np.max(hist_dct[mat][phase]["NPT"]))

        plt.bar(bin_dct[mat][phase]["NPT"], hist_dct[mat][phase]["NPT"], width=bin_size, color='tab:blue')
        plt.axvline(min_atoms_dct[mat][phase]["NPT"].get_potential_energy()*molmod.units.electronvolt/(8*molmod.units.kjmol), color='tab:blue', linestyle="--")
        plt.xlabel("Ground state energy pfu [kJ/mol]") 
        plt.ylabel("Count") 
        plt.savefig(Path_phase / "NPT_opt" / "histogram_opt_energies.pdf", bbox_inches='tight')
        plt.clf()

        for T_NVT in T_lst:
            Path_NVT = Path_phase / str("NVT_T" + str(T_NVT))
            min_atoms_dct[mat][phase][T_NVT] = read(Path_NVT / "NVT_opt" / "min_struc.xyz")
            Path_txtfile = Path_NVT / "NVT_opt" / "Compare_minima.txt"
            energies = get_energies_list(Path_txtfile)
            hist_dct[mat][phase][T_NVT], bin_dct[mat][phase][T_NVT] = get_hist_and_bin(energies, bin_size)
            max_count[T_NVT] = max(max_count[T_NVT], np.max(hist_dct[mat][phase][T_NVT]))

            plt.bar(bin_dct[mat][phase][T_NVT], hist_dct[mat][phase][T_NVT], width=bin_size, color='tab:blue')
            plt.axvline(min_atoms_dct[mat][phase][T_NVT].get_potential_energy()*molmod.units.electronvolt/(8*molmod.units.kjmol), color='tab:blue', linestyle="--")
            plt.xlabel("Ground state energy pfu [kJ/mol]") 
            plt.ylabel("Count") 
            plt.savefig(Path_NVT / "NVT_opt" / "histogram_opt_energies.pdf", bbox_inches='tight')
            plt.clf()

        bin_dct_sup[mat][phase] = {}
        hist_dct_sup[mat][phase] = {}
        min_atoms_dct_sup[mat][phase] = {}

        min_atoms_dct_sup[mat][phase]["NPT"] = read(Path_phase / "NPT_opt_sup_2" / "min_struc.xyz")
        Path_txtfile = Path_phase / "NPT_opt_sup_2" / "Compare_minima.txt"
        energies = get_energies_list(Path_txtfile)
        hist_dct_sup[mat][phase]["NPT"], bin_dct_sup[mat][phase]["NPT"] = get_hist_and_bin(energies, bin_size)
        max_count["NPT"] = max(max_count["NPT"], np.max(hist_dct_sup[mat][phase]["NPT"]))

        plt.bar(bin_dct_sup[mat][phase]["NPT"], hist_dct_sup[mat][phase]["NPT"], width=bin_size, color='tab:blue')
        plt.axvline(min_atoms_dct_sup[mat][phase]["NPT"].get_potential_energy()*molmod.units.electronvolt/(64*molmod.units.kjmol), color='tab:blue', linestyle="--")
        plt.xlabel("Ground state energy pfu [kJ/mol]") 
        plt.ylabel("Count") 
        plt.savefig(Path_phase / "NPT_opt_sup_2" / "histogram_opt_energies.pdf", bbox_inches='tight')
        plt.clf()

        for T_NVT in T_lst_sup:
            Path_NVT = Path_phase / str("NVT_T" + str(T_NVT))
            min_atoms_dct_sup[mat][phase][T_NVT] = read(Path_NVT / "NVT_opt_sup_2" / "min_struc.xyz")
            Path_NVT = Path_phase / str("NVT_T" + str(T_NVT))
            Path_txtfile = Path_NVT / "NVT_opt_sup_2" / "Compare_minima.txt"
            energies = get_energies_list(Path_txtfile)
            hist_dct_sup[mat][phase][T_NVT], bin_dct_sup[mat][phase][T_NVT] = get_hist_and_bin(energies, bin_size)
            max_count[T_NVT] = max(max_count[T_NVT], np.max(hist_dct_sup[mat][phase][T_NVT]))

            plt.bar(bin_dct_sup[mat][phase][T_NVT], hist_dct_sup[mat][phase][T_NVT], width=bin_size, color='tab:blue')
            plt.axvline(min_atoms_dct_sup[mat][phase][T_NVT].get_potential_energy()*molmod.units.electronvolt/(64*molmod.units.kjmol), color='tab:blue', linestyle="--")
            plt.xlabel("Ground state energy pfu [kJ/mol]") 
            plt.ylabel("Count") 
            plt.savefig(Path_NVT / "NVT_opt_sup_2" / "histogram_opt_energies.pdf", bbox_inches='tight')
            plt.clf()

    offset_dct[mat]["NPT"] = 0
    for i, T_NVT in enumerate(T_lst):
        """
        if i == 0:
            offset_dct[mat][T_NVT] = int(np.ceil((offset_dct[mat]["NPT"] + min(500, max_count["NPT"]))/100)*100)
        else:
            offset_dct[mat][T_NVT] = int(np.ceil((offset_dct[mat][T_lst[i-1]] +  min(500, max_count[T_lst[i-1]]))/100)*100)
        """
        offset_dct[mat][T_NVT] = 500*(i+1)

for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:
    Path_mat = path_cwd / mat

    for phase in ['gamma', 'Csdelta', 'FAdelta']:
        Path_phase = Path_mat / phase

        for i, key in enumerate(hist_dct[mat][phase]):
            plt.bar(bin_dct[mat][phase][key], hist_dct[mat][phase][key], width=bin_size, color=colors[i], alpha=0.5, label=str(key), bottom=offset_dct[mat][key])
            plt.axhline(offset_dct[mat][key], color=colors[i], linestyle=":")
            plt.axvline(min_atoms_dct[mat][phase][key].get_potential_energy()*molmod.units.electronvolt/(8*molmod.units.kjmol), color=colors[i], linestyle="--")

        plt.gcf().set_size_inches(8, 6)
        plt.legend()
        plt.xlabel("Ground state energy pfu [kJ/mol]") 
        plt.xlim(energy_range_dct[mat])
        plt.ylim(0, 6000)
        plt.title(mat + '_' + phase)
        plt.savefig(Path_mat / "Plots" / "opt" / str(phase + '.pdf'), bbox_inches='tight')
        plt.clf()

        for i, key in enumerate(hist_dct[mat][phase]):
            if key in hist_dct_sup[mat][phase]:
                plt.bar(bin_dct_sup[mat][phase][key], hist_dct_sup[mat][phase][key], width=bin_size, color=colors[i], alpha=0.5, label=str(key), bottom=offset_dct[mat][key])
                plt.axhline(offset_dct[mat][key], color=colors[i], linestyle=":")
                plt.axvline(min_atoms_dct_sup[mat][phase][key].get_potential_energy()*molmod.units.electronvolt/(64*molmod.units.kjmol), color=colors[i], linestyle="--")

        plt.gcf().set_size_inches(8, 6)
        plt.legend()
        plt.xlabel("Ground state energy pfu [kJ/mol]") 
        plt.xlim(energy_range_dct[mat])
        plt.ylim(0, 6000)
        plt.title(mat + '_' + phase)
        plt.savefig(Path_mat / "Plots" / "opt" / str(phase + '_sup.pdf'), bbox_inches='tight')
        plt.clf()