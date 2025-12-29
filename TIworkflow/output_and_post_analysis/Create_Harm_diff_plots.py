from lib.Free_energy_classes import Free_energy_contribution
import matplotlib.pyplot as plt
import numpy as np
import molmod.units
from pathlib import Path

def get_closest_temp(T_1, T_ref):
    T_0 = T_ref[0]
    for T_val in T_ref:
        if T_1 > 0.0:
            if np.abs(np.log(T_val/T_1)) < np.abs(np.log(T_0/T_1)):
                T_0 = T_val
        elif T_1 == 0.0:
            if np.abs(T_val) < np.abs(T_0):
                T_0 = T_val
    return T_0

tem_min = 150.0
tem_max = 600.0
num_tem = 32
tem_np = tem_min*np.exp(np.linspace(0.0, np.log(tem_max/tem_min), num_tem))
tem_np = np.insert(tem_np, 0, 0)

path_cwd = Path.cwd()

for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:

    Path_mat = path_cwd / mat
    f_dct = {}
    ref_phase = 'gamma'
    sim_dct = {
        'Harm': 'NVT_opt',
        'Harm_sup': 'NVT_opt_sup_1'
    }

    for ens, sim in sim_dct.items():
        if ens[-3:] == 'sup':
            sim_NVE = 'NVE_MD_sup'
            if mat == 'CsPbI3':
                T_lst = [150, 350, 600]
            else:
                T_lst = [234, 350, 501]
        else:
            T_lst = [150, 187, 234, 293, 350, 419, 501, 600]
            sim_NVE = 'NVE_MD'

        NPT_min = {}
        low_corr = {}
        for phase in ['gamma', 'Csdelta', 'FAdelta']:
            Path_phase = Path_mat / phase
            f_dct[phase] = {}
            f_dct[phase]['ave'] = {}
            f_dct[phase]['error'] = {}
            f_dct[phase]['ave']['NVE'] = np.zeros(len(T_lst))
            f_dct[phase]['error']['NVE'] = np.zeros(len(T_lst))

            with open(Path_phase / sim.replace("NVT", "NPT") / "Compare_minima.txt", 'r') as file:
                line = file.readline()
                values = line.split()
                NPT_min[phase] = float(values[7])

            low_corr[phase] = {}
            for j, T_NVT in enumerate(T_lst):
                fec = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT)) / sim / "fec.pickle")
                f_dct[phase]['ave'][str(T_NVT)] = fec.free_energy / (fec.n_fu * molmod.units.kjmol)
                f_dct[phase]['ave'][str(T_NVT)] = np.insert(f_dct[phase]['ave'][str(T_NVT)], 0, fec.GS_energy / (fec.n_fu * molmod.units.kjmol))
                f_dct[phase]['error'][str(T_NVT)] = fec.free_error / (fec.n_fu * molmod.units.kjmol)
                f_dct[phase]['error'][str(T_NVT)] = np.insert(f_dct[phase]['error'][str(T_NVT)], 0, 0.0)
                fec = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT)) / sim_NVE / "fec.pickle")
                f_dct[phase]['ave']['NVE'][j] = fec.free_energy / (fec.n_fu * molmod.units.kjmol)
                f_dct[phase]['error']['NVE'][j] = fec.free_error / (fec.n_fu * molmod.units.kjmol)

                if ens[-3:] == 'sup':
                    fec_low_corr = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT))/ "lmd_low_sup" / "fec.pickle")
                else:
                    fec_low_corr = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT))/ "lmd_low" / "fec.pickle")
                low_corr[phase][str(T_NVT)] = fec_low_corr.free_energy / (fec_low_corr.n_fu * molmod.units.kjmol)

            f_dct[phase]['ave']['combined'] = np.zeros(len(tem_np))
            f_dct[phase]['error']['combined'] = np.zeros(len(tem_np))
            for i, tem in enumerate(tem_np):
                T_close = get_closest_temp(tem, T_lst)
                f_dct[phase]['ave']['combined'][i] = f_dct[phase]['ave'][str(T_close)][i]

        for phase, color in {'Csdelta': 'tab:blue', 'FAdelta': 'tab:orange'}.items():
            df = np.zeros((len(f_dct[phase]['ave']) - 2, len(tem_np)))

            tel = 0
            for label, f in f_dct[phase]['ave'].items():
                diff_f = f - f_dct[ref_phase]['ave'][label]
                diff_fe = np.sqrt(f_dct[phase]['error'][label]**2 + f_dct[ref_phase]['error'][label]**2)
                if label == 'combined':
                    plt.errorbar(tem_np, diff_f, yerr=diff_fe, color=color, label=phase + "_" + label)
                elif label == 'NVE':
                    plt.errorbar(T_lst, diff_f, yerr=f_dct[phase]['error'][label], linestyle = ":", color=color, label=phase + "_" + label)
                else:
                    df[tel, :] = diff_f
                    df_corr = low_corr[phase][label] - low_corr[ref_phase][label]
                    if label == str(T_lst[0]):
                        alpha = 1.0
                    else:
                        alpha = 0.2
                    plt.arrow(150, diff_f[1], 0, df_corr-diff_f[1], head_width=5, head_length=0.2, fc=color, ec=color, alpha=alpha)
                    tel += 1
            plt.fill_between(tem_np, np.min(df, axis=0), np.max(df, axis=0), color=color, alpha=0.2)
            plt.plot(0.0, NPT_min[phase] - NPT_min[ref_phase], color=color, marker='o', markersize=5, label=phase + "_NPT")

        plt.legend()
        plt.gcf().set_size_inches(12, 10)
        plt.xlabel("temperature [K]") 
        plt.ylabel("free energy diff ref pfu [kJ/mol]") 
        plt.axhline(0.0, color='k', linestyle="--")
        plt.xlim(0, 600)
        plt.ylim(-14, 6)
        plt.title(mat + '_' + ens)
        plt.savefig( Path_mat / "Plots" / "fec" / str(ens + '.pdf'), bbox_inches='tight')
        plt.clf()