from lib.Free_energy_classes import Free_energy_contribution
import matplotlib.pyplot as plt
import numpy as np
import molmod.units
from pathlib import Path

def get_closest_temp(T_1, T_ref):
    T_0 = T_ref[0]
    for T_val in T_ref:
        if np.abs(np.log(T_val/T_1)) < np.abs(np.log(T_0/T_1)):
            T_0 = T_val
    return T_0

tem_min = 150.0
tem_max = 600.0
num_tem = 32
tem_np = tem_min*np.exp(np.linspace(0.0, np.log(tem_max/tem_min), num_tem))

path_cwd = Path.cwd()

for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:

    Path_mat = path_cwd / mat
    f_dct = {}
    ref_phase = 'gamma'
    sim_dct = {
        'Gibbs': 'REX_NPT',
        'Helm': 'REX_NVT',
        'int': 'lmd_REX',
        'Harm': 'NVT_opt',
        'Gibbs_sup': 'REX_NPT_sup',
        'Helm_sup': 'REX_NVT_sup',
        'int_sup': 'lmd_REX_sup',
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

        for phase in ['gamma', 'Csdelta', 'FAdelta']:
            Path_phase = Path_mat / phase
            f_dct[phase] = {}
            f_dct[phase]['ave'] = {}
            f_dct[phase]['error'] = {}

            if ens[:4] == 'Harm':
                f_dct[phase]['ave']['NVE'] = np.zeros(len(T_lst))
                f_dct[phase]['error']['NVE'] = np.zeros(len(T_lst))

            for j, T_NVT in enumerate(T_lst):
                if ens[:5] == 'Gibbs':
                    fec = Free_energy_contribution.from_p(Path_phase / sim / str("fec_ref_T" + str(T_NVT) +".pickle"))
                else:
                    fec = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT)) / sim / "fec.pickle")
                f_dct[phase]['ave'][str(T_NVT)] = fec.free_energy / (fec.n_fu * molmod.units.kjmol)
                f_dct[phase]['error'][str(T_NVT)] = fec.free_error / (fec.n_fu * molmod.units.kjmol)
                if ens[:4] == 'Harm':
                    fec = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT)) / sim_NVE / "fec.pickle")
                    f_dct[phase]['ave']['NVE'][j] = fec.free_energy / (fec.n_fu * molmod.units.kjmol)
                    f_dct[phase]['error']['NVE'][j] = fec.free_error / (fec.n_fu * molmod.units.kjmol)

            f_dct[phase]['ave']['combined'] = np.zeros(len(tem_np))
            f_dct[phase]['error']['combined'] = np.zeros(len(tem_np))
            for i, tem in enumerate(tem_np):
                T_close = get_closest_temp(tem, T_lst)
                f_dct[phase]['ave']['combined'][i] = f_dct[phase]['ave'][str(T_close)][i]
                if ens[:4] != 'Harm':
                    f_dct[phase]['error']['combined'][i] = f_dct[phase]['error'][str(T_close)][i]

        for phase, color in {'Csdelta': 'tab:blue', 'FAdelta': 'tab:orange'}.items():
            df = np.zeros((len(f_dct[phase]['ave']) - 1, len(tem_np)))
            if ens[:4] == 'Harm':
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
                    tel += 1
            plt.fill_between(tem_np, np.min(df, axis=0), np.max(df, axis=0), color=color, alpha=0.2)
            
        plt.legend()
        plt.xlabel("temperature [K]") 
        plt.ylabel("free energy diff ref pfu [kJ/mol]") 
        plt.axhline(0.0, color='k', linestyle="--")
        plt.xlim(150, 600)
        if mat == 'CsPbI3':
            plt.ylim(-12, 4)
        else:
            plt.ylim(-10, 6)
        plt.title(mat + '_' + ens)
        plt.savefig( Path_mat / "Plots" / "fec" / str(ens + '.pdf'), bbox_inches='tight')
        plt.clf()