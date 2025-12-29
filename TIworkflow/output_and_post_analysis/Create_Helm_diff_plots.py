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
        'Helm': 'REX_NVT',
        'Helm_sup': 'REX_NVT_sup',
    }

    for ens, sim in sim_dct.items():
        if ens[-3:] == 'sup':
            if mat == 'CsPbI3':
                T_lst = [150, 350, 600]
            else:
                T_lst = [234, 350, 501]
        else:
            T_lst = [150, 187, 234, 293, 350, 419, 501, 600]

        npt_corr = {}
        high_corr = {}
        for phase in ['gamma', 'Csdelta', 'FAdelta']:
            Path_phase = Path_mat / phase
            f_dct[phase] = {}
            f_dct[phase]['ave'] = {}
            f_dct[phase]['error'] = {}

            npt_corr[phase] = {}
            high_corr[phase] = {}
            for j, T_NVT in enumerate(T_lst):
                fec = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT)) / sim / "fec.pickle")
                f_dct[phase]['ave'][str(T_NVT)] = fec.free_energy / (fec.n_fu * molmod.units.kjmol)
                f_dct[phase]['error'][str(T_NVT)] = fec.free_error / (fec.n_fu * molmod.units.kjmol)

                if ens[-3:] == 'sup':
                    fec_npt_corr = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT))/ "NPT_MD_sup" / "fec.pickle")
                    fec_high_corr = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT))/ "lmd_high_sup" / "fec.pickle")
                else:
                    fec_npt_corr = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT))/ "NPT_MD" / "fec.pickle")
                    fec_high_corr = Free_energy_contribution.from_p(Path_phase / str("NVT_T" + str(T_NVT))/ "lmd_high" / "fec.pickle")
                npt_corr[phase][str(T_NVT)] = fec_npt_corr.free_energy[0] / (fec_npt_corr.n_fu * molmod.units.kjmol)
                high_corr[phase][str(T_NVT)] = fec_high_corr.calculate_prop(0.0) / molmod.units.kjmol

            f_dct[phase]['ave']['combined'] = np.zeros(len(tem_np))
            f_dct[phase]['error']['combined'] = np.zeros(len(tem_np))
            for i, tem in enumerate(tem_np):
                T_close = get_closest_temp(tem, T_lst)
                f_dct[phase]['ave']['combined'][i] = f_dct[phase]['ave'][str(T_close)][i]
                f_dct[phase]['error']['combined'][i] = f_dct[phase]['error'][str(T_close)][i]

        for phase, color in {'Csdelta': 'tab:blue', 'FAdelta': 'tab:orange'}.items():
            df = np.zeros((len(f_dct[phase]['ave']) - 1, len(tem_np)))

            tel = 0
            for label, f in f_dct[phase]['ave'].items():
                diff_f = f - f_dct[ref_phase]['ave'][label]
                diff_fe = np.sqrt(f_dct[phase]['error'][label]**2 + f_dct[ref_phase]['error'][label]**2)
                if label == 'combined':
                    plt.errorbar(tem_np, diff_f, yerr=diff_fe, color=color, label=phase + "_" + label)
                else:
                    df[tel, :] = diff_f
                    df_nptcorr = npt_corr[phase][label] - npt_corr[ref_phase][label]
                    df_highcorr = high_corr[phase][label] - high_corr[ref_phase][label]
                    if label == str(T_lst[-1]):
                        alpha_h = 1.0
                    else:
                        alpha_h = 0.2
                    plt.arrow(595, diff_f[-1] - df_highcorr, 0, df_highcorr, head_width=5, head_length=0.2, fc=color, ec=color, alpha=alpha_h)
                    npt_ind = np.where(np.floor(tem_np) == int(label))[0][0]
                    plt.arrow(int(label), diff_f[npt_ind], 0, df_nptcorr-diff_f[npt_ind], head_width=5, head_length=0.2, fc=color, ec=color, alpha=1.0)
                    tel += 1
            plt.fill_between(tem_np, np.min(df, axis=0), np.max(df, axis=0), color=color, alpha=0.2)
            
        plt.legend()
        plt.gcf().set_size_inches(9, 8.5)
        plt.xlabel("temperature [K]") 
        plt.ylabel("free energy diff ref pfu [kJ/mol]") 
        plt.axhline(0.0, color='k', linestyle="--")
        plt.xlim(150, 600)
        plt.ylim(-11, 6)
        plt.title(mat + '_' + ens)
        plt.savefig( Path_mat / "Plots" / "fec" / str(ens + '.pdf'), bbox_inches='tight')
        plt.clf()