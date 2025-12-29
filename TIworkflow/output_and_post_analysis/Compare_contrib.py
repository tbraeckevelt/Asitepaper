from lib.Free_energy_classes import Free_energy_contribution
import matplotlib.pyplot as plt
import numpy as np
import molmod.units
from pathlib import Path

tem_min = 150.0
tem_max = 600.0
num_tem = 32
tem_np = tem_min*np.exp(np.linspace(0.0, np.log(tem_max/tem_min), num_tem))

labels = ['gse', 'harm_grad', 'delta_tmin', 'int_grad', 'delta_tmax', 'nvt_grad']
#colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
#colors = ["#0000FF", "#3366FF", "#6699FF", "#66CCFF", "#FFCC66", "#FF9966", "#FF6633", "#FF0000"]
colors_blue = ["#001F3F", "#003F7F", "#005FBF", "#007FFF", "#339FFF", "#66BFFF", "#99DFFF", "#CCEFFF"]
colors_orange = ["#FF4500", "#FF6522", "#FF8544", "#FFA566", "#FFBD80", "#FFD199", "#FFE0B3", "#FFEBD4"]
width = 0.1
T_lst_all = [150, 187, 234, 293, 350, 419, 501, 600]

path_cwd = Path.cwd()

f_dct = {}
for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:
    Path_mat = path_cwd / mat

    for nfu, fnext in {8: '', 64: '_sup'}.items():
        f_dct[mat+fnext] = {}

        if nfu == 64:
            if mat == 'CsPbI3':
                T_lst = [150, 350, 600]
            else:
                T_lst = [234, 350, 501]
        else:
            T_lst = [150, 187, 234, 293, 350, 419, 501, 600]

        phr = 'gamma'
        phr_folder = Path_mat / phr

        for ph in ['Csdelta', 'FAdelta']:
            f_dct[mat+fnext][ph] = {}
            ph_folder = Path_mat / ph

            for j, T_NVT in enumerate(T_lst):
                f_dct[mat+fnext][ph][T_NVT] = {}
                path_nvt = ph_folder / str("NVT_T" + str(T_NVT))
                path_nvt_r = phr_folder / str("NVT_T" + str(T_NVT))
                if nfu == 8:
                    harm_fec = Free_energy_contribution.from_p(path_nvt / 'NVT_opt' / str("fec.pickle"))
                    harm_fec_r = Free_energy_contribution.from_p(path_nvt_r / 'NVT_opt' / str("fec.pickle"))
                else:
                    harm_fec = Free_energy_contribution.from_p(path_nvt / 'NVT_opt_sup_1' / str("fec.pickle"))
                    harm_fec_r = Free_energy_contribution.from_p(path_nvt_r / 'NVT_opt_sup_1' / str("fec.pickle"))

                gse = (harm_fec.GS_energy - harm_fec_r.GS_energy) / (molmod.units.kjmol * nfu)
                f_dct[mat+fnext][ph][T_NVT]['gse'] = gse
                harm_grad = ((harm_fec.free_energy[-1] - harm_fec_r.free_energy[-1]) / (molmod.units.kjmol * nfu) - gse) * (tem_max - tem_min)/ tem_max
                f_dct[mat+fnext][ph][T_NVT]['harm_grad'] = harm_grad

                int_fec = Free_energy_contribution.from_p(path_nvt / str('lmd_REX' + fnext) / str("fec.pickle"))
                int_fec_r = Free_energy_contribution.from_p(path_nvt_r / str('lmd_REX' + fnext) / str("fec.pickle"))
                delta_tmin = (int_fec.free_energy[0]- harm_fec.free_energy[0] - (int_fec_r.free_energy[0] - harm_fec_r.free_energy[0])) / (molmod.units.kjmol * nfu)
                f_dct[mat+fnext][ph][T_NVT]['delta_tmin'] = delta_tmin
                int_f_tmax = (int_fec.free_energy[-1] - int_fec_r.free_energy[-1]) / (molmod.units.kjmol * nfu)
                int_f_tmin = (int_fec.free_energy[0] - int_fec_r.free_energy[0]) / (molmod.units.kjmol * nfu)
                int_grad = int_f_tmax - int_f_tmin
                f_dct[mat+fnext][ph][T_NVT]['int_grad'] = int_grad

                nvt_fec = Free_energy_contribution.from_p(path_nvt / str('REX_NVT' + fnext) / str("fec.pickle"))
                nvt_fec_r = Free_energy_contribution.from_p(path_nvt_r / str('REX_NVT' + fnext) / str("fec.pickle"))
                delta_tmax = (nvt_fec.free_energy[-1] - int_fec.free_energy[-1] - (nvt_fec_r.free_energy[-1] - int_fec_r.free_energy[-1])) / (molmod.units.kjmol * nfu)
                f_dct[mat+fnext][ph][T_NVT]['delta_tmax'] = delta_tmax
                nvt_f_tmax = (nvt_fec.free_energy[-1] - nvt_fec_r.free_energy[-1]) / (molmod.units.kjmol * nfu)
                nvt_f_tmin = (nvt_fec.free_energy[0] - nvt_fec_r.free_energy[0]) / (molmod.units.kjmol * nfu)
                nvt_grad = nvt_f_tmax - nvt_f_tmin
                f_dct[mat+fnext][ph][T_NVT]['nvt_grad'] = nvt_grad

average = 0.0
tel=0
for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:
    print(mat)
    for j, T_NVT in enumerate(T_lst_all):
        if mat != 'MAPbI3' or j != 0:
            print(f_dct[mat]['Csdelta'][T_NVT]['nvt_grad']/4.5)
            average += f_dct[mat]['Csdelta'][T_NVT]['nvt_grad']/4.5
            tel+=1
print(mat, "average", average/tel)

average = 0.0
tel=0
for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:
    print(mat)
    for j, T_NVT in enumerate(T_lst_all):
        if mat != 'MAPbI3' or j != 0:
            print(f_dct[mat]['FAdelta'][T_NVT]['nvt_grad']/4.5)
            average += f_dct[mat]['FAdelta'][T_NVT]['nvt_grad']/4.5
            tel+=1
print(mat, "average", average/tel)

'''
for label in labels:
    for ph in ['Csdelta', 'FAdelta']:
        fig, ax = plt.subplots()
        if ph == 'Csdelta':
            colors = colors_blue
        else:
            colors = colors_orange

        x_pos = width/2.0
        num_tem = []
        for x, key in enumerate(f_dct.keys()):
            if key[-4:] == '_sup':
                if key[:2] == 'Cs':
                    T_lst = [150, 350, 600]
                else:
                    T_lst = [234, 350, 501]
            else:
                T_lst = [150, 187, 234, 293, 350, 419, 501, 600]

            for j, T_NVT in enumerate(T_lst_all):
                if T_NVT in T_lst:
                    if x == 0:
                        ax.bar(x_pos, f_dct[key][ph][T_NVT][label], width, color=colors[-1-j], label=str(T_NVT))
                    else:
                        ax.bar(x_pos, f_dct[key][ph][T_NVT][label], width, color=colors[-1-j])
                    x_pos += width
            x_pos += 4*width
            num_tem.append(len(T_lst))

        ax.set_xlabel('simulation')
        ax.set_ylabel('energy [kJ/mol] pfu')
        if "delta" in label:
            ax.set_ylim(-2, 8)
            plt.gcf().set_size_inches(10, 6)
        elif "grad" in label:
            ax.set_ylim(-3, 12)
            plt.gcf().set_size_inches(10, 9)
        else:
            ax.set_ylim(-15, 0)
            plt.gcf().set_size_inches(10, 9)
        plt.axhline(0.0, color='k', linestyle=":")
        ax.set_title(ph + '_' + label)
        x_ticks = []
        for i, nt in enumerate(num_tem):
            if len(x_ticks) == 0:
                x_ticks.append(nt*width/2)
            else:
                x_ticks.append(x_ticks[-1] + num_tem[i-1]*width/2 + nt*width/2 + 4*width)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(f_dct.keys())
        ax.legend()

        plt.savefig(path_cwd / "contrib_plots" / str(ph + '_' + label+'.pdf'), bbox_inches='tight')
        plt.clf()
'''
for label in labels:
    fig, ax = plt.subplots()

    x_pos = width/2.0
    x_ticks = [width*9, width*31, width*53]

    for x, key in enumerate(f_dct.keys()):
        if key[-4:] != '_sup':

            for ph in ['Csdelta', 'FAdelta']:
                if ph == 'Csdelta':
                    colors = colors_blue
                else:
                    colors = colors_orange

                for j, T_NVT in enumerate(T_lst_all):
                    if x == 0:
                        ax.bar(x_pos, f_dct[key][ph][T_NVT][label], width, color=colors[-1-j], label=str(T_NVT))
                    else:
                        ax.bar(x_pos, f_dct[key][ph][T_NVT][label], width, color=colors[-1-j])
                    x_pos += width
                x_pos += 2*width
            x_pos += 2*width

    ax.set_xlabel('simulation')
    ax.set_ylabel('energy [kJ/mol] pfu')
    if "delta" in label:
        ax.set_ylim(-2, 8)
        plt.gcf().set_size_inches(7.5, 5)
    elif "grad" in label:
        ax.set_ylim(-2, 12)
        plt.gcf().set_size_inches(7.5, 7)
    else:
        ax.set_ylim(-15, 0)
        plt.gcf().set_size_inches(7.5, 7.5)
    plt.axhline(0.0, color='k', linestyle=":")
    ax.set_title(label)
        
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([key for key in f_dct.keys()if key[-4:] != '_sup'])
    ax.legend()

    plt.savefig(path_cwd / "contrib_plots" / str(label+'_8fu.pdf'), bbox_inches='tight')
    plt.clf()


for label in labels:
    fig, ax = plt.subplots()

    x_pos = width/2.0
    x_ticks = [width*4, width*16, width*26]

    for x, key in enumerate(f_dct.keys()):
        if key[-4:] != '_sup':
            if key[:2] == 'Cs':
                T_lst_sup = [150, 350, 600]
            else:
                T_lst_sup = [234, 350, 501]

            for ph in ['Csdelta', 'FAdelta']:
                if ph == 'Csdelta':
                    colors = colors_blue
                else:
                    colors = colors_orange

                for j, T_NVT in enumerate(T_lst_all):
                    if T_NVT in T_lst_sup:
                        if x < 4:
                            ax.bar(x_pos, f_dct[key+'_sup'][ph][T_NVT][label]- f_dct[key][ph][T_NVT][label], width, color=colors[-1-j], label=str(T_NVT))
                        else:
                            ax.bar(x_pos, f_dct[key+'_sup'][ph][T_NVT][label] - f_dct[key][ph][T_NVT][label], width, color=colors[-1-j])
                        x_pos += width
                x_pos += 2*width
            x_pos += 2*width

    ax.set_xlabel('simulation')
    ax.set_ylabel('energy [kJ/mol] pfu')
    if "delta" in label:
        ax.set_ylim(-2, 2)
        plt.gcf().set_size_inches(7.5, 7.5)
    elif "grad" in label:
        ax.set_ylim(-2, 2)
        plt.gcf().set_size_inches(7.5, 7.5)
    else:
        ax.set_ylim(-2, 2)
        plt.gcf().set_size_inches(7.5, 7.5)
    plt.axhline(0.0, color='k', linestyle=":")
    ax.set_title(label)
        
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([key for key in f_dct.keys()if key[-4:] != '_sup'])
    ax.legend()

    plt.savefig(path_cwd / "contrib_plots" / str(label+'_supeffect.pdf'), bbox_inches='tight')
    plt.clf()
