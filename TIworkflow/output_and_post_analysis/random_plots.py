from lib.Free_energy_classes import Free_energy_contribution
import matplotlib.pyplot as plt
import numpy as np
import molmod.units
from pathlib import Path

mat = 'MAPbI3'
NVT_T = 350
path_cwd = Path.cwd()
freq_max = 2.0 * 10**12 / molmod.units.second
smear_freq = 0.02 * 10**12 / molmod.units.second

def plot_function(fec, x_np, smearing, pfu = True, label = None):
        y_np = np.zeros(len(x_np))
        for i,x in enumerate(x_np):
            for freq in fec.frequencies[3:]:
                y_np[i] += np.exp(- 0.5 * ((x-freq)/smearing)**2) / (np.sqrt(2*np.pi) * smearing)
        if pfu:
            y_np /= fec.n_fu
        plt.plot(x_np * molmod.units.second / 10**12 , y_np, label = label)


for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:
    Path_mat = path_cwd / mat
    for phase in ['gamma', 'Csdelta', 'FAdelta']:
        Path_phase = Path_mat / phase
        for nfu in [8, 64]:
            Path_sim = Path_phase / str("NVT_T" + str(NVT_T)) / str("NVT_opt" + ("_sup_1" if nfu == 64 else ""))
            harm_fec = Free_energy_contribution.from_p(Path_sim / "fec.pickle")
            plot_function(harm_fec, np.arange(0, freq_max, smear_freq/2), smear_freq, pfu = True, label = str(nfu) + ' formula units')
        plt.legend()
        #plt.gcf().set_size_inches(12, 10)
        plt.xlabel("frequency [Thz]") 
        plt.ylabel("phonon spectra") 
        plt.xlim(0, freq_max * molmod.units.second / 10**12)
        plt.ylim(0, 800000)
        plt.title(mat + '_' + phase + '_' + str(NVT_T))
        plt.savefig(Path_mat / 'Plots' / 'opt' / str('freq_' + phase + '.pdf'), bbox_inches='tight')
        plt.clf()