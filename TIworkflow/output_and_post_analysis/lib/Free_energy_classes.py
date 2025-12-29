import molmod.units
import molmod.constants
import numpy as np
import pickle
import h5py
from ase.io import read
import matplotlib.pyplot as plt
from lib.utils import get_frequencies, Running_Average, from_xyz_to_h5, check_validity_inputs
import os

class Free_energy_contribution():
    """super class that sets the structure of the four free energy contributions classes below"""

    def __init__(self, atoms_opt, free_energy = 0.0, free_error = 0.0, temperature = None, pressure = None):
        self.GS_energy = atoms_opt.get_potential_energy() * molmod.units.electronvolt
        self.atoms_opt = atoms_opt
        self.dof = int(len(atoms_opt))*3
        n_fu = 0
        for at in atoms_opt:
            if at.symbol == "Pb":    #This is hardcoded and thus only work for specific MHPs!!!
                n_fu += 1
        self.n_fu = n_fu
        self.free_energy = free_energy
        self.free_error  = free_error
        self.temperature = temperature
        self.pressure = pressure
        if pressure is not None:
            self.GS_enthalpy = atoms_opt.get_potential_energy() * molmod.units.electronvolt + pressure * atoms_opt.get_volume() * molmod.units.angstrom**3
        else:
            self.GS_enthalpy = None
    
    def calculate_prop():
        pass

    def calculate_error_prop():
        pass

    def plot_function():
        pass

    @classmethod
    def from_files():
        pass
    
    @classmethod
    def from_p(cls, filepath):
        """Loads a pickled file and returns a ``Free_energy_contribution`` instance"""
        fec = pickle.load(open(filepath, 'rb'))
        return fec

    def write_pickle_file(self, filepath_out):
        pickle.dump(self, open(filepath_out, 'wb'))


class PhononDensity_Hessian(Free_energy_contribution):
    """Class to represent a phonon density of states"""

    def __init__(self, atoms_opt, frequencies):
        super().__init__(atoms_opt)
        assert len(frequencies.shape) == 1
        self.frequencies = frequencies


    def calculate_prop(self, temperature, pfu = True, quantum = False):
        """Computes the free energy and entropy for the given temperature"""
        
        self.temperature = temperature
        free_energy_tot  = np.zeros(len(temperature))
        entropy_vib      = np.zeros(len(temperature))

        for i,tem in enumerate(temperature):
            # compute entropy for each frequency and integrate
            free_energy_vib_np, entropy_vib_np  = _free_energy_and_entropy(self.frequencies[3:], tem, quantum = quantum)
            free_energy_vib    = np.sum(free_energy_vib_np)
            free_energy_tot[i] = free_energy_vib + self.GS_energy
            entropy_vib[i]     = np.sum(entropy_vib_np)
        
        self.free_energy = free_energy_tot.copy()
        if pfu:
            free_energy_tot /= self.n_fu
            entropy_vib     /= self.n_fu
        return free_energy_tot, entropy_vib

    def plot_function(self, x_np, smearing, pfu = True, label = None):
        y_np = np.zeros(len(x_np))
        for i,x in enumerate(x_np):
            for freq in self.frequencies[3:]:
                y_np[i] += np.exp(- 0.5 * ((x-freq)/smearing)**2) / (np.sqrt(2*np.pi) * smearing)
        if pfu:
            y_np /= self.n_fu
        plt.plot(x_np, y_np, label = label)


    @classmethod
    def from_files(cls, File_atoms, File_hessian):
        assert File_atoms.filepath[-4:] == ".xyz", "file for optimized atoms does not have the right extension"
        assert File_hessian.filepath[-4:] == ".npy", "file for hessian does not have the right extension"
        atoms_opt = read(File_atoms.filepath)
        hes = np.load(File_hessian.filepath)
        frequencies = np.sort(get_frequencies(hes, atoms_opt))
        return cls(atoms_opt, frequencies)



class PhononDensity_vacf(Free_energy_contribution):
    """Class to represent a phonon density of states"""

    def __init__(self, atoms_opt, frequencies, spectrum, temperature, spectra_blocks):
        super().__init__(atoms_opt, temperature = temperature)
        assert len(frequencies.shape) == 1
        assert len(spectrum.shape) == 1
        self.frequencies    = frequencies
        self.spectrum       = spectrum
        self.spectra_blocks = spectra_blocks


    def calculate_prop(self, temperature = None, pfu = True, quantum = False, spectrum = None):
        if temperature == None:
            if self.temperature == None:
                raise ValueError('No temperature assigned')
            else:
                temperature = self.temperature
        if spectrum is None:
            spectrum_in = self.spectrum
        else:
            spectrum_in = spectrum

        # compute free energy and entropy for each frequency and integrate
        delta_f = self.frequencies[1] - self.frequencies[0]
        free_energy_vib_np, entropy_vib_np  = _free_energy_and_entropy(self.frequencies, temperature, quantum= quantum)
        free_energy_vib_spec = delta_f * spectrum_in * free_energy_vib_np
        entropy_vib_spec = delta_f * spectrum_in * entropy_vib_np

        free_energy_tot  = np.sum(free_energy_vib_spec) + self.GS_energy
        entropy_vib      = np.sum(entropy_vib_spec)
        if spectrum is None:
            self.free_energy = free_energy_tot.copy()
        if pfu:
            free_energy_tot /= self.n_fu
            entropy_vib /= self.n_fu
        return free_energy_tot, entropy_vib
    
    def calculate_error_prop(self, pfu = True):
        num_spectra = len(self.spectra_blocks)
        free_energy_tot = np.zeros(num_spectra)
        for i, spectrum in enumerate(self.spectra_blocks):
            free_energy_tot[i], _ = self.calculate_prop(pfu=False, spectrum = spectrum)
        self.free_error = np.std(free_energy_tot) / np.sqrt(num_spectra)
        if pfu:
            return self.free_error/ self.n_fu
        else:
            return self.free_error

    def plot_function(self, pfu = True, label = '', spectrum = None):
        if spectrum == None:
            spectrum = self.spectrum
        if pfu:
            plt.plot(self.frequencies, spectrum/self.n_fu, label = label + "_T" + str(self.temperature))
        else:
            plt.plot(self.frequencies, spectrum, label + "_T" + str(self.temperature))


    @classmethod
    def from_files(cls, File_atoms, output_files, temperature = None, delta_f=None, bsize = None):
        """Computes the phonon density from a trajectory of velocities

        By default, the FFTs are computed such that the frequency resolution
        (as specified by the delta_f argument) is at least 0.1 invcm. If
        delta_f and bsize is None, then a single FFT is used over the entire trajectory.
        """
        assert File_atoms.filepath[-4:] == ".xyz", "file for optimized atoms does not have the right extension"
        atoms_opt = read(File_atoms.filepath)
        assert len(output_files) == 1, "only one setting is possible for this contribution"
        #files_lst = output_files[0]         #Output files are a list of lists, for vacf len(output_files) = 1
        files_lst = []
        for j in range(len(output_files[0])):
            if check_validity_inputs([output_files[0][j]]):
                files_lst.append(output_files[0][j])
            else:
                print("removing NVE MD run " + str(j) + " because the simulation was failed.")

        spectra = None
        spectra_blocks =[]
        for File_h5 in files_lst:
            assert File_h5.filepath[-3:] == ".h5", "file for trajectory does not have the right extension"
            
            with h5py.File(File_h5.filepath, 'r') as f:
                time_signal = np.array(list(f['trajectory']['vel'])) #time_signal[time, atom, component]
                time = np.array(list(f['trajectory']['time']))
                masses = np.array(list(f['system']['masses']))

            sampling_period = time[1] - time[0]
            assert np.shape(time)[0] == len(time_signal)
            assert len(masses) == np.shape(time_signal)[1]

            dof = np.shape(time_signal)[1] * np.shape(time_signal)[2]

            tot_mass =  np.sum(masses)
            for i in range(len(time)):
                for j in range(3): 
                    time_signal[i,:,j] -= np.dot(masses,time_signal[i,:,j])/tot_mass #remove center of mass velocity

            if delta_f is not None:
                # determine block size
                size = _determine_blocksize(delta_f, sampling_period)
                if bsize is not None:
                    assert bsize == size
            elif bsize is not None:
                size = bsize
            else:
                # use entire signal as single block
                size = time_signal.shape[0]
            n_blocks = time_signal.shape[0] // size
            assert n_blocks >= 1

            # iterate over all blocks and compute spectrum
            for i in range(n_blocks):
                start = i * size
                end = start + size
                frequencies, block_spectrum = _compute_block(
                        time_signal[start:end, :],
                        sampling_period,
                        masses
                        )
                area_spectrum = np.trapz(block_spectrum, frequencies)
                block_spectrum *= (dof-3) / area_spectrum
                spectra_blocks.append(block_spectrum.copy())
                if spectra is None:
                    spectra = block_spectrum.copy()
                else:
                    spectra += block_spectrum

        #Norm the spectrum to the total number of degrees of freedom per formula unit
        area_spectrum = np.trapz(spectra, frequencies)
        spectrum = spectra * (dof-3) / area_spectrum
            
        return cls(atoms_opt, frequencies, spectrum, temperature, spectra_blocks)
    

class lambda_integration(Free_energy_contribution):
    """Class to collect the data needed to perform thermodynamic integration along a lambda-path"""

    def __init__(self, absci_dct, atoms_opt, temperature):
        super().__init__(atoms_opt, temperature = temperature)
        self.absci_dct = absci_dct

    def calculate_prop(self, A_beg, pfu = True):
        free_energy_tot = A_beg
        for y_dct in self.absci_dct.values():
            free_energy_tot += y_dct["weight"] * y_dct["delta_f"]
        self.free_energy = free_energy_tot
        if pfu:
            free_energy_tot /= self.n_fu
        return free_energy_tot

    def calculate_error_prop(self, A_beg_error = 0.0, pfu = True):
        free_error2 = A_beg_error ** 2
        for y_dct in self.absci_dct.values():
            free_error2 += (y_dct["weight"] * y_dct["delta_f_error"])**2
        self.free_error = np.sqrt(free_error2)
        if pfu:
            return self.free_error/ self.n_fu
        else:
            return self.free_error

    def plot_function(self, unit_kjmol = True, pfu = True, label = None):
        mult = 1
        unit_str = " [a.u.]"
        if unit_kjmol:
            mult /= molmod.units.kjmol
            unit_str = " [kjmol]"    
        if pfu:
            mult /= self.n_fu
            unit_str += " pfu" 
        x_np = [x for x in self.absci_dct if isinstance(x, float)]
        y_np = [y_dct["delta_f"] *mult for x, y_dct in self.absci_dct.items() if isinstance(x, float)] 
        e_np = [y_dct["delta_f_error"] *mult for x, y_dct in self.absci_dct.items() if isinstance(x, float)]
        plt.ylabel("E_end - E_beg" + unit_str)
        plt.errorbar(x_np, y_np, yerr = e_np, label = label + "_T" + str(self.temperature))

    @classmethod
    def from_files(cls, File_atoms, Files_h5, lmd_np, temperature, num_part = 1, calib = 0):
        
        assert File_atoms.filepath[-4:] == ".xyz", "file for optimized atoms does not have the right extension"
        atoms_opt = read(File_atoms.filepath)
        assert len(Files_h5) == len(lmd_np), "number of h5 files and number of lmd values do not match"
        
        lmd_ind_dct = {}
        new_lmd_lst = []
        for i, lmd in enumerate(lmd_np):
            ind_lst = []
            for j in range(len(Files_h5[i])):
                if check_validity_inputs([Files_h5[i][j]]):
                    ind_lst.append(j)
                else:
                    print("for lmd value " + str(lmd) + " removing MD run " + str(j) + " because the simulation was failed.")
            if len(ind_lst) > 0:
                lmd_ind_dct[lmd] = ind_lst
                new_lmd_lst.append(i)
            else:
                print("WARNING: All MD runs at lambda value " + str(lmd) + " lead to explosions for the MLP, removing this lambda value form discritized sum")
                print("WARNING: You have three options, 1. you try to run a new MD")
                print("WARNING: If multiple runs fail the MLP is just not trained on the region of the hessian at this temperature, so")
                print("WARNING: 2. you could add more training structures, preferably of the hessain PES at this or higher temperatures")
                print("WARNING: 3. You perform this lambda thermodynamic integration step at a lower temperature")

        assert len(new_lmd_lst) > 0, "All MDs are failed, we cannot calculate anything"
        absci_dct = {}
        for i, (weight, lmd) in zip(new_lmd_lst, get_weights(list(lmd_ind_dct.keys())).values()):
            num_runs = len(lmd_ind_dct[lmd]) * num_part
            absci_dct[lmd] = {"weight": weight, "delta_f_runs": np.zeros(num_runs)}
            for j_nr, j in enumerate(lmd_ind_dct[lmd]):
                Filepath = str(Files_h5[i][j].filepath)
                if Filepath[-4:] == ".xyz":
                    Filepath_h5 = str(Filepath)[:-4]+".h5"
                    from_xyz_to_h5(Filepath, Filepath_h5)
                else:
                    Filepath_h5 = Filepath
                assert Filepath_h5[-3:] == ".h5", "file for h5 trajectory does not have the right extension"
                with h5py.File(Filepath_h5, 'r') as f:
                    energys = np.array(f['trajectory']['epot'])[calib:] 
                step = int(np.floor(len(energys[:])/num_part))
                for run in range(num_part):
                    absci_dct[lmd]["delta_f_runs"][run + j_nr*num_part] = np.average(energys[(step*run):(step*(run+1))])

            absci_dct[lmd]["delta_f"]       = np.average(absci_dct[lmd]["delta_f_runs"])
            absci_dct[lmd]["delta_f_error"] = np.std(absci_dct[lmd]["delta_f_runs"]) /np.sqrt(num_runs)

        return cls(absci_dct, atoms_opt, temperature)



class thermo_integration(Free_energy_contribution):
    """Class to collect the data needed to perform thermodynamic integration along a temperature path"""

    def __init__(self, tem_dct, atoms_opt, temperature, pressure):
        super().__init__(atoms_opt, temperature = temperature, pressure = pressure)
        self.tem_dct = tem_dct
        

    def calculate_prop(self, T_ref, A_ref, pfu = True):

        self.free_energy = np.zeros(len(self.temperature))
        for i in range(len(self.temperature)-1):
            assert self.temperature[i+1] > self.temperature[i], "temperatures should be in ascending order"

        for i, T_1 in enumerate(self.temperature):
            T_0, A_beg = get_closest_temp(T_1, T_ref, A_ref)

            self.free_energy[i] = (T_1/T_0)* A_beg 
            self.free_energy[i] -= T_1 * (self.dof - 3) * np.log(T_1/T_0) /2 # forget a factor of k_B, does not matter as it is a constant term
            
            if T_0 != T_1: 
                min_tem = min(T_0, T_1)
                max_tem = max(T_0, T_1)
                tem_int = self.temperature[np.logical_and(self.temperature >= min_tem, self.temperature <= max_tem)]
                if min_tem != tem_int[0]:
                    tem_int = np.insert(tem_int, 0, min_tem)
                if max_tem != tem_int[-1]:
                    tem_int = np.append(tem_int, max_tem)
                if T_1 < T_0:
                    tem_int = tem_int[::-1]

                y_1 = 0.0
                for tem in tem_int[1:]:
                    y_2 = np.log(tem/T_0)
                    if self.pressure is None:
                        self.free_energy[i] -=  T_1 * self.tem_dct[tem]["ave_energy"] * (y_2 - y_1) / tem
                    else:
                        self.free_energy[i] -=  T_1 * self.tem_dct[tem]["ave_enthalpy"] * (y_2 - y_1) / tem
                    y_1 = y_2
        if pfu:
            return self.free_energy/ self.n_fu
        else:
            return self.free_energy


    def calculate_error_prop(self, T_ref, A_ref_error = 0.0, pfu = True):

        self.free_error = np.zeros(len(self.temperature))
        for i in range(len(self.temperature)-1):
            assert self.temperature[i+1] > self.temperature[i], "temperatures should be in ascending order"

        for i, T_1 in enumerate(self.temperature):
            T_0, A_beg_error = get_closest_temp(T_1, T_ref, A_ref_error)

            error2 = ((T_1/T_0)* A_beg_error )**2
            
            if T_0 != T_1: 
                min_tem = min(T_0, T_1)
                max_tem = max(T_0, T_1)
                tem_int = self.temperature[np.logical_and(self.temperature >= min_tem, self.temperature <= max_tem)]
                if min_tem != tem_int[0]:
                    tem_int = np.insert(tem_int, 0, min_tem)
                if max_tem != tem_int[-1]:
                    tem_int = np.append(tem_int, max_tem)
                if T_1 < T_0:
                    tem_int = tem_int[::-1]

                y_1 = 0.0
                for tem in tem_int[1:]:
                    y_2 = np.log(tem/T_0)
                    if self.pressure is None:
                        error2 +=  (T_1 * self.tem_dct[tem]["ave_energy_error"] * (y_2 - y_1) / (2*tem))**2
                    else:
                        error2 +=  (T_1 * self.tem_dct[tem]["ave_enthalpy_error"] * (y_2 - y_1) / (2*tem))**2
                    y_1 = y_2

            self.free_error[i] = np.sqrt(error2)
        if pfu:
            return self.free_error/ self.n_fu
        else:
            return self.free_error
        

    def plot_function(self, unit_kjmol= True, pfu = True, label = None):
        kb = molmod.constants.boltzmann
        mult = 1
        unit_str = " [a.u.]"
        if unit_kjmol:
            mult /= molmod.units.kjmol
            unit_str = " [kjmol]"    
        if pfu:
            mult /= self.n_fu
            unit_str += " pfu" 
        x_np = np.array(list(self.tem_dct.keys()))
        if self.pressure is None:
            anharm_energy = np.array([fdct["ave_energy"] for fdct in self.tem_dct.values()]) - self.GS_energy - (self.dof -3) * kb * x_np / 2
            y_np = anharm_energy * mult 
            e_np = np.array([fdct["ave_energy_error"] for fdct in self.tem_dct.values()]) * mult
            plt.ylabel("average_anharmonic_energy"+unit_str) 
        else:
            anharm_enthalpy = np.array([fdct["ave_enthalpy"] for fdct in self.tem_dct.values()]) - self.GS_enthalpy- (self.dof -3) * kb * x_np / 2
            y_np = anharm_enthalpy * mult
            e_np = np.array([fdct["ave_enthalpy_error"] for fdct in self.tem_dct.values()]) * mult
            plt.ylabel("average_anharmonic_enthalpy"+unit_str) 
        plt.errorbar(x_np, y_np, yerr = e_np, label = label )


    @classmethod
    def from_files(cls, File_atoms, Files_h5, temperature, num_part = 1, pressure = None, calib = 0):
        
        assert File_atoms.filepath[-4:] == ".xyz", "file for optimized atoms does not have the right extension"
        atoms_opt = read(File_atoms.filepath)
        assert len(Files_h5) == len(temperature), "number of h5 files and number of temperature values do not match"

        ind_lst = []
        for j in range(len(Files_h5[0])):
            files_inp = []
            for i,tem in enumerate(temperature):
                files_inp.append(Files_h5[i][j])
            if check_validity_inputs(files_inp):
                ind_lst.append(j)
            else:
                print("removing REX run " + str(j) + " because the simulation was failed.")
        assert len(ind_lst) > 0, "All REX simulations failed, redo MD possible retrain MLP or change settings MD"

        tem_dct = {}
        num_runs = len(ind_lst) * num_part  
        for i,tem in enumerate(temperature):
            tem_dct[tem] = {"ave_energy_runs": np.zeros(num_runs)}
            if pressure is not None:
                tem_dct[tem]["ave_enthalpy_runs"] = np.zeros(num_runs)
            for j_nr, j in enumerate(ind_lst):
                Filepath = str(Files_h5[i][j].filepath)
                if Filepath[-4:] == ".xyz":
                    Filepath_h5 = str(Filepath)[:-4]+".h5"
                    from_xyz_to_h5(Filepath, Filepath_h5)
                else:
                    Filepath_h5 = Filepath
                assert Filepath_h5[-3:] == ".h5", "file for h5 trajectory does not have the right extension"
                with h5py.File(Filepath_h5, 'r') as f:    
                    energys = np.array(f['trajectory']['epot'])[calib:] 
                    volumes = np.array(f['trajectory']['volume'])[calib:] 
                step = int(np.floor(len(energys[:])/num_part))
                for run in range(num_part):
                    ave_en = np.average(energys[(step*run):(step*(run+1))])
                    tem_dct[tem]["ave_energy_runs"][run + j_nr*num_part] = ave_en
                    if pressure is not None:
                        ave_vol = np.average(volumes[(step*run):(step*(run+1))])
                        tem_dct[tem]["ave_enthalpy_runs"][run + j_nr*num_part] = ave_en + pressure * ave_vol

            tem_dct[tem]["ave_energy"] = np.average(tem_dct[tem]["ave_energy_runs"])
            tem_dct[tem]["ave_energy_error"] = np.std(tem_dct[tem]["ave_energy_runs"]) /np.sqrt(num_runs)
            if pressure is not None:
                tem_dct[tem]["ave_enthalpy"] = np.average(tem_dct[tem]["ave_enthalpy_runs"])
                tem_dct[tem]["ave_enthalpy_error"] = np.std(tem_dct[tem]["ave_enthalpy_runs"]) /np.sqrt(num_runs)

        return cls(tem_dct, atoms_opt, temperature, pressure)
    


class NPT_distrubtion(Free_energy_contribution):
    """Class to collect the data needed to perform thermodynamic integration along a lambda-path"""

    def __init__(self, volumes, volumes_blocks, atoms_opt, temperature, pressure):
        super().__init__(atoms_opt, temperature = temperature, pressure = pressure)
        self.volumes        = volumes
        self.volumes_blocks = volumes_blocks


    def calculate_prop(self, A_beg = 0.0, temperature = None, pressure = None, bin_size_pfu = 0.1*molmod.units.angstrom**3, window = 9, pfu = True, volumes = None):
        k = molmod.constants.boltzmann
        if temperature == None:
            if self.temperature == None:
                raise ValueError('No temperature assigned')
            else:
                temperature = self.temperature
        if pressure == None:
            if self.pressure == None:
                raise ValueError('No pressure assigned')
            else:
                pressure = self.pressure
        if volumes is None:
            volumes_in = self.volumes
        else:
            volumes_in = volumes
        bin_size = bin_size_pfu * self.n_fu
        hist, bin = calculate_hist(volumes_in, bin_size)
        center_bin= Running_Average(bin, 2)
        run_hist  = Running_Average(hist, window)
        run_bin   = Running_Average(center_bin, window)
        vol = self.atoms_opt.get_volume() * molmod.units.angstrom
        min_diff = np.abs(vol - run_bin[0])
        hist_opt = run_hist[0]
        for hist_val, bin_val in zip(run_hist, run_bin):
            diff_vol = np.abs(vol - bin_val)
            if diff_vol <= min_diff:
                hist_opt = hist_val
                min_diff = diff_vol
        free_energy_tot = A_beg + pressure * vol + k*temperature * np.log(hist_opt)
        if volumes is None:
            self.free_energy = free_energy_tot.copy()
        if pfu:
            free_energy_tot /= self.n_fu
        return free_energy_tot
    

    def calculate_error_prop(self, A_beg_error = 0.0, temperature = None, pressure = None, bin_size_pfu = 0.1*molmod.units.angstrom**3, window = 9, pfu = True):
        num_volumes = len(self.volumes_blocks)
        free_energy_tot = np.zeros(num_volumes)
        for i, volumes in enumerate(self.volumes_blocks):
            free_energy_tot[i] = self.calculate_prop(temperature = temperature, pressure = pressure, bin_size_pfu = bin_size_pfu, window = window, pfu=False, volumes = volumes)
        free_error_dist = np.std(free_energy_tot) / np.sqrt(num_volumes)
        self.free_error = np.sqrt(free_error_dist**2 + A_beg_error**2)
        if pfu:
            return self.free_error/ self.n_fu
        else:
            return self.free_error


    def plot_function(self, unit_ang= True, bin_size_pfu = 0.1*molmod.units.angstrom**3, window = 9, pfu = True, label = None):
        unit_str = " [a.u.]"
        volumes = self.volumes   # I should make a .copy(), current volumes in fec files are normalized!
        opt_vol = self.atoms_opt.get_volume() * molmod.units.angstrom**3
        bin_size = bin_size_pfu * self.n_fu
        if unit_ang:  
            volumes /= molmod.units.angstrom**3
            opt_vol /= molmod.units.angstrom**3
            bin_size/= molmod.units.angstrom**3
            unit_str = " [kjmol]"  
        if pfu:
            volumes /= self.n_fu
            opt_vol /= self.n_fu
            bin_size/= self.n_fu
            unit_str += " pfu" 
        hist, bin = calculate_hist(volumes, bin_size)
        center_bin= Running_Average(bin, 2)
        run_hist  = Running_Average(hist, window)
        run_bin   = Running_Average(center_bin, window)
        plt.plot(center_bin, hist, label = label + "_T" + str(self.temperature))
        plt.plot(run_bin, run_hist, label = label + "_T" + str(self.temperature) + "_runave_" + str(window))
        plt.axvline(opt_vol, color='tab:red', linestyle="--")


    @classmethod
    def from_files(cls, File_atoms, Files_h5, temperature = None, pressure = None, num_part = 1, calib = 0):
        
        assert File_atoms.filepath[-4:] == ".xyz", "file for optimized atoms does not have the right extension"
        atoms_opt = read(File_atoms.filepath)
        assert len(Files_h5) == 1, "only one setting is possible for this contribution"
        #files_lst = Files_h5[0]
        files_lst = []
        for j in range(len(Files_h5[0])):
            if check_validity_inputs([Files_h5[0][j]]):
                files_lst.append(Files_h5[0][j])
            else:
                print("removing NPT MD run " + str(j) + " because the simulation was failed.")
        
        volumes = np.array([])
        volumes_blocks = []
        for file1 in files_lst:
            Filepath = str(file1.filepath)
            if Filepath[-4:] == ".xyz":
                Filepath_h5 = str(Filepath)[:-4]+".h5"
                from_xyz_to_h5(Filepath, Filepath_h5)
            else:
                Filepath_h5 = Filepath
            assert Filepath_h5[-3:] == ".h5", "file for h5 trajectory does not have the right extension"
            with h5py.File(Filepath_h5, 'r') as f:
                vol_arr  = np.array(list(f['trajectory']['volume']))[calib:]
            step = int(np.floor(len(vol_arr)/num_part))
            for run in range(num_part):
                volumes_blocks.append(vol_arr[(step*run):(step*(run+1))])
            volumes = np.concatenate((volumes, vol_arr))

        return cls(volumes, volumes_blocks, atoms_opt, temperature, pressure)



def _free_energy_and_entropy(f, T, quantum = True):
    if (f > 0).all():
        h = molmod.constants.planck
        k = molmod.constants.boltzmann
        beta = 1 / (molmod.constants.boltzmann * T)
        if quantum:
            q_quantum = np.exp(- (beta * h * f) / 2) / (1 - np.exp(- beta * h * f))
            f_quantum = - np.log(q_quantum) / beta
            s_quantum = -k * np.log(1 - np.exp(- beta * h * f)) + h * f / T * (np.exp(beta * h * f) - 1) ** (-1)    #Check this formula
            return f_quantum, s_quantum
        else:
            q_classical = 1 / (beta * h * f)
            f_classical = - np.log(q_classical) / beta
            s_classical = k * (1 + np.log(k * T / (h * f)))    #Check this formula
            return f_classical, s_classical
    else:
        raise ValueError('Entropy at 0Hz is infinite')


def _compute_block(time_signal, sampling_period, masses):
    """Computes the spectrum of the autocorrelation of a time signal

    This function computes the power spectrum of a time signal by computing
    the norm-squared DFT spectrum of the time signal using the FFT algorithm.

    Arguments
    ---------

    time_signal (ndarray of shape (nsteps, natoms, 3)):
        trajectory of samples of positions, velocities etc, in atomic units

    sampling_period (float):
        elapsed time between each of the samples, in atomic units.
    """
    nsteps = time_signal.shape[0]
    _all = np.reshape(time_signal, (nsteps, -1))
    N = 2 * nsteps - 1 # avoid periodic copies influencing the result
    out = np.fft.fft(_all, n=N, axis=0)
    power_spectrum = np.abs(out) ** 2
    
    masses_c = np.array([masses,masses,masses]).T
    masses_c_1 = np.reshape(masses_c, -1)
    
    spectrum = np.zeros(N)
    for freq in range(N):
        spectrum[freq] = np.dot(masses_c_1,power_spectrum[freq,:])

    # frequency axis of N-point DFT
    frequencies = 1 / N * np.arange(N) * (1 / sampling_period)

    # spectrum is even
    n = N // 2

    #print("removed spectrum(freq = 0): " + str(spectrum[0]))
    #print("Which is {:f} of the variance of the spectrum".format(spectrum[0]/np.var(spectrum)))
    return frequencies[1:n], spectrum[1:n]


def _determine_blocksize(delta_f, sampling_period):
    """Partitions the time_signal into blocks to fix the frequency resolution

    The return value is an integer representing the required block size, as
    obtained through the relation:

        1 / sampling_period = (2n - 1) * delta_f

    Arguments
    ---------

    delta_f (float):
        desired frequency resolution, in atomic units.

    sampling_period (float):
        total time inbetween two consecutive samples, in atomic units

    """
    return int(np.ceil((1 / (sampling_period * delta_f) + 1) / 2))


def get_weights(lmd_lst):
    m0_to_1= {}
    for i, absci in enumerate(lmd_lst):
        if i == 0:
            weight1 = lmd_lst[i] - 0.0
            weight2 = (lmd_lst[i+1]-lmd_lst[i])/2
            m0_to_1["begin_point"] = (weight1 + weight2, absci)
        elif i < len(lmd_lst)-1:
            weight1 = weight2
            weight2 = (lmd_lst[i+1]-lmd_lst[i])/2
            m0_to_1["point_"+str(i)] = (weight1 + weight2, absci)
        else:
            weight1 = weight2
            weight2 = 1.0 - lmd_lst[i]
            m0_to_1["end_point"] = (weight1 + weight2, absci)
    return m0_to_1


def get_closest_temp(T_1, T_ref, A_ref):
    if np.isscalar(T_ref) or len(T_ref) == 1:
        assert  np.isscalar(A_ref) or len(A_ref) == 1, "A_ref should also be a scalar like T_ref"
        T_0 = T_ref
        A_beg = A_ref
    else:
        assert len(T_ref) == len(A_ref), "A_beg[i] is the free energy at temperature T_0[i] in the reference free energy calculation"
        T_0 = T_ref[0]
        A_beg = A_ref[0]
        for T_val, A_val in zip(T_ref, A_ref):
            if np.abs(np.log(T_val/T_1)) < np.abs(np.log(T_0/T_1)):
                T_0 = T_val
                A_beg = A_val
    return T_0, A_beg


def calculate_hist(volumes, bin_size, density=True):
        vol_min = np.floor(np.min(volumes)/bin_size)*bin_size
        vol_max = np.ceil(np.max(volumes)/bin_size)*bin_size
        bin = np.arange(vol_min, vol_max, bin_size)
        return np.histogram(volumes, bins= bin, density=density)