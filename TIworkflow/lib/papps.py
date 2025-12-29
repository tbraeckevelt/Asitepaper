from parsl.app.app import python_app
from lib.bash_app_python import bash_app_python


def get_snaps_from_traj(inputs= [], outputs= [], scale_cell = False, calib_step = 0, samp_freq = 1, set_vel = True):
    from ase.io import read, write
    from lib.utils import from_h5_to_atoms_traj, check_validity_inputs, get_probable_cell_at_average_vol
    import numpy as np

    if not check_validity_inputs(inputs):
        for outp in outputs:
            f = open(str(outp), "w")
            f.write("failed output")
            f.close()
        return

    path_traj = str(inputs[0])
    
    num_samp = len(outputs)
    if len(outputs) == 0:
        num_samp = 1

    if path_traj[-4:] == ".xyz":
        traj = read(path_traj,index=slice(calib_step, None, samp_freq), parallel=False)
        if scale_cell == False and num_samp == 1:
            traj = [read(path_traj,index='-1', parallel=False)]
    elif path_traj[-3:] ==".h5":
        traj = from_h5_to_atoms_traj(path_traj, calib_step = calib_step, samp_freq = samp_freq)
        if scale_cell == False and num_samp == 1:
            traj = from_h5_to_atoms_traj(path_traj, get_last = True)
    
    if set_vel == False:
        for atoms in traj:
            atoms.set_momenta(None)

    len_traj = len(traj)
    if scale_cell == True:
        #Get the most probable cell chape at average volume
        #prob_cell_corr = get_probable_cell_at_average_vol(traj)

        #Get average cell
        for i,atoms in enumerate(traj):
            if i == 0:
                ave_cell = atoms.get_cell()/len_traj
                ave_vol  = atoms.get_volume()/len_traj
            else:
                ave_cell+= atoms.get_cell()/len_traj
                ave_vol += atoms.get_volume()/len_traj
        ave_cell_vol = np.linalg.det(ave_cell)
        ave_cell_corr = ave_cell* (ave_vol/ave_cell_vol)**(1.0/3.0)

    if num_samp == 1:
        atoms_ave = traj[-1].copy()
        if scale_cell == True:
            atoms_ave.set_cell(ave_cell_corr, scale_atoms = True)
        if len(outputs) != 0:
            write(str(outputs[0]), atoms_ave, append = False)
        return atoms_ave
    else:
        assert num_samp <= len_traj, "Make sure the sampling frequency is not too high such that there are enough samples in the trajectory"
        traj_out = []
        for i in range(num_samp):
            j = int((i+1)*np.floor(len_traj/num_samp)-1)
            atoms_ave = traj[j].copy()
            if scale_cell == True:
                atoms_ave.set_cell(ave_cell_corr, scale_atoms = True)
            write(str(outputs[i]), atoms_ave, append = False)
            traj_out.append(atoms_ave)
        return traj_out

app_get_snaps_from_traj = bash_app_python(get_snaps_from_traj, executors=['default'])

@bash_app_python(executors=['default'])
def app_get_weighted_random_snap(tem, inputs= [], outputs= [], frac_MLP = 1.0, frac_bias = 0.0, calib_step = 0, samp_freq = 1):
    from ase.io import write
    from lib.utils import from_h5_to_atoms_traj, from_xyz_to_h5, check_validity_inputs
    import numpy as np
    import molmod.units
    import h5py
    import os

    kb = molmod.constants.boltzmann
    beta = 1/(kb*tem)

    assert len(inputs) % 2 == 0, "For each trajectory file there should also be a recalculation file with the delta U"
    valid_inputs = []
    for i in range(int(len(inputs)/2)):
        if check_validity_inputs([inputs[2*i], inputs[2*i+1]]):
            valid_inputs.append(inputs[2*i])
            valid_inputs.append(inputs[2*i+1])

    f_MLP_lst = []
    f_rec_lst = []
    for i in range(len(valid_inputs)):
        if str(valid_inputs[i].filepath)[-4:] == ".xyz":
            path_in_h5 = str(valid_inputs[i].filepath)[:-4]+".h5"
            assert i% 2 == 0, "The input file for the recalculations files should be in h5 format to get the epot contributions"
            from_xyz_to_h5(valid_inputs[i].filepath, path_in_h5)
        else:
            path_in_h5 = str(valid_inputs[i].filepath)
        assert path_in_h5[-3:] == ".h5", "The input file is not in h5 format"
        if i% 2 == 0:
            f_MLP_lst.append(h5py.File(path_in_h5, 'r'))
        else:
            f_rec_lst.append(h5py.File(path_in_h5, 'r'))
        
    for i, (f_MLP, f_rec) in enumerate(zip(f_MLP_lst, f_rec_lst)):
        assert f_MLP['trajectory']['epot'].shape[0] == f_rec['trajectory']['epot'].shape[0], "The number of samples in the two trajectories are not the same"
        if i == 0:
            energy_MLP = f_MLP['trajectory']['epot'][calib_step::samp_freq]
            energy_rec = f_rec['trajectory']['epot_contribs'][calib_step::samp_freq,:]
        else:
            energy_MLP = np.concatenate((energy_MLP, f_MLP['trajectory']['epot'][calib_step::samp_freq]))
            energy_rec = np.concatenate((energy_rec, f_rec['trajectory']['epot_contribs'][calib_step::samp_freq,:]))

    rand_perm = np.random.permutation(energy_MLP.shape[0])
    
    En_MLP  = energy_MLP[rand_perm[0]]
    En_diff = energy_rec[rand_perm[0],0]
    En_bias = 0.0
    if frac_bias != 0.0:
        En_bias = energy_rec[rand_perm[0],1]
    current_energy = En_MLP - En_diff * (1.0 - frac_MLP) + En_bias * frac_bias
    current_index = rand_perm[0]

    for i in rand_perm[1:]:
        En_MLP  = energy_MLP[i]
        En_diff = energy_rec[i,0]
        if frac_bias != 0.0:
            En_bias = energy_rec[i,1]
        new_energy = En_MLP - En_diff * (1.0 - frac_MLP) + En_bias * frac_bias

        if new_energy > current_energy:
            P_acc = np.exp(-beta*(new_energy-current_energy))
        else: 
            P_acc = 2.0
        if np.random.rand() < P_acc:
            current_index = i
            current_energy = new_energy
    for f_rec in f_rec_lst:
        assert current_index >= 0, "The current index is negative" 
        if current_index < f_rec['trajectory']['epot'][calib_step::samp_freq].shape[0]:
            path_rec = f_rec.filename
            break
        else:
            current_index -= f_rec['trajectory']['epot'][calib_step::samp_freq].shape[0]
    atoms_fin = from_h5_to_atoms_traj(path_rec, calib_step = calib_step, samp_freq = samp_freq, get_frame = current_index, less_data = True)
    write(str(outputs[0]), atoms_fin, append = False)


@bash_app_python(executors=['default'])
def app_scale_cell(inputs=[], outputs = []):
    from ase.io import read, write
    from lib.utils import check_validity_inputs

    if not check_validity_inputs(inputs):
        for outp in outputs:
            f = open(str(outp), "w")
            f.write("failed output")
            f.close()
        return
    atoms = read(inputs[0].filepath)
    atoms_ref = read(inputs[1].filepath)    
    atoms.set_cell(atoms_ref.cell.array, scale_atoms = True)
    write(str(outputs[0].filepath), atoms)


@bash_app_python(executors=['default'])
def app_set_most_ortho_forall(inputs=[], outputs = []):
    from ase.io import read, write
    from numpy import linalg as LA
    import numpy as np
    from lib.utils import get_timestep, check_validity_inputs

    min_diff = 1000
    arg_min = None
    for i, input in enumerate(inputs):
        if check_validity_inputs([input]):
            atoms = read(input.filepath)
            angl = atoms.get_cell().angles()
            if min_diff > LA.norm(angl - 90):
                min_diff = LA.norm(angl - 90)
                arg_min = i
    for i, output in enumerate(outputs):
        atoms_out = read(inputs[arg_min].filepath)
        np.random.seed(i)
        timestep = get_timestep(atoms_out)
        pos=atoms_out.get_positions()
        max_disp = 0.01 * timestep + i**(1.0/3.0)/100 #in Angstrom, verry small change to adapt the trajectory
        new_pos = pos + np.random.rand(len(atoms_out),3) * 2 * max_disp - max_disp #in Angstrom
        atoms_out.set_positions(new_pos)

        write(str(output.filepath), atoms_out)


def plot_runave(inputs= [], outputs= [], calib_step = 0, samp_freq = 1, window = None):
    from ase.io import read
    from ase.cell import Cell
    from lib.utils import from_h5_to_atoms_traj, Running_Average, check_validity_inputs
    import numpy as np
    import matplotlib.pyplot as plt
    
    name_files = outputs[0].filepath
    name_files_noext = str(name_files)[:-4]

    traj = []
    num_part = len(inputs)
    for input in inputs:
        if check_validity_inputs([input]):
            path_traj = str(input)
            if path_traj[-4:] == ".xyz":
                traj_part = read(path_traj,index=slice(calib_step, None, samp_freq))
            elif path_traj[-3:] ==".h5":
                traj_part = from_h5_to_atoms_traj(path_traj, calib_step = calib_step, samp_freq = samp_freq)
            traj = traj + traj_part

    len_traj = len(traj)

    vol_lst   = []
    alpha_lst = []
    beta_lst  = []
    gamma_lst = []
    alen_lst = []
    blen_lst  = []
    clen_lst = []

    for i,atoms in enumerate(traj):
        vol_lst.append(atoms.get_volume())
        alpha_lst.append(atoms.get_cell().angles()[0])
        beta_lst.append(atoms.get_cell().angles()[1])
        gamma_lst.append(atoms.get_cell().angles()[2])
        alen_lst.append(atoms.get_cell().lengths()[0])
        blen_lst.append(atoms.get_cell().lengths()[1])
        clen_lst.append(atoms.get_cell().lengths()[2])
        if i == 0:
            ave_cell = atoms.get_cell()/len_traj
            ave_vol  = atoms.get_volume()/len_traj
        else:
            ave_cell+= atoms.get_cell()/len_traj
            ave_vol += atoms.get_volume()/len_traj

    f_out = open(name_files,'w')
    f_out.write("--------------------------------------------------\n")
    f_out.write("average volume: ")
    f_out.write(str(ave_vol)+"\n")
    f_out.write("--------------------------------------------------\n")
    f_out.write("average cell: ")
    f_out.write(str(ave_cell)+"\n")
    f_out.write("which has as volume: ")
    ave_cell_vol = np.linalg.det(ave_cell)
    f_out.write(str(ave_cell_vol)+"\n")
    f_out.write("average cell rescaled with correct average volume: ")
    ave_cell_corr = ave_cell* ave_vol/ave_cell_vol
    f_out.write(str(ave_cell_corr)+"\n")
    f_out.write("--------------------------------------------------\n")
    
    if window is None:
        window = int(len_traj/(2*num_part))

    fig, ax = plt.subplots()
    runave_vol_lst = Running_Average(vol_lst, window)
    plt.plot(range(len(runave_vol_lst)), runave_vol_lst)
    plt.axhline(ave_vol, color='r', linestyle="-")
    plt.xlabel('Step')
    plt.ylabel(r'Volume[A$^3$]')
    plt.savefig(name_files_noext+"_vol_runave_plot.pdf", format="pdf")
    plt.close()

    runave_alpha_lst = Running_Average(alpha_lst, window)
    runave_beta_lst  = Running_Average(beta_lst,  window)
    runave_gamma_lst = Running_Average(gamma_lst, window)

    fig, ax = plt.subplots()
    plt.plot(range(len(runave_alpha_lst)), runave_alpha_lst)
    plt.plot(range(len(runave_beta_lst)), runave_beta_lst)
    plt.plot(range(len(runave_gamma_lst)), runave_gamma_lst)
    plt.axhline(Cell(ave_cell).angles()[0], color='tab:blue', linestyle="--")
    plt.axhline(Cell(ave_cell).angles()[1], color='tab:orange', linestyle="--")
    plt.axhline(Cell(ave_cell).angles()[2], color='tab:green', linestyle="--")
    plt.xlabel('Step')
    plt.ylabel('Angle [degrees]')
    plt.savefig(name_files_noext+"_angle_runave_plot.pdf", format="pdf")
    plt.close()

    runave_alen_lst = Running_Average(alen_lst, window)
    runave_blen_lst = Running_Average(blen_lst,  window)
    runave_clen_lst = Running_Average(clen_lst, window)

    fig, ax = plt.subplots()
    plt.plot(range(len(runave_alen_lst)), runave_alen_lst)
    plt.plot(range(len(runave_blen_lst)), runave_blen_lst)
    plt.plot(range(len(runave_clen_lst)), runave_clen_lst)
    plt.axhline(Cell(ave_cell).lengths()[0], color='tab:blue', linestyle="--")
    plt.axhline(Cell(ave_cell).lengths()[1], color='tab:orange', linestyle="--")
    plt.axhline(Cell(ave_cell).lengths()[2], color='tab:green', linestyle="--")
    plt.xlabel('Step')
    plt.ylabel('lengths [Ang]')
    plt.savefig(name_files_noext+"_lengths_runave_plot.pdf", format="pdf")
    plt.close()

    hist_vol, bin_vol = np.histogram(vol_lst, bins= 20)
    hist_vol_norm = hist_vol / float(len(vol_lst))
    fig, ax = plt.subplots()
    plt.plot(bin_vol[1:], hist_vol_norm)
    plt.axvline(ave_vol, color='r', linestyle="-")
    plt.xlabel(r'Volume[A$^3$]')
    plt.ylabel('Probability')
    plt.savefig(name_files_noext+"_vol_hist.pdf", format="pdf")
    plt.close()

    hist_alpha, bin_alpha = np.histogram(alpha_lst, bins= 20)
    hist_beta,  bin_beta  = np.histogram(beta_lst,  bins= 20)
    hist_gamma, bin_gamma = np.histogram(gamma_lst, bins= 20)
    hist_alpha_norm = hist_alpha / float(len(alpha_lst))
    hist_beta_norm  = hist_beta  / float(len(beta_lst))
    hist_gamma_norm = hist_gamma / float(len(gamma_lst))
    fig, ax = plt.subplots()
    plt.plot(bin_alpha[1:], hist_alpha_norm)
    plt.plot(bin_beta[1:],  hist_beta_norm)
    plt.plot(bin_gamma[1:], hist_gamma_norm)
    plt.axvline(Cell(ave_cell).angles()[0], color='tab:blue', linestyle="--")
    plt.axvline(Cell(ave_cell).angles()[1], color='tab:orange', linestyle="--")
    plt.axvline(Cell(ave_cell).angles()[2], color='tab:green', linestyle="--")
    plt.xlabel('Angle [degrees]')
    plt.ylabel('Probability')
    plt.savefig(name_files_noext+"_angle_hist.pdf", format="pdf")
    plt.close()

    hist_alen, bin_alen = np.histogram(alen_lst, bins= 20)
    hist_blen, bin_blen = np.histogram(blen_lst, bins= 20)
    hist_clen, bin_clen = np.histogram(clen_lst, bins= 20)
    hist_alen_norm = hist_alen / float(len(alen_lst))
    hist_blen_norm = hist_blen / float(len(blen_lst))
    hist_clen_norm = hist_clen / float(len(clen_lst))
    fig, ax = plt.subplots()
    plt.plot(bin_alen[1:], hist_alen_norm)
    plt.plot(bin_blen[1:], hist_blen_norm)
    plt.plot(bin_clen[1:], hist_clen_norm)
    plt.axvline(Cell(ave_cell).lengths()[0], color='tab:blue', linestyle="--")
    plt.axvline(Cell(ave_cell).lengths()[1], color='tab:orange', linestyle="--")
    plt.axvline(Cell(ave_cell).lengths()[2], color='tab:green', linestyle="--")
    plt.xlabel('Length [Ang]')
    plt.ylabel('Probability')
    plt.savefig(name_files_noext+"_lengths_hist.pdf", format="pdf")
    plt.close()

    #Print warning about the convergence of the parameters!
    vol_diff_frac = (max(runave_vol_lst) -min(runave_vol_lst))/ ave_vol
    if vol_diff_frac > 0.03:
        f_out.write("WARNING: the fractional volume min-max difference of the running average with window "+str(window)+" is: "+ str(vol_diff_frac)+"\n")
    alpha_diff = max(runave_alpha_lst) -min(runave_alpha_lst)
    if alpha_diff > 3:
        f_out.write("WARNING: the alpha min-max difference of the running average with window "+str(window)+" is: "+ str(alpha_diff)+"\n")
    beta_diff = max(runave_beta_lst) -min(runave_beta_lst)
    if beta_diff > 3:
        f_out.write("WARNING: the beta min-max difference of the running average with window "+str(window)+" is: "+ str(beta_diff)+"\n")
    gamma_diff = max(runave_gamma_lst) -min(runave_gamma_lst)
    if gamma_diff > 3:
        f_out.write("WARNING: the gamma min-max difference of the running average with window "+str(window)+" is: "+ str(gamma_diff)+"\n")
    alen_diff_frac = (max(runave_alen_lst) -min(runave_alen_lst))/ Cell(ave_cell).lengths()[0]
    if alen_diff_frac > 0.01:
        f_out.write("WARNING: the fractional a length min-max difference of the running average with window "+str(window)+" is: "+str(alen_diff_frac)+"\n")
    blen_diff_frac = (max(runave_blen_lst) -min(runave_blen_lst))/ Cell(ave_cell).lengths()[1]
    if blen_diff_frac > 0.01:
        f_out.write("WARNING: the fractional b length min-max difference of the running average with window "+str(window)+" is: "+str(blen_diff_frac)+"\n")
    clen_diff_frac = (max(runave_clen_lst) -min(runave_clen_lst))/ Cell(ave_cell).lengths()[2]
    if clen_diff_frac > 0.01:
        f_out.write("WARNING: the fractional c length min-max difference of the running average with window "+str(window)+" is: "+str(clen_diff_frac)+"\n")
    
    f_out.close()
    
    
app_plot_runave = bash_app_python(plot_runave, executors=['default'])


def Compare_En_and_Vol(inputs = [], outputs = [], calib_step = 0, samp_freq = 1, check_vol = True, deltaU= False):
    from ase.io import read
    from lib.utils import from_h5_to_atoms_traj, check_validity_inputs
    import numpy as np
    import molmod.units
    import h5py
    
    with open(outputs[0].filepath, 'w') as f_out:
        assert len(inputs) %2 == 0, "The number of inputs must be even!"

        for i in range(int(len(inputs)/2)):
            input_1 = inputs[i]
            input_2 = inputs[i+int(len(inputs)/2)]

            if check_validity_inputs([input_1, input_2]):
                ave_en_pfu_lst = []
                ave_vol_pfu_lst = []
                for input_f in [input_1, input_2]:
                    path_traj = input_f.filepath
                    
                    if path_traj[-4:] == ".xyz":
                        traj = read(path_traj,index=slice(calib_step, None, samp_freq))
                    elif path_traj[-3:] ==".h5":
                        traj = from_h5_to_atoms_traj(path_traj, calib_step = calib_step, samp_freq = samp_freq, less_data = True)
                        f_h5 = h5py.File(path_traj, 'r')
                    
                    fu = 0
                    for at in traj[0]:
                        if at.symbol == "Pb":
                            fu +=1

                    if check_vol:
                        ave_vol = 0.0
                        for atoms in traj:
                            ave_vol += atoms.get_volume()                                                                         # in A**3
                        ave_vol_pfu_lst.append(ave_vol/ (len(traj)* fu))

                    if deltaU:
                        energy_eV_np = f_h5['trajectory']['epot'][calib_step::samp_freq] / molmod.units.electronvolt   # in eV
                        ave_en = np.average(energy_eV_np)
                        ave_en_pfu_lst.append(ave_en / fu)
                    else:
                        ave_en = 0.0
                        for j, atoms in enumerate(traj):
                            try:
                                ave_en += atoms.get_potential_energy()
                            except:
                                ave_en += atoms.info['energy']            # in eV
                        ave_en_pfu_lst.append(ave_en/ (len(traj)* fu))

                if check_vol:
                    diff_vol_pfu =  ave_vol_pfu_lst[1]-ave_vol_pfu_lst[0]
                    if  np.abs(diff_vol_pfu) > 30:
                        f_out.write("WARNING: volume difference between two methods is high, namely "+ str(diff_vol_pfu) + " Angstrom**3, for run " + str(i) + "\n")
                    else:
                        f_out.write("volume difference between two methods is relatively low, namely "+ str(diff_vol_pfu) + " Angstrom**3, for run " + str(i) + "\n")

                diff_en_kjmol_pfu =  (ave_en_pfu_lst[1]-ave_en_pfu_lst[0])*molmod.units.electronvolt/molmod.units.kjmol
                if  np.abs(diff_en_kjmol_pfu) > 0.2:
                    f_out.write("WARNING: eneryg difference between two methods is high, namely "+ str(diff_en_kjmol_pfu) + " eV, for run " + str(i) + "\n")
                else:
                    f_out.write("eneryg difference between two methods is relatively low, namely "+ str(diff_en_kjmol_pfu) + " eV, for run " + str(i) + "\n")
            else:
                f_out.write("WARNING: one or both of the files are not valid\n")

app_Compare_En_and_Vol = bash_app_python(Compare_En_and_Vol, executors=['default'])


@bash_app_python(executors=['default'])
def app_create_supercell(inputs = [], outputs = [], trans_mat = 2):
    from ase.io import read, write
    from ase.build.supercells import make_supercell
    from lib.utils import check_validity_inputs
    import numpy as np

    if not check_validity_inputs(inputs):
        for outp in outputs:
            f = open(str(outp), "w")
            f.write("failed output")
            f.close()
        return

    atoms = read(inputs[0].filepath)
    if isinstance(trans_mat, int):
        atoms_sup = make_supercell(atoms, trans_mat * np.identity(3))
    elif trans_mat.shape == (3,):
        atoms_sup = make_supercell(atoms, np.diag(trans_mat))
    elif trans_mat.shape == (3,3):
        atoms_sup = make_supercell(atoms, trans_mat)
    else:
        raise NotImplementedError
    write(outputs[0].filepath, atoms_sup)


@bash_app_python(executors=['default'])
def app_get_min_struc(inputs  = [], outputs = []):
    from ase.io import read, write
    import molmod.units
    from lib.DefineCV import Get_FAmol_lst, Get_MAmol_lst
    from lib.utils import get_cation_name, check_validity_inputs

    GS_energy = None
    for i, file_at in enumerate(inputs):
        if check_validity_inputs([file_at]):
            atoms= read(file_at.filepath)
            if GS_energy is None:
                atoms_min = atoms
                min_nr = i
                GS_energy = atoms.get_potential_energy()
            elif atoms.get_potential_energy() < GS_energy:
                atoms_min = atoms
                min_nr = i
                GS_energy = atoms.get_potential_energy()
    assert GS_energy is not None, "No valid inputs are provided to this app"
    
    # Addept atoms_min to get whole organic molecules
    cation_name = get_cation_name(atoms_min)
    if cation_name == "FA":
        index_lst = Get_FAmol_lst(atoms_min)
    elif cation_name == "MA":
        index_lst = Get_MAmol_lst(atoms_min)
    
    #write  out atoms_min
    write(str(outputs[0].filepath), atoms_min)
    
    with open(outputs[1].filepath, 'w') as f_out:
        fu = 0
        for at in atoms_min:
            if at.symbol == "Pb":
                fu += 1

        f_out.write("struc " + str(min_nr) + " has the mininmum energy of: " + str(GS_energy*molmod.units.electronvolt/(fu*molmod.units.kjmol))+" kj/mol\n")

        energy_diff_lst = []
        for i, file_at in enumerate(inputs):
            if check_validity_inputs([file_at]):
                atoms= read(file_at.filepath)
                energy_diff_lst.append((atoms.get_potential_energy() - GS_energy)*molmod.units.electronvolt/(fu*molmod.units.kjmol))
                f_out.write("diff with run "+ str(i) + " is " + str(energy_diff_lst[-1])+ " kj/mol\n")
            else: 
                f_out.write("File for run "+ str(i) + " is not valid kj/mol\n")
        energy_diff_lst.sort()
        checknr = min(4,len(energy_diff_lst)-1)
        if energy_diff_lst[checknr] > 0.2:
            f_out.write("Warning: The first " + str(checknr +1) + " lowest energy minima differ a lot in energy, namely: " + str(energy_diff_lst[checknr]) + " kj/mol. Maybe take more samples\n")
    

@bash_app_python(executors=['default'])
def app_transform_lower_triangular(inputs= [], outputs= [], reorder = False):
    from lib.RectMCbarostat import transform_lower_triangular
    from ase.io import read, write
    import molmod.units
    from lib.utils import check_validity_inputs

    if not check_validity_inputs(inputs):
        for outp in outputs:
            f = open(str(outp), "w")
            f.write("failed output")
            f.close()
        return

    path_in  = inputs[0].filepath
    path_out = outputs[0].filepath

    atoms = read(path_in)

    pos=atoms.get_positions() * molmod.units.angstrom
    rvecs=atoms.get_cell() * molmod.units.angstrom
    transform_lower_triangular(pos, rvecs, reorder= reorder)
    atoms.set_positions(pos / molmod.units.angstrom)
    atoms.set_cell(rvecs / molmod.units.angstrom)

    write(path_out, atoms)


@bash_app_python(executors=['default'])
def app_create_pickle_hes(temperature, inputs= [], outputs= [], freq_max = 5.0, smear_freq = 0.02):
    from lib.Free_energy_classes import PhononDensity_Hessian
    import matplotlib.pyplot as plt
    import numpy as np
    import molmod.units

    File_atoms   = inputs[0]
    File_hessian = inputs[1]
    path_pickle  = outputs[0].filepath
    
    pdos_hes = PhononDensity_Hessian.from_files(File_atoms, File_hessian)
    pdos_hes.calculate_prop(temperature)

    if len(outputs) == 2:
        freq_max    *= 10**12 / molmod.units.second     #convert to  atomic units
        smear_freq  *= 10**12 / molmod.units.second     #convert to  atomic units
        path_plot    = outputs[1].filepath
        pdos_hes.plot_function(np.arange(0, freq_max, smear_freq/2), smear_freq, label = "hessian")
        plt.xlim([0, freq_max])
        plt.legend()
        plt.savefig(path_plot, bbox_inches='tight')
        plt.clf()

    pdos_hes.write_pickle_file(path_pickle)


@bash_app_python(executors=['default'])
def app_create_pickle_vacf(num_val_prop, num_runs, temperature, inputs= [], outputs= [], bsize = 1024, freq_max = 5.0):
    from lib.Free_energy_classes import PhononDensity_vacf
    import matplotlib.pyplot as plt
    import molmod.units

    File_atoms    = inputs[0]
    output_files  = []
    for i in range(num_val_prop):
        output_files.append([])
        for j in range(num_runs):
            output_files[i].append(inputs[1 + i*num_runs + j])
    path_pickle   = outputs[0].filepath
    
    pdos_vacf = PhononDensity_vacf.from_files(File_atoms, output_files, temperature = temperature, bsize = bsize)
    pdos_vacf.calculate_prop()
    pdos_vacf.calculate_error_prop()

    if len(outputs) == 2:
        freq_max    *= 10**12 / molmod.units.second     #convert to  atomic units
        path_plot    = outputs[1].filepath
        pdos_vacf.plot_function(label = "vacf")
        plt.xlim([0, freq_max])
        plt.legend()
        plt.savefig(path_plot, bbox_inches='tight')
        plt.clf()

    pdos_vacf.write_pickle_file(path_pickle)


@bash_app_python(executors=['default'])
def app_create_pickle_lmdTI(num_val_prop, num_runs, lmd_np, inputs= [], outputs= [], temperature = None, num_part = 4, calib = 0):
    from lib.Free_energy_classes import lambda_integration, Free_energy_contribution
    import matplotlib.pyplot as plt
    import numpy as np

    File_atoms      = inputs[0]
    F_rec_lst       = []
    for i in range(num_val_prop):
        F_rec_lst.append([])
        for j in range(num_runs):
            F_rec_lst[i].append(inputs[1 + i*num_runs + j])
    Path_pickle_ref = inputs[-1].filepath
    fec_ref         = Free_energy_contribution.from_p(Path_pickle_ref)
    path_pickle     = outputs[0].filepath
    
    lmdTI = lambda_integration.from_files(File_atoms, F_rec_lst, lmd_np, temperature, num_part = num_part, calib = calib)
    if np.isscalar(fec_ref.free_energy) or len(fec_ref.free_energy) == 1:
        assert temperature == fec_ref.temperature, "The temperature of the reference free energy should be the same as the temperature at which the lambda TI is performed"
        lmdTI.calculate_prop(fec_ref.free_energy)
        lmdTI.calculate_error_prop(A_beg_error = fec_ref.free_error)
    else:
        assert len(fec_ref.free_energy) == len(fec_ref.temperature), "free_energy[i] is the free energy at temperature[i] in the reference free energy calculation"
        lmdTI.calculate_prop(float(fec_ref.free_energy[fec_ref.temperature == temperature]))
        if np.isscalar(fec_ref.free_error) or len(fec_ref.free_error) == 1:
            lmdTI.calculate_error_prop(A_beg_error = fec_ref.free_error)
        else:
            lmdTI.calculate_error_prop(A_beg_error = float(fec_ref.free_error[fec_ref.temperature == temperature]))     
    
    if len(outputs) == 2:
        path_plot    = outputs[1].filepath
        lmdTI.plot_function(label = "lmd_int")
        plt.xlim([0, 1])
        plt.legend()
        plt.savefig(path_plot, bbox_inches='tight')
        plt.clf()

    lmdTI.write_pickle_file(path_pickle)


@bash_app_python(executors=['default'])
def app_create_pickle_temTI(num_val_prop, num_runs, tem_np, inputs= [], outputs= [], pressure = None, ref_tem_lst = None, num_part = 4, calib = 0):
    from lib.Free_energy_classes import thermo_integration, Free_energy_contribution
    import matplotlib.pyplot as plt
    import numpy as np
    import molmod.units

    File_atoms      = inputs[0]
    F_run_lst       = []
    for i in range(num_val_prop):
        F_run_lst.append([])
        for j in range(num_runs):
            F_run_lst[i].append(inputs[1 + i*num_runs + j])
    path_pickle     = outputs[0].filepath
    
    temTI = thermo_integration.from_files(File_atoms, F_run_lst, tem_np, num_part = num_part, pressure = pressure, calib = calib)   

    A_ref = []
    A_ref_error = []
    T_ref = []
    for i, pfile in enumerate(inputs[(1+num_val_prop*num_runs):]):
        fec_ref = Free_energy_contribution.from_p(pfile.filepath)
        if ref_tem_lst is None:
            assert np.isscalar(fec_ref.temperature) or len(fec_ref.temperature) == 1, "If no reference temperature is given, the reference energy/temperature should be a scalar"
            assert np.isscalar(fec_ref.free_energy) or len(fec_ref.free_energy) == 1, "If no reference temperature is given, the reference energy/temperature should be a scalar"
            T_ref.append(fec_ref.temperature)
            A_ref.append(fec_ref.free_energy)
            A_ref_error.append(fec_ref.free_error)
        else:
            if np.isscalar(fec_ref.temperature) or len(fec_ref.temperature) == 1:
                assert fec_ref.temperature == ref_tem_lst[i], "The FEC file should contain the free energy at the same temperature as the reference temperature"
                T_ref.append(fec_ref.temperature)
                A_ref.append(fec_ref.free_energy)
                A_ref_error.append(fec_ref.free_error)
            else:
                assert len(fec_ref.temperature) == len(fec_ref.free_energy), "free_energy[i] is the free energy at temperature[i] in the reference free energy calculation"
                T_ref.append(ref_tem_lst[i])
                A_ref.append(float(fec_ref.free_energy[fec_ref.temperature == ref_tem_lst[i]]))
                if np.isscalar(fec_ref.free_error) or len(fec_ref.free_error) == 1:
                    A_ref_error.append(fec_ref.free_error)
                else:
                    A_ref_error.append(float(fec_ref.free_error[fec_ref.temperature == ref_tem_lst[i]]))
    temTI.calculate_prop(T_ref, A_ref)
    temTI.calculate_error_prop(T_ref, A_ref_error = A_ref_error)

    if len(outputs) == 2:
        path_plot    = outputs[1].filepath
        temTI.plot_function(label = "tem_int")
        plt.xlim([tem_np[0], tem_np[-1]])
        plt.legend()
        plt.savefig(path_plot, bbox_inches='tight')
        plt.clf()

    temTI.write_pickle_file(path_pickle)


@bash_app_python(executors=['default'])
def app_create_pickle_nptd(num_val_prop, num_runs, temperature, pressure, inputs= [], outputs= [], num_part = 4, calib = 0):
    from lib.Free_energy_classes import NPT_distrubtion, Free_energy_contribution
    import matplotlib.pyplot as plt
    import numpy as np

    File_atoms      = inputs[0]
    F_run_npt       = []
    for i in range(num_val_prop):
        F_run_npt.append([])
        for j in range(num_runs):
            F_run_npt[i].append(inputs[1 + i*num_runs + j])
    Path_pickle_ref = inputs[-1].filepath
    fec_ref         = Free_energy_contribution.from_p(Path_pickle_ref)
    path_pickle     = outputs[0].filepath

    pdos_nptd = NPT_distrubtion.from_files(File_atoms, F_run_npt, temperature = temperature, pressure = pressure, num_part = num_part, calib = calib)
    if np.isscalar(fec_ref.free_energy) or len(fec_ref.free_energy) == 1:
        assert temperature == fec_ref.temperature, "The temperature of the reference free energy should be the same as the temperature at which the lambda TI is performed"
        pdos_nptd.calculate_prop(fec_ref.free_energy)
        pdos_nptd.calculate_error_prop(A_beg_error = fec_ref.free_error)
    else:
        assert len(fec_ref.free_energy) == len(fec_ref.temperature), "free_energy[i] is the free energy at temperature[i] in the reference free energy calculation"
        pdos_nptd.calculate_prop(float(fec_ref.free_energy[fec_ref.temperature == temperature]))
        if np.isscalar(fec_ref.free_error) or len(fec_ref.free_error) == 1:
            pdos_nptd.calculate_error_prop(A_beg_error = fec_ref.free_error)
        else:
            pdos_nptd.calculate_error_prop(A_beg_error = float(fec_ref.free_error[fec_ref.temperature == temperature]))

    if len(outputs) == 2:
        path_plot    = outputs[1].filepath
        pdos_nptd.plot_function(label = "nptd")
        plt.legend()
        plt.savefig(path_plot, bbox_inches='tight')
        plt.clf()

    pdos_nptd.write_pickle_file(path_pickle)


@bash_app_python(executors=['default'])
def app_plot_fec(label_lst, inputs=[], outputs = []):
    from lib.Free_energy_classes import Free_energy_contribution
    import matplotlib.pyplot as plt
    import numpy as np
    import molmod.units

    path_plot = outputs[0].filepath
    fe_dct = {}
    for i, label in enumerate(label_lst):
        pfile = inputs[2*i]
        ref_pfile = inputs[2*i + 1]
        phase_fec = Free_energy_contribution.from_p(pfile.filepath)
        ref_fec = Free_energy_contribution.from_p(ref_pfile.filepath)

        tem = ref_fec.temperature
        phase_fe = phase_fec.free_energy  / phase_fec.n_fu
        phase_fe_error = phase_fec.free_error / phase_fec.n_fu
        ref_fe = ref_fec.free_energy / ref_fec.n_fu
        ref_fe_error = ref_fec.free_error / ref_fec.n_fu
        fe = (phase_fe - ref_fe) / molmod.units.kjmol
        fe_error = np.sqrt(phase_fe_error**2 + ref_fe_error**2) / molmod.units.kjmol
        if np.isscalar(tem):
            tem = [tem]
        if np.isscalar(fe):
            fe = [fe]
        if np.isscalar(fe_error):
            fe_error = [fe_error]

        if label in fe_dct:
            new_tem = []
            new_fe = []
            new_fe_error = []
            for tem_n, fe_n, fe_error_n in zip(tem, fe, fe_error):
                flag = False
                for tem_o, fe_o, fe_error_o in zip(fe_dct[label]["tem"], fe_dct[label]["fe"], fe_dct[label]["fe_error"]):
                    assert tem_n != tem_o, "Double value ecoutered for " + label + " at temperature " + str(tem_n)
                    if tem_n < tem_o and flag == False:
                        new_tem.append(tem_n)
                        new_fe.append(fe_n)
                        new_fe_error.append(fe_error_n)
                        flag = True
                    elif tem_o not in new_tem:
                        new_tem.append(tem_o)
                        new_fe.append(fe_o)
                        new_fe_error.append(fe_error_o)
            if tem_n > tem_o: #The beggist value is not added yet, so check which it is, and add it
                new_tem.append(tem_n)
                new_fe.append(fe_n)
                new_fe_error.append(fe_error_n)
            else:
                new_tem.append(tem_o)
                new_fe.append(fe_o)
                new_fe_error.append(fe_error_o)                
            fe_dct[label]= {"tem": new_tem, "fe": new_fe, "fe_error": new_fe_error}
        else:
            fe_dct[label]= {"tem": tem, "fe": fe, "fe_error": fe_error}

    for label, val_dct in fe_dct.items():
        plt.errorbar(val_dct["tem"], val_dct["fe"], yerr = val_dct['fe_error'], label = label)
    plt.legend()
    plt.ylabel("temperature [K]") 
    plt.ylabel("free energy diff ref pfu [kJ/mol]") 
    plt.axhline(0.0, color='k', linestyle="--")
    plt.savefig(path_plot, bbox_inches='tight')
    plt.clf()

