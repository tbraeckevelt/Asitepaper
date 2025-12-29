import molmod
import yaff
from ase.io import read, write
from lib.utils import ForcePartdifference, get_part_bias, get_cation_name
from lib.DefineCV import Get_FAmol_lst, Get_MAmol_lst
from lib.bash_app_python import bash_app_python


def run_MD(inputs=[], outputs=[], start = 0, steps = 100, step = 50, temperature = None, pressure = None, barostat = "Langevin", num_retries = 0, 
           num_checks = 10, frac_MLP = 1.0, frac_bias = 0.0, seed = 0, num_cores = 1, precision = "single", device = "cpu", restart = False):
    import torch
    import numpy as np
    import h5py
    import yaff
    import molmod

    import ase.units
    from ase.io import read, write
    from ase import Atoms
    
    from lib.utils import create_forcefield, ExtXYZHook, get_timestep, get_calculator, ForceThresholdExceededException, check_validity_inputs
    from lib.RectMCbarostat import RectangularMonteCarloBarostat, transform_lower_triangular
    from lib.papps import get_snaps_from_traj
    from lib.DefineCV import Get_FAmol_lst, Get_MAmol_lst

    if not check_validity_inputs(inputs):
        for outp in outputs:
            f = open(str(outp), "w")
            f.write("failed output")
            f.close()
        return

    path_atoms = str(inputs[0]) 
    path_calc  = str(inputs[1])
    path_traj  = str(outputs[0])
    assert path_atoms[-4:] == ".xyz", "The first input must be a .xyz file"
    assert path_calc[-6:] == ".model" or path_calc[-4:] == ".pth", "The second input must be a .model file or .pth file"
    if len(inputs) == 4:
        path_min_atoms = str(inputs[2])
        path_hess      = str(inputs[3])
        assert path_min_atoms[-4:] == ".xyz", "The third input must be a .xyz file"
        assert path_hess[-4:]      == ".pth", "The fourth input must be a .pth file"
    else:
        path_min_atoms = None
        path_hess      = None
    np.random.seed(seed)
    torch.set_num_threads(num_cores)

    if restart == True:
        atoms = get_snaps_from_traj(inputs= [path_traj], outputs= [])
    else:
        atoms = read(path_atoms)

    #Get Initial verlocities from ASE atoms object
    velocities_init = atoms.get_velocities() * molmod.units.angstrom * ase.units.fs / molmod.units.femtosecond
    if (velocities_init == 0).all():
        velocities_init = None
    timestep = get_timestep(atoms)

    dtype_str = 'float32'
    if precision == 'double':
        torch.set_default_dtype(torch.float64)
        dtype_str = 'float64'
    calculator = get_calculator(path_calc, atoms, device, dtype_str)
    atoms.calc = calculator
    if path_hess != None:
        Hess_model = torch.jit.load(path_hess)
    else:
        Hess_model = None

    if barostat == "MC":
        print("transform atoms cell to lower triangular for the Monte Carlo barostat")
        pos=atoms.get_positions() * molmod.units.angstrom
        rvecs=atoms.get_cell() * molmod.units.angstrom
        transform_lower_triangular(pos, rvecs, reorder=False)
        atoms.set_positions(pos / molmod.units.angstrom)
        atoms.set_cell(rvecs / molmod.units.angstrom)
    
    flag = True
    tel = 0
    while tel <= num_retries and flag:   #if the MLP exceed a force threshold (currently set to 30 eV/Ang), then retry the run with slightly adapted initial pos
        flag = False

        # create forcefield from atoms
        ff = create_forcefield(atoms, calculator, model = Hess_model, frac_MLP = frac_MLP, frac_bias = frac_bias, path_min_atoms = path_min_atoms)

        #Hooks
        hooks = []
        if restart == True:
            assert tel == 0, "The restart is not tested with retries probably problems will occur, use at own risk"
            if path_traj[-3:] == '.h5':
                h5file = h5py.File(path_traj, 'a')
                #Get counter and time from previous h5file
                count_pre = np.array(h5file["trajectory/counter"])[-1]
                time_pre  = np.array(h5file["trajectory/time"])[-1]
                hooks.append(yaff.HDF5Writer(h5file, step=step, start=count_pre + step))
            elif path_traj[-4:] == '.xyz':
                count_pre = len(read(path_traj, index =":")[1:]) *step   
                time_pre = count_pre * timestep
                hooks.append(ExtXYZHook(path_traj, start=count_pre + step, step=step, append = True))
            hooks.append(yaff.VerletScreenLog(step=step, start=count_pre + step))
        else:
            count_pre = start
            time_pre  = start
            if path_traj[-3:] == '.h5':
                h5file = h5py.File(path_traj, 'w')
                hooks.append(yaff.HDF5Writer(h5file, step=step, start=count_pre))
            elif path_traj[-4:] == '.xyz':
                hooks.append(ExtXYZHook(path_traj, start=count_pre, step=step))
            hooks.append(yaff.VerletScreenLog(step=step, start=count_pre))

        # temperature / pressure control
        if temperature is None:
            print('CONSTANT ENERGY, CONSTANT VOLUME')
        else:
            thermo = yaff.LangevinThermostat(temperature, timecon=100 * molmod.units.femtosecond)
            hooks.append(thermo)
            if pressure is None:
                print('CONSTANT TEMPERATURE, CONSTANT VOLUME')
            else:
                print('CONSTANT TEMPERATURE, CONSTANT PRESSURE')
                vol_constraint = False
                if barostat == "Langevin":
                    print('Langevin barostat')
                    baro = yaff.LangevinBarostat(
                            ff,
                            temperature,
                            pressure,
                            timecon=molmod.units.picosecond,
                            anisotropic=True,
                            vol_constraint=vol_constraint,
                            )
                    tbc = yaff.TBCombination(thermo, baro)
                    hooks.append(tbc)
                elif barostat == "MC":
                    print('MC barostat')
                    baro = RectangularMonteCarloBarostat(
                        temperature,
                        pressure,
                        )
                    hooks.append(baro)
    
        # integration
        try:
            verlet = yaff.VerletIntegrator(
                    ff,
                    timestep=timestep*molmod.units.femtosecond,
                    hooks=hooks,
                    vel0 = velocities_init,
                    temp0=temperature, # initialize velocities to correct temperature, if vel0 is None
                    time0=time_pre, 
                    counter0=count_pre
                    )
            yaff.log.set_level(yaff.log.medium)
            steps_check = int(np.floor(steps/num_checks))
            last_steps_check = steps_check + steps % num_checks
            for nch in range(num_checks):
                if nch == num_checks -1:
                    verlet.run(last_steps_check)
                else:
                    verlet.run(steps_check)
                atoms_check = Atoms(
                    numbers=verlet.ff.system.numbers.copy(),
                    positions=verlet.ff.system.pos / molmod.units.angstrom,
                    cell=verlet.ff.system.cell._get_rvecs() / molmod.units.angstrom,
                    pbc=True,
                    )
                cation_name = get_cation_name(atoms_check)
                if cation_name == "FA":
                    index_lst = Get_FAmol_lst(atoms_check)
                elif cation_name == "MA":
                    index_lst = Get_MAmol_lst(atoms_check)
            yaff.log.set_level(yaff.log.silent)
        except (ForceThresholdExceededException, AssertionError) as e:
            print("Exception caught at step " + str(verlet.counter - count_pre) + ":")
            print(e)
            print("Change initial coordinates and retry " + str(tel))
            tel += 1
            flag = True
            if restart == True:  #Use at owen risk, restart and retries not tested!
                atoms = get_snaps_from_traj(inputs= [path_traj], outputs= [])
            else:
                atoms = read(path_atoms)
            if verlet.counter - count_pre < 1000:
                if tel == num_retries and path_min_atoms is not None:
                    atoms.set_positions(read(path_min_atoms).get_positions())
                    print("As a last try, start from slightly perturbed positions of the minimum energy optimized structure")
                else:
                    from ase.optimize.precon import Exp, PreconLBFGS
                    atoms.calc = calculator
                    preconditioner = Exp(A=3) # from ASE docs
                    dof = atoms
                    optimizer = PreconLBFGS(
                            dof,
                            precon=preconditioner,
                            use_armijo=True,
                            )
                    optimizer.run(fmax=1.0/tel)   
            pos=atoms.get_positions()
            max_disp = 0.02 * timestep #in Angstrom, verry small change to adapt the trajectory
            new_pos = pos + np.random.rand(len(atoms),3) * 2 * max_disp - max_disp #in Angstrom
            atoms.set_positions(new_pos)
            atoms.calc = calculator
            if path_traj[-3:] == '.h5':
                h5file.close()
            

    if tel > num_retries:
        #raise ValueError("Retried " + str(num_retries) + " times, I will stop trying. Change MD settings or retrain MLP")
        print("Retried " + str(num_retries) + " times, I will stop trying. Change MD settings or retrain MLP")
        for outp in outputs:
            f = open(str(outp), "w")
            f.write("failed output")
            f.close()


def run_REX(inputs=[], outputs=[], MC_attempts = 10, MD_steps = 10, start = 0, step = 50, tem_min = None, tem_max = None, pressure = None, num_retries = 0, 
            num_checks = 10, barostat = "Langevin", frac_MLP = 1.0, frac_bias = 0.0, seed = 0, num_cores = 1, precision = "single", device = "cpu", restart = False):
    import torch
    import numpy as np
    import h5py
    import yaff
    import molmod
    from mpi4py import MPI

    import ase.units
    from ase.io import read, write
    from ase import Atoms
    from lib.utils import create_forcefield, ExtXYZHook, get_timestep, analyze_permutations, get_calculator, ForceThresholdExceededException, check_validity_inputs
    from lib.RectMCbarostat import RectangularMonteCarloBarostat, transform_lower_triangular
    from lib.papps import get_snaps_from_traj
    from lib.DefineCV import Get_FAmol_lst, Get_MAmol_lst

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_tem = comm.Get_size()

    if rank == 0:
        if not check_validity_inputs(inputs):
            for outp in outputs:
                f = open(str(outp), "w")
                f.write("failed output")
                f.close()
            return

    nr_replicas= len(outputs) - 1
    path_atoms = str(inputs[rank]) 
    path_calc  = str(inputs[nr_replicas])
    path_out   = str(outputs[rank])
    path_perm  = str(outputs[-1])
    assert path_atoms[-4:] == ".xyz", "The input structures must be a .xyz file"
    assert path_calc[-6:] == ".model" or path_calc[-4:] == ".pth", "The last input must be a .model file or .pth file"
    assert path_perm[-4:] == ".npy", "The last output must be a .npy file"
    if len(inputs) == nr_replicas + 3:
        path_min_atoms = str(inputs[nr_replicas+1])
        path_hess      = str(inputs[nr_replicas+2])
        assert path_min_atoms[-4:] == ".xyz", "The third input must be a .xyz file"
        assert path_hess[-4:]      == ".pth", "The fourth input must be a .pth file"
    else:
        path_min_atoms = None
        path_hess      = None
    np.random.seed(seed)
    torch.set_num_threads(num_cores)
    #Construct array with temperatures (preferred sampling method)
    y_max = np.log(tem_max/tem_min)
    y_np = np.arange(num_tem)*y_max/(num_tem-1)
    tem_np = np.exp(y_np)*tem_min
    
    if restart == True:
        atoms = get_snaps_from_traj(inputs= [path_out], outputs= [])
    else:
        atoms = read(path_atoms, parallel=False)

    #Get Initial verlocities from ASE atoms object
    velocities_init = atoms.get_velocities() * molmod.units.angstrom * ase.units.fs / molmod.units.femtosecond
    if (velocities_init == 0).all():
        velocities_init = None
    timestep = get_timestep(atoms)

    dtype_str = 'float32'
    if precision == 'double':
        torch.set_default_dtype(torch.float64)
        dtype_str = 'float64'
    calculator = get_calculator(path_calc, atoms, device, dtype_str)
    atoms.calc = calculator
    if path_hess != None:
        Hess_model = torch.jit.load(path_hess)
    else:
        Hess_model = None

    if barostat == "MC":
        print("rank "+str(rank)+": transform atoms cell to lower triangular for the Monte Carlo barostat")
        pos=atoms.get_positions() * molmod.units.angstrom
        rvecs=atoms.get_cell() * molmod.units.angstrom
        transform_lower_triangular(pos, rvecs, reorder=False)
        atoms.set_positions(pos / molmod.units.angstrom)
        atoms.set_cell(rvecs / molmod.units.angstrom)

    flag = True
    tel = 0
    while tel <= num_retries and flag:   #if the MLP exceed a force threshold (currently set to 30 eV/Ang), then retry the run with slightly adapted initial pos
        flag = False

        #Keep track of performed swaps
        prev_MC_Attemps = 0
        if rank == 0:
            if restart == True:
                permutations_old = np.load(path_perm)
                prev_MC_Attemps = np.shape(permutations_old)[0] -1
                permutations = np.zeros((prev_MC_Attemps + MC_attempts + 1, num_tem))
                for i in range(prev_MC_Attemps +1):
                    permutations[i,:] = permutations_old[i,:]
            else:
                permutations = np.zeros((MC_attempts+1,num_tem))
                permutations[0,:] = np.arange(num_tem)
        prev_MC_Attemps = comm.bcast(prev_MC_Attemps, root=0)

        # create forcefield from atoms
        ff = create_forcefield(atoms, calculator, model = Hess_model, frac_MLP = frac_MLP, frac_bias = frac_bias, path_min_atoms = path_min_atoms)

        #Hooks
        hooks = []
        if restart == True:
            assert tel == 0, "The restart is not tested with retries probably problems will occur, use at own risk"
            if path_out[-3:] == '.h5':
                h5file = h5py.File(path_out, 'a')
                #Get counter and time from previous h5file
                count_pre = np.array(h5file["trajectory/counter"])[-1]
                time_pre  = np.array(h5file["trajectory/time"])[-1]
                hooks.append(yaff.HDF5Writer(h5file, step=step, start=count_pre + step))
            elif path_out[-4:] == '.xyz':
                count_pre = len(read(path_out, index =":",parallel=False)[1:]) *step 
                time_pre = count_pre * timestep
                hooks.append(ExtXYZHook(path_out, start=count_pre + step, step=step, append = True))
                print(count_pre)
            hooks.append(yaff.VerletScreenLog(step=step, start=count_pre + step))
        else:
            count_pre = start
            time_pre  = start
            if path_out[-3:] == '.h5':
                h5file = h5py.File(path_out, 'w')
                hooks.append(yaff.HDF5Writer(h5file, step=step, start=count_pre))
            elif path_out[-4:] == '.xyz':
                hooks.append(ExtXYZHook(path_out, start=count_pre, step=step))
            hooks.append(yaff.VerletScreenLog(step=step, start=count_pre))


        thermo = yaff.LangevinThermostat(tem_np[rank], timecon=100 * molmod.units.femtosecond)
        if pressure is None:
            print('CONSTANT TEMPERATURE, CONSTANT VOLUME')
            hooks.append(thermo)
        else:
            print('CONSTANT TEMPERATURE, CONSTANT PRESSURE')
            vol_constraint = False
            if barostat == "Langevin":
                print('Langevin barostat')
                baro = yaff.LangevinBarostat(
                        ff,
                        tem_np[rank],
                        pressure,
                        timecon=molmod.units.picosecond,
                        anisotropic=True,
                        vol_constraint=vol_constraint,
                        )
                tbc = yaff.TBCombination(thermo, baro)
                hooks.append(tbc)
            elif barostat == "MC":
                print('MC barostat')
                baro = RectangularMonteCarloBarostat(
                    tem_np[rank],
                    pressure,
                    )
                hooks.append(baro)

        #Initialization verlet integrator for this rank
        verlet_rank = yaff.VerletIntegrator(
                ff,
                timestep=timestep*molmod.units.femtosecond,
                hooks=hooks,
                vel0 = velocities_init,
                temp0=tem_np[rank], # initialize velocities to correct temperature, if velocities_init is None
                time0=time_pre, 
                counter0=count_pre
                )
        
        #Do replica exchange run
        i = prev_MC_Attemps
        while i < prev_MC_Attemps+MC_attempts:
            #Gether the position, velocities and energies of each rank on root 0
            pos_lst = comm.gather(verlet_rank.pos.copy(), root=0)
            vel_lst = comm.gather(verlet_rank.vel.copy(), root=0)
            en_lst  = comm.gather(verlet_rank.epot, root=0)
            cell_lst= comm.gather(verlet_rank.rvecs, root=0)

            if rank == 0:

                #Attempt to swap positions
                perm_tuple_lst = []
                if np.random.rand()<0.5:
                    for j in range(int(num_tem/2)): #Chose an even number of temperatures!
                        perm_tuple_lst.append((2*j,2*j+1))
                else:
                    for j in range(int(num_tem/2)): #Chose an even number of temperatures!
                        if j == 0:
                            perm_tuple_lst.append((0,int(num_tem-1)))
                        else:
                            perm_tuple_lst.append((2*j-1,2*j))


                for (m,n) in perm_tuple_lst:

                    energy_m, beta_m, vol_m = en_lst[m], 1/(molmod.constants.boltzmann * tem_np[m]), np.linalg.det(cell_lst[m])
                    energy_n, beta_n, vol_n = en_lst[n], 1/(molmod.constants.boltzmann * tem_np[n]), np.linalg.det(cell_lst[n])

                    if pressure is None:
                        exponent = (beta_n-beta_m)*(energy_n - energy_m)
                    else:
                        exponent = (beta_n-beta_m)*((energy_n + pressure * vol_n) - (energy_m + pressure * vol_m) )
                    if exponent<0:
                        P_acc = np.exp(exponent)
                    else: 
                        P_acc = 2.0

                    #print("The acceptance probability for swap between "+str(m)+" and "+str(n)+" is: "+str(P_acc)+ ", with exponent: "+str(exponent))
                    if np.random.rand() < P_acc:
                        print("accepted swap between "+str(m)+" and "+str(n))
                        #actually swap positions
                        pos_lst[m], pos_lst[n] = pos_lst[n], pos_lst[m]
                        vel_lst[m], vel_lst[n] = vel_lst[n] * np.sqrt(tem_np[m] / tem_np[n]), vel_lst[m] * np.sqrt(tem_np[n] / tem_np[m])
                        cell_lst[m], cell_lst[n] = cell_lst[n], cell_lst[m]
                        permutations[i+1,m], permutations[i+1,n] = permutations[i,n], permutations[i,m]
                    else:
                        permutations[i+1,m], permutations[i+1,n] = permutations[i,m], permutations[i,n]

            #Scatter the positions and velocities back to each rank
            new_pos_rank = comm.scatter(pos_lst, root=0)
            new_vel_rank = comm.scatter(vel_lst, root=0)
            new_cell_rank= comm.scatter(cell_lst, root=0)
            #update positions and velocities of verlet instance on each rank
            if (new_pos_rank != verlet_rank.pos).all():
                verlet_rank.pos = new_pos_rank
                verlet_rank.vel = new_vel_rank
                verlet_rank.rvecs = new_cell_rank
            
            #Do MD
            try:
                verlet_rank.run(MD_steps)
                if i % (int(np.floor(MC_attempts/num_checks))) == 0:
                    atoms_check = Atoms(
                        numbers=verlet_rank.ff.system.numbers.copy(),
                        positions=verlet_rank.ff.system.pos / molmod.units.angstrom,
                        cell=verlet_rank.ff.system.cell._get_rvecs() / molmod.units.angstrom,
                        pbc=True,
                        )
                    cation_name = get_cation_name(atoms_check)
                    if cation_name == "FA":
                        index_lst = Get_FAmol_lst(atoms_check)
                    elif cation_name == "MA":
                        index_lst = Get_MAmol_lst(atoms_check)
                    excep = None
            except (ForceThresholdExceededException, AssertionError) as e:
                print("Exception caught on rank "+ str(rank) + " at step " + str(verlet_rank.counter - count_pre) + ":")
                print(e)
                print("Change all initial coordinates and retry " + str(tel))
                excep = e

            #Gather and scatter the possible exception such that all ranks get restarted
            excep_lst = comm.gather(excep, root=0)
            new_excep_lst = []
            if rank == 0:
                excep = None
                for el in excep_lst:
                    if el != None:
                        excep = el
                for el in excep_lst:
                    new_excep_lst.append(excep)
            exception = comm.scatter(new_excep_lst, root=0)

            # Handle the exception on all ranks
            if exception is not None:
                tel += 1
                flag = True
                i += MC_attempts  #Get out of while loop and immediately trigger the main while loop
                if restart == True:  #Use at owen risk, restart and retries not tested!
                    atoms = get_snaps_from_traj(inputs= [path_out], outputs= [])
                else:
                    atoms = read(path_atoms, parallel=False)
                if verlet_rank.counter - count_pre < 1000:
                    from ase.optimize.precon import Exp, PreconLBFGS
                    atoms.calc = calculator
                    preconditioner = Exp(A=3) # from ASE docs
                    dof = atoms
                    optimizer = PreconLBFGS(
                            dof,
                            precon=preconditioner,
                            use_armijo=True,
                            )
                    optimizer.run(fmax=1.0/tel) 
                pos=atoms.get_positions()
                max_disp = 0.02 * timestep #in Angstrom, verry small change to adapt the trajectory
                new_pos = pos + np.random.rand(len(atoms),3) * 2 * max_disp - max_disp #in Angstrom
                atoms.set_positions(new_pos)
                atoms.calc = calculator
                if path_out[-3:] == '.h5':
                    h5file.close()
            i += 1 #this is the counter for the while loop
    
    #Print out the performed swaps and analyze it
    if rank == 0:
        if tel > num_retries:
            #raise ValueError("Retried " + str(num_retries) + " times, I will stop trying. Change MD settings or retrain MLP")   should be done on all ranks!
            print("Retried " + str(num_retries) + " times, I will stop trying. Change MD settings or retrain MLP")
            for outp in outputs:
                f = open(str(outp), "w")
                f.write("failed output")
                f.close()
        else:
            np.save(path_perm, permutations)
            analyze_permutations(permutations, path_perm[:-4]+".pdf", num_tem = num_tem, tem_min = tem_min, tem_max = tem_max)
    


def optimize(inputs=[], outputs=[], num_cores = 1, precision = "double", device = "cpu", path_traj = None, constant_cell = True, constant_volume = True, fmax=1e-4):
    import time
    import torch
    from ase.io import read, write
    from ase.optimize.precon import Exp, PreconLBFGS
    from ase.constraints import ExpCellFilter
    from lib.utils import get_calculator, check_validity_inputs

    if not check_validity_inputs(inputs):
        for outp in outputs:
            f = open(str(outp), "w")
            f.write("failed output")
            f.close()
        return

    atoms = read(str(inputs[0])) 
    path_calc  = str(inputs[1])
    path_atoms_opt = str(outputs[0])
    assert str(inputs[0])[-4:] == ".xyz", "The first input must be a .xyz file"
    assert path_calc[-6:] == ".model" or path_calc[-4:] == ".pth", "The second input must be a .model file or .pth file"
    assert path_atoms_opt[-4:] == ".xyz", "The output must be a .xyz file"

    dtype_str = 'float32'
    if precision == 'double':
        torch.set_default_dtype(torch.float64)
        dtype_str = 'float64'
    calculator = get_calculator(path_calc, atoms, device, dtype_str)
    atoms.calc = calculator
    torch.set_num_threads(num_cores)

    preconditioner = Exp(A=3) # from ASE docs
    if constant_cell:
        dof = atoms
    else:
        dof = ExpCellFilter(
                atoms,
                mask=[True] * 6, # includes cell DOFs in optimization
                constant_volume = constant_volume, # Do not relax volume, only shape
                )
    optimizer = PreconLBFGS(
            dof,
            precon=preconditioner,
            use_armijo=True,
            trajectory=path_traj,
            )
    
    if constant_cell == False:
        print('initial box vectors:')
        for i in range(3):
            print('\t{}'.format(atoms.get_cell()[i]))
        print('')
    start = time.time()
    print('starting optimization ...')
    optimizer.run(fmax=fmax)                    #Verify that the forces decrease!
    print('')
    print('optimization completed in {} s'.format(time.time() - start))
    if constant_cell == False:
        print('final box vectors:')
        for i in range(3):
            print('\t{}'.format(atoms.get_cell()[i]))
    write(path_atoms_opt, atoms) # save optimized state


def calc_hessian(inputs=[], outputs=[], num_cores = 1, precision = "double", device = "cpu", num_retries = 0):
    import torch
    from ase.io import read, write
    from lib.utils import create_forcefield, get_frequencies, EffectiveHarmonicModel, get_calculator
    import yaff
    import numpy as np
    import molmod.units

    atoms = read(str(inputs[0])) 
    path_calc  = str(inputs[1])
    path_hessian = str(outputs[0])
    path_model = str(outputs[1])
    assert str(inputs[0])[-4:] == ".xyz", "The first input must be a .xyz file"
    assert path_calc[-6:] == ".model" or path_calc[-4:] == ".pth", "The second input must be a .model file or .pth file"

    dtype_str = 'float32'
    if precision == 'double':
        torch.set_default_dtype(torch.float64)
        dtype_str = 'float64'
    calculator = get_calculator(path_calc, atoms, device, dtype_str)
    atoms.calc = calculator
    
    torch.set_num_threads(num_cores)

    flag = True
    tel = 0
    while flag and tel <= num_retries:
        flag = False
        ff = create_forcefield(atoms, calculator)

        dof = yaff.CartesianDOF(ff, gpos_rms=1e-5, dpos_rms= 1e-3)
        hessian = yaff.estimate_hessian(dof, eps=1e-3)

        try:
            get_frequencies(hessian, atoms, printfreq = True)
        except AssertionError as e:
            print("Exception caught:")
            print(e)
            print("Change initial coordinates and retry " + str(tel))
            from parsl.data_provider.files import File
            outfile_md = File(str(inputs[0])[:-4] + "_md_"+str(tel)+".xyz")
            if tel == 0:
                md_input = [inputs[0], path_calc]
            else:
                md_input = [str(inputs[0])[:-4] + "_md_"+str(tel-1)+".xyz", path_calc]
            run_MD(inputs=md_input, outputs=[outfile_md], steps = 500, step = 500, temperature = 100, seed = tel, num_cores = num_cores, 
                   precision = precision, device = device)
            outfile_mdopt = File(str(inputs[0])[:-4] + "_mdopt"+str(tel)+".xyz")
            optimize(inputs=[outfile_md, path_calc], outputs=[outfile_mdopt], num_cores = num_cores, precision = precision, device = device)
            atoms = read(str(outfile_mdopt))
            atoms.calc = calculator
            tel += 1
            flag = True

    if tel > num_retries:
        raise ValueError("Retried " + str(num_retries) + " times, I will stop trying. Find a new minima to calculate the hessian")

    np.save(path_hessian, hessian)
    
    matrices = {
            'pos0': atoms.get_positions() *molmod.units.angstrom,
            'box0': np.array(atoms.get_cell()) *molmod.units.angstrom,
            'hessian': hessian,
            'GS_energy': atoms.get_potential_energy() *molmod.units.electronvolt,
            }
    model = EffectiveHarmonicModel(matrices)
    torch.set_default_dtype(torch.float32)
    model = model.float()
    pos = torch.tensor(matrices['pos0'])
    pos += torch.rand(pos.shape)
    box = torch.from_numpy(matrices['box0'])
    energy, forces = model(pos, box)

    scriptmodule = torch.jit.script(model, [pos, box])
    energy_, forces_ = scriptmodule(pos, box)
    assert torch.allclose(energy, energy_)
    assert torch.allclose(forces, forces_)
    scriptmodule.save(path_model)


def calc_hessian_autodiff(inputs=[], outputs=[], num_cores = 1, precision = "double", device = "cpu"):
    import torch
    import numpy as np
    from lib.utils import get_frequencies, EffectiveHarmonicModel
    from lib.ExtendedHessianOutput import get_model, check_sanity, wrap_model_extended
    from nequip.data import AtomicDataDict, AtomicData
    from pathlib import Path
    import molmod.units
    
    path_opt = str(inputs[0])
    atoms = read(path_opt) 
    path_calc  = str(inputs[1])
    path_config    = str(inputs[2])
    path_undep = str(inputs[3])
    path_hessian = str(outputs[0])
    path_model = str(outputs[1])
    assert path_opt[-4:] == ".xyz", "The first input must be a .xyz file"
    assert path_calc[-4:] == ".pth", "The second input must be a .pth file"
    assert path_config[-5:] == ".yaml", "The Third input must be a .yaml file"
    assert path_undep[-4:] == ".pth", "The fourth input must be a .pth file"
    assert precision == "double", "The precision must be double"
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(num_cores)
    
    # load and adapt model
    model, metadata, mapper = get_model(Path(path_undep), Path(path_config), device)
    model = model.double().eval()
    model = wrap_model_extended(model)
    
    # load data for atoms
    data = AtomicData.from_ase(atoms=atoms, r_max=metadata['r_max'])
    for k in AtomicDataDict.ALL_ENERGY_KEYS:
        if k in data:
            del data[k]
    data = AtomicData.to_AtomicDataDict(mapper(data).to(device=device))
    
    # evaluate model and get hessian
    out = model(data)
    hess = out['hessian'].detach().cpu().numpy()
    
    hess, hess_cart = check_sanity(hess, atoms)  

    hess_cart_noext_au = hess_cart[:-9,:-9]* molmod.units.electronvolt / molmod.units.angstrom**2

    np.save(path_hessian, hess_cart_noext_au)
    get_frequencies(hess_cart_noext_au, atoms, printfreq = True)

    #Create and write out hessian model
    matrices = {
            'pos0': atoms.get_positions() *molmod.units.angstrom,
            'box0': np.array(atoms.get_cell()) *molmod.units.angstrom,
            'hessian': hess_cart_noext_au,
            'GS_energy': atoms.get_potential_energy() *molmod.units.electronvolt,
            }
    model = EffectiveHarmonicModel(matrices)
    torch.set_default_dtype(torch.float32)
    model = model.float()
    pos = torch.tensor(matrices['pos0'])
    pos += torch.rand(pos.shape)
    box = torch.from_numpy(matrices['box0'])
    energy, forces = model(pos, box)

    scriptmodule = torch.jit.script(model, [pos, box])
    energy_, forces_ = scriptmodule(pos, box)
    assert torch.allclose(energy, energy_)
    assert torch.allclose(forces, forces_)
    scriptmodule.save(path_model)


def create_forcefield_recalc(atoms, calculator,  model = None, begin_frac_MLP = 0.0, end_frac_MLP = 1.0, begin_frac_bias = 0.0, end_frac_bias = 0.0, path_min_atoms = None):
    """Creates force field from ASE atoms instance"""
    system = yaff.System(
            numbers=atoms.get_atomic_numbers(), # dummy!
            pos=atoms.get_positions() * molmod.units.angstrom,
            rvecs=atoms.get_cell() * molmod.units.angstrom,
            )

    if path_min_atoms == None:
        assert begin_frac_MLP == 1.0, "you want a bias, but you do not have a reference structure"
        assert end_frac_MLP == 1.0, "you want a bias, but you do not have a reference structure"
        assert begin_frac_bias == 0.0, "you want a bias, but you do not have a reference structure"
        assert end_frac_bias == 0.0, "you want a bias, but you do not have a reference structure"
        min_atoms = None
        index_lst = None
    else:
        min_atoms = read(path_min_atoms)
        #No need to shift the positions anymore as the MD trajectories should already be translated in the correct way.
        
        #Get index_lst
        cation_name = get_cation_name(atoms)
        if cation_name == "FA":
            index_lst = Get_FAmol_lst(atoms)
        elif cation_name == "MA":
            index_lst = Get_MAmol_lst(atoms)

    part_lst = []
    if begin_frac_MLP != end_frac_MLP:
        part_diff = ForcePartdifference(system, atoms, model, calculator, end_frac_MLP - begin_frac_MLP)
        part_lst.append(part_diff)
    if begin_frac_bias != end_frac_bias:
        part_bias = get_part_bias(system, min_atoms, index_lst, end_frac_bias - begin_frac_bias)  #min_atoms is for the reference orientation
        part_lst.append(part_bias)

    return yaff.pes.ForceField(system, part_lst)


def run_recalc(inputs=[], outputs=[], begin_frac_MLP = 0.0, end_frac_MLP = 1.0, begin_frac_bias = 0.0, end_frac_bias = 0.0, seed = 0, num_cores = 1, 
               precision = "single", device = "cpu", trans_traj = False):
    import torch
    import numpy as np
    import h5py
    import yaff
    import os
    from ase.io import read, write
    from lib.utils import ExtXYZHook, from_xyz_to_h5, get_calculator, from_h5_to_atoms_traj, trans_traj_for_recalc, check_validity_inputs

    if not check_validity_inputs(inputs):
        for outp in outputs:
            f = open(str(outp), "w")
            f.write("failed output")
            f.close()
        return

    path_input = str(inputs[0]) 
    path_calc  = str(inputs[1])
    path_output  = str(outputs[0])
    assert path_input[-4:] == ".xyz" or path_input[-3:] == ".h5", "The first input must be a .xyz or h5 file"
    assert path_calc[-6:] == ".model" or path_calc[-4:] == ".pth", "The second input must be a .model file or .pth file"
    assert len(inputs) == 4, "The number of inputs must be 4"
    path_min_atoms = str(inputs[2])
    path_hess      = str(inputs[3])
    assert path_min_atoms[-4:] == ".xyz", "The third input must be a .xyz file"
    assert path_hess[-4:]      == ".pth", "The fourth input must be a .pth file"
    np.random.seed(seed)
    torch.set_num_threads(num_cores)


    atoms = read(path_min_atoms)
    dtype_str = 'float32'
    if precision == 'double':
        torch.set_default_dtype(torch.float64)
        dtype_str = 'float64'
    calculator = get_calculator(path_calc, atoms, device, dtype_str)
    atoms.calc = calculator

    if path_hess != None:
        Hess_model = torch.jit.load(path_hess)
    else:
        Hess_model = None
    
    torch.set_num_threads(num_cores)

    # create forcefield from atoms
    ff = create_forcefield_recalc(
        atoms, 
        calculator, 
        model = Hess_model, 
        begin_frac_MLP = begin_frac_MLP, 
        end_frac_MLP = end_frac_MLP, 
        begin_frac_bias = begin_frac_bias, 
        end_frac_bias = end_frac_bias,
        path_min_atoms = path_min_atoms, 
        )

    # hooks
    hooks = []
    loghook = yaff.sampling.trajectory.TrajScreenLog()
    hooks.append(loghook)
    if path_output[-4:] == '.xyz':
        hooks.append(ExtXYZHook(path_output, start=0, step=1, write_vel = False))
    elif path_output[-3:] == '.h5':
        h5file = h5py.File(path_output, 'w')
        hooks.append(yaff.HDF5Writer(h5file, step=1, start=0))

    if trans_traj:
        path_in_h5 = trans_traj_for_recalc(path_input, path_min_atoms)
    else:
        if path_input[-4:] == ".xyz":
            path_in_h5 = path_input[:-4]+".h5"
            from_xyz_to_h5(path_input, path_in_h5)
        else:
            path_in_h5 = path_input
    assert path_in_h5[-3:] == ".h5", "change format of the file to h5"
    reftrajcalc = yaff.sampling.trajectory.RefTrajectory(ff, path_in_h5, hooks=hooks)
    reftrajcalc.run()


def app_run_MD(device = "cpu", **kwargs):
    if device == "cpu":
        app_run_MD_cpu = bash_app_python(run_MD, precommand = "mpirun -wdir . -n 1 -rf configs/rankfiles/myrankfile", executors=['default_MD'])
        return app_run_MD_cpu(device = device, **kwargs)
    elif device == "cuda":
        app_run_MD_cuda = bash_app_python(run_MD, precommand = "mpirun -wdir . -n 1 -rf configs/rankfiles/myrankfile", executors=['cuda_default_MD'])
        return app_run_MD_cuda(device = device, **kwargs)

def app_run_REX(device = "cpu", num_replicas = 1, **kwargs):
    if device == "cpu":
        app_run_REX_cpu = bash_app_python(run_REX, precommand = "mpirun -wdir . -n "+str(num_replicas)+" -rf configs/rankfiles/myrankfile", executors=['default_replicas'])
        return app_run_REX_cpu(device = device, **kwargs)
    elif device == "cuda":
        app_run_REX_cuda = bash_app_python(run_REX, precommand = "mpirun -wdir . -n "+str(num_replicas)+" -rf configs/rankfiles/myrankfile", executors=['cuda_default_replicas'])
        return app_run_REX_cuda(device = device, **kwargs)
    
def app_optimize(device = "cpu", **kwargs):
    if device == "cpu":
        app_optimize_cpu = bash_app_python(optimize, precommand = "mpirun -wdir . -n 1 -rf configs/rankfiles/myrankfile", executors=['default_MD'])
        return app_optimize_cpu(device = device, **kwargs)
    elif device == "cuda":
        app_optimize_cuda = bash_app_python(optimize, precommand = "mpirun -wdir . -n 1 -rf configs/rankfiles/myrankfile", executors=['cuda_default_MD'])
        return app_optimize_cuda(device = device, **kwargs)

def app_calc_hessian(model_type = True, device = "cpu", **kwargs):
    if device == "cpu":
        if model_type.lower() == "nequip":
            app_calc_hessian_cpu = bash_app_python(calc_hessian_autodiff, precommand = "mpirun -wdir . -n 1 -rf configs/rankfiles/myrankfile", executors=['default_MD'])
        elif model_type.lower() == "mace":
            app_calc_hessian_cpu = bash_app_python(calc_hessian, precommand = "mpirun -wdir . -n 1 -rf configs/rankfiles/myrankfile", executors=['default_MD'])
        else:
            NotImplementedError
        return app_calc_hessian_cpu(device = device, **kwargs)
    elif device == "cuda":
        if model_type.lower() == "nequip":
            app_calc_hessian_cuda = bash_app_python(calc_hessian_autodiff, precommand = "mpirun -wdir . -n 1 -rf configs/rankfiles/myrankfile", executors=['cuda_default_MD'])
        elif model_type.lower() == "mace":
            app_calc_hessian_cuda = bash_app_python(calc_hessian, precommand = "mpirun -wdir . -n 1 -rf configs/rankfiles/myrankfile", executors=['cuda_default_MD'])
        else:
            NotImplementedError
        return app_calc_hessian_cuda(device = device, **kwargs)

def app_run_recalc(device = "cpu", **kwargs):
    if device == "cpu":
        app_run_recalc_cpu = bash_app_python(run_recalc, precommand = "mpirun -wdir . -n 1 -rf configs/rankfiles/myrankfile", executors=['default_MD'])
        return app_run_recalc_cpu(device = device, **kwargs)
    elif device == "cuda":
        app_run_recalc_cuda = bash_app_python(run_recalc, precommand = "mpirun -wdir . -n 1 -rf configs/rankfiles/myrankfile", executors=['cuda_default_MD'])
        return app_run_recalc_cuda(device = device, **kwargs)
