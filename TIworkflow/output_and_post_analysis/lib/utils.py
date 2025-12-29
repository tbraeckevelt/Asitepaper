from ase.io import read, iread, write
import h5py
import molmod.units
from ase import Atoms
import ase.units
import numpy as np
from ase.cell import Cell
import matplotlib.pyplot as plt
import yaff
from pathlib import Path
from ase.stress import voigt_6_to_full_3x3_stress
import torch
import os

from lib.DefineCV import CVoneFAorientation, CVoneMAorientation, CVoneFAgyration, CVoneMAgyration, Get_FAmol_lst, Get_MAmol_lst, FA_calc_CV_orient_value
from yaff import ForcePartBias, UpperWallBias, System
from yaff.conversion.xyz import xyz_to_hdf5
from mace.calculators import MACECalculator
from nequip.ase.nequip_calculator import NequIPCalculator


def from_h5_to_atoms_traj(path_traj, calib_step = 0, samp_freq = 1, get_last = False, get_frame = None, less_data = False):
    f = h5py.File(path_traj, 'r')
    at_numb = f['system']['numbers']
    if get_frame != None:
        frame = calib_step + get_frame * samp_freq
        pos_A = f['trajectory']['pos'][frame,:,:] / molmod.units.angstrom
        cell_A = f['trajectory']['cell'][frame,:,:] / molmod.units.angstrom
        vol_A3 = f['trajectory']['volume'][frame] / molmod.units.angstrom ** 3
        energy_eV = f['trajectory']['epot'][frame] / molmod.units.electronvolt
        atoms = Atoms(
                numbers=at_numb,
                positions=pos_A,
                pbc=True,
                cell=cell_A,
            )
        atoms.info['energy'] = energy_eV
        if not less_data:
            vel_ase = f['trajectory']['vel'][frame,:,:] * molmod.units.femtosecond / (ase.units.fs * molmod.units.angstrom)
            #forces_eVA = - f['trajectory']['gpos_contribs'][frame,0,:,:] * molmod.units.angstrom / molmod.units.electronvolt # forces = -gpos
            vtens_eV = f['trajectory']['vtens'][frame,:,:] / molmod.units.electronvolt
            stresses_eVA3 = vtens_eV / vol_A3
            atoms.set_velocities(vel_ase)
            #atoms.arrays['forces'] = forces_eVA            #Need gposcontribuation as state item
            atoms.info['stress'] = stresses_eVA3
        return atoms
    elif get_last:
        pos_A = f['trajectory']['pos'][-1,:,:] / molmod.units.angstrom
        cell_A = f['trajectory']['cell'][-1,:,:] / molmod.units.angstrom
        vol_A3 = f['trajectory']['volume'][-1] / molmod.units.angstrom ** 3
        energy_eV = f['trajectory']['epot'][-1] / molmod.units.electronvolt
        atoms = Atoms(
                numbers=at_numb,
                positions=pos_A,
                pbc=True,
                cell=cell_A,
            )
        atoms.info['energy'] = energy_eV
        if not less_data:
            vel_ase = f['trajectory']['vel'][-1,:,:] * molmod.units.femtosecond / (ase.units.fs * molmod.units.angstrom)
            #forces_eVA = - f['trajectory']['gpos_contribs'][-1,0,:,:] * molmod.units.angstrom / molmod.units.electronvolt # forces = -gpos
            vtens_eV = f['trajectory']['vtens'][-1,:,:] / molmod.units.electronvolt
            stresses_eVA3 = vtens_eV / vol_A3
            atoms.set_velocities(vel_ase)
            #atoms.arrays['forces'] = forces_eVA            #Need gposcontribuation as state item
            atoms.info['stress'] = stresses_eVA3
        return [atoms]
    else:
        traj = []
        for frame in range(calib_step, f['trajectory']['cell'].shape[0], samp_freq):
            pos_A = f['trajectory']['pos'][frame,:,:] / molmod.units.angstrom
            cell_A = f['trajectory']['cell'][frame,:,:] / molmod.units.angstrom
            vol_A3 = f['trajectory']['volume'][frame] / molmod.units.angstrom ** 3
            energy_eV = f['trajectory']['epot'][frame] / molmod.units.electronvolt
            atoms = Atoms(
                    numbers=at_numb,
                    positions=pos_A,
                    pbc=True,
                    cell=cell_A,
                )
            atoms.info['energy'] = energy_eV
            if not less_data:
                vel_ase = f['trajectory']['vel'][frame,:,:] * molmod.units.femtosecond / (ase.units.fs * molmod.units.angstrom)
                #forces_eVA = - f['trajectory']['gpos_contribs'][frame,0,:,:] * molmod.units.angstrom / molmod.units.electronvolt # forces = -gpos
                vtens_eV = f['trajectory']['vtens'][frame,:,:] / molmod.units.electronvolt
                stresses_eVA3 = vtens_eV / vol_A3
                atoms.set_velocities(vel_ase)
                #atoms.arrays['forces'] = forces_eVA       #Need gposcontribuation as state item
                atoms.info['stress'] = stresses_eVA3
            traj.append(atoms)
        return traj

def from_xyz_to_h5(path_xyz, path_h5):
    from filelock import Timeout, FileLock
    lock = FileLock(path_h5+".lock")
    with lock:
        create_h5 = True
        if os.path.exists(path_h5) == True: 
            try:
                f = h5py.File(path_h5, 'r')
                # File is readable
                if len(f.keys()) == 0:
                    # The file has no datasets
                    create_h5 = True
                else:
                    create_h5 = False
            except:
                #File is not readable, so create it again
                create_h5 = True
        if create_h5:
            cell_traj_lst = []
            epot_lst = []
            vel_lst = []
            volume_lst = []
            vtens_lst = []
            for ats in iread(path_xyz):
                cell_traj_lst.append(ats.get_cell()[:] * molmod.units.angstrom)
                epot_lst.append(ats.get_potential_energy() * molmod.units.electronvolt)
                vel_lst.append(ats.get_velocities() * ase.units.fs * molmod.units.angstrom / molmod.units.femtosecond)
                volume_lst.append(ats.get_volume() * molmod.units.angstrom ** 3)
                vtens_lst.append(ats.get_stress(voigt= False) * ats.get_volume() * molmod.units.electronvolt)
            cell_traj = np.asarray(cell_traj_lst, dtype='float')
            epot_traj = np.asarray(epot_lst, dtype='float')
            vel_traj = np.asarray(vel_lst, dtype='float')
            volume_traj = np.asarray(volume_lst, dtype='float')
            vtens_traj = np.asarray(vtens_lst, dtype='float')

            system = System.from_file(path_xyz)
            with h5py.File(path_h5, mode='w') as f:
                system.to_hdf5(f)
                xyz_to_hdf5(f, path_xyz)
                f.create_dataset('trajectory/cell', data = cell_traj)
                f.create_dataset('trajectory/epot', data = epot_traj)
                f.create_dataset('trajectory/vel', data = vel_traj)
                f.create_dataset('trajectory/volume', data = volume_traj)
                f.create_dataset('trajectory/vtens', data = vtens_traj)

def trans_traj_for_recalc(path_in, path_min_atoms):
    if path_in[-4:] == ".xyz":
        traj = read(path_in, index=":")
        path_xyz = path_in[:-4] + "_trans.xyz"
    elif path_in[-3:] == ".h5":
        traj = from_h5_to_atoms_traj(path_in)
        path_xyz = path_in[:-3] + "_trans.xyz"
    path_h5 = path_xyz[:-4] + ".h5"
    if os.path.exists(path_h5) == False:
        min_atoms = read(path_min_atoms)
        for atoms in traj:
            #Make sure that your min_struc and atoms object have the same center of mass and no atoms are at a periodic image compared to the other
            atoms.positions -= np.average(atoms.positions - min_atoms.get_positions(), axis = 0)
            inv_cell = np.linalg.inv(atoms.get_cell()[:])
            frac_dpos = np.dot(atoms.get_positions() - min_atoms.get_positions(), inv_cell)
            atoms.positions -= np.dot(np.round(frac_dpos), atoms.get_cell()[:])
            #Make sure that cations are whole molecules
            cation_name = get_cation_name(atoms)
            if cation_name == "FA":
                index_lst = Get_FAmol_lst(atoms)
                '''   We cannot do this switch because not all H are equivalent for the hessian model!!!
                #system invariant with respect to permutations of Nitrogen and hydrogen, but bias is not. Switch to representation with lowest bias energy
                new_pos = atoms.get_positions()
                masses = min_atoms.get_masses()
                for index_dct in index_lst:
                    swi_pos = new_pos.copy()
                    swi_pos[[index_dct["N1"], index_dct["N2"]],:] = swi_pos[[index_dct["N2"], index_dct["N1"]],:]
                    swi_pos[[index_dct["H1_N1"], index_dct["H1_N2"]],:] = swi_pos[[index_dct["H1_N2"], index_dct["H1_N1"]],:]
                    swi_pos[[index_dct["H2_N1"], index_dct["H2_N2"]],:] = swi_pos[[index_dct["H2_N2"], index_dct["H2_N1"]],:]
                    nor_val = FA_calc_CV_orient_value(new_pos, min_atoms.get_positions(), masses, index_dct)
                    swi_val = FA_calc_CV_orient_value(swi_pos, min_atoms.get_positions(), masses, index_dct)
                    if nor_val > swi_val:
                        new_pos = swi_pos
                atoms.set_positions(new_pos)
                '''
            elif cation_name == "MA":
                index_lst = Get_MAmol_lst(atoms)
            atoms.positions -= np.average(atoms.positions - min_atoms.get_positions(), axis = 0)
        write(path_xyz, traj)
        from_xyz_to_h5(path_xyz, path_h5)
        os.remove(path_xyz)
    return path_h5

def Running_Average(col,window):
    colavg=np.zeros((len(col)-window+1,1))
    for i in range(len(col)-window+1):
        colavg[i]=np.average(col[i:(i+window)])
    return colavg


def get_timestep(atoms):
    #Very crude, but very safe heuristic to set the timestep
    mass_min = min(atoms.get_masses())
    if mass_min < 5.0:
        return 0.5 #in femteseconds
    elif mass_min < 50.0:
        return 1.0 #in femteseconds
    else:
        return 2.0 #in femteseconds


class ForcePartdifference(yaff.pes.ForcePart):
    """Difference between ASE and torch ForceParts"""
    #This force part is used to determine the delta_F for TI

    def __init__(self, system, atoms, model, calculator, frac):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object

        atoms : ase.Atoms
            atoms object with calculator included.

        """
        yaff.pes.ForcePart.__init__(self, 'diff', system)
        self.system = system # store system to obtain current pos and box
        self.atoms = atoms
        self.model = model
        self.calculator = calculator
        self.frac = frac
        
    def _internal_compute(self, gpos=None, vtens=None):
        #torch
        pos = torch.tensor(self.system.pos)
        box = torch.tensor(self.system.cell._get_rvecs())
        energy_torch, forces_torch = self.model(pos, box)
        assert forces_torch is not None
        energy_torch = energy_torch.detach().numpy().item()
        
        #ase
        self.atoms.set_positions(self.system.pos / molmod.units.angstrom)
        self.atoms.set_cell(Cell(self.system.cell._get_rvecs() / molmod.units.angstrom))
        energy_ase = self.atoms.get_potential_energy() * molmod.units.electronvolt
        
        if gpos is not None:
            forces_torch = forces_torch.detach().numpy()
            forces_ase = self.atoms.get_forces() * molmod.units.electronvolt / molmod.units.angstrom
            gpos[:] = -self.frac * (forces_ase - forces_torch) 
        return self.frac* (energy_ase - energy_torch)

class ForceThresholdExceededException(Exception):
    pass

class ForcePartaddition(yaff.pes.ForcePart):
    """Linear combination of ASE and torch ForceParts"""

    def __init__(self, system, atoms, model, calculator, frac, force_threshold_eVA=30):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object

        atoms : ase.Atoms
            atoms object with calculator included.

        """
        yaff.pes.ForcePart.__init__(self, 'add', system)
        self.system = system # store system to obtain current pos and box
        self.atoms = atoms
        self.model = model
        self.calculator = calculator
        self.frac = frac
        self.force_threshold_eVA = force_threshold_eVA
        
    def _internal_compute(self, gpos=None, vtens=None):
        #torch
        pos = torch.tensor(self.system.pos )
        box = torch.tensor(self.system.cell._get_rvecs())
        energy_torch, forces_torch = self.model(pos, box)
        assert forces_torch is not None
        energy_torch = energy_torch.detach().numpy().item()
        
        #ase
        self.atoms.set_positions(self.system.pos / molmod.units.angstrom)
        self.atoms.set_cell(Cell(self.system.cell._get_rvecs() / molmod.units.angstrom))
        energy_ase_au = self.atoms.get_potential_energy() * molmod.units.electronvolt
        
        if gpos is not None:
            forces_torch = forces_torch.detach().numpy()
            forces_ase = self.atoms.get_forces()
            self.check_threshold(forces_ase)           #Only check forces of the MLP, because those forces can go wrong due to not sufficient training data
            forces_ase_au = forces_ase * molmod.units.electronvolt / molmod.units.angstrom
            gpos[:] = - (1-self.frac) * forces_torch - self.frac * forces_ase_au 
        return (1-self.frac)*energy_torch + self.frac* energy_ase_au
    
    def check_threshold(self, forces):   #Forces in eV/Ang
        max_force = np.max(np.linalg.norm(forces, axis=1))
        index = np.argmax(np.linalg.norm(forces, axis=1))
        if max_force > self.force_threshold_eVA:
            raise ForceThresholdExceededException(
                    'Max force exceeded: {} eV/A by atom index {}'.format(max_force, index),
                    )


class ForcePartASE(yaff.pes.ForcePart):
    """YAFF Wrapper around an ASE calculator"""

    def __init__(self, system, atoms, calculator, force_threshold_eVA=30):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object

        atoms : ase.Atoms
            atoms object with calculator included.

        """
        yaff.pes.ForcePart.__init__(self, 'ase', system)
        self.system = system # store system to obtain current pos and box
        self.atoms  = atoms
        self.calculator = calculator
        self.force_threshold_eVA = force_threshold_eVA

    def _internal_compute(self, gpos=None, vtens=None):
        self.atoms.set_positions(self.system.pos / molmod.units.angstrom)
        self.atoms.set_cell(Cell(self.system.cell._get_rvecs() / molmod.units.angstrom))
        energy = self.atoms.get_potential_energy() * molmod.units.electronvolt
        if gpos is not None:
            forces = self.atoms.get_forces()
            self.check_threshold(forces)
            gpos[:] = -forces * molmod.units.electronvolt / molmod.units.angstrom
        if vtens is not None:
            volume = np.linalg.det(self.atoms.get_cell())
            stress = voigt_6_to_full_3x3_stress(self.atoms.get_stress())
            vtens[:] = volume * stress * molmod.units.electronvolt
        return energy
    
    def check_threshold(self, forces):   #Forces in eV/Ang
        max_force = np.max(np.linalg.norm(forces, axis=1))
        index = np.argmax(np.linalg.norm(forces, axis=1))
        if max_force > self.force_threshold_eVA:
            raise ForceThresholdExceededException(
                    'Max force exceeded: {} eV/A by atom index {}'.format(max_force, index),
                    )

class ForcePartTorch(yaff.pes.ForcePart):
    """YAFF Wrapper around torch.ScriptModule"""

    def __init__(self, system, model):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object

        model : torch.ScriptModule
            torch module

        """
        yaff.pes.ForcePart.__init__(self, 'torch', system)
        self.system = system # store system to obtain current pos and box
        self.model  = model

    def _internal_compute(self, gpos=None, vtens=None):
        pos = torch.tensor(self.system.pos)
        box = torch.tensor(self.system.cell._get_rvecs())
        energy, forces = self.model(pos, box)
        assert forces is not None
        energy = energy.detach().numpy().item()
        forces = forces.detach().numpy()
        if gpos is not None:
            gpos[:] = -forces
        return energy

def get_part_bias(system, min_atoms, index_lst, frac):
    """Returns a ``ForcePartBias`` instance corresponding to the bias energy in mtd"""
    part = ForcePartBias(system)
    fc_or = frac * 800 *molmod.units.kjmol #Just a guess
    fc_gy = frac * 200 *molmod.units.kjmol #Just a guess
    rv = 0.5
    CV_lst=[]
    bias_lst=[]
    cation_name = get_cation_name(min_atoms)
    for index_dct in index_lst:
        if cation_name == "FA":
            CV_lst.append(CVoneFAorientation(system, min_atoms.get_positions() * molmod.units.angstrom, index_dct))
            CV_lst.append(CVoneFAgyration(system, min_atoms.get_positions() * molmod.units.angstrom, index_dct))
        elif cation_name == "MA":
            CV_lst.append(CVoneMAorientation(system, min_atoms.get_positions() * molmod.units.angstrom, index_dct))
            CV_lst.append(CVoneMAgyration(system, min_atoms.get_positions() * molmod.units.angstrom, index_dct))
        bias_lst.append(UpperWallBias(fc_or, rv, CV_lst[-2]))
        bias_lst.append(UpperWallBias(fc_gy, rv, CV_lst[-1]))
        part.add_term(bias_lst[-2])
        part.add_term(bias_lst[-1])
    return part

def create_forcefield(atoms, calculator,  model = None, frac_MLP = 1.0, frac_bias = 0.0, path_min_atoms = None):
    """Creates force field from ASE atoms instance"""

    if path_min_atoms == None:
        assert frac_bias == 0.0, "you want a bias, but you do not have a reference structure"
        min_atoms = None
        index_lst = None
    else:
        min_atoms = read(path_min_atoms, parallel=False)

        #Make sure that your min_struc and atoms object have the same center of mass and no atoms are at a periodic image compared to the other
        atoms.positions -= np.average(atoms.positions - min_atoms.get_positions(), axis = 0)
        inv_cell = np.linalg.inv(atoms.get_cell()[:])
        frac_dpos = np.dot(atoms.get_positions() - min_atoms.get_positions(), inv_cell)
        atoms.positions -= np.dot(np.round(frac_dpos), atoms.get_cell()[:])
        
        #Make sure that cations are whole molecules and get index_lst
        cation_name = get_cation_name(atoms)
        if cation_name == "FA":
            index_lst = Get_FAmol_lst(atoms)
        elif cation_name == "MA":
            index_lst = Get_MAmol_lst(atoms)
        atoms.positions -= np.average(atoms.positions - min_atoms.get_positions(), axis = 0)

    system = yaff.System(
            numbers=atoms.get_atomic_numbers(),
            pos=atoms.get_positions() * molmod.units.angstrom,
            rvecs=atoms.get_cell() * molmod.units.angstrom,
            )
    system.set_standard_masses()

    #if frac_MLP == 0.0:                                      #commented because we always to check the validity of the MLP
    #    part_torch = ForcePartTorch(system, model)
    #    ff = yaff.pes.ForceField(system, [part_torch])
    if frac_MLP == 1.0:
        part_ase = ForcePartASE(system, atoms, calculator)
        ff = yaff.pes.ForceField(system, [part_ase])
    else:
        part_add = ForcePartaddition(system, atoms, model, calculator, frac_MLP)
        ff = yaff.pes.ForceField(system, [part_add])
    if frac_bias != 0.0:
        part_bias = get_part_bias(system, min_atoms, index_lst, frac_bias)  #min_atoms is for the reference orientation
        ff.add_part(part_bias)
    return ff
    

class ExtXYZHook(yaff.sampling.iterative.Hook):

    def __init__(self, path_xyz, step=1, start=0, append = False, write_vel = True):
        super().__init__(step=step, start=start)
        if Path(path_xyz).exists() and (append == False):
            Path(path_xyz).unlink() # remove if exists
        self.path_xyz = path_xyz
        self.atoms = None
        self.write_vel = write_vel

    def init(self, iterative):
        self.atoms = Atoms(
                numbers=iterative.ff.system.numbers.copy(),
                positions=iterative.ff.system.pos / molmod.units.angstrom,
                cell=iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom,
                pbc=True,
                )

    def pre(self, iterative):
        pass

    def post(self, iterative):
        pass

    def __call__(self, iterative):
        if self.atoms is None:
            self.init(iterative)
        self.atoms.set_positions(iterative.ff.system.pos / molmod.units.angstrom)
        cell = iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom
        self.atoms.set_cell(cell)
        self.atoms.arrays['forces'] = -iterative.ff.gpos * molmod.units.angstrom / molmod.units.electronvolt
        self.atoms.info['energy'] = iterative.ff.energy / molmod.units.electronvolt
        volume = np.linalg.det(cell)
        self.atoms.info['stress'] = iterative.ff.vtens / (molmod.units.electronvolt * volume)
        if self.write_vel:
            self.atoms.set_velocities(iterative.vel * molmod.units.femtosecond / (ase.units.fs * molmod.units.angstrom) )
        write(self.path_xyz, self.atoms, parallel=False, append=True)

class EffectiveHarmonicModel(torch.nn.Module):
    """Represents a second-order approximation to the free energy"""
    # uses atomic units !!!

    def __init__(self, matrices):
        super().__init__()
        self.pos0      = torch.tensor(matrices['pos0']) # copies data!
        self.box0      = torch.tensor(matrices['box0']) # copies data!
        self.hessian   = torch.tensor(matrices['hessian'])
        self.GS_Energy = torch.tensor(matrices['GS_energy'])

    def forward(self, pos: torch.Tensor, box: torch.Tensor):
        pos.requires_grad_(True)

        dpos  = pos - self.pos0
        dpos  = dpos - torch.mean(dpos, dim = 0, keepdim = True)
        
        dpos  = dpos.reshape(-1)
        dfrac = torch.matmul(dpos.view(-1, 3), torch.linalg.inv(box))

        # sanity checks
        assert torch.all(torch.isclose(self.box0, box))
        
        # outputs list of Optional[torch.Tensor]
        energy = dpos @ self.hessian @ dpos  / 2

        #Do we need to take a translated image of certain atoms to get lower energies?
        if torch.all(torch.abs(dfrac) > 0.5):
            dpos2  = pos - self.pos0
            dpos2  = dpos2 - torch.mean(dpos2, dim = 0, keepdim = True)
            dpos2  = dpos2 - torch.matmul(torch.round(dfrac), box)
            dpos2  = dpos2 - torch.mean(dpos2, dim = 0, keepdim = True)
            dpos2  = dpos2.reshape(-1)
            energy2 = dpos2 @ self.hessian @ dpos2  / 2
            if energy2 < energy:
                raise ValueError("energy of tranlated atoms is lower ("+str(energy2)+"<"+str(energy)+"), did some atoms move with a lattice vector? Check your methodology")

        grads = torch.autograd.grad([energy], [pos])
        gradpos = grads[0] # type Optional[torch.Tensor] -> Tensor | None
        if gradpos is not None: # tells torch that this is Tensor, not None
            forces = torch.negative(gradpos)
        else:
            forces = None
        return energy + self.GS_Energy, forces   #Also added the GS energy to the final energy to easily compare to MLP/DFT energies


def analyze_permutations(permutations, path_out, num_tem = 32, tem_min = 100, tem_max = 600, print_nr_runs = 8):
    
    MC_steps = np.shape(permutations)[0]
    num_tem = np.shape(permutations)[1]

    y_max = np.log(tem_max/tem_min)
    y_np = np.arange(num_tem)*y_max/(num_tem-1)
    tem_np = np.exp(y_np)*tem_min

    print("Total number of MC steps: "+ str(MC_steps))

    print("expected frequency for a replica to be at a specific temperature: "+ str(MC_steps/num_tem))
    exp_freq = MC_steps/num_tem

    fig, ax = plt.subplots()
    random_permut = np.random.permutation(num_tem)
    plot_freq = 1

    #Print some warnings at the end
    warning_lst = []

    for run in random_permut[:print_nr_runs]:
        locations = []
        jumps_lst = []
        prog = 0
        for i in range(0,MC_steps,plot_freq):
            for loc in range(num_tem):
                if run == permutations[i, loc]:
                    locations.append(tem_np[loc])
                    if len(jumps_lst) < 1.5:
                        jumps_lst.append(loc)
                    elif loc != jumps_lst[-1]:
                        prog += np.abs(loc-jumps_lst[-2])
                        jumps_lst.append(loc)
        prog /= (len(jumps_lst) -1)
        print("Number of jumps: " + str(len(jumps_lst) -1))
        print("progression: " + str(prog))

        plt.plot(range(0,MC_steps,plot_freq),locations, label = str(run))
        bins, hist_run = np.unique(locations, return_counts=True)
        print("This for run: "+ str(run))
        print(hist_run)
        print("Max: "+ str(np.amax(hist_run)))

        not_incl = []
        for tem in tem_np:
            if int(tem) not in bins.astype(int):
                not_incl.append(tem)
        print("List of not included temperatures: "+ str(not_incl))
        print("Total number of zeros: "+ str(len(not_incl))+"\n")

        if prog < 0.75:
            warning_lst.append("progression for replica "+str(run)+" is quite small ("+str(prog)+") maybe increase the number of MD steps")
        if num_tem - len(not_incl) < 12.0:
            warning_lst.append("replica  "+str(run)+" did not swap to many different temperatures ("+str(num_tem-len(not_incl))+"), try to simulate for longer times")
        if np.amax(hist_run)/exp_freq > 12.0:
            warning_lst.append("replica  "+str(run)+" has mainly occupied a specific temperature, namely "+str(np.amax(hist_run)/exp_freq)+" times more thane expected")


    print("-----------------------------------------------------------------------------------------\n")

    for tem in [0, int(num_tem/2), num_tem-1]:
        bins, hist_tem = np.unique(permutations[::plot_freq, tem], return_counts=True)
        print("This is for temperature: " + str(tem_np[tem]))
        print(hist_tem)
        print("Max: "+ str(np.max(hist_tem)))

        not_incl = []
        for run in range(num_tem):
            if run not in bins.astype(int):
                not_incl.append(run)
        print("List of not included runs: "+ str(not_incl))
        print("Total number of zeros: "+ str(len(not_incl))+"\n")
        
        if num_tem - len(not_incl) < 12:
            warning_lst.append("at temperature  "+str(tem)+" not many replica's visisted ("+str(num_tem-len(not_incl))+"), try to simulate for longer times")
        if np.amax(hist_tem)/exp_freq > 12.0:
            warning_lst.append("at temperature  "+str(tem)+" at least one replica was very present, "+str(np.amax(hist_tem)/exp_freq)+" times more thane expected")

        
    plt.xlabel("MC_step")
    plt.ylabel("Temp")
    plt.yscale("log")
    plt.legend()
    plt.savefig(path_out)
    plt.close()

    print("-----------------------------------------------------------------------------------------\n")

    for warning in warning_lst:
        print("WARNING: "+ warning)


def get_probable_cell_at_average_vol(traj):

    cell_lst = []
    for atoms in traj:
        cell_lst.append(atoms.get_cell().cellpar())
    cell_np = np.asarray(cell_lst)
    cell_np_norm = np.zeros(cell_np.shape)
    for i in range(6):
        par_min = np.min(cell_np[:,i])
        par_max = np.max(cell_np[:,i])
        for j in range(len(traj)):
            cell_np_norm[j,i] = (cell_np[j,i] - par_min)/(par_max-par_min)

    flag = True
    atoms_prob_cell = None
    bin_size = 25   #Arbitrary starting bin size
    while flag:
        max_ind = np.zeros(6, dtype=int)
        bin = np.arange(0, 1.001, 1.0/bin_size)
        hist, bins = np.histogramdd(cell_np_norm[:,:3], bins = [bin] * 3)
        if np.max(hist) > 10:   #Otherwise the histogram is to noisy to be reliable
            for i in range(hist.shape[0]):
                for j in range(hist.shape[1]):
                    for k in range(hist.shape[2]):
                        if hist[i,j,k] == np.max(hist):
                            print("ind", i, j, k, "counts", hist[i,j,k])
                            max_ind[0], max_ind[1], max_ind[2] = i, j, k
            hist, bins = np.histogramdd(cell_np_norm[:,3:], bins = [bin] * 3)
            for i in range(hist.shape[0]):
                for j in range(hist.shape[1]):
                    for k in range(hist.shape[2]):            
                        if hist[i,j,k] == np.max(hist):
                            print("ind", i, j, k, "counts", hist[i,j,k])
                            max_ind[3], max_ind[4], max_ind[5] = i, j, k
            for i in range(len(cell_np)):
                if bin[max_ind[0]] < cell_np_norm[i,0] and bin[max_ind[0]+1] > cell_np_norm[i,0]:
                    if bin[max_ind[1]] < cell_np_norm[i,1] and bin[max_ind[1]+1] > cell_np_norm[i,1]:
                        if bin[max_ind[2]] < cell_np_norm[i,2] and bin[max_ind[2]+1] > cell_np_norm[i,2]:
                            if bin[max_ind[3]] < cell_np_norm[i,3] and bin[max_ind[3]+1] > cell_np_norm[i,3]:
                                if bin[max_ind[4]] < cell_np_norm[i,4] and bin[max_ind[4]+1] > cell_np_norm[i,4]:
                                    if bin[max_ind[5]] < cell_np_norm[i,5] and bin[max_ind[5]+1] > cell_np_norm[i,5]:
                                        atoms_prob_cell = traj[i]
        if atoms_prob_cell is not None:
            flag = False
        else:
            bin_size -=1
        assert bin_size != 0, "Something strange happenned, no structure can be found even if the two histograms are only one bin"
    
    prob_cell = atoms_prob_cell.get_cell()
    prob_cell_vol = atoms_prob_cell.get_volume()
    for i, atoms in enumerate(traj):
        if i == 0:
            ave_vol  = atoms.get_volume()/len(traj)
        else:
            ave_vol += atoms.get_volume()/len(traj)
    
    return prob_cell* (ave_vol/prob_cell_vol)**(1.0/3.0)

def get_frequencies(hessian, atoms_opt, printfreq = False):

    dof = np.shape(hessian)[0]
    mass_lst=[]
    for at in atoms_opt:
        for i in range(3):
            mass_lst.append(at.mass*molmod.units.amu)

    mass_hes = np.zeros([dof,dof])
    for i in range(dof):
        for j in range(dof):
            mass_hes[i,j] = hessian[i,j]/np.sqrt(mass_lst[i]*mass_lst[j])
            
    omega_au = np.sign(np.linalg.eig(mass_hes)[0]) *np.sqrt([np.abs(i) for i in np.linalg.eig(mass_hes)[0]])
    freq_THz = np.sort(omega_au * molmod.units.second / (2*np.pi * 10**12) )

    if printfreq:
        print("frequencies [THz]")
        print(freq_THz)
    assert np.abs(freq_THz[0]) < 0.0001, "The amplitude of the lowest frequency is too high to be seen as a translational move"

    return omega_au / (2*np.pi)


def get_cation_name(atoms):
    num_C = 0
    num_N = 0
    num_H = 0
    num_Cs= 0
    num_Pb= 0
    for at in atoms:
        if at.symbol == "Cs":
            num_Cs+= 1
        elif at.symbol == "C":
            num_C +=1
        elif at.symbol == "N":
            num_N +=1
        elif at.symbol == "H":
            num_H +=1
        elif at.symbol == "Pb":
            num_Pb+=1
    
    if num_Pb == num_Cs:
        return "Cs"
    elif 5*num_Pb == num_H:
        return "FA"
    elif 6*num_Pb == num_H:
        return "MA"
    else:
        raise NotImplementedError()
    
def get_calculator(path_calc, atoms, device, dtype_str):
    if path_calc[-6:] == ".model":
        calculator = MACECalculator(path_calc, device=device, default_dtype = dtype_str)
        if dtype_str == 'float64':
            calculator.model.double()                    #This is because MACE has a bug in converting the attributes!
    elif path_calc[-4:] == ".pth":
        if get_cation_name(atoms) == "Cs":
            calculator = NequIPCalculator.from_deployed_model(
                model_path = path_calc, 
                species_to_type_name = {"Cs": "Cs",
                                        "Pb": "Pb",
                                        "I" : "I" },
                device='cpu',
                )
        elif get_cation_name(atoms) == "FA" or get_cation_name(atoms) == "MA":
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return calculator


def check_validity_inputs(inputs):
    for inp in inputs:
        try:
            f = open(str(inp), "r")
            line = f.readline()
            f.close()
            if line == "failed output":
                return False
        except:
            pass
    return True
