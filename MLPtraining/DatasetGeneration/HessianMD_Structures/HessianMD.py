import torch
import numpy as np
import h5py
import yaff
import molmod
import ase.units
from ase.io import read, write
from ase import Atoms
from pathlib import Path
import sys

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


def get_timestep(atoms):
    #Very crude, but very safe heuristic to set the timestep
    mass_min = min(atoms.get_masses())
    if mass_min < 5.0:
        return 0.5 #in femteseconds
    elif mass_min < 50.0:
        return 1.0 #in femteseconds
    else:
        return 2.0 #in femteseconds

def create_forcefield(atoms, model = None, path_min_atoms = None):
    """Creates force field from ASE atoms instance"""
    min_atoms = read(path_min_atoms, parallel=False)
    #Make sure that your min_struc and atoms object have the same center of mass and no atoms are at a periodic image compared to the other
    atoms.positions -= np.average(atoms.positions - min_atoms.get_positions(), axis = 0)
    inv_cell = np.linalg.inv(atoms.get_cell()[:])
    frac_dpos = np.dot(atoms.get_positions() - min_atoms.get_positions(), inv_cell)
    atoms.positions -= np.dot(np.round(frac_dpos), atoms.get_cell()[:]) 
    atoms.positions -= np.average(atoms.positions - min_atoms.get_positions(), axis = 0)
    system = yaff.System(
            numbers=atoms.get_atomic_numbers(),
            pos=atoms.get_positions() * molmod.units.angstrom,
            rvecs=atoms.get_cell() * molmod.units.angstrom,
            )
    system.set_standard_masses()
    part_torch = ForcePartTorch(system, model)
    ff = yaff.pes.ForceField(system, [part_torch])
    return ff

def run_MD(path_atoms, path_min_atoms, path_hess, path_traj, steps = 10000, temperature = None, seed = 0):

    np.random.seed(seed)
    torch.set_num_threads(1)
    atoms = read(path_atoms)
    timestep = get_timestep(atoms)
    Hess_model = torch.jit.load(path_hess)

    # create forcefield from atoms
    ff = create_forcefield(atoms, model = Hess_model, path_min_atoms = path_min_atoms)

    #Hooks
    hooks = []
    if path_traj[-3:] == '.h5':
        h5file = h5py.File(path_traj, 'w')
        hooks.append(yaff.HDF5Writer(h5file, step=1, start=steps))   #Only get the last structure
    elif path_traj[-4:] == '.xyz':
        hooks.append(ExtXYZHook(path_traj, start=steps, step=1))     #Only get the last structure
    hooks.append(yaff.VerletScreenLog(step=1, start=steps))          #Only get the last structure

    # temperature / pressure control
    if temperature is None:
        print('CONSTANT ENERGY, CONSTANT VOLUME')
    else:
        thermo = yaff.LangevinThermostat(temperature, timecon=100 * molmod.units.femtosecond)
        hooks.append(thermo)
        print('CONSTANT TEMPERATURE, CONSTANT VOLUME')

    # integration
    verlet = yaff.VerletIntegrator(
            ff,
            timestep=timestep*molmod.units.femtosecond,
            hooks=hooks,
            vel0 = None,
            temp0=temperature, # initialize velocities to correct temperature, if vel0 is None
            time0=0, 
            counter0=0
            )
    yaff.log.set_level(yaff.log.medium)
    verlet.run(steps)
    yaff.log.set_level(yaff.log.silent)



#mat_lst = ["CsPbI3", "FAPbI3", "MAPbI3"]
mat_lst = [sys.argv[1]]
cl_lst= ["NVT_T150", "NVT_T600"]
phase_lst = ["gamma", "Csdelta", "FAdelta"]
run_np = np.arange(8)

for mat in mat_lst:
    for cl in cl_lst:
        for phase in phase_lst:
            for run in run_np:
                path_atoms = mat + "/" + cl + "/atoms_" + phase + "_" +str(run) + ".xyz"
                path_min_atoms = mat + "/" + cl + "/min_struc_"+phase+".xyz"
                path_hess  = mat + "/" + cl + "/Model_hessian_min_"+phase+".pth"
                path_traj  = mat + "/" + cl + "/atoms_out_" + phase + "_" +str(run) + ".xyz"
                if run < 2.5:
                    tem = 150
                elif run < 4.5:
                    tem = 245
                elif run < 6.5:
                    tem = 383
                else:
                    tem = 600
                if phase == "FAdelta" and cl == "NVT_T600" and mat == "FAPbI3":
                    run_MD(path_atoms, path_min_atoms, path_hess, path_traj,temperature = tem, seed = run)
