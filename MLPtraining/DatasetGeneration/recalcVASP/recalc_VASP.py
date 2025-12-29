import sys
import torch
import time
import numpy as np
from pathlib import Path
import yaff
import molmod
import h5py
import ase

from ase import Atoms
from ase.io import read, write, iread
from ase.calculators.vasp import Vasp
from ase.geometry import Cell
from nequip.ase.nequip_calculator import NequIPCalculator
from ase.stress import voigt_6_to_full_3x3_stress
from ase.calculators.cp2k import CP2K

import functools
print = functools.partial(print, flush=True)

# when evaluating a model that was converted to double precision, the default
# dtype of torch must be changed as well; otherwise the data loading within
# the calculator will still be performed in single precision
#torch.set_default_dtype(torch.float64) # DEPLOYING IN FLOAT32

if __name__ == '__main__':
    path_atoms     = sys.argv[1]
    path_traj      = sys.argv[1]
    path_out_traj  = sys.argv[2]
    atoms = read(path_atoms)

    #### Set VASP calculation ####
    calc_vasp = Vasp(gamma = True,
                     kpts  = [2,2,2],
                     xc    = 'pbe',
                     pp    = 'pbe',
                     setups= 'recommended')

    atoms.calc = calc_vasp

    calc_vasp.set(prec     ='Normal',
                  algo     ='Normal',
                  lreal    = False,
                  lwave    = False,
                  lorbit   = 11,
                  ismear   = 0,
                  lasph    = True,
                  ediff    = 1e-05,
                  encut    = 500,
                  sigma    = 0.05,
                  lcharg   = False,
                  ispin    = 1,
                  ivdw     = 12,
                  ncore    = 64,
                  kpar     = 8
                 )

    #recalc traj:
    for i, atoms_ in enumerate(iread(path_traj)):
        print(i, end="|")
        atoms_.calc = None
        atoms.set_positions(atoms_.get_positions())
        atoms.set_cell(atoms_.get_cell())
        atoms_.info['energy'] = atoms.get_potential_energy()
        atoms_.arrays['forces'] = atoms.get_forces()
        atoms_.info['stress'] = atoms.get_stress(voigt=False)
        print(atoms_.info['energy'])
        write(path_out_traj, atoms_, format='extxyz', append = True)
