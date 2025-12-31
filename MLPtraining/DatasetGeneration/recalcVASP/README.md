
# Recalculating VASP Energies, Forces, and Stresses

This document describes the procedure for recalculating energies, forces, and stresses using the VASP (Vienna Ab-initio Simulation Package) calculator object defined in ASE (Atomic Simulation Environment).

The script `recalc_VASP.py` executes VASP calculations utilizing a VASP calculator object within ASE. It accepts a trajectory extxyz file as input and produces an output trajectory extxyz file containing the VASP-calculated energies, forces, and stresses. Due to the significant computational demands, we could not perform all VASP calculations simultaneously; instead, we divided the calculations into multiple jobs, defining one job per structure. 

We consolidated the output from the initial CP2K-generated structures into training, validation, and test trajectories located in the CsPbI3, FAPbI3, and MAPbI3 subfolders. Subsequently, we expanded the ab initio dataset with Hessian structures, and the VASP output was merged into training and validation extxyz trajectory files named `train_hes_PBED3BJ_K222_E500.xyz` and `validation_hes_PBED3BJ_K222_E500.xyz`, respectively, found in the subfolders.
