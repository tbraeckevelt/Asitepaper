# MLPtraining

The 'DatasetGeneration' folder contains input files for generating structures via CP2K and Harmonic MD simulations, and recalculating energies, forces, and stresses with VASP for improved accuracy. See the folder's README.md for details.

run_passive.py trains a MACE neural network on ab initio datasets in the CsPbI3, FAPbI3, and MAPbI3 folders using the psiflow package (v1.0.0rc0, defined in submitscript.sh). Training was performed on LUMI with resources configured in lumi_native.py. For more information, visit https://github.com/molmod/psiflow.