
# CP2K MD Structures

## Overview

This directory contains the structures and scripts for performing molecular dynamics (MD) simulations using a CP2K calculator defined in ASE and integrated as a force field in YAFF. These simulations were conducted to generate initial atomic configurations for further ab initio calculations and machine learning potential training. 

## Contents

- **yaff_CP2K_MD.py**: Main script that executes NVT MD simulations using a CP2K calculator integrated as a force field in YAFF
- **CP2K_para.inp**: Configuration file containing CP2K computational parameters
- **BASIS_SETS**: Basis set definitions for CP2K calculations
- **GTH_POTENTIALS**: Pseudopotential files
- **dftd3.dat**: Dispersion correction parameters
- **InitialStruc/**: Initial structures organized by material and phase, containing XYZ files at various volumes

## Sampling Strategy

To improve configurational sampling and overcome hindered rotation of FA and MA molecules, we employed random orientation protocols:
- **RotateFAMolecules.ipynb**: Randomly rotates FA molecules (FAPbI3 structures)
- **RotateMAMolecules.ipynb**: Randomly rotates MA molecules (MAPbI3 structures)

These notebooks ensure that each MD simulation begins with diverse organic molecule orientations, enabling more comprehensive sampling of the conformational space.
