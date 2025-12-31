
# HessianMD_Structures

## Overview

This repository contains the necessary files and scripts for generating datasets using Hessian-based molecular dynamics (MD) simulations. The primary script, `HessianMD.py`, utilizes the YAFF package to perform MD simulations based on a harmonic potential energy surface (PES) modeled with PyTorch.

### Directory Structure

The main subfolders include:

- **CsPbI3**
- **FAPbI3**
- **MAPbI3**

Each of these folders contains two additional subfolders:

- **NVT_T150**: Contains simulation files at 150 K.
- **NVT_T600**: Contains simulation files at 600 K.

In these subfolders, you will find the following key files:

### Model Files

- `Model_hessian_min_<phase>.pth`: This file contains the harmonic model for the specified phase.
- `min_struc_<phase>.xyz`: Provides the minimum energy structures around which the harmonic PES is constructed.

### Input and Output

The input and output structures for each MD simulation are organized as follows:

- `atoms_<phase>_<run>.xyz`: Input structure files.
- `atoms_out_<phase>_<run>.xyz`: Output structure files.
