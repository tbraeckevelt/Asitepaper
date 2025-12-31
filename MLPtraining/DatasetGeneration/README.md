# Dataset Generation for MLP Training

## Overview

This dataset was generated through a multi-step process:

1. **CP2K MD Simulations** - Created initial atomic structures via molecular dynamics wiht CP2K
2. **VASP Recalculation** - Generated ab initio training data from CP2K structures
3. **Create MACE MLP** - Trained machine learning potential from VASP data, which was used to optimize structures and create the harmonic PES
4. **Harmonic PES Sampling** - Create structures from harmonic potential energy surfaces
5. **Final VASP Recalculation** - Recalculated the harmonic structures
6. **Train Final MACE MLP** - Trained final machine learning potential from all recalculated structures

### Directory Structure

- `CP2K_MD_Structures/` - CP2K MD input files
- `HessianMD_Structures/` - Harmonic MD simulation input files
- `recalcVASP/` - VASP recalculation input files for all structures

