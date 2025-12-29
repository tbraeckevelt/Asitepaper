import yaff
from yaff.pes.colvar import CollectiveVariable, CVLinCombIC
from yaff.log import log
from ase import Atoms
from ase.io import read, write
import numpy as np
from numpy import linalg as LA
import autograd.numpy as anp
from autograd import grad  
from autograd.numpy import linalg as aLA
from scipy.spatial.transform import Rotation as R

import molmod.units
import molmod.constants

def my_distance_func(at1, at2, cell):
    inv_cell = LA.inv(cell)
    dir_at1 = np.dot(at1.position,inv_cell)
    dir_at2 = np.dot(at2.position,inv_cell)
    dir_at2 -=np.around(dir_at2 - dir_at1)
    new_pos2 = np.dot(dir_at2,cell)
    return LA.norm(at1.position - new_pos2), new_pos2

def Get_FAmol_lst(atoms):
    FAmol_lst = []
    for at in atoms:
        if at.symbol == 'C':
            FAmol_lst.append({'C' : at.index})

    for FAmol in FAmol_lst:
        at_c = atoms[FAmol['C']]
        i=1
        for at in atoms:
            dist, new_pos = my_distance_func(at_c, at, atoms.cell)
            if dist < 1.9 and at.symbol == 'N':    #heuristic, seems to work fine
                at.position = new_pos
                FAmol['N'+str(i)] = at.index
                i+=1
        assert i -1 == 2, "too many or too little Nitrogen bounded to the Carbon"

    for FAmol in FAmol_lst:
        at_c = atoms[FAmol['C']]
        H_C_tel = 0
        at_N1 = atoms[FAmol['N1']]
        j=1
        at_N2 = atoms[FAmol['N2']]
        k=1
        for at in atoms:
            dist, new_pos = my_distance_func(at_c, at, atoms.cell)
            if dist < 1.5 and at.symbol == 'H':    #heuristic, seems to work fine
                at.position = new_pos
                FAmol['H_C'] = at.index
                H_C_tel += 1
            dist, new_pos = my_distance_func(at_N1, at, atoms.cell)
            if dist < 1.5 and at.symbol == 'H':
                at.position = new_pos
                FAmol['H'+str(j)+'_N1'] = at.index
                j+=1
            dist, new_pos = my_distance_func(at_N2, at, atoms.cell)
            if dist < 1.5 and at.symbol == 'H':
                at.position = new_pos
                FAmol['H'+str(k)+'_N2'] = at.index
                k+=1
        assert H_C_tel == 1, "too many or too little Hydrogens bounded to the Carbon"
        assert j - 1   == 2, "too many or too little Hydrogens bounded to a Nitrogen"
        assert k - 1   == 2, "too many or too little Hydrogens bounded to a Nitrogen"

    return FAmol_lst

def FA_calc_unit_vector_N(pos, masses, index_dct):
    com_1 = pos[index_dct['N1'],:] * masses[index_dct['N1']] 
    com_1+= pos[index_dct['H1_N1'],:] * masses[index_dct['H1_N1']]
    com_1+= pos[index_dct['H2_N1'],:]* masses[index_dct['H2_N1']]
    com_1/= (masses[index_dct['N1']] + masses[index_dct['H1_N1']] + masses[index_dct['H2_N1']])
    com_2 = pos[index_dct['N2'],:]* masses[index_dct['N2']] 
    com_2+= pos[index_dct['H1_N2'],:]* masses[index_dct['H1_N2']]
    com_2+= pos[index_dct['H2_N2'],:]* masses[index_dct['H2_N2']]
    com_2/= (masses[index_dct['N2']] + masses[index_dct['H1_N2']] + masses[index_dct['H2_N2']])
    unit_vec_N = (com_2 - com_1)/aLA.norm(com_2 - com_1)
    return unit_vec_N

def FA_calc_unit_vector_C(pos, masses, index_dct):
    com_1 = pos[index_dct['N1'],:] * masses[index_dct['N1']] 
    com_1+= pos[index_dct['H1_N1'],:] * masses[index_dct['H1_N1']]
    com_1+= pos[index_dct['H2_N1'],:]* masses[index_dct['H2_N1']]
    com_1/= (masses[index_dct['N1']] + masses[index_dct['H1_N1']] + masses[index_dct['H2_N1']])
    com_2 = pos[index_dct['N2'],:]* masses[index_dct['N2']] 
    com_2+= pos[index_dct['H1_N2'],:]* masses[index_dct['H1_N2']]
    com_2+= pos[index_dct['H2_N2'],:]* masses[index_dct['H2_N2']]
    com_2/= (masses[index_dct['N2']] + masses[index_dct['H1_N2']] + masses[index_dct['H2_N2']])
    com_3 = pos[index_dct['C'],:]* masses[index_dct['C']] 
    com_3+= pos[index_dct['H_C'],:]* masses[index_dct['H_C']]
    com_3/= (masses[index_dct['C']] + masses[index_dct['H_C']])
    unit_vec_N = (com_2 - com_1)/aLA.norm(com_2 - com_1)
    vec_CN1 =  com_3 - com_1 
    vec_C = vec_CN1 - (anp.dot(vec_CN1, unit_vec_N))* unit_vec_N
    unit_vec_C = vec_C/aLA.norm(vec_C)
    return unit_vec_C

def FA_calc_CV_orient_value(pos, ref_pos, masses, index_dct):
    unit_vec_N = FA_calc_unit_vector_N(pos, masses, index_dct)
    ref_unit_vec_N = FA_calc_unit_vector_N(ref_pos, masses, index_dct)
    dot_prod = anp.dot(unit_vec_N, ref_unit_vec_N)
    return sigmoid_fct(dot_prod, mult = -5.0, rc = 0.95)

def FA_calc_CV_gyrat_value(pos, ref_pos, masses, index_dct):
    unit_vec_C = FA_calc_unit_vector_C(pos, masses, index_dct)
    ref_unit_vec_C = FA_calc_unit_vector_C(ref_pos, masses, index_dct)
    dot_prod = anp.dot(unit_vec_C, ref_unit_vec_C)
    return sigmoid_fct(dot_prod, mult = -5.0, rc = 0.95)

def Get_MAmol_lst(atoms):
    MAmol_lst = []
    for at in atoms:
        if at.symbol == 'C':
            MAmol_lst.append({'C' : at.index})

    for MAmol in MAmol_lst:
        at_c = atoms[MAmol['C']]
        N_tel = 0
        for at in atoms:
            dist, new_pos = my_distance_func(at_c, at, atoms.cell)
            if dist < 1.9 and at.symbol == 'N':    #heuristic, seems to work fine
                at.position = new_pos
                MAmol['N'] = at.index
                N_tel += 1
        assert N_tel == 1, "too many or too little Nitrogen bounded to the Carbon"

    for MAmol in MAmol_lst:
        at_c = atoms[MAmol['C']]
        at_N = atoms[MAmol['N']]
        j=1
        k=1
        for at in atoms:
            dist, new_pos = my_distance_func(at_c, at, atoms.cell)
            if dist < 1.5 and at.symbol == 'H':    #heuristic, seems to work fine
                at.position = new_pos
                MAmol['H'+str(j)+'_C'] = at.index
                j+=1
            dist, new_pos = my_distance_func(at_N, at, atoms.cell)
            if dist < 1.5 and at.symbol == 'H':
                at.position = new_pos
                MAmol['H'+str(k)+'_N'] = at.index
                k+=1
        assert j - 1 == 3, "too many or too little Hydrogens bounded to the Carbon"
        assert k - 1 == 3, "too many or too little Hydrogens bounded to the Nitrogen"

    return MAmol_lst

def MA_calc_unit_vector_CN(pos, masses, index_dct):
    com_1 = pos[index_dct['C'],:] * masses[index_dct['C']] 
    com_1+= pos[index_dct['H1_C'],:] * masses[index_dct['H1_C']]
    com_1+= pos[index_dct['H2_C'],:] * masses[index_dct['H2_C']]
    com_1+= pos[index_dct['H3_C'],:] * masses[index_dct['H3_C']]
    com_1/= (masses[index_dct['C']] + masses[index_dct['H1_C']] + masses[index_dct['H2_C']] + masses[index_dct['H3_C']])
    com_2 = pos[index_dct['N'],:] * masses[index_dct['N']] 
    com_2+= pos[index_dct['H1_N'],:] * masses[index_dct['H1_N']]
    com_2+= pos[index_dct['H2_N'],:] * masses[index_dct['H2_N']]
    com_2+= pos[index_dct['H3_N'],:] * masses[index_dct['H3_N']]
    com_2/= (masses[index_dct['N']] + masses[index_dct['H1_N']] + masses[index_dct['H2_N']] + masses[index_dct['H3_N']])
    unit_vec_CN = (com_2 - com_1)/aLA.norm(com_2 - com_1)
    return unit_vec_CN

def MA_calc_unit_vector_for_Hs(pos, masses, index_dct):
    vec_Hs_C = (4/3)*pos[index_dct['H1_C'],:] - (2/3)*pos[index_dct['H2_C'],:] - (2/3)*pos[index_dct['H3_C'],:]
    unit_vec_Hs_C = vec_Hs_C /aLA.norm(vec_Hs_C)
    vec_Hs_N = (4/3)*pos[index_dct['H1_N'],:] - (2/3)*pos[index_dct['H2_N'],:] - (2/3)*pos[index_dct['H3_N'],:]
    unit_vec_Hs_N = vec_Hs_N /aLA.norm(vec_Hs_N)
    return unit_vec_Hs_C, unit_vec_Hs_N

def MA_calc_CV_orient_value(pos, ref_pos, masses, index_dct):
    unit_vec_CN = MA_calc_unit_vector_CN(pos, masses, index_dct)
    ref_unit_vec_CN = MA_calc_unit_vector_CN(ref_pos, masses, index_dct)
    dot_prod = anp.dot(unit_vec_CN, ref_unit_vec_CN)
    return sigmoid_fct(dot_prod, mult = -5.0, rc = 0.95)

def MA_calc_CV_gyrat_value(pos, ref_pos, masses, index_dct):
    unit_vec_Hs_C, unit_vec_Hs_N = MA_calc_unit_vector_for_Hs(pos, masses, index_dct)
    ref_unit_vec_Hs_C, ref_unit_vec_Hs_N = MA_calc_unit_vector_for_Hs(ref_pos, masses, index_dct)
    dot_prod = (anp.dot(unit_vec_Hs_C, ref_unit_vec_Hs_C) + anp.dot(unit_vec_Hs_N, ref_unit_vec_Hs_N))/2
    return sigmoid_fct(dot_prod, mult = -5.0, rc = 0.95)

def sigmoid_fct(val, mult = 1.0, rc = 0.0):
    return 1/(1+anp.exp(mult*(rc-val)))

class CVoneFAorientation(CollectiveVariable):

    def __init__(self, system, pos_min_au, index_dct):

        CollectiveVariable.__init__(self, 'CVoneFAorientation', system)
        self.index_dct = index_dct
        self.pos_min_au = pos_min_au
        # Safety checks
        assert len(index_dct)==8, "Exactly 8 atoms should be defined to construct a full FA molecule"
        assert "C" in index_dct, "index for the carbon atom not in dict"
        assert "N1" in index_dct, "index for the first nitrogen atom not in dict"
        assert "N2" in index_dct, "index for the second nitrogen atom not in dict"
        assert "H_C" in index_dct, "index for the hydrogen atom bounded to the carbon not in dict"
        assert "H1_N1" in index_dct, "index for the first hydrogen atom bounded to the first nitrogen not in dict"
        assert "H2_N1" in index_dct, "index for the second hydrogen atom bounded to the first nitrogen not in dict"
        assert "H1_N2" in index_dct, "index for the first hydrogen atom bounded to the second nitrogen not in dict"
        assert "H2_N2" in index_dct, "index for the second hydrogen atom bounded to the second nitrogen not in dict"
        inv_rvecs = LA.inv(system.cell._get_rvecs())  
        self.inv_rvecs =  inv_rvecs  
        pos_c = system.pos[index_dct['C'],:]        
        for ind in index_dct.values():
            pos = system.pos[ind,:]
            diff_dir = np.dot(pos_c - pos, inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        pos_c = pos_min_au[index_dct['C'],:]        
        for ind in index_dct.values():
            pos = pos_min_au[ind,:]
            diff_dir = np.dot(pos_c - pos, inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        # Masses need to be defined in order to compute centers of mass
        if self.system.masses is None:
            self.system.set_standard_masses()

    def get_conversion(self):          
        return 1.0 #log.length.conversion   #If we use angles or dotprod!!!

    def compute(self, gpos=None, vtens=None):
        pos_c = self.system.pos[self.index_dct['C'],:]        
        for ind in self.index_dct.values():
            pos = self.system.pos[ind,:]
            diff_dir = np.dot(pos_c - pos, self.inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        self.value = FA_calc_CV_orient_value(self.system.pos, self.pos_min_au, self.system.masses, self.index_dct)
        if gpos is not None:
            calc_grad_CV = grad(FA_calc_CV_orient_value)
            gpos[:] = calc_grad_CV(self.system.pos, self.pos_min_au, self.system.masses, self.index_dct)
        if vtens is not None: vtens[:] = 0.0
        return self.value

class CVoneFAgyration(CollectiveVariable):

    def __init__(self, system, pos_min_au, index_dct):

        CollectiveVariable.__init__(self, 'CVoneFAgyration', system)
        self.index_dct = index_dct
        self.pos_min_au = pos_min_au
        # Safety checks
        assert len(index_dct)==8, "Exactly 8 atoms should be defined to construct a full FA molecule"
        assert "C" in index_dct, "index for the carbon atom not in dict"
        assert "N1" in index_dct, "index for the first nitrogen atom not in dict"
        assert "N2" in index_dct, "index for the second nitrogen atom not in dict"
        assert "H_C" in index_dct, "index for the hydrogen atom bounded to the carbon not in dict"
        assert "H1_N1" in index_dct, "index for the first hydrogen atom bounded to the first nitrogen not in dict"
        assert "H2_N1" in index_dct, "index for the second hydrogen atom bounded to the first nitrogen not in dict"
        assert "H1_N2" in index_dct, "index for the first hydrogen atom bounded to the second nitrogen not in dict"
        assert "H2_N2" in index_dct, "index for the second hydrogen atom bounded to the second nitrogen not in dict"
        inv_rvecs = LA.inv(system.cell._get_rvecs())
        self.inv_rvecs =  inv_rvecs  
        pos_c = system.pos[index_dct['C'],:]        
        for ind in index_dct.values():
            pos = system.pos[ind,:]
            diff_dir = np.dot(pos_c - pos, inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        pos_c = pos_min_au[index_dct['C'],:]        
        for ind in index_dct.values():
            pos = pos_min_au[ind,:]
            diff_dir = np.dot(pos_c - pos, inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        # Masses need to be defined in order to compute centers of mass
        if self.system.masses is None:
            self.system.set_standard_masses()

    def get_conversion(self): 
        return 1.0 #log.length.conversion   #If we use angles or dotprod!!!

    def compute(self, gpos=None, vtens=None):
        pos_c = self.system.pos[self.index_dct['C'],:]        
        for ind in self.index_dct.values():
            pos = self.system.pos[ind,:]
            diff_dir = np.dot(pos_c - pos, self.inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        self.value = FA_calc_CV_gyrat_value(self.system.pos, self.pos_min_au, self.system.masses, self.index_dct)
        if gpos is not None:
            calc_grad_CV = grad(FA_calc_CV_gyrat_value)
            gpos[:] = calc_grad_CV(self.system.pos, self.pos_min_au, self.system.masses, self.index_dct)
        if vtens is not None: vtens[:] = 0.0
        return self.value

class CVoneMAorientation(CollectiveVariable):

    def __init__(self, system, pos_min_au, index_dct):

        CollectiveVariable.__init__(self, 'CVoneMAorientation', system)
        self.index_dct = index_dct
        self.pos_min_au = pos_min_au
        # Safety checks
        assert len(index_dct)==8, "Exactly 8 atoms should be defined to construct a full MA molecule"
        assert "C" in index_dct, "index for the carbon atom not in dict"
        assert "N" in index_dct, "index for the nitrogen atom not in dict"
        assert "H1_C" in index_dct, "index for the first hydrogen atom bounded to the carbon not in dict"
        assert "H2_C" in index_dct, "index for the second hydrogen atom bounded to the carbon not in dict"
        assert "H3_C" in index_dct, "index for the third hydrogen atom bounded to the carbon not in dict"
        assert "H1_N" in index_dct, "index for the first hydrogen atom bounded to the nitrogen not in dict"
        assert "H2_N" in index_dct, "index for the second hydrogen atom bounded to the nitrogen not in dict"
        assert "H3_N" in index_dct, "index for the third hydrogen atom bounded to the nitrogen not in dict"
        inv_rvecs = LA.inv(system.cell._get_rvecs())
        self.inv_rvecs =  inv_rvecs  
        pos_c = system.pos[index_dct['C'],:]        
        for ind in index_dct.values():
            pos = system.pos[ind,:]
            diff_dir = np.dot(pos_c - pos, inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        pos_c = pos_min_au[index_dct['C'],:]        
        for ind in index_dct.values():
            pos = pos_min_au[ind,:]
            diff_dir = np.dot(pos_c - pos, inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        # Masses need to be defined in order to compute centers of mass
        if self.system.masses is None:
            self.system.set_standard_masses()

    def get_conversion(self):          
        return 1.0 #log.length.conversion   #If we use angles or dotprod!!!

    def compute(self, gpos=None, vtens=None):
        pos_c = self.system.pos[self.index_dct['C'],:]        
        for ind in self.index_dct.values():
            pos = self.system.pos[ind,:]
            diff_dir = np.dot(pos_c - pos, self.inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        self.value = MA_calc_CV_orient_value(self.system.pos, self.pos_min_au, self.system.masses, self.index_dct)
        if gpos is not None:
            calc_grad_CV = grad(MA_calc_CV_orient_value)
            gpos[:] = calc_grad_CV(self.system.pos, self.pos_min_au, self.system.masses, self.index_dct)
        if vtens is not None: vtens[:] = 0.0
        return self.value

class CVoneMAgyration(CollectiveVariable):

    def __init__(self, system, pos_min_au, index_dct):

        CollectiveVariable.__init__(self, 'CVoneMAgyration', system)
        self.index_dct = index_dct
        self.pos_min_au = pos_min_au
        # Safety checks
        assert len(index_dct)==8, "Exactly 8 atoms should be defined to construct a full MA molecule"
        assert "C" in index_dct, "index for the carbon atom not in dict"
        assert "N" in index_dct, "index for the nitrogen atom not in dict"
        assert "H1_C" in index_dct, "index for the first hydrogen atom bounded to the carbon not in dict"
        assert "H2_C" in index_dct, "index for the second hydrogen atom bounded to the carbon not in dict"
        assert "H3_C" in index_dct, "index for the third hydrogen atom bounded to the carbon not in dict"
        assert "H1_N" in index_dct, "index for the first hydrogen atom bounded to the nitrogen not in dict"
        assert "H2_N" in index_dct, "index for the second hydrogen atom bounded to the nitrogen not in dict"
        assert "H3_N" in index_dct, "index for the third hydrogen atom bounded to the nitrogen not in dict"
        inv_rvecs = LA.inv(system.cell._get_rvecs())
        self.inv_rvecs =  inv_rvecs  
        pos_c = system.pos[index_dct['C'],:]        
        for ind in index_dct.values():
            pos = system.pos[ind,:]
            diff_dir = np.dot(pos_c - pos, inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        pos_c = pos_min_au[index_dct['C'],:]        
        for ind in index_dct.values():
            pos = pos_min_au[ind,:]
            diff_dir = np.dot(pos_c - pos, inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        # Masses need to be defined in order to compute centers of mass
        if self.system.masses is None:
            self.system.set_standard_masses()

    def get_conversion(self): 
        return 1.0 #log.length.conversion   #If we use angles or dotprod!!!

    def compute(self, gpos=None, vtens=None):
        pos_c = self.system.pos[self.index_dct['C'],:]        
        for ind in self.index_dct.values():
            pos = self.system.pos[ind,:]
            diff_dir = np.dot(pos_c - pos, self.inv_rvecs)
            assert (np.abs(diff_dir) < 0.5).all(), "The distance of the absolute positions are not equal to the distances from the minimal image convention"
        self.value = MA_calc_CV_gyrat_value(self.system.pos, self.pos_min_au, self.system.masses, self.index_dct)
        if gpos is not None:
            calc_grad_CV = grad(MA_calc_CV_gyrat_value)
            gpos[:] = calc_grad_CV(self.system.pos, self.pos_min_au, self.system.masses, self.index_dct)
        if vtens is not None: vtens[:] = 0.0
        return self.value
