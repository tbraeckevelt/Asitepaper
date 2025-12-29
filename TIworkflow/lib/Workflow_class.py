from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Any
import numpy as np
import os
import molmod.units

from lib.Simulation_class import Simulation
from lib.papps import app_Compare_En_and_Vol, app_plot_runave
from parsl.data_provider.files import File


class Workflow:
    input_folder     : Optional[Any]        = None
    output_folder    : Optional[Any]        = None
    MLP_model_dct    : Optional[Any]        = None
    init_struc       : Optional[Any]        = None
    min_struc_npt    : Optional[Any]        = None               #Get from NPT_opt simulation
    min_struc_npt_sup: Optional[Any]        = None               #Get from NPT_opt_sup simulation
    nsteps_dct       : Optional[Any]        = None               #Dictionary with the number of steps for each simulation
    pressure         : Optional[float]      = None               #Pressure in atomic units
    tem_min          : Optional[float]      = None               #Minimum temperature in K
    tem_max          : Optional[float]      = None               #Maximum temperature in K
    num_tem          : Optional[int]        = None               #Number of temperatures
    tem_np           : Optional[np.float]   = None               #numpy array of temperatures - set by init function
    num_opt          : Optional[int]        = None               #Number of optimizations
    NVE_print        : Optional[int]        = None               #Print frequency for the NVE simulations
    lmd_np           : Optional[np.float]   = None               #numpy array of lambda values for the bias
    num_lmd          : Optional[int]        = None               #Number of lambda values - set by init function
    MLP_beg          : Optional[float]      = None               #fraction of MLP at the begin of the low temperature TI step
    MLP_int          : Optional[float]      = None               #fraction of MLP at the end of the low temperature TI step and begin of the high temperature TI step
    MLP_end          : Optional[float]      = None               #fraction of MLP at the end of the high temperature TI step
    bias_beg         : Optional[float]      = None               #fraction of bias at the begin of the low temperature TI step
    bias_int         : Optional[float]      = None               #fraction of bias at the end of the low temperature TI step and begin of the high temperature TI step
    bias_end         : Optional[float]      = None               #fraction of bias at the end of the high temperature TI step
    trans_mat        : Optional[Any]        = None               #sets transformation matrix for the supercell
    freq_max         : Optional[float]      = None               #in THz, maximum frequency plotted in the frequency spectra of hessian and vacf
    smear_freq       : Optional[float]      = None               #in THz, width of the gaussian used to smear the frequency spectra of hessian
    bsize            : Optional[int]        = None               #block size for the block average of the vacf
    num_part         : Optional[int]        = None               #number of parts to split the MD runs to calculate the error on free energy
    check_apps       : Optional[list]       = None               #list of applications that should run, typically comparisons betweent simulations
    #Simulations  -  all simulations are set by the set simulations function
    init             : Optional[Simulation] = None
    equi_NPT         : Optional[Simulation] = None               #For equilibration
    REX_NPT          : Optional[Simulation] = None
    NPT_opt          : Optional[Simulation] = None
    REX_NPT_sup      : Optional[Simulation] = None
    NPT_opt_sup_1    : Optional[Simulation] = None   
    NPT_opt_sup_2    : Optional[Simulation] = None               #For check
    vol_ind_lst      : Optional[list]       = None               #The indices which are used to determine at which temperature we 
                                                                 #determine the average volume to perform the NVT simulations 
    vol_ind_lst_sup  : Optional[list]       = None               #same but for supercell simulations, leave the list empty if you do not want to perform any supercell simulations
    NVT_Workflow_lst : Optional[list]       = None               #List of NVT_Workflow objects

    def __init__(self, **kwargs): 
        for key, val in kwargs.items():
            assert key in self.__dir__(), f"key {key} not in Workflow class"
            setattr(self, key, val)
        self.tem_np = self.tem_min*np.exp(np.linspace(0.0, np.log(self.tem_max/self.tem_min), self.num_tem)) #exponential temperature distribution T_(i+1)/T_i=cte
        self.num_lmd = len(self.lmd_np)     #Make sure number of lambda values is correct
        #Try to create the input and output folders
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        #Set the simulations
        self.set_simulations()
        self.NVT_Workflow_lst = []
        #Create sub NVT workflows for each index in vol_ind
        assert set(self.vol_ind_lst_sup).issubset(set(self.vol_ind_lst)), "vol_ind_lst_sup is not a subset of vol_ind_lst"
        for ind in self.vol_ind_lst:
            NVT_workflow_obj = NVT_Workflow(
                input_folder  = self.input_folder / str("NVT_T" + str(int(self.tem_np[ind]))),
                output_folder = self.output_folder / str("NVT_T" + str(int(self.tem_np[ind]))),
                vol_ind       = ind,
                vol_tem       = self.tem_np[ind],
                vol_pres      = self.pressure,
                )
            #set the attributes of the NVT_workflow from the attributes of the main workflow
            for key in self.__annotations__: 
                if key in NVT_workflow_obj.__dir__() and key not in ["input_folder", "output_folder"]:
                    setattr(NVT_workflow_obj, key, getattr(self, key))
            if ind in self.vol_ind_lst_sup:
                NVT_workflow_obj.do_sup = True
            #Set the attributes of the NVT_workflow
            NVT_workflow_obj.set_simulations()
            self.NVT_Workflow_lst.append(NVT_workflow_obj)
        
    def set_simulations(self):

        #Initial run
        self.init = Simulation(
            input_folder   = self.input_folder / "init",
            output_folder  = self.output_folder / "init",
            MLP_model_dct  = self.MLP_model_dct,
            pre_Files      = [[self.init_struc]],
            num_runs       = self.nsteps_dct["init"][0],
            )
        self.init.create_settings(nsteps=self.nsteps_dct["init"][2], calib_steps=self.nsteps_dct["init"][1], temperature = self.tem_max, 
                                  pressure = self.pressure)

        #equilibration NPT
        self.equi_NPT = Simulation(
            input_folder   = self.input_folder / "equi_NPT",
            output_folder  = self.output_folder / "equi_NPT",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.init,
            num_val_prop   = self.num_tem,
            num_runs       = self.nsteps_dct["equi_NPT"][0],
            )
        self.equi_NPT.create_settings(nsteps=self.nsteps_dct["equi_NPT"][2], calib_steps=self.nsteps_dct["equi_NPT"][1], temperature = self.tem_np, 
                                      pressure = self.pressure, set_vel = False)
        
        #REX NPT
        self.REX_NPT = Simulation(
            input_folder   = self.input_folder / "REX_NPT",
            output_folder  = self.output_folder / "REX_NPT",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.equi_NPT,
            num_val_prop   = self.num_tem,
            num_runs       = self.nsteps_dct["REX_NPT"][0],
            )
        self.REX_NPT.create_settings(MD_steps_REX = self.nsteps_dct["REX_NPT"][3], nsteps = self.nsteps_dct["REX_NPT"][2], 
                                     calib_steps=self.nsteps_dct["REX_NPT"][1], temperature = self.tem_np, pressure = self.pressure)

        #NPT optimization
        self.NPT_opt = Simulation(
            input_folder   = self.input_folder / "NPT_opt",
            output_folder  = self.output_folder / "NPT_opt",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.REX_NPT,
            pre_val        = 0,
            num_runs       = self.num_opt,
            )
        self.NPT_opt.create_settings(pressure = self.pressure, set_vel = False)

        if self.vol_ind_lst_sup != []:
            #REX NPT supercell
            self.REX_NPT_sup = Simulation(
                input_folder   = self.input_folder / "REX_NPT_sup",
                output_folder  = self.output_folder / "REX_NPT_sup",
                MLP_model_dct  = self.MLP_model_dct,
                pre_simulation = self.REX_NPT,
                num_val_prop   = self.num_tem,
                num_runs       = self.nsteps_dct["REX_NPT_sup"][0],
                )
            self.REX_NPT_sup.create_settings(MD_steps_REX = self.nsteps_dct["REX_NPT_sup"][3], nsteps = self.nsteps_dct["REX_NPT_sup"][2], 
                                            calib_steps=self.nsteps_dct["REX_NPT_sup"][1], temperature = self.tem_np, pressure = self.pressure, 
                                            trans_mat = self.trans_mat)

            #NPT optimization supercell
            #from NPT_opt
            self.NPT_opt_sup_1 = Simulation(
                input_folder   = self.input_folder / "NPT_opt_sup_1",
                output_folder  = self.output_folder / "NPT_opt_sup_1",
                MLP_model_dct  = self.MLP_model_dct,
                )
            self.NPT_opt_sup_1.create_settings(pressure = self.pressure, set_vel = False, trans_mat = self.trans_mat) 
            #From NPT REX sup
            self.NPT_opt_sup_2 = Simulation(
                input_folder   = self.input_folder / "NPT_opt_sup_2",
                output_folder  = self.output_folder / "NPT_opt_sup_2",
                MLP_model_dct  = self.MLP_model_dct,
                pre_simulation = self.REX_NPT_sup,
                pre_val        = 0,
                num_runs       = self.num_opt,
                )
            self.NPT_opt_sup_2.create_settings(pressure = self.pressure, set_vel = False, trans_mat = self.trans_mat)

    def perform_simulations(self):
        self.init.get_input_struc(transform_low_tri = True)
        self.init.perform_MD()
        self.equi_NPT.get_input_struc()
        self.equi_NPT.perform_MD()                                        #For equilibration only
        self.REX_NPT.get_input_struc()
        self.REX_NPT.perform_REX()
        self.NPT_opt.get_input_struc()
        self.min_struc_npt = self.NPT_opt.perform_optimization()
        set_min_struc_and_hessians([self.REX_NPT], min_struc = self.min_struc_npt)
        if self.vol_ind_lst_sup != []:
            self.REX_NPT_sup.get_input_struc(sup = True)
            self.REX_NPT_sup.perform_REX()
            self.NPT_opt_sup_1.pre_Files = [[self.min_struc_npt]]
            self.NPT_opt_sup_1.get_input_struc(sup=True)
            self.min_struc_npt_sup = self.NPT_opt_sup_1.perform_optimization()
            set_min_struc_and_hessians([self.REX_NPT_sup], min_struc = self.min_struc_npt_sup)
            self.NPT_opt_sup_2.get_input_struc()
            self.NPT_opt_sup_2.perform_optimization()
        for NVT_workflow_obj in self.NVT_Workflow_lst:
            NVT_workflow_obj.perform_simulations()

    def create_pickle_files(self):
        self.REX_NPT.ref_pickle_file_lst  = []
        if self.vol_ind_lst_sup != []:
            self.REX_NPT_sup.ref_pickle_file_lst  = []
        for NVT_workflow_obj in self.NVT_Workflow_lst:
            NVT_workflow_obj.create_pickle_files()
            self.REX_NPT.ref_pickle_file_lst.append(NVT_workflow_obj.NPT_MD.pickle_file)
            if NVT_workflow_obj.do_sup:
                self.REX_NPT_sup.ref_pickle_file_lst.append(NVT_workflow_obj.NPT_MD_sup.pickle_file)
        self.REX_NPT.create_pickle(num_part = self.num_part)
        if self.vol_ind_lst_sup != []:
            self.REX_NPT_sup.create_pickle(num_part = self.num_part)

    def do_checks(self):
        self.check_apps = []
        #Print distributions of NPT REX
        for i in range(len(self.REX_NPT.output_files)):
            outfile = File(str(self.REX_NPT.output_folder / str("distr_"+str(i)+".txt")))
            if os.path.exists(outfile.filepath) == False:
                check_app = app_plot_runave(
                    execution_folder = self.REX_NPT.output_folder,
                    inputs = self.REX_NPT.output_files[i], 
                    outputs = [outfile], 
                    calib_step = self.REX_NPT.settings.calib_steps,
                    )
                self.check_apps.append(check_app)

        if self.vol_ind_lst_sup != []:
            #Check NPT REX sup
            for i in range(len(self.REX_NPT.output_files)):
                outfile = File(str(self.REX_NPT_sup.output_folder / str("Compare_sup_" + str(i) + ".txt")))
                if os.path.exists(outfile.filepath) == False:
                    min_runs = min(len(self.REX_NPT.output_files[i]), len(self.REX_NPT_sup.output_files[i]))
                    check_app = app_Compare_En_and_Vol(
                        execution_folder = self.REX_NPT_sup.output_folder,
                        inputs = self.REX_NPT.output_files[i][:min_runs] + self.REX_NPT_sup.output_files[i][:min_runs], 
                        outputs = [outfile],
                        calib_step = max(self.REX_NPT.settings.calib_steps, self.REX_NPT_sup.settings.calib_steps),
                        )
                    self.check_apps.append(check_app)
            #Check NPT opt sup
            outfile = File(str(self.NPT_opt_sup_1.output_folder / str("Compare_minima_supmethods.txt")))
            if os.path.exists(outfile.filepath) == False:
                check_app = app_Compare_En_and_Vol(
                    execution_folder = self.NPT_opt_sup_1.output_folder,
                    inputs = [self.NPT_opt_sup_1.min_struc, self.NPT_opt_sup_2.min_struc],
                    outputs = [outfile],
                    check_vol= False,
                    )
                self.check_apps.append(check_app)
        #Do checks for NVT workflows
        for NVT_workflow_obj in self.NVT_Workflow_lst:
            NVT_workflow_obj.do_checks(self.check_apps)
        
class NVT_Workflow:
    MLP_model_dct    : Optional[Any]        = None               #Set by main workflow
    min_struc_nvt    : Optional[Any]        = None               #Get from NVT_opt simulation
    Hessian_npy      : Optional[Any]        = None               #Get from NVT_opt simulation
    Hessian_model    : Optional[Any]        = None               #Get from NVT_opt simulation
    min_struc_nvt_sup: Optional[Any]        = None               #Get from NVT_opt_sup simulation
    Hessian_npy_sup  : Optional[Any]        = None               #Get from NVT_opt_sup simulation    
    Hessian_model_sup: Optional[Any]        = None               #Get from NVT_opt_sup simulation
    nsteps_dct       : Optional[Any]        = None               #Dictionary with the number of steps for each simulation - set by main workflow
    input_folder     : Optional[Any]        = None               
    output_folder    : Optional[Any]        = None               
    vol_ind          : Optional[int]        = None               #index to select the NPT run of REX simualtions to proceed from - set by main workflow
    vol_tem          : Optional[float]      = None               #Temperature in K - set by main workflow
    vol_pres         : Optional[float]      = None               #Pressure in MPa - set by main workflow
    do_sup           : bool                 = False              #If true, the calculations for the supercell are performed
    tem_min          : Optional[float]      = None               #Minimum temperature in K - set by main workflow
    tem_max          : Optional[float]      = None               #Maximum temperature in K - set by main workflow
    num_tem          : Optional[int]        = None               #Number of temperatures - set by main workflow
    tem_np           : Optional[float]      = None               #Temperature array in K for REX simulations - set by main workflow
    num_opt          : Optional[int]        = None               #Number of optimizations - set by main workflow
    NVE_print        : Optional[int]        = None               #print frequency for the NVE simulations - set by main workflow
    lmd_np           : Optional[list[float]]= None               #numpy array of lambda values for the REX simulations - set by main workflow
    num_lmd          : Optional[int]        = None               #length of lmd_lst - set by main workflow
    MLP_beg          : Optional[float]      = None               #fraction of MLP at the begin of the low temperature TI step - set by main workflow
    MLP_int          : Optional[float]      = None               #fraction of MLP at the end of the low tem TI step and begin of the high tem TI step - set by main workflow
    MLP_end          : Optional[float]      = None               #fraction of MLP at the end of the high temperature TI step - set by main workflow
    bias_beg         : Optional[float]      = None               #fraction of bias at the begin of the low temperature TI step - set by main workflow
    bias_int         : Optional[float]      = None               #fraction of bias at the end of the low tem TI step and begin of the high tem TI step - set by main workflow
    bias_end         : Optional[float]      = None               #fraction of bias at the end of the high temperature TI step - set by main workflow
    trans_mat        : Optional[Any]        = None               #sets transformation matrix for the supercell - set by main workflow
    freq_max         : Optional[float]      = None               #in THz, maximum frequency plotted in the frequency spectra of hessian and vacf - set by main workflow
    smear_freq       : Optional[float]      = None               #in THz, width of the gaussian used to smear the frequency spectra of hessian - set by main workflow
    bsize            : Optional[int]        = None               #block size for the block average of the vacf
    num_part         : Optional[int]        = None               #number of parts to split the MD runs to calculate the error on free energy
    #main workflow simulations
    init             : Optional[Simulation] = None               #Reference to Simulation in main workflow
    equi_NPT         : Optional[Simulation] = None               #Reference to Simulation in main workflow
    REX_NPT          : Optional[Simulation] = None               #Reference to Simulation in main workflow
    NPT_opt          : Optional[Simulation] = None               #Reference to Simulation in main workflow
    REX_NPT_sup      : Optional[Simulation] = None               #Reference to Simulation in main workflow
    NPT_opt_sup_1    : Optional[Simulation] = None               #Reference to Simulation in main workflow
    NPT_opt_sup_2    : Optional[Simulation] = None               #Reference to Simulation in main workflow
    #NVT_workflow simulations  -  all simulations are set by the set simulations function
    NPT_MD           : Optional[Simulation] = None 
    equi_NVT         : Optional[Simulation] = None               #For equilibration
    REX_NVT          : Optional[Simulation] = None 
    NVT_MD           : Optional[Simulation] = None               #For check
    NVE_MD           : Optional[Simulation] = None 
    NVT_opt          : Optional[Simulation] = None
    lmd_low          : Optional[Simulation] = None 
    lmd_REX          : Optional[Simulation] = None 
    lmd_high         : Optional[Simulation] = None 
    #if self.do_sup, than the following simulations are also defined
    NPT_MD_sup       : Optional[Simulation] = None 
    REX_NVT_sup      : Optional[Simulation] = None 
    NVE_MD_sup       : Optional[Simulation] = None 
    NVT_opt_sup_1    : Optional[Simulation] = None
    NVT_opt_sup_2    : Optional[Simulation] = None               #For check
    lmd_low_sup      : Optional[Simulation] = None 
    lmd_REX_sup      : Optional[Simulation] = None
    lmd_high_sup     : Optional[Simulation] = None

    def __init__(self, **kwargs): 
        for key, val in kwargs.items():
            assert key in self.__dir__(), f"key {key} not in NVT_Workflow class"
            setattr(self, key, val)
        #Try to create the input and output folders
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)

    def set_simulations(self):
        
        #NPT MD
        self.NPT_MD = Simulation(
            input_folder   = self.input_folder / "NPT_MD",
            output_folder  = self.output_folder / "NPT_MD",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.REX_NPT,
            pre_val        = self.vol_ind,
            num_runs       = self.nsteps_dct["NPT_MD"][0],
            )
        self.NPT_MD.create_settings(nsteps=self.nsteps_dct["NPT_MD"][2], calib_steps=self.nsteps_dct["NPT_MD"][1], temperature = self.vol_tem, 
                                    pressure = self.vol_pres)

        #equilibration NVT
        self.equi_NVT = Simulation(
            input_folder   = self.input_folder / "equi_NVT",
            output_folder  = self.output_folder / "equi_NVT",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.REX_NPT,
            num_val_prop   = self.num_tem,
            num_runs       = self.nsteps_dct["equi_NVT"][0],
            )
        self.equi_NVT.create_settings(nsteps=self.nsteps_dct["equi_NVT"][2], calib_steps=self.nsteps_dct["equi_NVT"][1], temperature = self.tem_np, 
                                      set_vel = False)  #Use self.NPT_MD to get reference cell

        #REX NVT
        self.REX_NVT = Simulation(
            input_folder   = self.input_folder / "REX_NVT",
            output_folder  = self.output_folder / "REX_NVT",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.equi_NVT,
            num_val_prop   = self.num_tem,
            num_runs       = self.nsteps_dct["REX_NVT"][0],
            )
        MLP_beg  = min(self.MLP_beg, self.MLP_int, self.MLP_end)
        MLP_end  = max(self.MLP_beg, self.MLP_int, self.MLP_end)
        bias_beg = min(self.bias_beg, self.bias_int, self.bias_end)
        bias_end = max(self.bias_beg, self.bias_int, self.bias_end)
        self.REX_NVT.create_settings(MD_steps_REX = self.nsteps_dct["REX_NVT"][3], nsteps = self.nsteps_dct["REX_NVT"][2], 
                                     calib_steps=self.nsteps_dct["REX_NVT"][1], temperature = self.tem_np, MLP_beg = MLP_beg, MLP_end = MLP_end, 
                                     bias_beg = bias_beg, bias_end = bias_end)

        #NVT MD at highest temperature to Check REX
        self.NVT_MD = Simulation(
            input_folder   = self.input_folder / "NVT_MD",
            output_folder  = self.output_folder / "NVT_MD",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.REX_NVT,
            pre_val        = self.num_tem-1,
            num_runs       = self.nsteps_dct["NVT_MD"][0],
            )
        self.NVT_MD.create_settings(nsteps=self.nsteps_dct["NVT_MD"][2], calib_steps=self.nsteps_dct["NVT_MD"][1], temperature = self.tem_max)

        #NVE MD
        self.NVE_MD = Simulation(
            input_folder   = self.input_folder / "NVE_MD",
            output_folder  = self.output_folder / "NVE_MD",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.REX_NVT,
            pre_val        = self.vol_ind,
            num_runs       = self.nsteps_dct["NVE_MD"][0],
            )
        self.NVE_MD.create_settings(nsteps=self.nsteps_dct["NVE_MD"][2], calib_steps=self.nsteps_dct["NVE_MD"][1], print_freq = self.NVE_print)

        #NVT optimization
        self.NVT_opt = Simulation(
            input_folder   = self.input_folder / "NVT_opt",
            output_folder  = self.output_folder / "NVT_opt",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.REX_NVT,
            pre_val        = 0,
            num_runs       = self.num_opt,
        )
        self.NVT_opt.create_settings(set_vel = False) 

        #lmd low
        self.lmd_low = Simulation(
            input_folder   = self.input_folder / "lmd_low",
            output_folder  = self.output_folder / "lmd_low",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.REX_NVT,  #also uses recalculated snapshots
            pre_val        = 0,
            num_val_prop   = self.num_lmd,
            num_runs       = self.nsteps_dct["lmd_low"][0],
            )
        frac_MLP  = self.MLP_beg + self.lmd_np*(self.MLP_int- self.MLP_beg)
        frac_bias = self.bias_beg + self.lmd_np*(self.bias_int- self.bias_beg)
        self.lmd_low.create_settings(nsteps=self.nsteps_dct["lmd_low"][2], calib_steps=self.nsteps_dct["lmd_low"][1], temperature = self.tem_min, 
                                     frac_MLP = frac_MLP, frac_bias = frac_bias, set_vel = False, MLP_beg = self.MLP_beg, MLP_end = self.MLP_int, 
                                     bias_beg = self.bias_beg, bias_end = self.bias_int)
        
        #lmd REX
        self.lmd_REX = Simulation(
            input_folder   = self.input_folder / "lmd_REX",
            output_folder  = self.output_folder / "lmd_REX",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.REX_NVT,  #also uses recalculated snapshots      
            pre_val        = 0,  #We need to sample minima before, 
                                 #because the bias can introduce activated events which lead to long equilibration times for the biased REX simulations of FAPbI3
            num_val_prop   = self.num_tem,
            num_runs       = self.nsteps_dct["lmd_REX"][0],
            )
        self.lmd_REX.create_settings(MD_steps_REX = self.nsteps_dct["lmd_REX"][3], nsteps = self.nsteps_dct["lmd_REX"][2], 
                                     calib_steps=self.nsteps_dct["lmd_REX"][1], temperature = self.tem_np, frac_MLP = self.MLP_int, 
                                     frac_bias = self.bias_int, set_vel = False)

        #lmd high
        self.lmd_high = Simulation(
            input_folder   = self.input_folder / "lmd_high",
            output_folder  = self.output_folder / "lmd_high",
            MLP_model_dct  = self.MLP_model_dct,
            pre_simulation = self.REX_NVT,  #also uses recalculated snapshots
            pre_val        = self.num_tem-1,
            num_val_prop   = self.num_lmd,
            num_runs       = self.nsteps_dct["lmd_high"][0],
            )
        frac_MLP  = self.MLP_int + self.lmd_np*(self.MLP_end- self.MLP_int)
        frac_bias = self.bias_int + self.lmd_np*(self.bias_end- self.bias_int)
        self.lmd_high.create_settings(nsteps=self.nsteps_dct["lmd_high"][2], calib_steps=self.nsteps_dct["lmd_high"][1], temperature = self.tem_max, 
                                      frac_MLP = frac_MLP, frac_bias = frac_bias, set_vel = False, MLP_beg = self.MLP_int, MLP_end = self.MLP_end, 
                                      bias_beg = self.bias_int, bias_end = self.bias_end)
        
        if self.do_sup:
            #NPT MD supercell
            self.NPT_MD_sup = Simulation(
                input_folder   = self.input_folder / "NPT_MD_sup",
                output_folder  = self.output_folder / "NPT_MD_sup",
                MLP_model_dct  = self.MLP_model_dct,
                pre_simulation = self.NPT_MD,
                num_runs       = self.nsteps_dct["NPT_MD_sup"][0],
                )
            self.NPT_MD_sup.create_settings(nsteps=self.nsteps_dct["NPT_MD_sup"][2], calib_steps=self.nsteps_dct["NPT_MD_sup"][1], 
                                            temperature = self.vol_tem, pressure = self.vol_pres, trans_mat = self.trans_mat)

            #REX NVT supercell
            self.REX_NVT_sup = Simulation(
                input_folder   = self.input_folder / "REX_NVT_sup",
                output_folder  = self.output_folder / "REX_NVT_sup",
                MLP_model_dct  = self.MLP_model_dct,
                pre_simulation = self.REX_NVT,
                num_val_prop   = self.num_tem,
                num_runs       = self.nsteps_dct["REX_NVT_sup"][0],
                )
            self.REX_NVT_sup.create_settings(MD_steps_REX = self.nsteps_dct["REX_NVT_sup"][3], nsteps = self.nsteps_dct["REX_NVT_sup"][2], 
                                             calib_steps=self.nsteps_dct["REX_NVT_sup"][1], temperature = self.tem_np, trans_mat = self.trans_mat)

            #NVE MD supercell
            self.NVE_MD_sup = Simulation(
                input_folder   = self.input_folder / "NVE_MD_sup",
                output_folder  = self.output_folder / "NVE_MD_sup",
                MLP_model_dct  = self.MLP_model_dct,
                pre_simulation = self.REX_NVT_sup,
                pre_val        = self.vol_ind,
                num_runs       = self.nsteps_dct["NVE_MD_sup"][0],
                )
            self.NVE_MD_sup.create_settings(nsteps=self.nsteps_dct["NVE_MD_sup"][2], calib_steps=self.nsteps_dct["NVE_MD_sup"][1], 
                                            print_freq = self.NVE_print, trans_mat = self.trans_mat)

            #NVT optimization supercell
            #from NVT_opt
            self.NVT_opt_sup_1 = Simulation(
                input_folder   = self.input_folder / "NVT_opt_sup_1",
                output_folder  = self.output_folder / "NVT_opt_sup_1",
                MLP_model_dct  = self.MLP_model_dct,
                )
            self.NVT_opt_sup_1.create_settings(set_vel = False, trans_mat = self.trans_mat) 
            #From REX_NVT_sup
            self.NVT_opt_sup_2 = Simulation(
                input_folder   = self.input_folder / "NVT_opt_sup_2",
                output_folder  = self.output_folder / "NVT_opt_sup_2",
                MLP_model_dct  = self.MLP_model_dct,
                pre_simulation = self.REX_NVT_sup,
                pre_val        = 0,
                num_runs       = self.num_opt,
                )
            self.NVT_opt_sup_2.create_settings(set_vel = False, trans_mat = self.trans_mat) 

            #lmd low supercell
            self.lmd_low_sup = Simulation(
                input_folder   = self.input_folder / "lmd_low_sup",
                output_folder  = self.output_folder / "lmd_low_sup",
                MLP_model_dct  = self.MLP_model_dct,
                pre_simulation = self.lmd_low, 
                num_val_prop   = self.num_lmd,
                num_runs       = self.nsteps_dct["lmd_low_sup"][0],
                )
            frac_MLP  = self.MLP_beg + self.lmd_np*(self.MLP_int- self.MLP_beg)
            frac_bias = self.bias_beg + self.lmd_np*(self.bias_int- self.bias_beg)
            self.lmd_low_sup.create_settings(nsteps=self.nsteps_dct["lmd_low_sup"][2], calib_steps=self.nsteps_dct["lmd_low_sup"][1], 
                                             temperature = self.tem_min, frac_MLP = frac_MLP, frac_bias = frac_bias, trans_mat = self.trans_mat,
                                             MLP_beg = self.MLP_beg, MLP_end = self.MLP_int, bias_beg = self.bias_beg, bias_end = self.bias_int)

            #lmd REX supercell
            self.lmd_REX_sup = Simulation(
                input_folder   = self.input_folder / "lmd_REX_sup",
                output_folder  = self.output_folder / "lmd_REX_sup",
                MLP_model_dct  = self.MLP_model_dct,
                pre_simulation = self.lmd_REX, 
                num_val_prop   = self.num_tem,
                num_runs       = self.nsteps_dct["lmd_REX_sup"][0],
                )
            self.lmd_REX_sup.create_settings(MD_steps_REX = self.nsteps_dct["lmd_REX_sup"][3], nsteps = self.nsteps_dct["lmd_REX_sup"][2], 
                                             calib_steps=self.nsteps_dct["lmd_REX_sup"][1], temperature = self.tem_np, frac_MLP = self.MLP_int, 
                                             frac_bias = self.bias_int, trans_mat = self.trans_mat)

            #lmd high supercell
            self.lmd_high_sup = Simulation(
                input_folder   = self.input_folder / "lmd_high_sup",
                output_folder  = self.output_folder / "lmd_high_sup",
                MLP_model_dct  = self.MLP_model_dct,
                pre_simulation = self.lmd_high,  
                num_val_prop   = self.num_lmd,
                num_runs       = self.nsteps_dct["lmd_high_sup"][0],
                )
            frac_MLP  = self.MLP_int + self.lmd_np*(self.MLP_end- self.MLP_int)
            frac_bias = self.bias_int + self.lmd_np*(self.bias_end- self.bias_int)
            self.lmd_high_sup.create_settings(nsteps=self.nsteps_dct["lmd_high_sup"][2], calib_steps=self.nsteps_dct["lmd_high_sup"][1], 
                                              temperature = self.tem_max, frac_MLP = frac_MLP, frac_bias = frac_bias, trans_mat = self.trans_mat,
                                              MLP_beg = self.MLP_int, MLP_end = self.MLP_end, bias_beg = self.bias_int, bias_end = self.bias_end)
    
    def perform_simulations(self):
        self.NPT_MD.get_input_struc(set_most_ortho = True)
        self.NPT_MD.perform_MD()
        self.equi_NVT.set_ref_cell(self.NPT_MD)
        self.equi_NVT.get_input_struc(scale_cell = True)         #For equilibration only
        self.equi_NVT.perform_MD()                           
        self.REX_NVT.get_input_struc()
        self.REX_NVT.perform_REX()
        self.NVT_MD.get_input_struc()
        self.NVT_MD.perform_MD()                                 #For checks only
        self.NVE_MD.get_input_struc()
        self.NVE_MD.perform_MD()
        self.NVT_opt.get_input_struc()
        self.min_struc_nvt = self.NVT_opt.perform_optimization()
        self.Hessian_npy, self.Hessian_model = self.NVT_opt.calculate_hessian()
        sim_lst = [self.NPT_MD, self.REX_NVT, self.NVE_MD, self.lmd_low, self.lmd_REX, self.lmd_high]
        set_min_struc_and_hessians(sim_lst, min_struc = self.min_struc_nvt, Hessian_model = self.Hessian_model)
        self.REX_NVT.perform_recalc(trans_traj= True)
        self.lmd_low.get_weighted_snap()
        self.lmd_low.perform_MD()
        self.lmd_low.perform_recalc()
        self.lmd_REX.get_weighted_snap()
        self.lmd_REX.perform_REX()
        self.lmd_REX.settings.MLP_beg = self.MLP_beg
        self.lmd_REX.settings.MLP_end = self.MLP_int
        self.lmd_REX.settings.bias_beg = self.bias_beg
        self.lmd_REX.settings.bias_end = self.bias_int
        self.lmd_REX.perform_recalc(ind_lst = [0])               #For checks only
        self.lmd_REX.settings.MLP_beg = self.MLP_int
        self.lmd_REX.settings.MLP_end = self.MLP_end
        self.lmd_REX.settings.bias_beg = self.bias_int
        self.lmd_REX.settings.bias_end = self.bias_end
        self.lmd_REX.perform_recalc(ind_lst = [-1])              #For checks only
        self.lmd_high.get_weighted_snap()
        self.lmd_high.perform_MD()
        self.lmd_high.perform_recalc()
        if self.do_sup:
            self.NPT_MD_sup.get_input_struc(sup=True)
            self.NPT_MD_sup.perform_MD()
            self.REX_NVT_sup.get_input_struc(sup=True)
            self.REX_NVT_sup.perform_REX()
            self.NVE_MD_sup.get_input_struc()
            self.NVE_MD_sup.perform_MD()
            self.NVT_opt_sup_1.pre_Files = [[self.min_struc_nvt]]
            self.NVT_opt_sup_1.get_input_struc(sup=True)
            self.min_struc_nvt_sup = self.NVT_opt_sup_1.perform_optimization()
            self.Hessian_npy_sup, self.Hessian_model_sup = self.NVT_opt_sup_1.calculate_hessian()
            sim_lst = [self.NPT_MD_sup, self.REX_NVT_sup, self.NVE_MD_sup, self.lmd_low_sup, self.lmd_REX_sup, self.lmd_high_sup]
            set_min_struc_and_hessians(sim_lst, min_struc = self.min_struc_nvt_sup, Hessian_model = self.Hessian_model_sup)
            self.NVT_opt_sup_2.get_input_struc()
            self.NVT_opt_sup_2.perform_optimization()            #For checks only
            self.lmd_low_sup.get_input_struc(sup=True)
            self.lmd_low_sup.perform_MD()
            self.lmd_low_sup.perform_recalc()
            self.lmd_REX_sup.get_input_struc(sup=True)
            self.lmd_REX_sup.perform_REX()
            self.lmd_REX_sup.settings.MLP_beg = self.MLP_beg
            self.lmd_REX_sup.settings.MLP_end = self.MLP_int
            self.lmd_REX_sup.settings.bias_beg = self.bias_beg
            self.lmd_REX_sup.settings.bias_end = self.bias_int
            self.lmd_REX_sup.perform_recalc(ind_lst = [0])       #For checks only
            self.lmd_REX_sup.settings.MLP_beg = self.MLP_int
            self.lmd_REX_sup.settings.MLP_end = self.MLP_end
            self.lmd_REX_sup.settings.bias_beg = self.bias_int
            self.lmd_REX_sup.settings.bias_end = self.bias_end
            self.lmd_REX_sup.perform_recalc(ind_lst = [-1])      #For checks only
            self.lmd_high_sup.get_input_struc(sup=True)
            self.lmd_high_sup.perform_MD()
            self.lmd_high_sup.perform_recalc()

    def create_pickle_files(self):
        self.NVT_opt.create_pickle(temperature = self.tem_np, freq_max = self.freq_max, smear_freq = self.smear_freq)
        self.NVE_MD.create_pickle(temperature = self.vol_tem, bsize = self.bsize, freq_max = self.freq_max)
        self.lmd_low.ref_pickle_file_lst = [self.NVT_opt.pickle_file]
        self.lmd_low.create_pickle(lmd_np = self.lmd_np, num_part = self.num_part)
        self.lmd_REX.ref_pickle_file_lst = [self.lmd_low.pickle_file]
        self.lmd_REX.create_pickle(num_part = self.num_part)
        self.lmd_high.ref_pickle_file_lst= [self.lmd_REX.pickle_file]
        self.lmd_high.create_pickle(lmd_np = self.lmd_np, num_part = self.num_part)
        self.REX_NVT.ref_pickle_file_lst = [self.lmd_high.pickle_file]
        self.REX_NVT.create_pickle(num_part = self.num_part)
        self.NPT_MD.ref_pickle_file_lst  = [self.REX_NVT.pickle_file]
        self.NPT_MD.create_pickle(num_part = self.num_part)
        if self.do_sup:
            self.NVT_opt_sup_1.create_pickle(temperature = self.tem_np, freq_max = self.freq_max, smear_freq = self.smear_freq)
            self.NVE_MD_sup.create_pickle(temperature = self.vol_tem, bsize = self.bsize, freq_max = self.freq_max)
            self.lmd_low_sup.ref_pickle_file_lst = [self.NVT_opt_sup_1.pickle_file]
            self.lmd_low_sup.create_pickle(lmd_np = self.lmd_np, num_part = self.num_part)
            self.lmd_REX_sup.ref_pickle_file_lst = [self.lmd_low_sup.pickle_file]
            self.lmd_REX_sup.create_pickle(num_part = self.num_part)
            self.lmd_high_sup.ref_pickle_file_lst= [self.lmd_REX_sup.pickle_file]
            self.lmd_high_sup.create_pickle(lmd_np = self.lmd_np, num_part = self.num_part)
            self.REX_NVT_sup.ref_pickle_file_lst = [self.lmd_high_sup.pickle_file]
            self.REX_NVT_sup.create_pickle(num_part = self.num_part)
            self.NPT_MD_sup.ref_pickle_file_lst  = [self.REX_NVT_sup.pickle_file]
            self.NPT_MD_sup.create_pickle(num_part = self.num_part)

    def do_checks(self, check_apps_lst):
        #Check NPT REX at voltem
        outfile = File(str(self.NPT_MD.output_folder / str("Compare_REX_voltem.txt")))
        if os.path.exists(outfile.filepath) == False:
            min_runs = min(len(self.NPT_MD.output_files[0]), len(self.REX_NPT.output_files[self.vol_ind]))
            check_app = app_Compare_En_and_Vol(
                execution_folder=self.NPT_MD.output_folder,
                inputs = self.NPT_MD.output_files[0][:min_runs] + self.REX_NPT.output_files[self.vol_ind][:min_runs], 
                outputs = [outfile],
                calib_step = max(self.NPT_MD.settings.calib_steps, self.REX_NPT.settings.calib_steps),
                )
            check_apps_lst.append(check_app)
        #Print distributions for NPT MD
        outfile = File(str(self.NPT_MD.output_folder / str("distr.txt")))
        if os.path.exists(outfile.filepath) == False:
            check_app = app_plot_runave(
                execution_folder=self.NPT_MD.output_folder,
                inputs = self.NPT_MD.output_files[0], 
                outputs = [outfile], 
                calib_step = self.NPT_MD.settings.calib_steps,
                )
            check_apps_lst.append(check_app)
        #Check NVT REX at tmax
        outfile = File(str(self.NVT_MD.output_folder / str("Compare_REX_tmax.txt")))
        if os.path.exists(outfile.filepath) == False:
            min_runs = min(len(self.REX_NVT.output_files[-1]), len(self.NVT_MD.output_files[0]))
            check_app = app_Compare_En_and_Vol(
                execution_folder=self.NVT_MD.output_folder,
                inputs = self.REX_NVT.output_files[-1][:min_runs] + self.NVT_MD.output_files[0][:min_runs], 
                outputs = [outfile],
                check_vol= False,
                calib_step = max(self.REX_NVT.settings.calib_steps, self.NVT_MD.settings.calib_steps),
                )
            check_apps_lst.append(check_app)
        #Check lmd REX min tem
        outfile = File(str(self.lmd_REX.output_folder / str("Compare_deltaU_REX_tmin.txt")))
        if os.path.exists(outfile.filepath) == False:
            min_runs = min(len(self.lmd_low.output_files_rec[-1]), len(self.lmd_REX.output_files_rec[0]))
            check_app = app_Compare_En_and_Vol(
                execution_folder=self.lmd_REX.output_folder,
                inputs = self.lmd_low.output_files_rec[-1][:min_runs] + self.lmd_REX.output_files_rec[0][:min_runs], 
                outputs = [outfile],
                check_vol= False,
                deltaU = True,
                calib_step = max(self.lmd_low.settings.calib_steps, self.lmd_REX.settings.calib_steps),
                )
            check_apps_lst.append(check_app)
        #Check lmd REX max tem
        outfile = File(str(self.lmd_REX.output_folder / str("Compare_deltaU_REX_tmax.txt")))
        if os.path.exists(outfile.filepath) == False:
            min_runs = min(len(self.lmd_high.output_files_rec[0]), len(self.lmd_REX.output_files_rec[-1]))
            check_app = app_Compare_En_and_Vol(
                execution_folder=self.lmd_REX.output_folder,
                inputs = self.lmd_high.output_files_rec[0][:min_runs] + self.lmd_REX.output_files_rec[-1][:min_runs], 
                outputs = [outfile],
                check_vol= False,
                deltaU = True,
                calib_step = max(self.lmd_high.settings.calib_steps, self.lmd_REX.settings.calib_steps),
                )
            check_apps_lst.append(check_app)
        
        if self.do_sup:
            #Check NPT MD sup
            outfile = File(str(self.NPT_MD_sup.output_folder / str("Compare_sup.txt")))
            if os.path.exists(outfile.filepath) == False:
                min_runs = min(len(self.NPT_MD.output_files[0]), len(self.NPT_MD_sup.output_files[0]))
                check_app = app_Compare_En_and_Vol(
                    execution_folder=self.NPT_MD_sup.output_folder,
                    inputs = self.NPT_MD.output_files[0][:min_runs] + self.NPT_MD_sup.output_files[0][:min_runs], 
                    outputs = [outfile],
                    calib_step = max(self.NPT_MD.settings.calib_steps, self.NPT_MD_sup.settings.calib_steps),
                    )
                check_apps_lst.append(check_app)
            #Print distributions for NPT MD
            outfile = File(str(self.NPT_MD_sup.output_folder / str("distr.txt")))
            if os.path.exists(outfile.filepath) == False:
                check_app = app_plot_runave(
                    execution_folder=self.NPT_MD_sup.output_folder,
                    inputs = self.NPT_MD_sup.output_files[0], 
                    outputs = [outfile], 
                    calib_step = self.NPT_MD_sup.settings.calib_steps,
                    )
                check_apps_lst.append(check_app)

            #Check NVT REX sup
            for i in range(len(self.REX_NVT.output_files)):
                outfile = File(str(self.REX_NVT_sup.output_folder / str("Compare_sup_" + str(i) + ".txt")))
                if os.path.exists(outfile.filepath) == False:
                    min_runs = min(len(self.REX_NVT.output_files[i]), len(self.REX_NVT_sup.output_files[i]))
                    check_app = app_Compare_En_and_Vol(
                        execution_folder=self.REX_NVT_sup.output_folder,
                        inputs = self.REX_NVT.output_files[i][:min_runs] + self.REX_NVT_sup.output_files[i][:min_runs], 
                        outputs = [outfile],
                        check_vol= False,
                        calib_step = max(self.REX_NVT.settings.calib_steps, self.REX_NVT_sup.settings.calib_steps),
                        )
                    check_apps_lst.append(check_app)

            #Check NVT opt sup
            outfile = File(str(self.NVT_opt_sup_1.output_folder / str("Compare_minima_supmethods.txt")))
            if os.path.exists(outfile.filepath) == False:
                check_app = app_Compare_En_and_Vol(
                    execution_folder=self.NVT_opt_sup_1.output_folder,
                    inputs = [self.NVT_opt_sup_1.min_struc, self.NVT_opt_sup_2.min_struc], 
                    outputs = [outfile],
                    check_vol= False,
                    )
                check_apps_lst.append(check_app)

            #Check lmd REX min tem sup
            outfile = File(str(self.lmd_REX_sup.output_folder / str("Compare_deltaU_REX_tmin.txt")))
            if os.path.exists(outfile.filepath) == False:
                min_runs = min(len(self.lmd_low_sup.output_files_rec[-1]), len(self.lmd_REX_sup.output_files_rec[0]))
                check_app = app_Compare_En_and_Vol(
                    execution_folder=self.lmd_REX_sup.output_folder,
                    inputs = self.lmd_low_sup.output_files_rec[-1][:min_runs] + self.lmd_REX_sup.output_files_rec[0][:min_runs], 
                    outputs = [outfile],
                    check_vol= False,
                    deltaU = True,
                    calib_step = max(self.lmd_low_sup.settings.calib_steps, self.lmd_REX_sup.settings.calib_steps),
                    )
                check_apps_lst.append(check_app)
            #Check lmd REX max tem sup
            outfile = File(str(self.lmd_REX_sup.output_folder / str("Compare_deltaU_REX_tmax.txt")))
            if os.path.exists(outfile.filepath) == False:
                min_runs = min(len(self.lmd_high_sup.output_files_rec[0]), len(self.lmd_REX_sup.output_files_rec[-1]))
                check_app = app_Compare_En_and_Vol(
                    execution_folder=self.lmd_REX_sup.output_folder,
                    inputs = self.lmd_high_sup.output_files_rec[0][:min_runs] + self.lmd_REX_sup.output_files_rec[-1][:min_runs], 
                    outputs = [outfile],
                    check_vol= False,
                    deltaU = True,
                    calib_step = max(self.lmd_high_sup.settings.calib_steps, self.lmd_REX_sup.settings.calib_steps),
                    )
                check_apps_lst.append(check_app)


def set_min_struc_and_hessians(sim_lst, min_struc = None, Hessian_model = None):
    for sim in sim_lst:
        if min_struc is not None:
            sim.min_struc = min_struc
        if Hessian_model is not None:
            sim.Hessian_model = Hessian_model


