from pathlib import Path
import parsl
from parsl.data_provider.files import File
import configs.local_htex
import configs.vsc_hortense
import configs.lumi_native

from lib.papps import  app_plot_fec
from lib.Workflow_class import Workflow

import os
import molmod.units
import numpy as np


if __name__ == '__main__':

    #Define input settings for the workflow
    phase_lst     = ["gamma", "Csdelta", "FAdelta"]
    num_replicas  = 32
    MLP_model_dct = {
                "model_type"   : "MACE", #MACE or NEQUIP - nequip config.yaml and undeployed model are necessary for autodiff - for mace finite diff is used to determine hessian
                "cpu_single"   : File(str(Path.cwd() / "data" / "MACE_Cs_cpu_float32.model")),
                "cpu_double"   : File(str(Path.cwd() / "data" / "MACE_Cs_cpu_float64.model")),
                "cuda_single"  : None, 
                "cuda_double"  : None,
                "config"       : None, 
                "undeployed"   : None,
                }
    init_struc = {}
    for phase in phase_lst:
        init_struc[phase] = File(str(Path.cwd() / "data" / str("atoms_Cs_"+phase+".xyz")))
    #Set the number of steps for each simulation
    #For MD: (number of runs, calib_steps, total number of MD steps per run)
    #For REX: (number of runs, calib_steps, total number of MC steps per run, number of MD steps between each replica exchange attempt)
    #For opt: the total number of optimization runs is determined by num_opt in kwargs, shown below
    #Calib steps are number of snapshot in the written out trajectory which are discarded for postprocessing of this run (so in time units: calib_steps * print_freq * dt)
    nsteps_dct = {
        "init"             : (1, 10, 100000),
        "equi_NPT"         : (5, 0, 1000),
        "REX_NPT"          : (5, 0, 1000, 200),
        "REX_NPT_sup"      : (10, 0, 250, 200),
        "NPT_MD"           : (32, 0, 100000),
        "equi_NVT"         : (5, 0, 1000),
        "REX_NVT"          : (5, 0, 2000, 200),
        "NVT_MD"           : (5, 0, 400000),
        "NVE_MD"           : (5, 0, 400000),
        "lmd_low"          : (5, 100, 400000),
        "lmd_REX"          : (5, 500, 2000, 200),
        "lmd_high"         : (5, 100, 400000),
        "NPT_MD_sup"       : (32, 0, 25000),
        "REX_NVT_sup"      : (10, 0, 500, 200),
        "NVE_MD_sup"       : (10, 0, 100000),
        "lmd_low_sup"      : (10, 10, 100000), 
        "lmd_REX_sup"      : (10, 10, 500, 200),
        "lmd_high_sup"     : (10, 10, 100000),
    }
    kwargs = {
        "pressure"         : 0.1 * 1e6 * molmod.units.pascal,   #Pressure in atomic units
        "tem_min"          : 150.0,              #Minimum temperature in K
        "tem_max"          : 600.0,              #Maximum temperature in K
        "num_tem"          : num_replicas,       #Number of temperatures
        "num_opt"          : 4000,               #Number of optimizations
        "NVE_print"        : 5,                  #Print frequency for the NVE simulations
        "lmd_np"           : np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]), #numpy array of lambda values for the bias
        "MLP_beg"          : 0.0,                #fraction of MLP at the begin of the low temperature TI step
        "MLP_int"          : 0.7,                #fraction of MLP at the end of the low temperature TI step and begin of the high temperature TI step
        "MLP_end"          : 1.0,                #fraction of MLP at the end of the high temperature TI step
        "bias_beg"         : 0.0,                #fraction of bias at the begin of the low temperature TI step
        "bias_int"         : 0.0,                #fraction of bias at the end of the low temperature TI step and begin of the high temperature TI step
        "bias_end"         : 0.0,                #fraction of bias at the end of the high temperature TI step
        "trans_mat"        : np.array([2,2,2]),  #sets transformation matrix for the supercell
        "freq_max"         : 5.0,                #in THz, maximum frequency plotted in the frequency spectra of hessian and vacf
        "smear_freq"       : 0.02,               #in THz, width of the gaussian used to smear the frequency spectra of hessian
        "bsize"            : 1024,               #block size for the block average of the vacf
        "num_part"         : 4,                  #number of parts to split the MD runs to calculate the error on free energy
        "vol_ind_lst"      : [0,5,10,15,19,23,27,31], #The indices which are used to determine at which temperature we determine the average volume to perform the NVT simulations 
        "vol_ind_lst_sup"  : [0,19,31],          #same but for supercell simulations, leave the list empty if you do not want to perform any supercell simulations
    }
    

    #Set up Parsl configuration
    config = configs.vsc_hortense.get_config(Path.cwd(), num_replicas = num_replicas)
    parsl.load(config)

    #Create input and output folders
    Path_folder = Path.cwd() 
    main_input_folder = Path_folder / "input"
    main_output_folder = Path_folder / "output"
    main_input_folder.mkdir(exist_ok=True)
    main_output_folder.mkdir(exist_ok=True)
    print("Start constructing the simulations")
    #Create workflow
    workflow_dct = {}
    for phase in phase_lst:
        workflow_dct[phase] = Workflow(
            input_folder = main_input_folder / phase, 
            output_folder = main_output_folder / phase, 
            MLP_model_dct = MLP_model_dct,
            init_struc = init_struc[phase],
            nsteps_dct = nsteps_dct,
            **kwargs
            )
        #run workflow
        workflow_dct[phase].perform_simulations()
        print("firsttest")
        workflow_dct[phase].create_pickle_files()
        print("test0")
        workflow_dct[phase].do_checks()
    print("test1")
    #Postprocessing an create plots of the free energy
    ref_phase= "gamma"
    ref_workflow = workflow_dct[ref_phase]
    noref_workflow = {}
    for phase, workflow in workflow_dct.items():
        if phase != ref_phase:
            noref_workflow[phase] = workflow
    print("test2")
    label_lst_3 = []
    inputs_3 = []
    app_plot_lst = []
    for phase, workflow in noref_workflow.items():
        label_lst_3.append(phase)
        inputs_3.append(workflow.REX_NPT.pickle_file)
        inputs_3.append(ref_workflow.REX_NPT.pickle_file)
        if kwargs["vol_ind_lst_sup"] != []:
            label_lst_3.append(phase+"_sup")
            inputs_3.append(workflow.REX_NPT_sup.pickle_file)
            inputs_3.append(ref_workflow.REX_NPT_sup.pickle_file)

        label_lst_2 = ["NPT_REX"]
        inputs_2 = [workflow.REX_NPT.pickle_file, ref_workflow.REX_NPT.pickle_file]
        if kwargs["vol_ind_lst_sup"] != []:
            label_lst_2_sup = ["NPT_REX"]
            inputs_2_sup = [workflow.REX_NPT_sup.pickle_file, ref_workflow.REX_NPT_sup.pickle_file]
        print("test3")
        for i, (NVT_workflow, ref_NVT_workflow) in enumerate(zip(workflow.NVT_Workflow_lst, ref_workflow.NVT_Workflow_lst)):
            label_lst_2.append("NVT_REX_T" +str(int(NVT_workflow.vol_tem)))
            inputs_2.append(NVT_workflow.REX_NVT.pickle_file)
            inputs_2.append(ref_NVT_workflow.REX_NVT.pickle_file)
            label_lst_2.append("NVE")
            inputs_2.append(NVT_workflow.NVE_MD.pickle_file)
            inputs_2.append(ref_NVT_workflow.NVE_MD.pickle_file)

            File_plot = File(str(NVT_workflow.output_folder / str("plot_free_energy_contributions.pdf")))
            if os.path.exists(File_plot.filepath) == False:
                label_lst = ["hessian", "vacf", "lmdTI_low", "temTI_lmd", "lmdTI_high", "tem_TI", "NPT_corr"]
                inputs = [NVT_workflow.NVT_opt.pickle_file, ref_NVT_workflow.NVT_opt.pickle_file, 
                          NVT_workflow.NVE_MD.pickle_file,  ref_NVT_workflow.NVE_MD.pickle_file,
                          NVT_workflow.lmd_low.pickle_file, ref_NVT_workflow.lmd_low.pickle_file,
                          NVT_workflow.lmd_REX.pickle_file, ref_NVT_workflow.lmd_REX.pickle_file,
                          NVT_workflow.lmd_high.pickle_file,ref_NVT_workflow.lmd_high.pickle_file,
                          NVT_workflow.REX_NVT.pickle_file, ref_NVT_workflow.REX_NVT.pickle_file,
                          NVT_workflow.NPT_MD.pickle_file,  ref_NVT_workflow.NPT_MD.pickle_file]
                app_plot = app_plot_fec(
                    label_lst, 
                    execution_folder = NVT_workflow.output_folder, 
                    inputs=inputs, 
                    outputs = [File_plot],
                    )
                app_plot_lst.append(app_plot)
            print("test4")
            if NVT_workflow.do_sup:  #Ik moet nog iets fixen met de sup plots want dit wilt parsl liek niet doen + NPT TI gebruikt niet de juiste NVT_TI !!!!
                label_lst_2_sup.append("NVT_REX_T" +str(int(NVT_workflow.vol_tem)))
                inputs_2_sup.append(NVT_workflow.REX_NVT_sup.pickle_file)
                inputs_2_sup.append(ref_NVT_workflow.REX_NVT_sup.pickle_file)
                label_lst_2_sup.append("NVE")
                inputs_2_sup.append(NVT_workflow.NVE_MD_sup.pickle_file)
                inputs_2_sup.append(ref_NVT_workflow.NVE_MD_sup.pickle_file)

                File_plot = File(str(NVT_workflow.output_folder / str("plot_free_energy_contributions_sup.pdf")))
                if os.path.exists(File_plot.filepath) == False:
                    label_lst = ["hessian", "vacf", "lmdTI_low", "temTI_lmd", "lmdTI_high", "tem_TI", "NPT_corr"]
                    inputs = [NVT_workflow.NVT_opt_sup_1.pickle_file, ref_NVT_workflow.NVT_opt_sup_1.pickle_file, 
                              NVT_workflow.NVE_MD_sup.pickle_file,  ref_NVT_workflow.NVE_MD_sup.pickle_file,
                              NVT_workflow.lmd_low_sup.pickle_file, ref_NVT_workflow.lmd_low_sup.pickle_file,
                              NVT_workflow.lmd_REX_sup.pickle_file, ref_NVT_workflow.lmd_REX_sup.pickle_file,
                              NVT_workflow.lmd_high_sup.pickle_file,ref_NVT_workflow.lmd_high_sup.pickle_file,
                              NVT_workflow.REX_NVT_sup.pickle_file, ref_NVT_workflow.REX_NVT_sup.pickle_file,
                              NVT_workflow.NPT_MD_sup.pickle_file,  ref_NVT_workflow.NPT_MD_sup.pickle_file]
                    app_plot = app_plot_fec(
                        label_lst, 
                        execution_folder = NVT_workflow.output_folder, 
                        inputs=inputs, 
                        outputs = [File_plot],
                        )
                    app_plot_lst.append(app_plot)
        print("test5")
        File_plot = File(str(workflow.output_folder / str("Compare_fe_volumes.pdf")))
        if os.path.exists(File_plot.filepath) == False:  
            app_plot = app_plot_fec(
                label_lst_2, 
                execution_folder = workflow.output_folder, 
                inputs=inputs_2, 
                outputs = [File_plot],
                )
            app_plot_lst.append(app_plot)
        if kwargs["vol_ind_lst_sup"] != []:
            File_plot = File(str(workflow.output_folder / str("Compare_fe_volumes_sup.pdf")))
            if os.path.exists(File_plot.filepath) == False:  
                app_plot = app_plot_fec(
                    label_lst_2_sup, 
                    execution_folder = workflow.output_folder, 
                    inputs=inputs_2_sup, 
                    outputs = [File_plot],
                    )
                app_plot_lst.append(app_plot)
    print("test6")
    File_plot_overview = File(str(main_output_folder / str("Overview_plot.pdf")))
    if os.path.exists(File_plot_overview.filepath) == False:     
        app_plot = app_plot_fec(
            label_lst_3, 
            execution_folder = main_output_folder, 
            inputs=inputs_3, 
            outputs = [File_plot_overview],
            )
        app_plot_lst.append(app_plot)
    
    print("Start calculations")

    #Make sure all calculations are finished
    for app_plot in app_plot_lst:
        app_plot.result()
    for phase, workflow in workflow_dct.items():
        for check_app in workflow.check_apps:
            check_app.result()


