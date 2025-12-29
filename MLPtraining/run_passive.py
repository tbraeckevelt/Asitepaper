import argparse
import shutil
import requests
import logging
import yaml
from pathlib import Path
import numpy as np
import time

from ase.io import read

import psiflow
from psiflow.models import  MACEModel, MACEConfig
from psiflow.data import Dataset


def get_mace_model():
    MACEModel.create_apps()
    config = MACEConfig()
    config.hidden_irreps    = "8x0e + 8x1o"
    config.r_max            = 6.0
    config.max_ell          = 2
    config.correlation      = 3
    config.num_interactions = 2
    config.energy_weight    = 10.0
    config.forces_weight    = 1.0
    config.lr               = 0.02
    config.batch_size       = 1
    config.max_num_epochs   = 2000
    config.ema              = True
    config.ema_decay        = 0.99
    return MACEModel(config)


def main(context):
    """Simple training based on existing data"""
    train = Dataset.load('train_PBED3BJ_K222_E500.xyz')
    valid = Dataset.load('validation_PBED3BJ_K222_E500.xyz')
    test_d= Dataset.load('test_PBED3BJ_K222_E500.xyz')
    model = get_mace_model()
    model.initialize(train)
    model.train(train, valid)
    model.deploy()
    errors = Dataset.get_errors(
            test_d,
            model.evaluate(test_d),
            )
    errors = np.mean(errors.result(), axis=0)
    print('energy error [RMSE, meV/atom]: {}'.format(errors[0]))
    print('forces error [RMSE, meV/A]   : {}'.format(errors[1]))
    print('stress error [RMSE, MPa]     : {}'.format(errors[2]))
    model.save(path_output)

if __name__ == '__main__':
    psiflow.load(
            'lumi_native.py',      # path to psiflow config file
            'psiflow_internal',        # internal psiflow cache dir
            logging.DEBUG,             # psiflow log level
            logging.INFO,              # parsl log level
            )

    path_output = Path.cwd() / 'output' # stores final model
    main(path_output)
