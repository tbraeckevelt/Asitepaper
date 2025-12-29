
from pathlib import Path
import os
from parsl.executors import HighThroughputExecutor
from parsl.config import Config
from psiflow.external import SlurmProviderVSC # fixed SlurmProvider



def get_config(path_internal, num_replicas=1):

    executors = {}
    cluster = 'dodrio' # all partitions reside on a single cluster

    if 'HOSTNAME' in os.environ.keys():
        htex_address = os.environ['HOSTNAME']
    else:
        htex_address = 'localhost'

    label = 'default'
    worker_init = 'source /dodrio/scratch/projects/starting_2022_006/Forinstall/psiflow_env/activate.sh\n'
    worker_init += 'export PYTHONPATH='+str(path_internal)+'/:$PYTHONPATH'
    provider = SlurmProviderVSC(       # one block == one slurm job to submit
        cluster=cluster,
        partition='cpu_milan',
        account='2023_029',
        nodes_per_block=1,      # each block fits on (less than) one node
        cores_per_node=1,       # number of cores per slurm job, 1 is OK
        init_blocks=1,          # initialize a block at the start of the workflow
        min_blocks=1,           # always keep at least one block open
        max_blocks=500,          # do not use more than fifty blocks
        walltime='71:59:00',    # walltime per block
        mem_per_node = 2,       #memory per node in GB
        worker_init=worker_init,
        exclusive=False,
        )
    executor = HighThroughputExecutor(
                address=htex_address,
                label=label,
                working_dir=str(Path(path_internal) / label),
                cores_per_worker=1,
                provider=provider,
                )
    executors[label] = executor

    label = 'default_MD'
    worker_init = 'source /dodrio/scratch/projects/starting_2022_006/Forinstall/psiflow_env/activate.sh\n'
    worker_init += 'export PYTHONPATH='+str(path_internal)+'/:$PYTHONPATH'
    worker_init += 'export SLURM_TASKS_PER_NODE={}\n'.format(128)
    worker_init += 'export SLURM_NTASKS={}\n'.format(128)
    worker_init += 'export SLURM_NPROCS={}\n'.format(128)
    provider = SlurmProviderVSC(       # one block == one slurm job to submit
        cluster=cluster,
        partition='cpu_milan',
        account='2023_029',
        nodes_per_block=1,      # each block fits on (less than) one node
        cores_per_node=128,       # number of cores per slurm job, 1 is OK
        init_blocks=0,          # initialize a block at the start of the workflow
        min_blocks=0,           # always keep at least one block open
        max_blocks=50,           # do not use more than one block
        walltime='71:59:00',    # walltime per block
        worker_init=worker_init,
        exclusive=False,
        )
    executor = HighThroughputExecutor(
                address=htex_address,
                label=label,
                working_dir=str(Path(path_internal) / label),
                cores_per_worker=1,
                max_workers=128,
                provider=provider,
                )
    executors[label] = executor

    label = 'default_replicas'
    worker_init += 'source /dodrio/scratch/projects/starting_2022_006/Forinstall/psiflow_env/activate.sh\n'
    worker_init += 'export PYTHONPATH='+str(path_internal)+'/:$PYTHONPATH'
    worker_init += 'export SLURM_TASKS_PER_NODE={}\n'.format(128)
    worker_init += 'export SLURM_NTASKS={}\n'.format(128)
    worker_init += 'export SLURM_NPROCS={}\n'.format(128)
    provider = SlurmProviderVSC(       # one block == one slurm job to submit
        cluster=cluster,
        partition='cpu_milan',
        account='2023_029',
        nodes_per_block=1,      # each block fits on (less than) one node
        cores_per_node=128,       # number of cores per slurm job, 1 is OK
        init_blocks=0,          # initialize a block at the start of the workflow
        min_blocks=0,           # always keep at least one block open
        max_blocks=50,           # do not use more than one block
        walltime='71:59:00',    # walltime per block
        worker_init=worker_init,
        exclusive=False,
        )
    executor = HighThroughputExecutor(
        address=htex_address,
        label=label,
        working_dir=str(Path(path_internal) / label),
        cores_per_worker=num_replicas,
        max_workers=int(128/num_replicas),
        provider=provider,
        )
    executors[label] = executor

    config = Config(
        executors=list(executors.values()),
        run_dir=str(path_internal),
        usage_tracking=True,
        app_cache=False,
        retries=1,
        initialize_logging=True,
        strategy='simple',
        max_idletime=10,
        )
    return config



