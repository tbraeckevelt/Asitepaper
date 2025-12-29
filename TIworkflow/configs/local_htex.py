
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SingleNodeLauncher, MpiExecLauncher
from parsl.providers import LocalProvider
from parsl.config import Config
from parsl.monitoring.monitoring import MonitoringHub
from parsl.addresses import address_by_hostname


def get_config(path_internal, num_replicas=1):
    provider = LocalProvider(
        min_blocks=1,
        max_blocks=1,
        nodes_per_block=1,
        parallelism=0.5,
        launcher=SingleNodeLauncher(),
        worker_init='export PYTHONPATH='+str(path_internal)+'/:$PYTHONPATH',
        )
    config = Config(
        executors=[
            HighThroughputExecutor(
                label="default",
                working_dir=str(path_internal / 'default_executor'),
                provider=provider,
                cores_per_worker=1,
                max_workers=1,
                address=address_by_hostname(),
                ),
            HighThroughputExecutor(
                label="default_MD",
                working_dir=str(path_internal / 'default_MD_executor'),
                provider=provider,
                cores_per_worker=1,
                max_workers=2,
                address=address_by_hostname(),
                ),
            HighThroughputExecutor(
                label="default_replicas",
                working_dir=str(path_internal / 'default_replicas_executor'),
                provider=provider,
                cores_per_worker=num_replicas,
                max_workers=1,
                address=address_by_hostname(),
                )
            ],
        monitoring=MonitoringHub(
            hub_address=address_by_hostname(),
            hub_port=55015,
            monitoring_debug=False,
            resource_monitoring_interval=10,
            ),
        strategy='none',
        retries=1,
    )
    return config
