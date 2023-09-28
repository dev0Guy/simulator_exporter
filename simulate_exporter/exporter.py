from prometheus_client import Gauge, start_http_server
from prometheus_client.metrics import MetricWrapperBase
from dataclasses import dataclass, field
from . import utils, models
import time, logging
import random


@dataclass
class MetricSimulatorExporter:
    """ """

    _prom_port: int
    _parser: utils.AnnotationParser
    _metrics: dict = field(default_factory=dict)
    _registered: dict[str, models.SimulatedPod] = field(default_factory=dict)

    def _init_metrics(self):
        self._metrics["cpu"] = Gauge(
            "container_cpu_usage",
            "cpu usage of containers in the machine.",
            ["namespace", "node", "pod", "container"],
        )
        self._metrics["memory"] = Gauge(
            "container_memory_usage",
            "memory usage of the containers in machine.",
            ["namespace", "node", "pod", "container"],
        )
        self._metrics["gpu"] = Gauge(
            "gpu_usage_percent",
            "gpu usage of the containers in machine.",
            ["namespace", "node", "pod", "container"],
        )
        self._metrics["pvc"] = Gauge(
            "pvc_storage_usage_bytes",
            "PVC Storage Usage in Bytes",
            ["namespace", "node", "pod", "pvc"],
        )

    @property
    def metrics(self) -> dict[str, MetricWrapperBase]:
        return self._metrics

    def __post_init__(self):
        utils.kubernetese_load_config()
        self._init_metrics()
        logging.info(
            f"Starting Exporter at `http://localhost:{self._prom_port}/metrics`"
        )
        start_http_server(port=self._prom_port)

    @staticmethod
    def _push_pod_metric(name: str, metric: MetricWrapperBase, pod: models.Pod):
        # before had list of pods and for loop to all
        if name == "pvc":
            for volume_name, _ in pod.volumes.items():
                metric.labels(
                    node=pod.node,
                    pod=pod.name,
                    namespace=pod.namespace,
                    pvc=volume_name,
                ).set(random.random() * 100)
        else:
            for container in pod.containers.values():
                metric.labels(
                    node=pod.node,
                    pod=pod.name,
                    container=container.name,
                    namespace=pod.namespace,
                ).set(random.random() * 100)

    def register_pod(self, logger: logging, body: dict, name: str, uid: str, **_):
        """
        Get Pod in the moment it created, maybe hasn't been scheduler yet [just created]
        """
        try:
            pod_already_registered: bool = uid in self._registered
            if not pod_already_registered:
                pod: models.Pod = utils.create_pod_object(pod=body)
                if pod:
                    print("pod has annotations")
                    metrics: set = set(self.metrics.keys())
                    dist_cfg, cfg = self._parser.parse(
                        annotations=pod.annotations, metrics=metrics
                    )
                    self._registered[uid] = models.SimulatedPod(
                        pod=pod,
                        resource_usage_generator=dist_cfg,
                        **cfg,
                    )
                    logging.info(f"Registration of pod {uid}")
                else:
                    logger.info(f"Pod {uid} has no simulate annotations")
        except StopIteration:
            logger.error(f"Object pod: {uid} has no simulate annotation, {name}")
        except Exception as e:
            logger.error(e)

    def interval(
        self, logger: logging, uid: str, name: str, spec: dict, body: dict, namespace: str, **kwargs
    ):
        try:
            pod_should_be_generated_for: bool = uid in self._registered
            if pod_should_be_generated_for:
                simulate_pod = self._registered[uid]
                if not simulate_pod.pod.node:
                    simulate_pod.pod.node = spec["nodeName"]
                    simulate_pod.is_schedule = True
                for _name, metric in self.metrics.items():
                    self._push_pod_metric(
                        name=_name, metric=metric, pod=simulate_pod.pod
                    )
                    logging.info(f"Pushed '{_name}' metric of pod {name}")
        except Exception as e:
            logger.error(e)
