from simulate_exporter.utils import (
    NestedDict,
    diffrent,
    LogColor,
    kubernetese_load_config,
)
from typing import Optional, Dict, Tuple, Any, List
from datetime import datetime, timedelta
from prometheus_client import Gauge, CollectorRegistry
from functools import cache
import scipy.stats as stats
import numpy as np
import pydantic
import re

API, V1 = kubernetese_load_config()
METRICS_SET = dict()

KUBE_UNITS: dict[str, float] = {
    "n": 1e-9,  # nano
    "u": 1e-6,  # micro
    "m": 1e-3,  # milli
    "": 1.0,  # no suffix (default)
    "K": 1e3,  # Kilo
    "M": 1e6,  # Mega
    "G": 1e9,  # Giga
    "T": 1e12,  # Tera
    "P": 1e15,  # Peta
    "E": 1e18,  # Exa
    "Ki": 1024,  # Kibi
    "Mi": 1024**2,  # Mebi
    "Gi": 1024**3,  # Gibi
    "Ti": 1024**4,  # Tebi
}


def kubernetese_unit_convertor(resource: dict[str, str]) -> float:
    def func(unit: str):
        match = re.match(r"^(\d+(?:\.\d+)?)\s*([a-zA-Z]*)$", unit)
        if match:
            value_str, unit = match.groups()
            value = float(value_str)
            if unit in KUBE_UNITS:
                return value * KUBE_UNITS[unit]

    if resource:
        return {key: func(val) for key, val in resource.items()}


def get_deployment_name(pod: dict) -> Optional[Tuple[str, str]]:
    for owner_reference in pod.get("metadata", {}).get("ownerReferences", []):
        kind = owner_reference.get("kind")
        owner_name: Optional[str] = None
        match kind:
            case "Deployment":
                owner_name = owner_reference.get("kind")
            case "ReplicaSet":
                name = owner_reference.get("name")
                replicaset = API.read_namespaced_replica_set(
                    name=name, namespace=pod.metadata.namespace
                )
                owner_references = replicaset.metadata.owner_references
                if owner_references:
                    # Iterate through owner references
                    for owner_reference in owner_references:
                        owner_kind = owner_reference.kind  # Kind of the owning resource
                        owner_name = owner_reference.name  # Name of the owning resource
                        owner_uid = owner_reference.uid
            case _:
                pass
        return owner_name, owner_uid
    return None


class PVC(pydantic.BaseModel):
    namespace: str
    name: str
    pres_v_claim: dict = pydantic.Field(default_factory=dict)
    capacity: float = pydantic.Field(default=0.0)

    @pydantic.validator("capacity", pre=True, always=True)
    def validate_capacity(cls, capacity: float, values: dict) -> float:
        v_claim: dict = values.get("pres_v_claim")
        namespace: str = values.get("namespace")
        pvc_name = v_claim.get("claimName", "")
        if pvc_name:
            pvc = V1.read_namespaced_persistent_volume_claim(
                namespace=namespace, name=pvc_name
            )
            capacity = kubernetese_unit_convertor(pvc.status.capacity)
            capacity = capacity.get("storage")
        return capacity


class Container(pydantic.BaseModel):
    name: str
    image: str
    ports: List[dict]
    resources: Dict[str, dict]
    volumeMounts: List[dict]
    terminationMessagePath: str
    terminationMessagePolicy: str
    imagePullPolicy: str
    limits: Dict[str, float] = pydantic.Field(
        default_factory=dict, alias="resources.limits"
    )
    requests: Dict[str, float] = pydantic.Field(
        default_factory=dict, alias="resources.requests"
    )

    @pydantic.validator("limits", pre=True, always=True)
    def validate_limits(cls, _, values):
        limits = values.get("resources").get("limits", {})
        limits = kubernetese_unit_convertor(limits)
        return limits

    @pydantic.validator("requests", pre=True, always=True)
    def validate_requests(cls, _, values):
        requests = values.get("resources").get("requests", {})
        requests = kubernetese_unit_convertor(requests)
        return requests


class Pod(pydantic.BaseModel):
    deployment: str
    metadata: dict
    spec: dict
    status: dict
    kind: str
    apiVersion: str
    namespace: str = pydantic.Field(default="", alias="metadata.namespace")
    volumes: Dict[str, PVC] = pydantic.Field(default_factory=dict, alias="spec.volumes")
    containers: Dict[str, Container] = pydantic.Field(
        default_factory=dict, alias="spec.containers"
    )
    annotations: dict = pydantic.PrivateAttr(default_factory=dict)
    _start_time: datetime = pydantic.PrivateAttr(default_factory=datetime.now)
    _schedule_time: Optional[datetime] = None

    @property
    def name(self) -> str:
        return self.metadata["name"]

    @property
    def namespace(self) -> str:
        return self.metadata["namespace"]

    @property
    def node_name(self) -> Optional[str]:
        return self.spec.get("node_name")

    @node_name.setter
    def node_name(self, name: str) -> None:
        self.spec["node_name"] = name
        if not self._schedule_time:
            self._schedule_time = datetime.now()

    @property
    def is_assigned(self) -> bool:
        return bool(self.node_name)

    @property
    def assignment_time(self) -> datetime:
        return self._schedule_time

    @property
    def annotations(self) -> NestedDict:
        if not self._annotations:
            annotations = self.metadata.annotations
            self._annotations = NestedDict.create_from_dot_string(value=annotations)
        return self._annotations

    @pydantic.validator("namespace", pre=True, always=True)
    def validate_namespace(cls, _, values):
        metadata = values.get("metadata")
        return metadata["namespace"]

    @pydantic.validator("volumes", pre=True, always=True)
    def validate_volumes(cls, _, values):
        namespace: str = values.get("namespace")
        spec: dict = values.get("spec")
        volumes: dict = dict(
            map(
                lambda volume: (volume.name, volume),
                map(
                    lambda v: PVC(namespace=namespace, **v),
                    filter(lambda x: x, spec.get("volumes", [])),
                ),
            )
        )
        return volumes

    @pydantic.validator("containers", pre=True, always=True)
    def validate_containers(cls, _, values):
        spec: dict = values.get("spec")
        containers: dict = dict(
            map(
                lambda container: (container.name, container),
                map(lambda container: Container(**container), spec["containers"]),
            )
        )
        return containers

    @pydantic.validator("annotations", pre=True, always=True)
    def validate_annotations(cls, _, values):
        metadata: dict = values.get("metadata")
        annnotations = NestedDict.create_from_dot_string(
            value=metadata.get("annotations")
        )
        return annnotations


class SimulatedPod(Pod):
    _shutdown: Optional[timedelta] = pydantic.PrivateAttr(default=None)
    _generator: dict = pydantic.PrivateAttr(default_factory=dict)
    _metrics: dict = pydantic.PrivateAttr(default={})

    def __init__(self, **data: dict):
        super().__init__(**data)
        shudown_time: str = self.annotations[self.prefix][self.shutdown_key]
        self._shutdown = timedelta(seconds=int(shudown_time))
        self.__init_metrics__()

    def __init_metrics__(self):
        sim_annotations: dict = self.annotations.pop(self.prefix)
        sim_annotations, sim_cfg = self.add_defualt_values(annotations=sim_annotations)
        for name, v in sim_cfg.items():
            is_metric = self.distrib_prefix in v
            if not is_metric:
                continue
            kwargs = v[self.distrib_prefix]
            func = kwargs.pop(self.distrib_key)
            distribution = getattr(stats, func)
            if name not in METRICS_SET:
                METRICS_SET[name] = Gauge(
                    name=name,
                    documentation=f"container_{name}_usage",
                    labelnames=["namespace", "node", "pod", "container", "deployment"],
                )
            self.metrics[name] = METRICS_SET[name]
            # create resource generator
            if not distribution:
                raise ValueError(f"Unsupported distribution type: {distribution}")
            try:
                kwargs: dict = dict(map(lambda x: (x[0], float(x[1])), kwargs.items()))
                self._generator[name] = distribution(**kwargs)
            except TypeError as e:
                e = str(e).strip("_parse_args()")
                raise ValueError(e)

    @property
    def settings(self) -> set:
        return set(["shutdown", "distribution", "interval"])

    @property
    def shutdown(self) -> Optional[timedelta]:
        return self._shutdown

    @property
    def metrics(cls) -> dict:
        return cls._metrics

    @classmethod
    @property
    def shutdown_key(self) -> str:
        return "shutdown"

    @classmethod
    @property
    def prefix(cls) -> str:
        return "simulate"

    @classmethod
    @property
    def distrib_prefix(cls) -> str:
        return "distribution"

    @classmethod
    @property
    def distrib_key(cls) -> str:
        return "type"

    @classmethod
    def add_defualt_values(
        cls, annotations: dict, inplace: bool = False
    ) -> Tuple[dict, dict]:
        if not inplace:
            annotations = annotations.copy()
        config: dict = {
            "shutdown": annotations.get("shutdown", "5m"),
            cls.distrib_prefix: {
                cls.distrib_key: annotations.get(cls.distrib_key, "norm")
            },
        }
        metrics: dict = dict(
            map(
                lambda x: x if isinstance(x[1], dict) else (x[0], NestedDict()),
                diffrent(first=annotations, second=config).items(),
            )
        )
        metrics_with_no_distribution = filter(
            lambda m: cls.distrib_prefix not in m[1], metrics.items()
        )
        for _, metric_cfg in metrics_with_no_distribution:
            metric_cfg[cls.distrib_prefix].update(config[cls.distrib_prefix])
        config.update(metrics)
        return annotations, config

    @classmethod
    def should_be_simualted(cls, annotations: List[str]) -> bool:
        return any(filter(lambda name: name.startswith(cls.prefix + "."), annotations))

    def push_metrics(self) -> None:
        for _name in self._generator:
            _metric = self.metrics[_name]
            containers: Dict[str, Container] = self.containers
            namespace: str = self.namespace
            deployment: str = self.deployment
            node: str = self.node_name
            pod: str = self.name
            value = self._generator[_name].rvs(size=len(containers))
            for idx, (c_name, container) in enumerate(containers.items()):
                a_max: float = container.limits.get(_name, np.inf)
                a_min: Optional[float] = container.requests.get(_name)
                a_min: float = a_min / 1.5 if a_min else 0
                _value = np.clip(value, a_min=a_min, a_max=a_max)[idx]
                _metric.labels(
                    pod=pod,
                    container=c_name,
                    node=node,
                    namespace=namespace,
                    deployment=deployment,
                ).set(_value)
