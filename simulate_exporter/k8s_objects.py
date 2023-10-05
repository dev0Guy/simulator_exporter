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
    name: str
    namespace: str
    capacity: Optional[float]

    def __pre_init__(self, key: str, **data):
        volume: dict = data.pop(key)
        name = volume["name"]
        namespace: str = data.pop("namespace")
        pvc_name = volume.get("persistentVolumeClaim", {})
        pvc_name = pvc_name.get("claimName", "")
        if pvc_name:
            pvc = V1.read_namespaced_persistent_volume_claim(
                namespace=namespace, name=pvc_name
            )
            capacity = kubernetese_unit_convertor(pvc.status.capacity)
            capacity = capacity.get("storage")
        else:
            capacity = 0
        ##############################################################
        data["name"] = name
        data["namespace"] = namespace
        data["capacity"] = capacity
        return data

    def __init__(self, **data):
        cls_name: str = __class__.__name__.lower()
        if cls_name not in data:
            raise ValueError(f"`{cls_name}` doesn't exist in initilization parmaters")
        data = self.__pre_init__(key=cls_name, **data)
        super().__init__(**data)


class Container(pydantic.BaseModel):
    name: str
    limits: Dict[str, float]
    requests: Dict[str, float]

    def __pre_init__(self, key: str, **data):
        container: dict = data.pop(key)

        resources: dict = container["resources"]
        requests: Optional[Dict[str, float]] = kubernetese_unit_convertor(
            resources.get("requests", {})
        )
        limits: Optional[Dict[str, float]] = kubernetese_unit_convertor(
            resources.get("limits", {})
        )

        #########################################
        data["name"]: str = container.get("name")
        data["requests"] = requests
        data["limits"] = limits
        #######################
        return data

    def __init__(self, **data):
        cls_name: str = __class__.__name__.lower()
        if cls_name not in data:
            raise ValueError(f"`{cls_name}` doesn't exist in initilization parmaters")
        data = self.__pre_init__(key=cls_name, **data)
        super().__init__(**data)


class Pod(pydantic.BaseModel):
    name: str
    annotations: Dict[str, dict]
    namespace: str
    containers: Dict[str, Container]
    volumes: Dict[str, PVC]
    deployment: str
    _node: Optional[str] = pydantic.PrivateAttr(default=None)
    _start_time: datetime = pydantic.PrivateAttr(default_factory=datetime.now)
    _schedule_time: Optional[datetime] = None

    def __pre_init__(self, **data):
        pod: dict = data.pop("pod")
        node: str = pod.spec.get("node_name", None)
        namespace: str = pod.metadata["namespace"]
        name: str = pod.metadata["name"]
        annotations: NestedDict = self._extract_annotations(pod=pod)
        #############################################################
        volumes: Dict[str, PVC] = dict(
            map(
                lambda volume: (volume.name, volume),
                map(
                    lambda v: PVC(pvc=v, namespace=namespace),
                    filter(lambda x: x, pod.spec.get("volumes", [])),
                ),
            )
        )
        containers: Dict[str, Container] = dict(
            map(
                lambda container: (container.name, container),
                map(lambda x: Container(container=x), pod.spec["containers"]),
            )
        )
        #####################################################
        data["containers"]: Dict[str, Container] = containers
        data["annotations"]: NestedDict = annotations
        data["volumes"]: Dict[str, PVC] = volumes
        data["namespace"]: str = namespace
        data["_node"]: str = node
        data["name"]: str = name
        return data

    def __init__(self, **data):
        cls_name: str = __class__.__name__.lower()
        if cls_name not in data:
            raise ValueError(f"`{cls_name}` doesn't exist in initilization parmaters")
        data = self.__pre_init__(**data)
        super().__init__(**data)

    @classmethod
    def _extract_annotations(cls, pod: dict) -> NestedDict:
        annotations = pod.metadata.annotations
        return NestedDict.create_from_dot_string(value=annotations)

    @cache
    def get_annotations_by_preifx(self, prefix: str) -> Dict[str, str]:
        return list(filter(lambda k: k.startswith(prefix), self.annotations))

    @property
    def node(self) -> str:
        return self._node

    @node.setter
    def node(self, value: str):
        self._node = value
        if not self._schedule_time:
            self._schedule_time = datetime.now()

    @property
    def is_assigned(self) -> bool:
        return bool(self.node)

    @property
    def assignment_time(self) -> datetime:
        return self._schedule_time


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
            node: str = self.node
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
