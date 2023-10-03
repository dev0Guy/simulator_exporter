from typing import Optional, Dict, Tuple, Any, Annotated, Type, List
from simulate_exporter.utils import NestedDict, diffrent, LogColor
from simulate_exporter.prom import MetricSetter
from functools import cache
from pprint import pprint
import kubernetes as k8s
import scipy.stats as stats
import logging
import pydantic
import re


def kubernetese_load_config() -> Tuple[k8s.client.AppsV1Api, k8s.client.CoreV1Api]:
    """
    Load k8s config acording to running env, Incluster or in minikube.

    Returns:
    - (k8s.client.AppsV1Api): k8s apps api
    - ( k8s.client.CoreV1Api): k8s core api
    """
    try:  # Inside the cluster
        logging.info("in cluster load")
        k8s.config.load_incluster_config()
    except (
        k8s.config.config_exception.ConfigException
    ):  # outside of the cluster a.k.a minikube
        logging.info("minikube load")
        k8s.config.load_kube_config()
    except Exception as e:  # maybe minikube is not set (cannot connect)
        return logging.fatal(e)
    finally:  # At the end
        return k8s.client.AppsV1Api(), k8s.client.CoreV1Api()


_, V1 = kubernetese_load_config()

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
    node: Optional[str]
    annotations: Dict[str, dict]
    namespace: str
    containers: Dict[str, Container]
    volumes: Dict[str, PVC]

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
        data["node"]: str = node
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
    def is_assigned(self) -> bool:
        return bool(self.node)

    @property
    def labels(self) -> dict:
        return {"node": self.node, "namespace": self.namespace}


class SimulatedPod(Pod):
    _resource_usage_generator: dict = pydantic.PrivateAttr(default_factory=dict)

    @property
    def resource_generator(self):
        if not self._resource_usage_generator:
            if self.prefix in self.annotations:
                sim_annotations: dict = self.annotations.pop(self.prefix)
                sim_annotations, sim_cfg = self.add_defualt_values(
                    annotations=sim_annotations
                )
                for k, v in sim_cfg.items():
                    if self.distrib_prefix in v:
                        kwargs = v[self.distrib_prefix]
                        func = kwargs.pop(self.distrib_key)
                        distribution = getattr(stats, func)
                        if not distribution:
                            raise ValueError(
                                f"Unsupported distribution type: {distribution}"
                            )
                        try:
                            kwargs: dict = dict(map(lambda x: (x[0],float(x[1])), kwargs.items()))
                            sim_cfg[k] = distribution(**kwargs)
                        except TypeError as e:
                            e = str(e).strip("_parse_args()")
                            raise ValueError(e)
                self._resource_usage_generator = sim_cfg
        return self._resource_usage_generator

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
            "interval": annotations.get("interval", "30s"),
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

    def push_metrics(
        self, metrics: Dict[str, Tuple[Type[MetricSetter], Annotated[Any, "Metric"]]]
    ) -> None:
        for _name, (_setter, metric) in metrics.items():
            if _name in self.resource_generator:
                value = self.resource_generator[_name]  # .rvs(size=1)
                value = next(iter(self.resource_generator[_name].rvs(size=1)))
                _setter.set(
                    metric=metric,
                    value=value,
                    namespace=self.namespace,
                    name=self.name,
                    containers=self.containers,
                    volumes=self.volumes,
                    node=self.node,
                )
            else:
                LogColor.warn(
                    f"Metric: {_name} has not been configured in pod {self.name} "
                )
