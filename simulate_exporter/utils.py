import kubernetes as k8s
from typing import Optional, Dict, Tuple
from . import models
import logging, re


V1: k8s.client.CoreV1Api = None

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


def kubernetese_load_config():
    try:
        logging.info("in cluster load")
        k8s.config.load_incluster_config()
    except k8s.config.config_exception.ConfigException:
        logging.info("minikube load")
        k8s.config.load_kube_config()
    except Exception as e:
        logging.fatal(e)
        return
    finally:
        global V1
        V1 = k8s.client.CoreV1Api()


def get_all_running_simulative_pods_in_namespace(
    namespace: Optional[str] = "default", name: Optional[str] = None
):
    def filter_func(p: dict):
        simulate_annotation = any(
            map(lambda k: k.startswith("simulate."), p.metadata.annotations)
        )
        return simulate_annotation

    if not name:
        pods = V1.read_namespaced_pod(namespace=namespace)
    else:
        pods = V1.read_namespaced_pod(namespace=namespace, name=name)
    pods: list = pods.items
    return list(filter(filter_func, pods))


def _extract_pod_pvc(volume: dict, namespace: str) -> Tuple[str, models.PVC]:
    name = volume["name"]
    pvc_name = volume.get("persistentVolumeClaim")
    if pvc_name:
        pvc_name = pvc_name["claimName"]
        pvc = V1.read_namespaced_persistent_volume_claim(
            namespace=namespace, name=pvc_name
        )
        pvc_capacity = kubernetese_unit_convertor(pvc.status.capacity)
        pvc = models.PVC(name=pvc_name, capacity=pvc_capacity.get("storage"))
        return name, pvc


def create_pod_object(pod: k8s.client.V1Pod) -> Optional[models.Pod]:
    p_name: str = pod.metadata["name"]
    namespace: str = pod.metadata["namespace"]
    node_name: str = pod.spec.get("node_name", None)
    volumes: list = pod.spec.get("volumes", [])
    annotations = pod.metadata.annotations
    simulate_annotations = dict(
        map(
            lambda tup: (tup[0][len("simulate.") :], tup[1]),
            filter(lambda tup: tup[0].startswith("simulate."), annotations.items()),
        )
    )
    if simulate_annotations:
        volumes: Dict[str, models.PVC] = dict(
            filter(
                lambda x: x,
                map(lambda v: _extract_pod_pvc(volume=v, namespace=namespace), volumes),
            )
        )
        containers = dict()
        for container in pod.spec["containers"]:
            c_name: str = container["name"]
            requests: Optional[Dict[str, float]] = kubernetese_unit_convertor(
                container["resources"].get("requests", {})
            )
            limits: Optional[Dict[str, float]] = kubernetese_unit_convertor(
                container["resources"].get("limits", {})
            )
            containers[c_name] = models.Container(
                name=c_name, limits=limits, requests=requests
            )
        return models.Pod(
            annotations=simulate_annotations,
            name=p_name,
            node=node_name,
            namespace=namespace,
            containers=containers,
            volumes=volumes,
        )


################

# TODO: reprase as funcitonal programming

import scipy.stats as stats


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = NestedDict()
        return self[key]


class AnnotationParser:
    ANNOTATION_PREFIX: str = "simulate"
    DISTRIBUTION_PREFIX: str = "distribution"
    DISTRIBUTION_KEY_NAME: str = "type"

    @classmethod
    def parse(cls, annotations: dict[str, str], metrics: set[str]) -> tuple[dict, dict]:
        distrib_config = cls._create_string_config(config=annotations, options=metrics)
        # take importent values
        config: dict = {
            "shutdown": distrib_config.get("shutdown", "5m"),
            "interval": distrib_config.get("interval", "30s"),
        }
        config.update({k: annotations[k] for k in config})
        #  user didnt specify the global distriv skip
        if cls.DISTRIBUTION_PREFIX in distrib_config:
            _ = distrib_config.pop(cls.DISTRIBUTION_PREFIX)
        # convert into stats distribution
        distrib_config = dict(
            filter(
                lambda itm: cls.DISTRIBUTION_PREFIX in itm[1], distrib_config.items()
            )
        )
        cls._covert_config_to_distrib(config=distrib_config)
        # update config by user information
        return distrib_config, config

    @classmethod
    def _covert_config_to_distrib(cls, config: dict[str, dict]):
        for k, v in config.items():
            kwargs = v[cls.DISTRIBUTION_PREFIX]
            func = kwargs.pop(cls.DISTRIBUTION_KEY_NAME)
            distribution = getattr(stats, func)
            if not distribution:
                raise ValueError(f"Unsupported distribution type: {distribution}")
            try:
                config[k] = distribution(**kwargs)
            except TypeError as e:
                e = str(e).strip("_parse_args()")
                raise ValueError(e)

    @classmethod
    def _create_string_config(cls, config: dict, options: list[str]) -> dict:
        prefix: str = cls.DISTRIBUTION_PREFIX
        distrib_type_k: str = cls.DISTRIBUTION_KEY_NAME
        defualt_distrib: str = "norm"
        user_config: NestedDict = cls.convert_dot_notation(config)[
            cls.ANNOTATION_PREFIX
        ]
        if prefix in user_config:
            if distrib_type_k not in user_config[prefix]:
                user_config[prefix][distrib_type_k] = defualt_distrib
            defualt_distrib = user_config[prefix]
        else:
            defualt_distrib = {distrib_type_k: defualt_distrib}
        # remove distribution from all gpu
        config: dict = {
            k: {prefix: defualt_distrib.copy()} for k in options
        }  # defualt config
        config.update(user_config)
        # # update global distrib if arguemnts are providded but no tpye
        for v in config.values():
            if prefix in v:
                _cfg = v[prefix]
                if distrib_type_k not in _cfg:
                    _cfg[distrib_type_k] = defualt_distrib[prefix][distrib_type_k]
        return config

    @classmethod
    def convert_dot_notation(cls, annotations) -> NestedDict:
        def dot_notation(keys: list[str], v: str, prev: dict):
            for _k in keys[:-1]:
                prev = prev[_k]
            prev[keys[-1]] = v

        result = NestedDict()
        if annotations:
            annotations = sorted(
                map(lambda item: (item[0].split("."), item[1]), annotations.items()),
                key=lambda item: len(item[0]),
                reverse=True,
            )
            for item in annotations:
                dot_notation(*item, prev=result)
        return result


#
