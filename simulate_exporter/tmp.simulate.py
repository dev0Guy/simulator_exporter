from typing import (
    Protocol,
    Optional,
    Dict,
    List,
    Any,
    Tuple,
    Type,
    runtime_checkable,
    Annotated,
)
from prometheus_client import start_http_server, Metric
from .utils import LogColor
from datetime import datetime, timedelta
import scipy.stats as stats
import kubernetes as k8s
import pathlib
import pydantic
import logging
import kopf
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


API, V1 = kubernetese_load_config()


class KopFunctions:

    """
    All the KOPF function needed in order to run.
    """

    @staticmethod
    def startup(
        log_level: Optional["logging._Level"] = logging.ERROR, enabled: bool = True
    ):
        def inner_startup(settings: kopf.OperatorSettings, **_):
            pass

        return inner_startup


class Structs:
    """
    Contains all of the struct need for simulation
    """

    class Original:
        # all the defualt k8s defention
        class Container(pydantic.BaseModel):
            name: str
            limits: Dict[str, float]
            requests: Dict[str, float]

        class PVC(pydantic.BaseModel):
            name: str
            capacity: Optional[float]

        class Pod(pydantic.BaseModel):
            name: str
            node: Optional[str]
            annotations: Dict[str, str]
            namespace: str
            containers: Dict[str, "Structs.Original.Container"]
            volumes: Dict[str, "Structs.Original.PVC"]

    class Simulated:
        class Pod(pydantic.BaseModel):
            pod: "Structs.Original.Pod"
            shutdown: timedelta
            interval: timedelta
            resource_usage_generator: dict
            _is_schedule: bool = False
            _start_time: datetime = pydantic.PrivateAttr(default_factory=datetime.now)
            _schedule_time: Optional[datetime] = None

            @pydantic.validator("shutdown", pre=True, always=True)
            def _shutdown(cls, value: str):
                return timedelta(seconds=int(value))

            @pydantic.validator("interval", pre=True, always=True)
            def _interval(cls, value: str):
                return timedelta(seconds=int(value))

            @property
            def volumes(self) -> Dict[str, "Structs.Original.PVC"]:
                return self.pod.volumes

            @property
            def is_schedule(self) -> bool:
                return self._is_schedule

            @property
            def namespace(self) -> str:
                return self.pod.namespace

            @property
            def name(self) -> str:
                return self.pod.name

            @property
            def node(self) -> str:
                return self.pod.name

            @property
            def containers(self) -> Dict[str, "Structs.Original.Container"]:
                return self.pod.containers

            @property
            def assignment_time(self) -> datetime:
                return self._schedule_time

            @property
            def queue_enter_time(self) -> datetime:
                return self._start_time

            @node.setter
            def node(self, value: str):
                self.pod.node = value
                if not self._schedule_time:
                    self._schedule_time = datetime.now()


class Setters:
    @runtime_checkable
    class MetricSetter(Protocol):
        @staticmethod
        def set(metric: Metric, pod: dict, value: Any):
            ...

    class CPUMetricSetter(MetricSetter):
        @staticmethod
        def set(metric: Metric, pod: dict, value: Any):
            for container in pod.containers.values():
                metric.labels(
                    node=pod.node,
                    pod=pod.name,
                    container=container.name,
                    namespace=pod.namespace,
                ).set(value)

    class GPUMetricSetter(MetricSetter):
        @staticmethod
        def set(metric: Metric, pod: dict, value: Any):
            for container in pod.containers.values():
                metric.labels(
                    node=pod.node,
                    pod=pod.name,
                    container=container.name,
                    namespace=pod.namespace,
                ).set(value)

    class MemoryMetricSetter(MetricSetter):
        @staticmethod
        def set(metric: Metric, pod: dict, value: Any):
            for container in pod.containers.values():
                metric.labels(
                    node=pod.node,
                    pod=pod.name,
                    container=container.name,
                    namespace=pod.namespace,
                ).set(value)

    class PVCMetricSetter(MetricSetter):
        @staticmethod
        def set(metric: Metric, pod: dict, value: Any):
            for volume_name, _ in pod.volumes.items():
                metric.labels(
                    node=pod.node,
                    pod=pod.name,
                    namespace=pod.namespace,
                    pvc=volume_name,
                ).set(value)


class Utils:
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

    @staticmethod
    def kubernetese_unit_convertor(resource: dict[str, str]) -> float:
        def func(unit: str):
            match = re.match(r"^(\d+(?:\.\d+)?)\s*([a-zA-Z]*)$", unit)
            if match:
                value_str, unit = match.groups()
                value = float(value_str)
                if unit in __class__.KUBE_UNITS:
                    return value * __class__.KUBE_UNITS[unit]

        if resource:
            return {key: func(val) for key, val in resource.items()}

    @staticmethod
    def extract_pod_pvc(
        volume: dict, namespace: str
    ) -> Tuple[str, Structs.Original.PVC]:
        name = volume["name"]
        pvc_name = volume.get("persistentVolumeClaim")
        if pvc_name:
            pvc_name = pvc_name["claimName"]
            pvc = V1.read_namespaced_persistent_volume_claim(
                namespace=namespace, name=pvc_name
            )
            pvc_capacity = __class__.kubernetese_unit_convertor(pvc.status.capacity)
            pvc = Structs.Original.PVC(
                name=pvc_name, capacity=pvc_capacity.get("storage")
            )
            return name, pvc

    @staticmethod
    def get_deployment_name(pod: dict) -> Optional[str]:
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
                            owner_kind = (
                                owner_reference.kind
                            )  # Kind of the owning resource
                            owner_name = (
                                owner_reference.name
                            )  # Name of the owning resource
                case _:
                    pass
            return owner_name
        return None


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = NestedDict()
        return self[key]


class Config(Protocol):
    def prefix(self):
        ...

    def parse(
        self, annotations: dict[str, str], metrics: set[str]
    ) -> tuple[dict, dict]:
        ...


class SimulateConfig(Config):
    DISTRIBUTION_KEY_NAME: str = "type"
    DISTRIBUTION_PREFIX: str = "distribution"

    def prefix(self):
        return "simulate."

    def parse(
        self, annotations: dict[str, str], metrics: set[str]
    ) -> tuple[dict, dict]:
        distrib_config = self._create_string_config(config=annotations, options=metrics)
        config: dict = {
            "shutdown": distrib_config.get("shutdown", "5m"),
            "interval": distrib_config.get("interval", "30s"),
        }
        config.update({k: annotations[k] for k in config})
        #  user didnt specify the global distriv skip
        if self.DISTRIBUTION_PREFIX in distrib_config:
            _ = distrib_config.pop(self.DISTRIBUTION_PREFIX)
        # convert into stats distribution
        distrib_config = dict(
            filter(
                lambda itm: self.DISTRIBUTION_PREFIX in itm[1], distrib_config.items()
            )
        )
        self._covert_config_to_distrib(config=distrib_config)
        # update config by user information
        return distrib_config, config

    def _covert_config_to_distrib(self, config: dict[str, dict]):
        for k, v in config.items():
            kwargs = v[self.DISTRIBUTION_PREFIX]
            func = kwargs.pop(self.DISTRIBUTION_KEY_NAME)
            distribution = getattr(stats, func)
            if not distribution:
                raise ValueError(f"Unsupported distribution type: {distribution}")
            try:
                config[k] = distribution(**kwargs)
            except TypeError as e:
                e = str(e).strip("_parse_args()")
                raise ValueError(e)

    def _convert_dot_notation(self, annotations) -> NestedDict:
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

    def _create_string_config(self, config: dict, options: list[str]) -> dict:
        prefix: str = self.DISTRIBUTION_PREFIX
        distrib_type_k: str = self.DISTRIBUTION_KEY_NAME
        defualt_distrib: str = "norm"
        user_config: NestedDict = self._convert_dot_notation(config)[self.prefix()]
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


class Simulate(pydantic.BaseModel):
    file: pathlib.Path
    prom_port: int
    push_interval: int
    shudown_interval: int
    metrics: Dict[str, Tuple[Type[Setters.MetricSetter], Annotated[Any, "Metric"]]]
    _registration: Dict[str, Structs.Simulated.Pod] = pydantic.PrivateAttr(
        default_factory=dict
    )
    _config: Config = pydantic.PrivateAttr(default_factory=SimulateConfig)

    def _register_kopf_functions(self):
        kopf.timer("v1", "pods", interval=self.push_interval)(self._push_pod_metrics)
        kopf.timer("v1", "pods", interval=self.shudown_interval)(self._pod_shutdowner)
        kopf.on.create("v1", "pods")(self._registeration_of_pod)
        kopf.on.create("apps", "v1", "deployments")(self._inherent_deployment_anotation)
        kopf.on.startup()(KopFunctions.startup())

    def _inherent_deployment_anotation(
        self,
        logger: logging,
        uid: str,
        name: str,
        spec: dict,
        body: dict,
        namespace: str,
        **kwargs,
    ):
        prefix: str = self._config.prefix()
        annotations: dict = dict(
            map(
                lambda itm: (prefix + itm[0], itm[1]),
                self._extract_annotations(config=self._config, body=body).items(),
            )
        )
        pod_template = spec["template"]
        if "metadata" not in pod_template:
            pod_template["metadata"] = {}
        if "annotations" not in pod_template["metadata"]:
            pod_template["metadata"]["annotations"] = {}
        pod_template["metadata"]["annotations"].update(annotations)
        try:
            API.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body={"spec": {"template": pod_template}},
            )
            LogColor.info(
                f"Annotations copied from Deployment to Pod template in {name}"
            )
        except k8s.client.exceptions.ApiException as e:
            LogColor.error(f"Error updating Deployment {name}: {str(e)}")

    def _pod_shutdowner(
        self,
        logger: logging,
        uid: str,
        name: str,
        spec: dict,
        body: dict,
        namespace: str,
        **kwargs,
    ):
        try:
            if uid in self._registration:
                pod: Structs.Simulated.Pod = self._registration[uid]
                dep_name = Utils.get_deployment_name(body)
                assignment_time: Optional[datetime] = pod.assignment_time
                current_time: datetime = datetime.now()
                pod_not_assigned_yet: bool = not assignment_time
                if pod_not_assigned_yet:
                    return
                pod_as_expired = (assignment_time + pod.shutdown) <= current_time
                if pod_as_expired:
                    del self._registration[uid]
                    if dep_name:
                        print(dep_name)
                        scale = API.read_namespaced_deployment_scale(
                            name=dep_name, namespace=namespace
                        )
                        scale.spec.replicas = 0
                        API.replace_namespaced_deployment_scale(
                            dep_name, namespace, scale
                        )
                    patch = [
                        {"op": "replace", "path": "/status/phase", "value": "Succeeded"}
                    ]
                    V1.patch_namespaced_pod_status(
                        name=name, namespace=namespace, body=patch
                    )
                    LogColor.info(f"Shutdown pod: {name} with id {uid}")
        except Exception as e:
            LogColor.error(e)

    @classmethod
    def should_pod_be_simualted(cls, config: Config, pod: dict) -> bool:
        return bool(cls._extract_annotations(config=config, body=pod))

    @classmethod
    def _extract_annotations(cls, config: Config, body: dict) -> Dict[str, Any]:
        prefix = config.prefix()
        annotations: dict = body.metadata.annotations
        return dict(
            map(
                lambda tup: (tup[0][len(prefix) :], tup[1]),
                filter(lambda tup: tup[0].startswith(prefix), annotations.items()),
            )
        )

    @classmethod
    def create_pod(cls, config: Config, pod: dict) -> Structs.Original.Pod:
        node: str = pod.spec.get("node_name", None)
        namespace: str = pod.metadata["namespace"]
        name: str = pod.metadata["name"]
        volumes: list = pod.spec.get("volumes", [])
        annotations: dict = cls._extract_annotations(config=config, body=pod)
        volumes: Dict[str, Structs.Original.PVC] = dict(
            filter(
                lambda x: x,
                map(
                    lambda v: Utils.extract_pod_pvc(volume=v, namespace=namespace),
                    volumes,
                ),
            )
        )
        containers = dict()
        for container in pod.spec["containers"]:
            c_name: str = container["name"]
            requests: Optional[Dict[str, float]] = Utils.kubernetese_unit_convertor(
                container["resources"].get("requests", {})
            )
            limits: Optional[Dict[str, float]] = Utils.kubernetese_unit_convertor(
                container["resources"].get("limits", {})
            )
            containers[c_name] = Structs.Original.Container(
                name=c_name, limits=limits, requests=requests
            )
        return Structs.Original.Pod(
            annotations=annotations,
            name=name,
            node=node,
            namespace=namespace,
            containers=containers,
            volumes=volumes,
        )

    @property
    def metrics_set(self) -> set:
        return set(self.metrics.keys())

    def _push_pod_metrics(
        self,
        logger: logging,
        uid: str,
        name: str,
        spec: dict,
        body: dict,
        namespace: str,
        **kwargs,
    ):
        try:
            if uid in self._registration:
                pod: Structs.Simulated.Pod = self._registration[uid]
                pod_as_been_assigned: bool = not pod.node
                if pod_as_been_assigned:
                    pod.node = spec["nodeName"]
                for _name, (_setter, metric) in self.metrics.items():
                    value = next(iter(pod.resource_usage_generator[_name].rvs(size=1)))
                    _setter.set(metric=metric, pod=pod, value=value)
            elif self.should_pod_be_simualted(config=self._config, pod=body):
                LogColor.warn(f"Pod {name} was created before the script run time.")
                self._registeration_of_pod(
                    logger=logger,
                    uid=uid,
                    name=name,
                    spec=spec,
                    body=body,
                    namespace=namespace,
                    **kwargs,
                )
        except Exception as e:
            LogColor.error(e)

    def _registeration_of_pod(
        self,
        logger: logging,
        uid: str,
        name: str,
        spec: dict,
        body: dict,
        namespace: str,
        **kwargs,
    ):
        try:
            should_be_simulated = self.should_pod_be_simualted(
                config=self._config, pod=body
            )
            if not should_be_simulated:
                logging.debug(
                    f"Pod {name} [{uid}],  has no {self._config.prefix()} prefix in its  annotations."
                )
            elif uid not in self._registration:
                pod: Structs.Original.Pod = self.create_pod(
                    config=self._config, pod=body
                )
                dist_cfg, cfg = self._config.parse(
                    annotations=pod.annotations, metrics=self.metrics_set
                )
                self._registration[uid] = Structs.Simulated.Pod(
                    pod=pod,
                    resource_usage_generator=dist_cfg,
                    **cfg,
                )
                self._registration[uid].node = spec.get("nodeName")
                LogColor.info(f"Pod {name} [{uid}], has been registered")
            else:
                LogColor.warn(f"Pod {name} [{uid}],  already as been registered")
        except Exception as e:
            LogColor.error(e)

    def run(self):
        LogColor.info("[bold][Intializing][/bold] Register kube hooks ...")
        self._register_kopf_functions()
        LogColor.info(
            f"[bold][Starting][/bold] Exporter running on: [bold]`http://localhost:{self.prom_port}`[/bold]"
        )
        start_http_server(port=self.prom_port)
        kopf.run()
