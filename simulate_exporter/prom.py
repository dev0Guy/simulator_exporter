from typing import runtime_checkable, Protocol, Any
from prometheus_client import start_http_server, Metric


@runtime_checkable
class MetricSetter(Protocol):
    @staticmethod
    def set(metric: Metric, value: Any, **kwargs):
        pod: str = kwargs.pop("name")
        containers: dict = kwargs.pop("containers")
        node: dict = kwargs.pop("node")
        namespace: dict = kwargs.pop("namespace")
        for c_name, _ in containers.items():
            metric.labels(
                pod=pod, container=c_name, node=node, namespace=namespace
            ).set(value)


class PVCMetricSetter(MetricSetter):
    @staticmethod
    def set(metric: Metric, value: Any, **kwargs):
        pod: str = kwargs.pop("name")
        node: dict = kwargs.pop("node")
        namespace: dict = kwargs.pop("namespace")
        volumes = kwargs.pop("volumes")
        for volume_name, _ in volumes.items():
            metric.labels(pod=pod, pvc=volume_name, node=node, namespace=namespace).set(
                value
            )
