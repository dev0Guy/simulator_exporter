from typing import Dict, Tuple, Annotated, Type, Any
from simulate_exporter.k8s_objects import Pod, SimulatedPod
from simulate_exporter.prom import MetricSetter, Metric, start_http_server
from simulate_exporter.utils import LogColor
import pydantic
import logging
import kopf


class Simulate(pydantic.BaseModel):
    prom_port: int
    push_interval: int
    shudown_interval: int
    metrics: Dict[str, Tuple[Type[MetricSetter], Annotated[Any, "Metric"]]]
    _registration: Dict[str, SimulatedPod] = pydantic.PrivateAttr(default_factory=dict)

    def register_kopf_functions(self) -> None:
        kopf.timer("v1", "pods", interval=self.push_interval)(self.push_metrics)
        kopf.timer("v1", "pods", interval=self.shudown_interval)(self.shutdown)
        kopf.on.create("v1", "pods")(self.register)

    def register(
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
            already_registered: bool = uid in self._registration
            annotations: list = list(body.metadata.annotations.keys())
            if not SimulatedPod.should_be_simualted(annotations=annotations):
                logger.debug(
                    f"Pod {name} [{uid}],  has no {SimulatedPod.prefix} prefix in its  annotations."
                )
            elif not already_registered:
                self._registration[uid] = SimulatedPod(
                    pod=body,
                )
                self._registration[uid].node = spec.get("nodeName")
                LogColor.info(f"Pod {name} [{uid}], has been registered")
            else:
                LogColor.warn(f"Pod {name} [{uid}],  already as been registered")
        except Exception as e:
            e.with_traceback()
            LogColor.error(e)

    def push_metrics(
        self,
        logger: logging,
        uid: str,
        name: str,
        spec: dict,
        body: dict,
        namespace: str,
        **kwargs,
    ):
        """
        Push register pod metrics, into prometheus.
        """

        try:
            already_registered: bool = uid in self._registration
            annotations: list[str] = list(body.metadata.annotations.keys())
            if already_registered:
                LogColor.info("Pod Metrics are been pushed")
                pod: SimulatedPod = self._registration[uid]
                if not pod.is_assigned:
                    self.node = spec["nodeName"]
                else:
                    pod.push_metrics(metrics=self.metrics)
            elif SimulatedPod.should_be_simualted(annotations=annotations):
                self.register(
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

    def shutdown(
        self,
        logger: logging,
        uid: str,
        name: str,
        spec: dict,
        body: dict,
        namespace: str,
        **kwargs,
    ):
        pass

    def run(self):
        LogColor.info("[bold][Intializing][/bold] Register kube hooks ...")
        self.register_kopf_functions()
        LogColor.info(
            f"[bold][Starting][/bold] Exporter running on: [bold]`http://localhost:{self.prom_port}`[/bold]"
        )
        start_http_server(port=self.prom_port)
        kopf.run()
