from typing import Dict, Tuple, Annotated, Type, Any, List
from simulate_exporter.prom import start_http_server
from simulate_exporter.utils import LogColor, kubernetese_load_config
from simulate_exporter.k8s_objects import get_deployment_name, SimulatedPod
from datetime import datetime
from typing import Optional
import kubernetes as k8s
import pydantic
import logging
import kopf

API, V1 = kubernetese_load_config()


class Simulate(pydantic.BaseModel):
    prom_port: int
    push_interval: int
    shudown_interval: int
    _registration: Dict[Annotated[str, "POD_UID"], SimulatedPod] = pydantic.PrivateAttr(
        default_factory=dict
    )

    def register_kopf_functions(self) -> None:
        kopf.on.field("apps", "v1", "deployments", field="status.replicas")(
            self.inherent_deployment_anotation
        )
        kopf.timer("v1", "pods", interval=self.push_interval)(self.push_metrics)
        kopf.timer("v1", "pods", interval=self.shudown_interval)(self.shutdown)
        kopf.on.create("v1", "pods")(self.register)
        kopf.on.delete("v1", "pods")(self.delete)

    def inherent_deployment_anotation(
        self,
        logger: logging,
        uid: str,
        name: str,
        spec: dict,
        body: dict,
        namespace: str,
        status: dict,
        **kwargs,
    ):
        annotations: dict = dict(
            filter(
                lambda itm: SimulatedPod.prefix in itm[0],
                body.metadata.annotations.items(),
            )
        )
        if not annotations:
            return
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
            should_simulate_pod: bool = SimulatedPod.should_be_simualted(
                annotations=annotations
            )
            if not should_simulate_pod:
                logger.debug(
                    f"Pod {name} [{uid}],  has no {SimulatedPod.prefix} prefix in its  annotations."
                )
            elif not already_registered:
                _, dep_uid = get_deployment_name(body)
                self._registration[uid] = SimulatedPod(deployment=dep_uid,**body)
                self._registration[uid].node = spec.get("nodeName")
                LogColor.info(f"Pod {name} [{uid}], has been registered")
            else:
                LogColor.warn(f"Pod {name} [{uid}],  already as been registered")
        except Exception as e:
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
            should_simulate_pod: bool = SimulatedPod.should_be_simualted(
                annotations=annotations
            )
            if already_registered:
                LogColor.info("Pod Metrics are been pushed")
                pod: SimulatedPod = self._registration[uid]
                if not pod.is_assigned:
                    pod.node = spec["nodeName"]
                else:
                    pod.push_metrics()
            elif should_simulate_pod:
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
        try:
            if uid in self._registration:
                pod: SimulatedPod = self._registration[uid]
                dep_name, dep_uid = get_deployment_name(body)
                assignment_time: Optional[datetime] = pod.assignment_time
                current_time: datetime = datetime.now()
                pod_not_assigned_yet: bool = not assignment_time
                if pod_not_assigned_yet:
                    return
                pod_as_expired = (assignment_time + pod.shutdown) <= current_time
                if pod_as_expired:
                    LogColor.info(f"Deleting pod {uid}")
                    del self._registration[uid]
                    if dep_name:
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

    def run(self):
        LogColor.info("[bold][Intializing][/bold] Register kube hooks ...")
        self.register_kopf_functions()
        LogColor.info(
            f"[bold][Starting][/bold] Exporter running on: [bold]`http://localhost:{self.prom_port}`[/bold]"
        )
        start_http_server(port=self.prom_port)
        kopf.run()

    def delete(
        self,
        logger: logging,
        uid: str,
        name: str,
        spec: dict,
        body: dict,
        namespace: str,
        **kwargs,
    ):
        if uid in self._registration:
            del self._registration[uid]
            LogColor.info(f"Remove pod: {name} {uid}")
