from typing import Optional
from simulate_exporter import MetricSimulatorExporter, AnnotationParser
import logging
import typer
import kopf

"""
Entry point of the script: 
only has one script, the one that run the simulator it self.
"""


def kopf_startup(settings: kopf.OperatorSettings, **_):
    settings.posting.level = logging.ERROR
    # don't show kopf defualt logs only the one providded
    settings.posting.enabled = False


def main(prom_port: Optional[int] = 9153, simulate_interval: Optional[int] = 5):
    """
    Get optional arguments to the script, run the metric it'self
    """
    parser = AnnotationParser
    exporter = MetricSimulatorExporter(prom_port, parser)
    kopf.timer("v1", "pods", interval=simulate_interval)(exporter.interval)
    kopf.on.create("v1", "pods")(exporter.register_pod)
    kopf.on.startup()(kopf_startup)
    kopf.run()


if __name__ == "__main__":
    typer.run(main)
