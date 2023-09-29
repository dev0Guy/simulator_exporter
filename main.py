from typing import Optional, Annotated, Dict
from simulate_exporter import Simulate, Setters, LogColor
from prometheus_client import Gauge
from pathlib import Path
import warnings
import typer

app = typer.Typer(rich_markup_mode="rich")
warnings.filterwarnings("ignore", category=FutureWarning)


METRICS = {
    "cpu": (
        Setters.CPUMetricSetter,
        Gauge(
            "container_cpu_usage",
            "cpu usage of containers in the machine.",
            ["namespace", "node", "pod", "container"],
        ),
    ),
    "memory": (
        Setters.MemoryMetricSetter,
        Gauge(
            "container_memory_usage",
            "memory usage of the containers in machine.",
            ["namespace", "node", "pod", "container"],
        ),
    ),
    "gpu": (
        Setters.GPUMetricSetter,
        Gauge(
            "gpu_usage_percent",
            "gpu usage of the containers in machine.",
            ["namespace", "node", "pod", "container"],
        ),
    ),
    "pvc": (
        Setters.PVCMetricSetter,
        Gauge(
            "pvc_storage_usage_bytes",
            "PVC Storage Usage in Bytes",
            ["namespace", "node", "pod", "pvc"],
        ),
    ),
}


@app.command()
def simulate(
    prom_port: int,
    path: Annotated[Optional[Path], typer.Option(help="Replay load usecase.")] = Path(
        "/"
    ),
    push_interval: Annotated[
        Optional[int], typer.Option(help="Prometheus push interval")
    ] = 5,
    shudown_interval: Annotated[
        Optional[int],
        typer.Option(help="How nuch time to delete pod after shutdown time"),
    ] = 1,
):
    """
    Simulate pod metrics
    """
    simulate = Simulate(
        file=path,
        push_interval=push_interval,
        shudown_interval=shudown_interval,
        prom_port=prom_port,
        metrics=METRICS,
    )
    simulate.run()


@app.command()
def generate(
    path: Annotated[Path, typer.Option(help="File generate metatadata Path")] = None
):
    """
    Generate Use case of pod load
    """
    pass


if __name__ == "__main__":
    import logging

    # TODO: today only work in deployment with one pod, if have couple the moment one of them stop all of them will be killed as well, nned to change
    logging.basicConfig(
        level=logging.ERROR,  # Set the desired logging level (e.g., logging.DEBUG, logging.INFO)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # Customize the date and time format
    )
    app()
