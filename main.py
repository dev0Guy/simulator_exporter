from typing import Optional, Annotated, List, Union
from simulate_exporter import utils
from simulate_exporter import *
from prometheus_client import Gauge
from pathlib import Path
import warnings
import secrets
import string
import typer
import yaml
import os

# TODO: add clip acordeing to pod/deployment limits

app = typer.Typer(rich_markup_mode="rich")
warnings.filterwarnings("ignore", category=FutureWarning)


class Helper:
    @staticmethod
    def load_yaml_file(path):
        try:
            with open(path, "r") as file:
                return yaml.safe_load(file)
        except yaml.error.YAMLError as e:
            return LogColor.error(f"[bold]{path} is't valid yaml file:[/bold] \n  {e}")
        except FileNotFoundError as e:
            return LogColor.error(f"{e}")

    @staticmethod
    def generate_name(length: int = 10, prefix: str = "", suffix: str = ""):
        characters = string.ascii_letters + string.digits
        random_string = "".join(secrets.choice(characters) for _ in range(length))
        return prefix + random_string + suffix


# METRICS = {
#     "cpu": (
#         CPUMetricSetter,
#         Gauge(
#             "container_cpu_usage",
#             "cpu usage of containers in the machine.",
#             ["namespace", "node", "pod", "container"],
#         ),
#     ),
#     # "memory": (
#     #     MemoryMetricSetter,
#     #     Gauge(
#     #         "container_memory_usage",
#     #         "memory usage of the containers in machine.",
#     #         ["namespace", "node", "pod", "container"],
#     #     ),
#     # ),
#     "gpu": (
#         GPUMetricSetter,
#         Gauge(
#             "gpu_usage_percent",
#             "gpu usage of the containers in machine.",
#             ["namespace", "node", "pod", "container"],
#         ),
#     ),
#     "pvc": (
#         PVCMetricSetter,
#         Gauge(
#             "pvc_storage_usage_bytes",
#             "PVC Storage Usage in Bytes",
#             ["namespace", "node", "pod", "pvc"],
#         ),
#     ),
# }

METRICS = [PVCMetricSetter]


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
    # TODO: create a replay from file (of existing promethues records)
    simulate = Simulate(
        push_interval=push_interval,
        shudown_interval=shudown_interval,
        prom_port=prom_port,
    )
    simulate.run()


@app.command()
def generate(
    files: Annotated[List[Path], typer.Argument(help="dsaasdsa")] = None,
    regex: Annotated[str, typer.Option(help="Mock File regex")] = "*",
    # mock_files: Annotated[List[Path], typer.Argument(help="Mock Files")],
    # output_path: Annotated[Path, typer.Argument(help="Destenation of generated file")]
):
    """
    Generate Use case of pod load
    """
    if not files:
        files: List[Path] = utils.FilesHelper.select_file_by_regex(regex=regex)
    gen = Generator(to_mock=files)
    # if not output_path or output_path == Path("."):
    #     output_path = Helper.generate_name(length=10,prefix=os.getcwd(),suffix=".yaml")
    # out_dir_name: str = os.path.dirname(output_path)
    # if not os.path.exists(out_dir_name):
    #     return LogColor.error(f"Folder [bold]'{out_dir_name}'[bold] doesn't exist.")


if __name__ == "__main__":
    import logging

    # TODO: today only work in deployment with one pod, if have couple the moment one of them stop all of them will be killed as well, nned to change
    # TODO: change to work with an deployment annotation as well
    logging.basicConfig(
        level=logging.ERROR,  # Set the desired logging level (e.g., logging.DEBUG, logging.INFO)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # Customize the date and time format
    )
    app()
