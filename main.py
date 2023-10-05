from typing import Optional, Annotated, List
from simulate_exporter import utils
from simulate_exporter import *
from pathlib import Path
import numpy as np
from time import sleep
import warnings
import typer
import os

app = typer.Typer(rich_markup_mode="rich")
warnings.filterwarnings("ignore", category=FutureWarning)

METRICS = [PVCMetricSetter]


@app.command()
def simulate(
    files: List[Path] = typer.Argument(help="List of specific files or directories"),
    regex: Optional[str] = typer.Option(default=None,help="Regex pattern to match files"),
    prometheus_port: int = typer.Option(default=9090, help="Prometheus port"),
    interval: int = typer.Option(default=5, help="Promethues push interval in seconds"),
    shutdown: int = typer.Option(default=1, help="Await time after pod deletion"),
    seed: Optional[int] = typer.Option(default=None,help="Random seed"),
):
    """
    Simulate pod metrics
    """
    np.random.seed(seed)
    matching_files = utils.get_matching_files(files, regex) if regex else files
    Simulate(
        push_interval=interval,
        shudown_interval=shutdown,
        prom_port=prometheus_port,
        files=matching_files
    ).run()



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
    if not output_path or output_path == Path("."):
        output_path = utils.Helper.generate_name(length=10,prefix=os.getcwd(),suffix=".yaml")
    out_dir_name: str = os.path.dirname(output_path)
    if not os.path.exists(out_dir_name):
        return LogColor.error(f"Folder [bold]'{out_dir_name}'[bold] doesn't exist.")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.ERROR,  # Set the desired logging level (e.g., logging.DEBUG, logging.INFO)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # Customize the date and time format
    )
    app()
