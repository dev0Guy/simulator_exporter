from typing import Optional, Annotated, List
from simulate_exporter import utils
from simulate_exporter import *
from pathlib import Path
import numpy as np
import warnings
import typer
import os

app = typer.Typer(rich_markup_mode="rich")
warnings.filterwarnings("ignore", category=FutureWarning)

METRICS = [PVCMetricSetter]


@app.command()
def simulate(
    files: List[Path] = typer.Argument(help="List of specific files or directories"),
    regex: Optional[str] = typer.Option(default="tmp/*",help="Regex pattern to match files"),
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
    config: Path = typer.Argument(help="Config file location"),
    template: Path = typer.Argument(help="Template to generate from"),
    seed: Optional[int] = typer.Option(default=None,help="Random seed"),
    number: Optional[int] = typer.Option(default=5, help="Number of generator output"),
    out: Optional[Path] = typer.Option(default=Path("tmp"),help="Output directory")
):
    """
    Generate Use case of pod load
    """
    np.random.seed(seed)
    config = utils.YamlHelper.assert_and_return_file(path=config)
    generator = Generator(
        config=config,
    )
    file_names: list[str] = generator.create(
        template_file=template,
        number=number,
        out=out
    )
    LogColor.info(f"[bold][Writting][/bold] into files \n\t {file_names} ")



if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.ERROR,  # Set the desired logging level (e.g., logging.DEBUG, logging.INFO)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # Customize the date and time format
    )
    app()
