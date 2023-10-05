from pathlib import Path
from typing import Optional, Dict, Tuple, Any, Annotated, Type, List
import kubernetes as k8s
import pydantic
import logging
import rich
import yaml
import os
import glob
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


def diffrent(first: dict, second: dict) -> dict:
    key_diff: set = set(first) - set(second)
    return {k: first[k] for k in key_diff}


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = NestedDict()
        return self[key]

    @classmethod
    def create_from_dot_string(cls, value: dict) -> "NestedDict":
        def dot_notation(keys: list[str], v: str, prev: dict):
            for _k in keys[:-1]:
                prev = prev[_k]
            prev[keys[-1]] = v

        result = NestedDict()
        if value:
            value = sorted(
                map(lambda item: (item[0].split("."), item[1]), value.items()),
                key=lambda item: len(item[0]),
                reverse=True,
            )
            for item in value:
                dot_notation(*item, prev=result)
        return result


class LogColor:
    def warn(log: str):
        rich.print(f"[yellow]{log}[/yellow]")

    def info(log: str):
        rich.print(f"[blue]{log}[/blue]")

    def error(log: str):
        rich.print(f"[red]{log}[/red]")

    def regular(log: str):
        rich.print(log)


###################################
class FilesHelper:
    @staticmethod
    def select_file_by_regex(regex: str) -> List[Path]:
        """
        Select all files that fit the regex.
        """
        files: List[Path] = glob.glob(regex)
        return files


class YamlHelper:
    _EXTENSION_OPTIONS = [".yaml", ".yml"]

    @classmethod
    def assert_and_return_file(cls, path: Path) -> dict:
        """
        Assert yaml file end with the write extenstions. {_EXTENSION_OPTIONS}.

        Raise:
        - ValueError: the file doesn't have the right extenstion.

        Returns:
        - (Path): original argument file content
        """
        _, extension = os.path.splitext(path)
        file_extension_is_yaml: bool = extension in cls._EXTENSION_OPTIONS
        if not file_extension_is_yaml:
            raise ValueError(
                f"File extention'{extension}' doesn't match {cls._EXTENSION_OPTIONS}"
            )
        with open(path, "r") as file:
            return yaml.safe_load(file)
