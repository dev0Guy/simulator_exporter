from typing import List, Any, Dict
from simulate_exporter.k8s_objects import Pod
from simulate_exporter.utils import NestedDict, YamlHelper
from pathlib import Path
import pydantic
import yaml


class Generator(pydantic.BaseModel):
    to_mock: List[Pod]

    @pydantic.validator("to_mock", pre=True, always=True)
    def _read_mock_files(cls, to_mock: List[Path]) -> List[Pod]:
        """
        Load content of yaml files.
        Arguemnts:
        - to_mock (List[Path]): files to read from
        Returns:
        - (list): all files content
        """
        to_mock: List[dict] = map(YamlHelper.assert_and_return_file, to_mock)
        from pprint import pprint

        for x in to_mock:
            print(x, end="\n" + "#" * 28 + "\n")
            pprint(Pod(pod=x))
        return []
        # return list(
        #     map(
        #         lambda x: Pod({'pod': x}),
        #         to_mock
        #     )
        # )

    @classmethod
    def __parse_yml(cls, file: dict) -> Pod:
        return


def filter_by_prefix(values: List[str], prefix: str) -> List[str]:
    """
    Filter String by thier prefix

    Returns:
    - (List[str]): all elements that start with the providded prefix
    """
    return list(filter(lambda value: value.startswith(prefix), values))
