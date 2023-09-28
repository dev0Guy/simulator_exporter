from typing_extensions import Unpack
import pydantic
from typing import List, Dict, Optional
from datetime import datetime

from pydantic.config import ConfigDict


class Container(pydantic.BaseModel):
    name: str
    limits: Dict[str, float]
    requests: Dict[str, float]


class PVC(pydantic.BaseModel):
    name: str
    capacity: Optional[float]


class Pod(pydantic.BaseModel):
    name: str
    node: Optional[str]
    annotations: Dict[str, str]
    namespace: str
    containers: Dict[str, Container]
    volumes: Dict[str, PVC]


class SimulatedPod(pydantic.BaseModel):
    pod: Pod
    shutdown: str
    interval: str
    resource_usage_generator: dict
    _is_schedule: bool = False
    _start_time: Optional[datetime] = None
    _schedule_time: Optional[datetime] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._start_time = datetime.now()

    @property
    def is_schedule(self):
        return self._is_schedule

    @is_schedule.setter
    def is_schedule(self, value):
        if value and self._schedule_time is None:
            self._schedule_time = datetime.now()
        self._is_schedule = value
