from dataclasses import dataclass
from typing import Callable

from lmclient import LMClient

from redstar.pipeline import EvaluationPipeline
from redstar.types import Records


@dataclass
class Task:
    task_name: str
    load_dataset_func: Callable[[], Records]
    create_pipeline_func: Callable[[LMClient], EvaluationPipeline]
