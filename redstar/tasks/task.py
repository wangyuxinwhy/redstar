import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from redstar.datasets import DatasetRegistry
from redstar.pipeline import EvaluationPipeline
from redstar.types import Records

TaskRegistry: dict[str, 'Task'] = {}
FixturesDir = Path(__file__).parent / 'fixtures'
logger = logging.getLogger(__name__)


@dataclass
class Task:
    task_name: str
    dataset_name: str
    pipeline: EvaluationPipeline
    load_records_function: Callable[[], Records] | None = None
    tags: set[str] = field(default_factory=set)

    def __post_init__(self):
        TaskRegistry[self.task_name] = self
        self._records = None

    @property
    def records(self):
        if self._records is None:
            if self.load_records_function is None:
                self._records = DatasetRegistry[self.dataset_name]()
            else:
                self._records = self.load_records_function()
        return self._records

    @records.setter
    def records(self, records: Records):
        self._records = records


def load_tasks(task_name: str | None = None, task_filter: Callable[[Task], bool] | None = None):
    if task_name is not None and task_filter is not None:
        raise ValueError('Either task_name or task_filter should be specified, not both')

    if task_name:
        tasks = [TaskRegistry[task_name]]
    elif task_filter is not None:
        tasks = [task for task in TaskRegistry.values() if task_filter(task)]
    else:
        tasks = [task for task in TaskRegistry.values()]
    return tasks


def run_tasks(
    model, tasks: Sequence[Task], output_dir: Path | None = None, debug: bool = False, max_records: int | None = None
):
    logger.info(f'Running {len(tasks)} tasks: {", ".join(task.task_name for task in tasks)}')

    for task in tasks:
        if debug:
            task.pipeline.debug(model, task.records[0])
            continue

        if max_records is not None:
            task.records = task.records[:max_records]

        logger.info(f'Running {task.task_name} of {model}')
        evaluation_result = task.pipeline(model, task.records)

        if output_dir is None:
            continue

        subdir = output_dir / model.value / task.task_name
        subdir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving {task.task_name} of {model} results to {subdir}')

        with open(subdir / 'records.jsonl', 'w') as f:
            for result in evaluation_result.records:
                f.write(json.dumps(result) + '\n')

        with open(subdir / 'metrics.json', 'w') as f:
            json.dump(evaluation_result.metric, f)

        logger.info(f'Evaluation result for {task.task_name}: {evaluation_result.metric}')
