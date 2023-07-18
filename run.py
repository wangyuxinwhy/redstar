import logging
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, Sequence

import typer
from lmclient import AzureChat, LMClient, MinimaxChat, OpenAIChat

from redstar.model import Model
from redstar.tasks.task import TaskRegistry, load_tasks, run_tasks
from redstar.types import Messages
from redstar.utils import is_jupyter

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    azure_gpt_3_5 = 'azure_gpt_3_5'
    openai_gpt_4 = 'openai_gpt_4'
    openai_gpt_3_5_turbo = 'openai_gpt_3_5_turbo'
    minimax_5_5 = 'minimax_5_5'
    minimax_5 = 'minimax_5'


TaskType = Enum('TaskType', names={key: key for key in TaskRegistry}, type=str)


class LMClientWrapper(Model):
    def __init__(self, client: LMClient, async_run: bool | None = None) -> None:
        self.client = client
        self.async_run = async_run or (not is_jupyter())

    def __call__(self, prompts: Sequence[Messages], **kwargs) -> Sequence[str]:
        if self.async_run:
            return self.client.async_run(prompts, **kwargs)
        else:
            return self.client.run(prompts, **kwargs)


def load_client_model(model_type: ModelType, **kwargs):
    match model_type:
        case ModelType.azure_gpt_3_5:
            model = AzureChat()
        case ModelType.openai_gpt_4:
            model = OpenAIChat(model_name='gpt-4')
        case ModelType.openai_gpt_3_5_turbo:
            model = OpenAIChat(model_name='gpt-3.5-turbo')
        case ModelType.minimax_5_5:
            model = MinimaxChat('abab5.5-chat')
        case ModelType.minimax_5:
            model = MinimaxChat('abab5-chat')
    client = LMClient(model, **kwargs)
    client = LMClientWrapper(client)
    return client


def generate_task_filter_function(filter_code: str):
    if not filter_code.startswith('lambda task:'):
        filter_code = 'lambda task: ' + filter_code
    return eval(filter_code)


def main(
    model: Annotated[ModelType, typer.Option(...)],
    task: TaskType | None = None,  # type: ignore
    filter: Optional[str] = None,
    output_dir: Path = Path('outputs'),
    debug: bool = False,
    max_records: Optional[int] = None,
    timeout: int = 40,
    max_requests_per_minute: int = 30,
    async_capacity: int = 5,
    error_mode: str = 'ignore',
    cache_dir: Optional[str] = 'restar_cache',
):
    task_name = task.value if task is not None else None
    task_filter = generate_task_filter_function(filter) if filter is not None else None
    tasks = load_tasks(task_name=task_name, task_filter=task_filter)

    client_model = load_client_model(
        model,
        timeout=timeout,
        max_requests_per_minute=max_requests_per_minute,
        async_capacity=async_capacity,
        error_mode=error_mode,
        cache_dir=cache_dir,
    )

    run_tasks(
        model=client_model,
        tasks=tasks,
        output_dir=output_dir,
        debug=debug,
        max_records=max_records,
    )


if __name__ == '__main__':
    typer.run(main)
