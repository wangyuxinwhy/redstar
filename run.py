import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, cast

import typer
from lmclient import AzureChat, LMClient, MinimaxChat, OpenAIChat

from redstar.tasks.task import TaskRegistry


class ModelType(str, Enum):
    azure_gpt_3_5 = 'azure_gpt_3_5'
    openai_gpt_4 = 'openai_gpt_4'
    openai_gpt_3_5_turbo = 'openai_gpt_3_5_turbo'
    minimax_5_5 = 'minimax_5_5'
    minimax_5 = 'minimax_5'


TaskType = Enum('TaskType', names={key: key for key in TaskRegistry}, type=str)


def load_lm_client(model_type: ModelType, **kwargs):
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
    return client


def generate_task_filter_function(filter_code: str):
    if not filter_code.startswith('lambda task:'):
        filter_code = 'lambda task: ' + filter_code
    return eval(filter_code)


def main(
    model: Annotated[ModelType, typer.Option(...)],
    task_name: TaskType | None = None,  # type: ignore
    task_filter: Optional[str] = None,
    output_dir: Path = Path('outputs'),
    show: bool = False,
    max_records: Optional[int] = None,
    timeout: int = 40,
    max_requests_per_minute: int = 30,
    async_capacity: int = 5,
    error_mode: str = 'ignore',
    cache_dir: Optional[str] = 'restar_cache',
):
    if task_name is not None and task_filter is not None:
        raise ValueError('Either task_name or task_filter should be specified, not both')

    if task_name:
        tasks = [TaskRegistry[task_name.value]]
    elif task_filter is not None:
        task_filter = cast(str, task_filter)
        filter_function = generate_task_filter_function(task_filter)
        tasks = [task for task in TaskRegistry.values() if filter_function(task)]
    else:
        tasks = [task for task in TaskRegistry.values()]

    print(f'Running {len(tasks)} tasks: {", ".join(task.task_name for task in tasks)}')

    client = load_lm_client(
        model,
        timeout=timeout,
        max_requests_per_minute=max_requests_per_minute,
        async_capacity=async_capacity,
        error_mode=error_mode,
        cache_dir=cache_dir,
    )

    for task in tasks:
        if show:
            task.pipeline.show(client, task.records[0])
            continue

        if max_records is not None:
            records = task.records[:max_records]
        else:
            records = task.records

        evaluation_result = task.pipeline(client, records)

        subdir = output_dir / model.value / task.task_name
        subdir.mkdir(parents=True, exist_ok=True)

        with open(subdir / 'records.jsonl', 'w') as f:
            for result in evaluation_result.records:
                f.write(json.dumps(result) + '\n')

        with open(subdir / 'metrics.json', 'w') as f:
            json.dump(evaluation_result.metric, f, indent=2)

        print(evaluation_result.metric)


if __name__ == '__main__':
    typer.run(main)
