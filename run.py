import json
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from lmclient import LMClient, AzureChat, OpenAIChat, MinimaxChat

from redstar.tasks.task import TaskRegistry


class ModelType(str, Enum):
    azure_gpt_3_5 = 'azure_gpt_3_5'
    openai_gpt_4 = 'openai_gpt_4'
    openai_gpt_3_5_turbo = 'openai_gpt_3_5_turbo'
    minimax_5_5 = 'minimax_5_5'


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
    client = LMClient(model, **kwargs)
    return client


def main(
    model: ModelType,
    task: TaskType,  # type: ignore
    output_dir: Path = Path('outputs'),
    show: bool = False,
    max_records: Optional[int] = None,
    timeout: int = 40,
    max_requests_per_minute: int = 30,
    async_capacity: int = 5,
    error_mode: str = 'ignore',
    cache_dir: Optional[str] = 'restar_cache',
):
    task_instance = TaskRegistry[task.value]
    client = load_lm_client(
        model,
        timeout=timeout,
        max_requests_per_minute=max_requests_per_minute,
        async_capacity=async_capacity,
        error_mode=error_mode,
        cache_dir=cache_dir,
    )
    pipeline = task_instance.create_pipeline_func(client)
    dataset = task_instance.load_dataset_func()
    if show:
        pipeline.show(dataset[0])
        return

    if max_records is not None:
        dataset = dataset[:max_records]

    evaluation_result = pipeline(dataset)

    subdir = output_dir / model.value / task.value
    subdir.mkdir(parents=True, exist_ok=True)

    with open(subdir / 'records.jsonl', 'w') as f:
        for result in evaluation_result.records:
            f.write(json.dumps(result) + '\n')

    with open(subdir / 'metrics.json', 'w') as f:
        json.dump(evaluation_result.metric, f, indent=2)

    print(evaluation_result.metric)


if __name__ == '__main__':
    typer.run(main)
