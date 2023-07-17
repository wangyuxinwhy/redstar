import json
from enum import Enum
from pathlib import Path

import typer
from lmclient import AzureChat, LMClient

from redstar.tasks.gsm8k import gsm8k_zero_shot_task
from redstar.tasks.task import Task


class ModelType(str, Enum):
    azure_gpt_3_5 = 'azure_gpt_3_5'


class TaskType(str, Enum):
    gsm8k_zero_shot = 'gsm8k_zero_shot'


task_mapping: dict[TaskType, Task] = {
    TaskType.gsm8k_zero_shot: gsm8k_zero_shot_task,
}


def load_lm_client(model_type: ModelType, **kwargs):
    match model_type:
        case ModelType.azure_gpt_3_5:
            model = AzureChat()
            client = LMClient(model, **kwargs)

    return client


def main(
    model: ModelType = ModelType.azure_gpt_3_5,
    task: TaskType = TaskType.gsm8k_zero_shot,
    output_dir: Path = Path('outputs'),
):
    task_instance = task_mapping[task]
    client = load_lm_client(model)
    pipeline = task_instance.create_pipeline_func(client)
    dataset = task_instance.load_dataset_func()
    evaluation_result = pipeline(dataset)

    subdir = output_dir / task
    subdir.mkdir(parents=True, exist_ok=True)

    with open(subdir / f'records.jsonl', 'w') as f:
        for result in evaluation_result.records:
            f.write(json.dumps(result) + '\n')

    with open(subdir / f'metrics.json', 'w') as f:
        json.dump(evaluation_result.metric, f, indent=2)

    print(evaluation_result.metric)


if __name__ == '__main__':
    typer.run(main)
