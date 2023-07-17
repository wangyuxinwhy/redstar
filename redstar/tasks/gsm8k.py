from typing import cast

from datasets import DatasetDict, load_dataset

from redstar.metrics import Accuracy
from redstar.parsers import RegexParser
from redstar.pipeline import EvaluationPipeline
from redstar.processors import BaseSingleProcessor, LambdaProcessor
from redstar.prompt import BasePrompt
from redstar.tasks.task import Task
from redstar.types import Record, Records


def _convert_to_int(record: Record):
    try:
        record['pred'] = int(record['parsed_result'])
    except ValueError:
        print(f'Cannot convert {record["parsed_result"]} to int, set pred to 0')
        record['pred'] = 0
    return record


class GSM8KZeroShotPrompt(BasePrompt):
    def __init__(self, system_content: str | None = None):
        default_system_content = (
            'Answer the primary school math problem, YOU MUST end with this format: the answer is |<YOUR NUMBER ANSWER>|'
        )

        self.system_content = system_content or default_system_content

    def compile(self, question: str, **kwargs):
        messages = [{'role': 'system', 'content': self.system_content}, {'role': 'user', 'content': question}]
        return messages


class GSM8KNumberAnswerExtractor(BaseSingleProcessor):
    def _process(self, record: Record) -> Record:
        answer = record['answer']
        number_answer = answer.rsplit('####')[1].strip()
        record['target'] = int(number_answer)
        return record


def load_gsm8k_dataset(split: str = 'test'):
    dataset_dict = load_dataset('gsm8k', 'main')
    dataset_dict = cast(DatasetDict, dataset_dict)
    records = [i for i in dataset_dict[split]][:10]
    records = cast(Records, records)
    return records


def create_zero_shot_pipeline(lmclient, async_run: bool = True):
    zero_shot_pipeline = EvaluationPipeline(
        lmclient,
        preprocessors=GSM8KNumberAnswerExtractor(),
        prompt=GSM8KZeroShotPrompt(),
        parser=RegexParser(r'\|.*?(\d+).*?\|'),
        postprocessors=LambdaProcessor(_convert_to_int),
        metrics=Accuracy(),
        default_client_kwargs={'temperature': 0.0},
        async_run=async_run,
    )
    return zero_shot_pipeline


gsm8k_zero_shot_task = Task(
    task_name='gsm8k_zero_shot',
    load_dataset_func=load_gsm8k_dataset,
    create_pipeline_func=create_zero_shot_pipeline,
)
