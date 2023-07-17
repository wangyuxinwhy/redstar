from functools import partial
from pathlib import Path
from typing import Sequence, cast

from datasets import DatasetDict, load_dataset

from redstar.metrics import Accuracy
from redstar.parsers import RegexParser
from redstar.pipeline import EvaluationPipeline
from redstar.processors import BaseSingleProcessor, LambdaProcessor
from redstar.prompt import BasePrompt
from redstar.tasks.task import FixturesDir, Task
from redstar.types import Record, Records
from redstar.utils import load_from_json


def _convert_to_float(record: Record):
    try:
        record['pred'] = float(record['parsed_result'])
    except ValueError:
        print(f'Cannot convert {record["parsed_result"]} to float, set pred to 0')
        record['pred'] = 0.0
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


class GSM8KFewShotPrompt(BasePrompt):
    def __init__(
        self, few_shot_examples: Sequence[dict[str, str]], dialog_style: bool = False, system_content: str | None = None
    ):
        self.few_shot_examples = few_shot_examples
        self.dialog_style = dialog_style
        default_system_content = 'Follow the given examples and answer the question.'
        self.system_content = system_content or default_system_content

    @classmethod
    def from_file(cls, few_shot_file: str | Path, **kwargs):
        examples = load_from_json(few_shot_file)
        return cls(examples, **kwargs)

    def compile(self, question: str, **kwargs):
        if self.dialog_style:
            messages = [{'role': 'system', 'content': self.system_content}]
            for example in self.few_shot_examples:
                messages.append({'role': 'user', 'content': example['question']})
                messages.append({'role': 'assistant', 'content': example['answer']})
            messages.append({'role': 'user', 'content': question + "\nLet's think step by step"})
        else:
            messages = [{'role': 'system', 'content': self.system_content}]
            prompt_base = ''
            for example in self.few_shot_examples:
                prompt_base += example['question'] + '\n' + example['answer'] + '\n'
            prompt_question = prompt_base + '\nQuestion: ' + question + "\nLet's think step by step"
            messages.append({'role': 'user', 'content': prompt_question})
        return messages


class GSM8KNumberAnswerExtractor(BaseSingleProcessor):
    def _process(self, record: Record) -> Record:
        answer = record['answer']
        number_answer = answer.rsplit('####')[1].strip()
        record['target'] = float(number_answer)
        return record


def load_gsm8k_dataset(split: str = 'test'):
    dataset_dict = load_dataset('gsm8k', 'main')
    dataset_dict = cast(DatasetDict, dataset_dict)
    records = [i for i in dataset_dict[split]]
    records = cast(Records, records)
    return records


def create_zero_shot_pipeline(lmclient, async_run: bool = True):
    zero_shot_pipeline = EvaluationPipeline(
        lmclient,
        preprocessors=GSM8KNumberAnswerExtractor(),
        prompt=GSM8KZeroShotPrompt(),
        parser=RegexParser(r'\|.*?(\d+).*?\|'),
        postprocessors=LambdaProcessor(_convert_to_float),
        metrics=Accuracy(),
        default_client_kwargs={'temperature': 0.0},
        async_run=async_run,
    )
    return zero_shot_pipeline


def create_few_shot_pipeline(
    lmclient,
    examples: Sequence[dict[str, str]] | str | Path = FixturesDir / 'gsm8k_hardest.json',
    system_content: str | None = None,
    dialog_style: bool = False,
    default_client_kwargs: dict | None = None,
):
    if isinstance(examples, (str, Path)):
        prompt = GSM8KFewShotPrompt.from_file(examples, dialog_style=dialog_style, system_content=system_content)
    else:
        prompt = GSM8KFewShotPrompt(examples, dialog_style=dialog_style, system_content=system_content)

    few_shot_pipeline = EvaluationPipeline(
        lmclient,
        preprocessors=GSM8KNumberAnswerExtractor(),
        prompt=prompt,
        parser=RegexParser(r'answer is .*?(\d+).*?'),
        postprocessors=LambdaProcessor(_convert_to_float),
        metrics=Accuracy(),
        default_client_kwargs=default_client_kwargs,
    )
    return few_shot_pipeline


gsm8k_zero_shot_task = Task(
    task_name='gsm8k_zero_shot',
    load_dataset_func=load_gsm8k_dataset,
    create_pipeline_func=create_zero_shot_pipeline,
)

gsm8k_few_shot_task = Task(
    task_name='gsm8k_few_shot',
    load_dataset_func=load_gsm8k_dataset,
    create_pipeline_func=partial(
        create_few_shot_pipeline,
        examples=FixturesDir / 'gsm8k_hardest.json',
        dialog_style=False,
        default_client_kwargs={'temperature': 0.01},
    ),
)

gsm8k_dialog_few_shot_task = Task(
    task_name='gsm8k_dialog_few_shot',
    load_dataset_func=load_gsm8k_dataset,
    create_pipeline_func=partial(
        create_few_shot_pipeline,
        examples=FixturesDir / 'gsm8k_hardest.json',
        dialog_style=True,
        default_client_kwargs={'temperature': 0.01},
    ),
)
