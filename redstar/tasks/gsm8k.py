from pathlib import Path
from typing import Sequence

from redstar.metrics import Accuracy
from redstar.parsers import RegexParser
from redstar.pipeline import EvaluationPipeline
from redstar.processors import BaseSingleProcessor, LambdaProcessor
from redstar.prompt import FewShotQAPrompt, ZeroShotQAPrompt
from redstar.tasks.task import FixturesDir, Task
from redstar.types import Record


def _convert_to_float(record: Record):
    try:
        parsed_result = record['parsed_result']
        if parsed_result:
            parsed_result = parsed_result.replace(',', '')
            record['pred'] = float(parsed_result)
        else:
            record['pred'] = 0.0
    except ValueError:
        print(f'Cannot convert {record["parsed_result"]} to float, set pred to 0')
        record['pred'] = 0.0
    return record


class GSM8KNumberAnswerExtractor(BaseSingleProcessor):
    def _process(self, record: Record) -> Record:
        answer = record['answer']
        number_answer = answer.rsplit('####')[1].strip()
        number_answer = number_answer.replace(',', '')
        record['target'] = float(number_answer)
        return record


gsm8k_number_pattren = r'(\d+(?:,\d+)*(?:\.\d+)?)'


def create_few_shot_pipeline(
    examples: Sequence[dict[str, str]] | str | Path = FixturesDir / 'gsm8k_hardest.json',
    system_content: str | None = None,
    dialog_style: bool = False,
    default_client_kwargs: dict | None = None,
    cot: bool = False,
):
    if cot:
        question_template = "Question: {question}\nLet's think step by step"
    else:
        question_template = 'Question: {question}'
    if isinstance(examples, (str, Path)):
        prompt = FewShotQAPrompt.from_file(
            examples,
            dialog_style=dialog_style,
            system_content=system_content,
            question_template=question_template,
        )
    else:
        prompt = FewShotQAPrompt(
            examples,
            dialog_style=dialog_style,
            system_content=system_content,
            question_template=question_template,
        )

    few_shot_pipeline = EvaluationPipeline(
        preprocessors=GSM8KNumberAnswerExtractor(),
        prompt=prompt,
        parser=RegexParser(rf'answer is .*?{gsm8k_number_pattren}.*?'),
        postprocessors=LambdaProcessor(_convert_to_float),
        metrics=Accuracy(),
        default_model_kwargs=default_client_kwargs,
    )
    return few_shot_pipeline


gsm8k_zero_shot_pipeline = EvaluationPipeline(
    preprocessors=GSM8KNumberAnswerExtractor(),
    prompt=ZeroShotQAPrompt.from_file(FixturesDir / 'gsm8k_zero_shot.txt'),
    parser=RegexParser(rf'\|.*?{gsm8k_number_pattren}.*?\|'),
    postprocessors=LambdaProcessor(_convert_to_float),
    metrics=Accuracy(),
    default_model_kwargs={'temperature': 0.0},
)


gsm8k_zero_shot_task = Task(
    task_name='gsm8k_zero_shot',
    dataset_name='gsm8k',
    pipeline=gsm8k_zero_shot_pipeline,
    tags={'zero_shot'},
)

gsm8k_few_shot_task = Task(
    task_name='gsm8k_few_shot',
    dataset_name='gsm8k',
    pipeline=create_few_shot_pipeline(
        examples=FixturesDir / 'gsm8k_hardest.json',
        dialog_style=False,
        default_client_kwargs={'temperature': 0.0},
        cot=False,
    ),
    tags={'few_shot', 'instruction-style'},
)

gsm8k_dialog_few_shot_task = Task(
    task_name='gsm8k_dialog_few_shot',
    dataset_name='gsm8k',
    pipeline=create_few_shot_pipeline(
        examples=FixturesDir / 'gsm8k_hardest.json',
        dialog_style=True,
        default_client_kwargs={'temperature': 0.01},
    ),
    tags={'few_shot', 'dialog-style'},
)

gsm8k_few_shot_task = Task(
    task_name='gsm8k_cot_few_shot',
    dataset_name='gsm8k',
    pipeline=create_few_shot_pipeline(
        examples=FixturesDir / 'gsm8k_hardest_cot.json',
        dialog_style=False,
        default_client_kwargs={'temperature': 0.0},
        cot=True,
    ),
    tags={'few_shot', 'cot', 'instruction-style'},
)

gsm8k_dialog_few_shot_task = Task(
    task_name='gsm8k_dialog_cot_few_shot',
    dataset_name='gsm8k',
    pipeline=create_few_shot_pipeline(
        examples=FixturesDir / 'gsm8k_hardest_cot.json',
        dialog_style=True,
        default_client_kwargs={'temperature': 0.01},
        cot=True,
    ),
    tags={'few_shot', 'cot', 'dialog-style'},
)
