from copy import deepcopy
from inspect import signature
from pathlib import Path
from typing import Sequence

from redstar.types import Messages
from redstar.utils import load_from_json


class BasePrompt:
    def compile(self, **kwargs) -> Messages:
        raise NotImplementedError

    @property
    def compile_parameters(self) -> list[str]:
        parameters = signature(ZeroShotQAPrompt.compile).parameters
        have_var_keyword = False

        parameter_names: list[str] = []
        for parameter_name, parameter in parameters.items():
            if parameter_name == 'self':
                continue
            if parameter.kind is parameter.VAR_POSITIONAL:
                raise ValueError(
                    f'Parameter {parameter_name} of {self.__class__.__name__} is VAR_POSITIONAL,'
                    'compile() does not support VAR_POSITIONAL.'
                )
            if parameter.kind is parameter.VAR_KEYWORD:
                have_var_keyword = True
            else:
                parameter_names.append(parameter_name)
        if have_var_keyword:
            return []
        return parameter_names


class ZeroShotQAPrompt(BasePrompt):
    def __init__(self, system_content: str | None = None, question_template: str | None = None):
        default_system_content = "You are a helpful assistant to answer user's questions"
        self.system_content = system_content or default_system_content
        self.question_template = question_template or '{question}'
        self.base_messages = [{'role': 'system', 'content': self.system_content}]

    @classmethod
    def from_file(cls, system_prompt_file: str | Path):
        system_prompt_file = Path(system_prompt_file)
        system_prompt = system_prompt_file.read_text()
        return cls(system_prompt)

    def compile(self, question: str):
        messages = deepcopy(self.base_messages)
        question = self.question_template.format(question=question)
        messages.append({'role': 'user', 'content': question})
        return messages


class FewShotQAPrompt(BasePrompt):
    def __init__(
        self,
        examples: Sequence[dict[str, str]],
        dialog_style: bool = False,
        system_content: str | None = None,
        question_template: str | None = None,
        answer_template: str | None = None,
    ):
        self.few_shot_examples = examples
        self.dialog_style = dialog_style

        default_system_content = 'Follow the given examples and answer the question.'
        self.system_content = system_content or default_system_content
        self.question_template = question_template or '{question}'
        self.answer_template = answer_template or '{answer}'

        if dialog_style:
            self.base_messages = [{'role': 'system', 'content': self.system_content}]
            for example in self.few_shot_examples:
                question = self.question_template.format(question=example['question'])
                answer = self.answer_template.format(answer=example['answer'])
                self.base_messages.append({'role': 'user', 'content': question})
                self.base_messages.append({'role': 'assistant', 'content': answer})
        else:
            self.base_messages = [{'role': 'system', 'content': self.system_content}]

    @classmethod
    def from_file(cls, examples_file: str | Path, **kwargs):
        examples = load_from_json(examples_file)
        return cls(examples, **kwargs)

    def compile(self, question: str):
        messages = deepcopy(self.base_messages)
        if self.dialog_style:
            question = self.question_template.format(question=question)
            messages.append({'role': 'user', 'content': question})
        else:
            prompt_base = ''
            for example in self.few_shot_examples:
                example_question = self.question_template.format(question=example['question'])
                example_answer = self.answer_template.format(answer=example['answer'])
                prompt_base += example_question + '\n' + example_answer + '\n'
            prompt_question = prompt_base + self.question_template.format(question=question)
            messages.append({'role': 'user', 'content': prompt_question})
        return messages
