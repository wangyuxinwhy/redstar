from copy import copy
from dataclasses import dataclass
from typing import Sequence

from redstar.metrics import BaseMetric
from redstar.model import Model
from redstar.parsers import BaseParser
from redstar.processors import BaseProcessor
from redstar.prompt import BasePrompt
from redstar.types import Record, Records


@dataclass
class EvaluationResult:
    metric: dict[str, float]
    records: Records


class EvaluationPipeline:
    def __init__(
        self,
        prompt: BasePrompt,
        preprocessors: Sequence[BaseProcessor] | BaseProcessor | None = None,
        parser: BaseParser | None = None,
        postprocessors: Sequence[BaseProcessor] | BaseProcessor | None = None,
        metrics: Sequence[BaseMetric] | BaseMetric | None = None,
        default_model_kwargs: dict | None = None,
    ) -> None:
        self.prompt = prompt

        if preprocessors is None:
            preprocessors = []
        elif isinstance(preprocessors, BaseProcessor):
            preprocessors = [preprocessors]
        else:
            preprocessors = list(preprocessors)
        self.preprocessors = preprocessors

        self.parser = parser

        if postprocessors is None:
            postprocessors = []
        elif isinstance(postprocessors, BaseProcessor):
            postprocessors = [postprocessors]
        else:
            postprocessors = list(postprocessors)
        self.postprocessor = postprocessors

        if metrics is None:
            metrics = []
        elif isinstance(metrics, BaseMetric):
            metrics = [metrics]
        else:
            metrics = list(metrics)
        self.metrics = metrics

        self.default_model_kwargs = default_model_kwargs or {}

    def __call__(self, model: Model, records: Records, **model_kwargs) -> EvaluationResult:
        records = copy(records)
        model_kwargs = {**self.default_model_kwargs, **model_kwargs}

        for preprocessor in self.preprocessors:
            records = preprocessor.process(records)

        prompts = []
        for record in records:
            compile_parameters = self.prompt.compile_parameters
            if compile_parameters:
                compile_kwargs = {key: record[key] for key in compile_parameters}
            else:
                compile_kwargs = record
            prompt = self.prompt.compile(**compile_kwargs)
            prompts.append(prompt)
            record['prompt'] = prompt

        results = model(prompts, **model_kwargs)

        for record, result in zip(records, results):
            record['result'] = result
            if self.parser is not None:
                parsed_result = self.parser.parse(result)
                record['parsed_result'] = parsed_result

        for postprocessor in self.postprocessor:
            records = postprocessor.process(records)

        metrics = {}
        for metric in self.metrics:
            metrics.update(metric(records))

        return EvaluationResult(metric=metrics, records=records)

    def debug(self, model: Model, record: Record):
        import rich
        from rich.columns import Columns
        from rich.panel import Panel
        from rich.pretty import Pretty

        original_record = copy(record)
        panels = []
        panels.append(Panel.fit(Pretty(record), title='Record'))
        for preprocessor in self.preprocessors:
            record = preprocessor.process([record])[0]
            panels.append(Panel.fit(Pretty(record), title=f'{preprocessor.__class__.__name__} Record'))

        compile_parameters = self.prompt.compile_parameters
        if compile_parameters:
            compile_kwargs = {key: record[key] for key in compile_parameters}
        else:
            compile_kwargs = record
        prompt = self.prompt.compile(**compile_kwargs)
        panels.append(Panel.fit(Pretty(prompt), title='Prompt'))

        result = model([prompt])[0]
        record['result'] = result
        panels.append(Panel.fit(Pretty(result), title=f'{model.__class__.__name__} Result'))

        if self.parser is not None:
            parsed_result = self.parser.parse(record['result'])
            record['parsed_result'] = parsed_result
            panels.append(Panel.fit(Pretty(parsed_result), title=f'{self.parser.__class__.__name__} Result'))

        for postprocessor in self.postprocessor:
            record = postprocessor.process([record])[0]
            panels.append(Panel.fit(Pretty(record), title=f'{postprocessor.__class__.__name__} Record'))

        original_record.update(**record)
        panels.append(Panel.fit(Pretty(original_record), title='Final Record'))

        rich.print(Columns(panels, equal=True))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
