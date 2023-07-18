from typing import Protocol, Sequence

from redstar.types import Messages


class Model(Protocol):
    identifier: str

    def __call__(self, prompts: Sequence[Messages], **kwargs) -> Sequence[str]:
        ...
