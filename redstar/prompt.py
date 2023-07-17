from redstar.types import Messages


class BasePrompt:
    def compile(self, **kwargs) -> Messages:
        raise NotImplementedError
