import re


class BaseParser:
    def parse(self, text: str) -> str:
        raise NotImplementedError


class RegexParser(BaseParser):
    def __init__(self, regex: str):
        self.pattern = re.compile(regex)

    def parse(self, text: str) -> str:
        match = self.pattern.search(text)
        if match is None:
            return ''
        return match.group(1)
