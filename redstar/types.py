from typing import Any, Sequence, TypeAlias, TypedDict

from typing_extensions import NotRequired


class Message(TypedDict):
    role: str
    content: str
    name: NotRequired[str]
    function_call: NotRequired[str]


Messages = Sequence[Message]

Record: TypeAlias = dict[str, Any]
Records: TypeAlias = Sequence[Record]
