from typing import Any, Sequence, TypeAlias

from lmclient.types import Message as Message
from lmclient.types import Messages as Messages

Record: TypeAlias = dict[str, Any]
Records: TypeAlias = Sequence[Record]
