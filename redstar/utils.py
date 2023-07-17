import json
from pathlib import Path


def load_from_json(json_file: str | Path):
    json_file = Path(json_file)
    if not json_file.exists():
        raise FileExistsError(f'File {json_file} does not exist')

    with open(json_file, 'r') as f:
        return json.load(f)


def is_jupyter():
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return True
    except NameError:
        return False
    return False
