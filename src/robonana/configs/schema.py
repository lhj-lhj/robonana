"""显式 JSON 配置：拒绝拼错的字段/类型，相对路径以配置文件所在目录为准。"""
from dataclasses import fields
import json
import math
import os
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints, Union


def _convert(value, annotation, base, name):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        for option in args:
            try:
                return _convert(value, option, base, name)
            except (TypeError, ValueError):
                pass
        raise ValueError(f'{name}: expected {annotation}, got {value!r}')
    if annotation is type(None):
        if value is not None: raise TypeError(name)
        return None
    if annotation is Path:
        if not isinstance(value, str) or not value.strip(): raise ValueError(f'{name}: required path')
        path = Path(value).expanduser()
        path = base / path if not path.is_absolute() else path
        # venv 的 python 通常是符号链接：不能 resolve 到系统解释器，否则丢失 pyvenv.cfg。
        return Path(os.path.abspath(path)) if name == 'sim_python' else path.resolve()
    if origin is tuple:
        if not isinstance(value, list): raise ValueError(f'{name}: expected JSON array')
        return tuple(_convert(v, args[0], base, name) for v in value)
    if origin is dict:
        if not isinstance(value, dict): raise ValueError(f'{name}: expected JSON object')
        return {_convert(k, args[0], base, name): _convert(v, args[1], base, name) for k,v in value.items()}
    if annotation is float:
        if type(value) not in (int, float) or not math.isfinite(value): raise ValueError(f'{name}: finite number required')
        return float(value)
    if type(value) is not annotation: raise ValueError(f'{name}: expected {annotation.__name__}')
    return value


def load_options(cls, path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict): raise ValueError('Config must be a JSON object')
    unknown = set(payload) - {f.name for f in fields(cls)}
    if unknown: raise ValueError(f'Unknown config fields: {sorted(unknown)}')
    missing = {f.name for f in fields(cls)} - set(payload)
    if missing: raise ValueError(f'Missing explicit config fields: {sorted(missing)}')
    hints = get_type_hints(cls)
    try:
        return cls(**{k: _convert(v, hints[k], path.parent, k) for k,v in payload.items()})
    except TypeError as exc:
        raise ValueError(f'Missing required config fields: {exc}') from exc


def accumulation(gpus, microbatch, global_batch):
    if not gpus or len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError('gpus must contain distinct nonnegative integers')
    if type(microbatch) is not int or type(global_batch) is not int or min(microbatch, global_batch) < 1:
        raise ValueError('microbatch and global_batch must be positive integers')
    divisor = len(gpus) * microbatch
    if global_batch % divisor: raise ValueError('global_batch must be divisible by GPU count * microbatch')
    return global_batch // divisor
