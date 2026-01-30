from __future__ import annotations
from typing import Sequence
import numpy as np
from tinygrad.helpers import prod
from tinygrad.dtype import _to_np_dtype
from tinygrad.device import MultiBuffer, Buffer
from tinygrad.uop.ops import Ops, UOp, sym_infer

def _shape_elems(ast:UOp, var_vals:dict[str, int]) -> tuple[int, ...]|None:
  try:
    shp = ast._shape
  except Exception:
    return None
  if shp is None: return None
  return tuple(int(sym_infer(s, var_vals)) if isinstance(s, UOp) else int(s) for s in shp)

def _as_bytes(buf:Buffer) -> bytes:
  buf.ensure_allocated()
  return memoryview(buf.as_buffer()).tobytes()

def _expected_sizes(shape:tuple[int, ...]|None, parts:int) -> tuple[int, int]:
  if shape is None: return 0, 0
  total = prod(shape)
  return total, total // parts

def host_allgather(ast:UOp, out_mb:MultiBuffer, in_mb:MultiBuffer, var_vals:dict[str, int]):
  parts = len(out_mb.bufs)
  shape = None
  try: shape = _shape_elems(ast, var_vals)
  except Exception: shape = None
  if shape is None:
    expected_chunk = in_mb.bufs[0].size
    total = expected_chunk * parts
  else:
    total, expected_chunk = _expected_sizes(shape, parts)

  pieces = [_as_bytes(b) for b in in_mb.bufs]
  if expected_chunk: assert all(len(p) == expected_chunk * in_mb.bufs[0].dtype.itemsize for p in pieces), "allgather chunk size mismatch"
  full = b"".join(pieces)
  if total: assert len(full) == total * in_mb.bufs[0].dtype.itemsize, "allgather total size mismatch"
  for ob in out_mb.bufs:
    ob.ensure_allocated()
    assert ob.nbytes == len(full), f"allgather output size mismatch {ob.nbytes} vs {len(full)}"
    ob.copyin(memoryview(full))

def host_reducescatter(ast:UOp, out_mb:MultiBuffer, in_mb:MultiBuffer, var_vals:dict[str, int]):
  op, _ = ast.arg if isinstance(ast.arg, tuple) else (ast.arg, None)
  if op is not Ops.ADD: raise NotImplementedError(f"reducescatter only supports ADD, got {op}")

  parts = len(out_mb.bufs)
  shape = None
  try: shape = _shape_elems(ast, var_vals)
  except Exception: shape = None
  if shape is None:
    total = in_mb.bufs[0].size
    expected_chunk = total // parts
  else:
    total, expected_chunk = _expected_sizes(shape, parts)
  dtype = in_mb.bufs[0].dtype.base
  np_dtype = _to_np_dtype(dtype)
  assert np_dtype is not None, f"cannot map dtype {dtype} to numpy"

  arrays = [np.frombuffer(_as_bytes(b), dtype=np_dtype, count=None if total == 0 else total) for b in in_mb.bufs]
  reduced = arrays[0].copy()
  for arr in arrays[1:]: reduced += arr

  chunk = expected_chunk if expected_chunk else (len(reduced) // parts)
  assert len(reduced) % parts == 0, "reducescatter requires divisible chunks"
  for i, ob in enumerate(out_mb.bufs):
    start, end = i*chunk, (i+1)*chunk
    shard = reduced[start:end]
    assert ob.size == shard.size, f"reducescatter shard size mismatch {ob.size} vs {shard.size}"
    ob.ensure_allocated()
    ob.copyin(memoryview(shard.tobytes()))
