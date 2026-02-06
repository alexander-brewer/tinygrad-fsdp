#!/usr/bin/env python3
"""
Minimal FSDP smoke test for Llama 3 that *requires* more than one GPU.

Designed for 2×5090 runpod boxes. It shards the 70B checkpoint across devices,
runs a single forward+backward step on a short prompt, and logs throughput so
you can confirm ALLGATHER/REDUCESCATTER runners are exercised.

Example (after weights are in ~/.cache/tinygrad/DeepSeek-R1-Distill-Llama-70B):
  DEBUG=2 TINYGRAD_PROFILER=1 PYTHONPATH=. \\
  CUDA_VISIBLE_DEVICES=0,1 \\
  python examples/fsdp_train_llama3.py --size 70B --shard 2 --steps 1
"""

from pathlib import Path
from collections import deque
import argparse, time
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import fetch, colored, Timing, Profiling
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict
from tinygrad.uop.ops import UOp, Ops
from tinygrad.examples.llama3 import build_transformer, Tokenizer


TRACE_OPS = (Ops.MULTI, Ops.ALLREDUCE, Ops.ALLGATHER, Ops.REDUCESCATTER, Ops.COPY)

def ensure_weights(size: str, download_dir: Path) -> Path:
  assert size == "70B", "only the 70B checkpoint is large enough for this smoke test"
  download_dir.mkdir(parents=True, exist_ok=True)
  base = "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main"
  index = fetch(f"{base}/model.safetensors.index.json?download=true",
                "model.safetensors.index.json", subdir=str(download_dir))
  tok_src = "https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model"
  fetch(tok_src, "tokenizer.model", subdir=str(download_dir))
  for i in range(17):
    fetch(f"{base}/model-{i+1:05d}-of-000017.safetensors?download=true",
          f"model-{i+1:05d}-of-000017.safetensors", subdir=str(download_dir))
  return Path(index)


def make_batch(tokenizer: Tokenizer, prompt: str, seq_len: int, devices):
  toks = [tokenizer.bos_id] + tokenizer.encode(prompt, allow_special=True)
  toks = toks[:seq_len]
  if len(toks) < 4:
    toks += [tokenizer.bos_id] * (4 - len(toks))
  inp, tgt = toks[:-1], toks[1:]
  x = Tensor([inp], device=Device.DEFAULT)
  y = Tensor([tgt], device=Device.DEFAULT)
  if isinstance(devices, tuple): x, y = x.shard_(devices, axis=None), y.shard_(devices, axis=None)
  return x, y


def _names_with_flag(named_params, pred):
  return [name for name, t in named_params if pred(t)]


def _suspect_router_names(names):
  flags = ("moe", "router", "gate", "expert", "experts", "topk", "route")
  return [name for name in names if any(flag in name for flag in flags)]


def _grad_axis_mismatch(param: Tensor) -> bool:
  if param.grad is None: return False
  if not isinstance(param.device, tuple): return False
  if not isinstance(param.grad.device, tuple): return True
  return param.uop.axis != param.grad.uop.axis

def _layer_idx(name: str) -> int|None:
  parts = name.split(".")
  if len(parts) < 2 or parts[0] != "layers": return None
  return int(parts[1]) if parts[1].isdigit() else None

def _missing_key(item):
  name, _ = item
  li = _layer_idx(name)
  return (li is None, li if li is not None else 10**9, name)

def _first_missing(named_params):
  missing = [(name, t) for name, t in named_params if t.requires_grad and t.grad is None]
  if not missing: return None
  return sorted(missing, key=_missing_key)[0]

def _trace_ops_to_loss(src: UOp, dst: UOp, watch_ops=TRACE_OPS) -> list[str]:
  if src is dst: return [src.op.name]
  topo = src.toposort()
  if dst not in topo: return []
  consumers: dict[UOp, list[UOp]] = {}
  for u in topo:
    for s in u.src: consumers.setdefault(s, []).append(u)
  prev: dict[UOp, UOp|None] = {dst: None}
  q = deque([dst])
  while q:
    cur = q.popleft()
    if cur is src: break
    for nxt in consumers.get(cur, []):
      if nxt in prev: continue
      prev[nxt] = cur
      q.append(nxt)
  if src not in prev: return []
  path = [src]
  while prev[path[-1]] is not None: path.append(prev[path[-1]])
  traced = [u.op.name for u in path if u.op in watch_ops]
  return traced if traced else [u.op.name for u in path[:8]]

def _valid_trainable(params):
  return [p for p in params if p.grad is not None and not _grad_axis_mismatch(p)]

def _ensure_valid_optimizer(opt, params, lr: float):
  trainable = _valid_trainable(params)
  if not trainable: raise RuntimeError("no parameters have usable gradients after backward")
  if len(trainable) != len(opt.params) or any(a is not b for a, b in zip(trainable, opt.params)):
    return optim.Adam(trainable, lr=lr)
  return opt


def log_grad_status(named_params, log_path: Path|None, loss: Tensor|None = None):
  trainable = _names_with_flag(named_params, lambda t: t.requires_grad is True)
  frozen = _names_with_flag(named_params, lambda t: t.requires_grad is False)
  unknown = _names_with_flag(named_params, lambda t: t.requires_grad is None)
  missing = [name for name, t in named_params if t.requires_grad and t.grad is None]
  mismatched = []
  for name, t in named_params:
    if not _grad_axis_mismatch(t): continue
    if t.grad is None:
      mismatched.append(name)
      continue
    if not isinstance(t.grad.device, tuple):
      mismatched.append(f"{name} (param axis {t.uop.axis}, grad device {t.grad.device})")
    else:
      mismatched.append(f"{name} (param axis {t.uop.axis}, grad axis {t.grad.uop.axis})")

  lines = []
  summary = f"Grad status: total={len(named_params)} trainable={len(trainable)} frozen={len(frozen)} unknown={len(unknown)}"
  if loss is not None:
    summary += f" loss_requires_grad={loss.requires_grad}"
    summary += f" loss_uop={loss.uop.op}"
  lines.append(summary)
  lines.append(f"missing_grads_count={len(missing)}")
  lines.append(f"mismatched_grads_count={len(mismatched)}")
  if frozen:
    lines.append("FROZEN_PARAMS (requires_grad=False):")
    lines.extend(frozen)
  if unknown:
    lines.append("UNKNOWN_GRAD_PARAMS (requires_grad=None):")
    lines.extend(unknown)
  if missing:
    lines.append("MISSING_GRADS (requires_grad=True, grad=None):")
    lines.extend(missing)
  if mismatched:
    lines.append("MISMATCHED_GRADS (param axis != grad axis):")
    lines.extend(mismatched)
  if loss is not None and (first_missing := _first_missing(named_params)) is not None:
    name, t = first_missing
    lines.append(f"FIRST_MISSING_PARAM={name}")
    if (li := _layer_idx(name)) is not None: lines.append(f"FIRST_MISSING_LAYER={li}")
    lines.append(f"FIRST_MISSING_REQUIRES_GRAD={t.requires_grad}")
    lines.append(f"FIRST_MISSING_PARAM_AXIS={t.uop.axis}")
    lines.append(f"FIRST_MISSING_PARAM_DEVICE={t.device}")
    in_graph = t.uop in loss.uop.toposort()
    lines.append(f"FIRST_MISSING_IN_LOSS_GRAPH={in_graph}")
    if in_graph:
      traced = _trace_ops_to_loss(loss.uop, t.uop)
      lines.append("FIRST_MISSING_BACKWARD_TRACE_OPS:")
      lines.append(" -> ".join(traced))

  suspect = _suspect_router_names(missing or frozen)
  if suspect:
    lines.append("SUSPECT_ROUTER_PARAMS:")
    lines.extend(suspect)

  for line in lines:
    print(colored(line, "yellow") if line.endswith(":") or line.startswith("Grad status:") else line)
  if log_path is not None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n")
    print(colored(f"Wrote grad report to {log_path}", "yellow"))


def missing_grad_names(named_params):
  return [name for name, t in named_params if t.requires_grad and t.grad is None]


def train_step(model, x: Tensor, y: Tensor, opt, params, named_params, profile: bool, strict_grads: bool, lr: float,
               grad_log: Path|None):
  GlobalCounters.reset()
  with Tensor.train():
    opt.zero_grad()
    with Profiling(enabled=profile):
      with Timing("forward+backward ", on_exit=lambda dt: f"{dt*1e3:.2f} ms, {GlobalCounters.global_mem*1e-9/dt:.2f} GB/s"):
        logits = model(x, 0, float("nan"), 0, 0, 0, 0)
        log_probs = logits.log_softmax()
        loss = -log_probs.gather(-1, y.unsqueeze(-1)).mean()
        loss.backward()
    if not getattr(train_step, "_logged_grad_status", False):
      log_grad_status(named_params, grad_log, loss)
      setattr(train_step, "_logged_grad_status", True)
    missing = missing_grad_names(named_params)
    mismatched = [p for p in params if _grad_axis_mismatch(p)]
    if missing:
      preview = ", ".join(missing[:8])
      msg = f"{len(missing)} params had no grad (e.g. {preview})"
      if strict_grads: raise RuntimeError(msg)
      if not getattr(train_step, "_warned_missing_grads", False):
        print(colored(f"WARNING: {msg}. Optimizer will skip them.", "yellow"))
        setattr(train_step, "_warned_missing_grads", True)
    if mismatched:
      msg = f"{len(mismatched)} params had grad axis mismatch (see grad report)"
      if strict_grads: raise RuntimeError(msg)
      if not getattr(train_step, "_warned_mismatch_grads", False):
        print(colored(f"WARNING: {msg}. Optimizer will skip them.", "yellow"))
        setattr(train_step, "_warned_mismatch_grads", True)
    if missing or mismatched:
      opt = _ensure_valid_optimizer(opt, params, lr)
    opt.step()
  return loss, opt


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=Path, help="Path to model.safetensors.index.json (70B). If omitted, downloads.")
  parser.add_argument("--cache_dir", type=Path, default=Path("~/.cache/tinygrad/DeepSeek-R1-Distill-Llama-70B").expanduser())
  parser.add_argument("--size", choices=["70B"], default="70B")
  parser.add_argument("--shard", type=int, default=2, help="Number of GPUs; must be >=2 for FSDP.")
  parser.add_argument("--steps", type=int, default=1, help="Training steps to run.")
  parser.add_argument("--seq_len", type=int, default=64, help="Truncate prompt to this many tokens.")
  parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
  parser.add_argument("--prompt", type=str, default="Explain Fully Sharded Data Parallel in one sentence.",
                      help="Short prompt to drive the tiny training step.")
  parser.add_argument("--profile", action="store_true", help="Enable tinygrad profiling context.")
  parser.add_argument("--quantize", choices=["int8", "nf4", "fp8", "float16"], help="Optional quantization to reduce memory.")
  parser.add_argument("--strict_grads", action="store_true", help="Error if any parameter has no grad after backward.")
  parser.add_argument("--grad_log", type=Path, help="Write grad report to this file (default: cache_dir/fsdp_grad_report.txt).")
  args = parser.parse_args()

  assert args.shard >= 2, "Use at least two GPUs to prove FSDP works."
  devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(args.shard))
  print(colored(f"Using devices: {devices}", "green"))

  model_path = args.model or ensure_weights(args.size, args.cache_dir)
  tokenizer = Tokenizer(str((model_path if model_path.is_dir() else model_path.parent) / "tokenizer.model"))

  model = build_transformer(model_path, model_size=args.size, device=devices, max_context=args.seq_len,
                            load_weights=True, quantize=args.quantize)
  params = get_parameters(model)
  named_params = list(get_state_dict(model).items())
  param_bytes = sum(p.uop.size * p.dtype.itemsize for p in params)
  print(colored(f"Sharded params: {param_bytes/1e9:.2f} GB across {len(devices)} devices", "yellow"))
  grad_log = args.grad_log or (args.cache_dir / "fsdp_grad_report.txt")

  opt = optim.Adam(params, lr=args.lr)
  x, y = make_batch(tokenizer, args.prompt, args.seq_len, devices)

  for step in range(args.steps):
    t0 = time.time()
    loss, opt = train_step(model, x, y, opt, params, named_params, args.profile, args.strict_grads, args.lr, grad_log)
    dt = time.time() - t0
    print(colored(f"[step {step}] loss={loss.item():.4f} time={dt*1e3:.1f} ms", "cyan"))

  print(colored("FSDP smoke run finished. Check DEBUG=2 logs for ALLGATHER/REDUCESCATTER runners.", "green"))


if __name__ == "__main__":
  main()
