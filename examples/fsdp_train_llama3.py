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
import argparse, time
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import fetch, colored, Timing, Profiling
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from examples.llama3 import build_transformer, Tokenizer


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


def train_step(model, x: Tensor, y: Tensor, opt, profile: bool):
  GlobalCounters.reset()
  with Tensor.train():
    opt.zero_grad()
    with Profiling(enabled=profile):
      with Timing("forward+backward ", on_exit=lambda dt: f"{dt*1e3:.2f} ms, {GlobalCounters.global_mem*1e-9/dt:.2f} GB/s"):
        logits = model(x, 0, float("nan"), 0, 0, 0, 0)
        log_probs = logits.log_softmax()
        loss = -log_probs.gather(-1, y.unsqueeze(-1)).mean()
        loss.backward()
    opt.step()
  return loss


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
  args = parser.parse_args()

  assert args.shard >= 2, "Use at least two GPUs to prove FSDP works."
  devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(args.shard))
  print(colored(f"Using devices: {devices}", "green"))

  model_path = args.model or ensure_weights(args.size, args.cache_dir)
  tokenizer = Tokenizer(str((model_path if model_path.is_dir() else model_path.parent) / "tokenizer.model"))

  model = build_transformer(model_path, model_size=args.size, device=devices, max_context=args.seq_len, load_weights=True)
  params = get_parameters(model)
  param_bytes = sum(p.uop.size * p.dtype.itemsize for p in params)
  print(colored(f"Sharded params: {param_bytes/1e9:.2f} GB across {len(devices)} devices", "yellow"))

  opt = optim.Adam(params, lr=args.lr)
  x, y = make_batch(tokenizer, args.prompt, args.seq_len, devices)

  for step in range(args.steps):
    t0 = time.time()
    loss = train_step(model, x, y, opt, args.profile)
    dt = time.time() - t0
    print(colored(f"[step {step}] loss={loss.item():.4f} time={dt*1e3:.1f} ms", "cyan"))

  print(colored("FSDP smoke run finished. Check DEBUG=2 logs for ALLGATHER/REDUCESCATTER runners.", "green"))


if __name__ == "__main__":
  main()
