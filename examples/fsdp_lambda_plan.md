# FSDP real-GPU smoke plan (Lambda 2×5090 or similar)

Goal: prove the FSDP path (ALLGATHER/REDUCESCATTER) can run a model that does not fit on a single GPU. Use the 70B Llama 3
variant in `examples/llama3.py`, sharded across two 5090s on a Lambda Cloud instance (or any pair with ≥32 GB each). Capture a
single-GPU failure for contrast, then show the two-GPU run succeeds with collectives.

Note on `examples/llama3.py` device usage (pre-flight check):
- The script builds `device = tuple(f\"{Device.DEFAULT}:{i}\" for i in range(args.shard)) if args.shard > 1 else Device.DEFAULT`.
  Weight loading shreds per-parameter via `shard_` helpers (axis-aware) to place shards on each device.
  There is no explicit FSDP toggle yet; correctness hinges on the runtime recognizing multi-device shards and lowering to the
  new collectives (ALLGATHER/REDUCESCATTER). The runplan assumes those ops are wired in; otherwise it behaves as pure tensor
  sharding without parameter streaming.

## Node setup
- Provision a Lambda Cloud instance with 2×5090 (or A100/H100 pair if 5090 is unavailable). Choose an image with CUDA 12.x and
  Python 3.12. If uv is missing: `curl -Ls https://astral.sh/uv/install.sh | sh` and `export PATH="$HOME/.local/bin:$PATH"`.
- Clone the repo: `git clone https://github.com/geohot/tinygrad.git && cd tinygrad`.
- Create an env and install deps with uv:
  ```
  uv venv .venv
  source .venv/bin/activate
  uv pip install -e .
  ```

## Environment
- Exports:
  - `export HF_TOKEN=...` (for gated weights), `export HF_HUB_ENABLE_HF_TRANSFER=1` for faster pulls.
  - `export HUGGINGFACE_HUB_CACHE=/data/hf_cache` (or other fast volume) to avoid re-downloading.
  - `export LIBCLANG_PATH=$(llvm-config --libdir 2>/dev/null || echo /usr/lib/llvm-18/lib)` if clang is needed.
  - `export PYTHONPATH=$(pwd)` so the local checkout is used.
- Optional logging: `export DEBUG=2` and `export TINYGRAD_PROFILER=1` to see collective runners and timing.

## Model download (70B, too large for one 5090)
- Kick off the download through the example so weights stream via `tinygrad.helpers.fetch`:
  ```
  HF_HUB_ENABLE_HF_TRANSFER=1 uv run python examples/llama3.py --size 70B --download_model --shard 2 --benchmark --no_api
  ```
  This populates `~/.cache/tinygrad/DeepSeek-R1-Distill-Llama-70B` with tokenizer + 17 safetensor parts and immediately runs a
  20-token benchmark across two devices. First pull can take a while on fresh disks.

## Single-GPU failure check (control)
- Prove it doesn’t fit on one GPU:
  ```
  CUDA_VISIBLE_DEVICES=0 uv run python examples/llama3.py --size 70B \
    --model ~/.cache/tinygrad/DeepSeek-R1-Distill-Llama-70B/model.safetensors.index.json \
    --shard 1 --benchmark --no_api
  ```
  Record the OOM / CUDA out-of-memory log.

## FSDP two-GPU run (target validation)
- Exercise collectives on both GPUs:
  ```
  CUDA_VISIBLE_DEVICES=0,1 DEBUG=2 TINYGRAD_PROFILER=1 uv run python examples/llama3.py --size 70B \
    --model ~/.cache/tinygrad/DeepSeek-R1-Distill-Llama-70B/model.safetensors.index.json \
    --shard 2 --benchmark --no_api --profile
  ```
  Expect 20 tokens to print and logs to show collective runners without host stalls once warmed up.

## Optional backward micro-loop (hit gradients)
- Build the model with `device=("CUDA:0","CUDA:1")` in a tiny scratch script on the node, feed a short prompt batch, compute
  cross-entropy loss, and call `loss.backward()`. With `DEBUG=2`, you should see REDUCESCATTER/ALLGATHER during backward.

## Artifacts to capture
- Save stdout/stderr from the two-GPU run as `fsdp_70b_two_gpu.log`, plus `nvidia-smi` before/after.
- Keep the single-GPU failure log to demonstrate the “does not fit” baseline.
- If profiling is enabled, archive any emitted traces (e.g., `profile.json`) alongside the logs.
