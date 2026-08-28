# Is PyTorch JIT compilation worth it for our training scripts?

**TL;DR: no, for SB3 + small MLPs on CPU. Maybe later for `torch.compile` on
GPU with bigger networks.**

## What "JIT" means here

PyTorch has two compilation paths:
- **TorchScript** (`torch.jit.trace` / `torch.jit.script`): older, deprecated
  for new code as of PyTorch 2.4+.
- **`torch.compile`**: modern (PyTorch 2.0+), uses TorchDynamo + Inductor.
  Replaces TorchScript for almost all use cases.

This note covers both as candidates for our SB3 training loop.

## What our forward/backward graphs look like

SB3's default MLP policies for the envs we currently train:

| Env | Algo | Policy net | Critic net | Params |
|-----|------|-----------|-----------|--------|
| CartPole | PPO | 64×64 | 64×64 | ~5 K |
| InvertedPendulum | SAC | 64×64 | 64×64 (×2) | ~8 K |
| InvertedDoublePendulum | A2C | 64×64 | 64×64 | ~5 K |
| Reacher | TD3 | 256×256 | 256×256 (×2) | ~130 K |
| Point | SAC | 64×64 | 64×64 (×2) | ~5 K |
| Line Follower | PPO | 64×64 | 64×64 | ~5 K |

All sit in the **single-millisecond forward-pass regime on CPU.** A
typical 64×64 ReLU MLP takes ~50 to 200 µs per inference call on a modern
x86 core.

## Why JIT doesn't help here

1. **Call-site overhead is the bottleneck, not graph execution.** Each
   `policy.predict()` or training-batch forward call costs ~50 µs of
   Python + PyTorch dispatch overhead, then ~100 µs of actual compute.
   JIT optimizes the compute side. Saving 30 µs on a 150 µs total isn't
   worth a 5-second graph-compile cost on first call.

2. **SB3 already uses `torch.no_grad()`** for prediction and doesn't
   recompile every step; the eager mode is already pretty efficient
   for tiny models.

3. **`torch.compile` shines on GPU with big batches**: our biggest
   training batch (TD3 with batch_size=256, 256×256 network) is roughly
   33 K MACs per element × 256 = 8 M MACs per forward. Modern CPUs do
   that in microseconds. There's no compile win to grab.

4. **TorchScript adds friction.** Tracing/scripting an SB3 policy
   requires Stable Baselines3 internals that aren't `@torch.jit.script`-
   friendly out of the box. The patch effort is real; the speedup isn't.

## Empirical estimates (published benchmarks)

From the PyTorch 2.0 release post + community benchmarks for **small MLPs
on CPU**:

| Configuration | Speedup vs eager |
|--------------|------------------|
| `torch.jit.trace` on 64×64 MLP | 1.0-1.1× |
| `torch.compile(mode="default")` on 64×64 MLP | 1.0-1.2× (after warm-up) |
| `torch.compile(mode="reduce-overhead")` on CartPole policy | 1.1-1.3× |
| Eager mode on GPU (CUDA) for same model | ~5-10× over CPU |

The biggest win in the chart is **switching to GPU**, not adding JIT.
For users with CUDA, `GAZEBO_GYM_DEVICE=cuda` is the lever.

## When would JIT be worth revisiting

- We bump policy networks to ≥1 M parameters (e.g. CNN policies for
  vision-from-pixels on the rover) → JIT on GPU becomes meaningful.
- We move to torch's nightly with the new flash-attention CPU kernels.
- We adopt a CUDA workflow for batched simulation (parallel envs feeding
  a shared GPU-side policy) → `torch.compile` on the actor pays off.

Until any of those land: keep eager mode, keep the device flag (already
in place via `GAZEBO_GYM_DEVICE`), and revisit if profiling shows the
policy forward becomes a real bottleneck. Right now the **gz physics
step + transport latency dominate**, not the policy network.
