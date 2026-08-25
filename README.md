# Accelerating MoE Inference

CUDA optimization of prefill inference for **Phi-tiny-MoE-instruct** (3.8B total /
1.1B active) on a **single NVIDIA RTX 3090**, starting from a sequential
CPU/OpenMP reference.

**0.012 → ~455 sequences/sec at batch 1024 — roughly 38,000×**, with output that
stays **bit-identical** to the reference throughout (max abs diff `1.87e-4`
against a `3e-3` tolerance, unchanged since the numerics were fixed).

## Constraints

- One RTX 3090 (sm_86), no multi-GPU, no Tensor Cores
- **No cuBLAS, cuDNN, or any other math library** — every kernel is hand-written
- All inference inside the timed region; only model-weight loading may be preloaded
- Output must match a reference produced by an FP32 CPU implementation

That last one shapes everything. The MoE router snaps gate scores onto a 1e-3
grid, so a last-place bit can send a token to a different expert and move a logit
by 0.09. Several kernels therefore use a deliberately *less* accurate formulation
because it reproduces the reference's rounding — a tree reduction in `layer_norm`
broke validation once and is not coming back.

## What moved the needle

| | change | seq/s |
|---|---|---|
| — | CPU/OpenMP reference | 0.012 |
| batching | sequences packed into chunks so matmul `M` stops being tiny | 13.9 |
| device residency | the whole 32-layer stack runs on device pointers; only embeddings down and logits up | 45 |
| numerics | RoPE/`layer_norm`/softmax matched to the reference's rounding — **first passing run at n=1024** | 48.4 |
| register blocking | 4×4 then 8×8 micro-tiles: 0.5 → 2 FMAs per shared load | 130.8 |
| adaptive tiles | tile geometry chosen per call — 128×128 costs the MoE experts 4× fewer blocks | 174.7 |
| `float4` + bank conflicts | vectorised loads and a 4+4 micro-tile split; the reflex fix (padding) addresses the wrong access | 251.8 |
| RoPE table | the angle depends only on (position, dim pair) — 811M evaluations collapse to 2,048 | 280.2 |
| grouped expert GEMM | 16 serialised expert launches become one; the problem was block count, not tile size | 312.6 |
| parallel softmax `exp` | only the accumulation is order-dependent, not the `exp` beside it | 325.7 |
| **prefix deduplication** | shared prefixes give bit-identical hidden states — **19,803 rows → 15,583** | **421** |
| LayerNorm split | its map is order-independent and half its traffic; the reduction is not | 435 |
| double buffering | `__launch_bounds__(NT, 2)` — one register decided between −7% and +8% | **451** |

## Build and run

```sh
make
./run.sh -v -n 1024        # -v validates against the reference output
```

`APS_PROFILE=1` enables a per-section timing breakdown (a no-op otherwise).

## Layout

Implementation — `include/{tensor,layer,model,model_loader}.h`,
`src/{tensor,layer,model,model_loader}.cu`. Everything else
(`src/main.cpp`, `include/config.h`, `Makefile`, `submit.sh`, `tools/`) is
course-provided harness and is unmodified.
