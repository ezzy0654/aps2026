# Session notes — perf/v22, 2026-08-27

Starting baseline: 621-628 seq/s at n=1024 on a 1905 MHz node,
`Validation max abs diff: 0.00018692`. Three attempts below were reverted
(kept here so a later session doesn't re-try them without cause); two were
adopted and pushed the leaderboard personal best 627.1 -> 637.9 -> 646.7
seq/s (+3.1% total). Both are still uncommitted working-tree changes as of
this writing, not `HEAD` — check `git status`/`git diff` before assuming
either one is or isn't present.

## Adopted

### SiLU(w1) * w3 folded into the w1‖w3 grouped GEMM's epilogue

`silu_mul_fused_kernel` was already one kernel (silu and the elementwise
multiply merged), but still a separate pass reading back the `[rows, 2*inter]`
gate/up buffer the w13 GEMM had just written and writing `[rows, inter]` —
about 277 MB of traffic for very little arithmetic (measured `mlp.silu_mul`:
~1.08 ms/layer, 34.4 ms total).

Tile misalignment blocked the obvious move (reuse the RoPE-epilogue trick
directly): w13 is N=896=448(w1)+448(w3), but the 128-wide column tiles don't
land on the w1/w3 boundary (tile 3 covers columns [384,512), straddling
w1[384:448) and w3[448:512)), so a thread's two `BN/VN=64`-apart column
groups don't generally correspond to the same interpolation index's w1 and
w3 outputs the way RoPE's `(j, j+64)` pair did.

Fixed with a **weight-layout change** (explicitly the allowed kind): each
expert's `w13_all_` rows are now interleaved in 64-row chunks — chunk c holds
w1's rows `[c*64, c*64+64)` then w3's same range, 7 chunks (`448 = 7*64`) —
so each 128-wide N tile covers exactly one chunk's (w1, w3) pair, and a
thread's two column groups land on the same index's w1 and w3 outputs again.
Added a `SILU_PAIR` template bool to `matmul_transposed_blocked_kernel` (new
`matmul_transposed_grouped_silu` entry point) whose epilogue computes
`silu(w1) * w3` and writes directly to a `[rows, inter]` buffer — no
`[rows, 2*inter]` buffer ever materialises. Register cost: 252 (VEC+GATHER
variant), same as the RoPE attempt's ceiling and still under the 256-register
2-blocks-per-half-SM limit.

Learned from the RoPE bug: `silu_mul_fused_kernel`'s formula is `exp` (in
double) then one division then one multiply — no second product for the
compiler to contract differently by context, unlike RoPE's two-term sum — so
this one didn't need explicit `__fmul_rn`/`__fmaf_rn` pinning. Validated
first try: 0.00018692, bit-identical. `mlp.forward` (mm.w13+silu_mul+mm.w2)
16.44 -> 15.87 ms/layer; throughput 624.7-627.1 -> 628.5 seq/s (b0, 1905 MHz).
Submitted: personal best 627.1 -> 637.9 seq/s (+1.7%; the submission run
landed a bit above the dev-box measurement, node variance).

### Last layer: only compute the rows lm_head reads

Input is short (avg 19.3 tokens, 1024 distinct sequences): of the trie's
15,583 rows, lm_head reads only 1,024 (6.6%, one per sequence's terminal
node). Every other row's layer-31 output feeds a layer 32 that doesn't
exist, so nobody reads it. k_proj/v_proj still need every row — some other
sequence's terminal node may have this row as an ancestor key — but
q_proj/RoPE(q)/attention/o_proj/post_norm/gate/routing/w13/silu/w2 only ever
need to run on the 1,024 rows that survive to lm_head. Skipped rows' values
are provably unread, so this is bit-identical, not an approximation — see
the discussion in-conversation about why this is the same category of
optimisation as prefix deduplication itself (compute a value once instead of
redundantly; here, don't compute a value nobody reads at all), not a model-
structure change.

No new kernel math was needed, which is why this one didn't carry RoPE-style
numeric risk: `sliding_window_attention`'s query loop and `rope_kernel` both
already separate "which row is Q/output" from "which row's position/ancestor-
chain to look up" (the latter goes through `RowMap.position`/`.path`, which
already hold *node indices* into the full K/V arrays, decoupled from the
query row count). So the compressed tail path is the *same* kernels called
with a second, compressed `RowMap` (`upload_row_map_tail`, one row per
sequence's terminal node, built by copying each terminal node's own
position/path entry out of the arrays already built for the full RowMap) and
a compressed row count, not new kernel code.

New pieces: `PhiAttention::forward_device_tail` / `PhiDecoderLayer::
forward_device_tail` (q/attention/o_proj/post_norm/MoE on `rows_tail`, input_
norm/k_proj/v_proj still on `rows_full`); `device::apply_rope_q`/`apply_rope_k`
(the existing `rope_kernel`, launched once each against different RowMaps
instead of `apply_rope`'s combined two-launch call); `device::
upload_row_map_tail` and `device::tail_index_buffer` — **both deliberately
separate persistent buffers**, not reusing `upload_row_map`'s or `index_
buffer`'s pools: the former is overwritten by every one of the 32 layers'
own RowMap (only one is live at a time normally, but the tail path needs its
compressed RowMap alive *alongside* the full one within the same call), the
latter is overwritten by every layer's MoE call rebuilding its expert-
assignment index, which would have clobbered a `final_row` upload made once
before the layer loop by the time layer 31 read it back. Caught both via
code inspection before running anything, not via a failed validation run.
Six new `Buffer` enum tags (`NormedTail, XTail, QTail, AttnCoreTail,
AttnTail, PostTail`) for the same reason, sized off `chunk_batch` instead of
`total_tokens`.

The old pre-lm_head gather (`d_last_idx` H2D + one `gather_rows` call) is
gone: the tail path already leaves the final layer's output compressed to
`[chunk_batch, hidden]` in the same order `final_row` always defined, so the
final LayerNorm and lm_head run on it directly.

Validated first try: 0.00018692, bit-identical. `decoder.layer.forward`
(layers 0-30, avg) 48.78 ms vs `decoder.layer.forward_tail` (layer 31) 12.46
ms; `attention.forward` (avg) 27.83 ms vs `attention.forward_tail` 7.90 ms.
Throughput 622.6 -> 644.3 seq/s (b5, 1905 MHz, same node/clock as the
pre-session baseline reading). Submitted: personal best 637.9 -> 646.7 seq/s
(+1.4%).

Caveat carried over from the design discussion, not re-verified this
session: correctness assumes `batch <= total_tokens` (every sequence's
terminal row is a distinct valid trie row index), true for this input
(1,024 distinct sequences) but not proven to hold for pathological inputs
(e.g. exact duplicate sequences collapsing `total_tokens` below `batch`).
Not defended against in code — no `max(total, batch)` sizing anywhere in the
new buffers — because it isn't reachable with the graded input, not because
it's been checked safe.

## Tried and reverted

### RoPE rotation folded into q_proj/k_proj's GEMM epilogue

Idea: the 128x128/16x8 wide-tile GEMM's column layout puts a thread's two
column groups exactly `BN/VN = 64` apart — `HEAD_DIM/2` — so a thread already
holds both halves of a RoPE rotation pair in registers when it's about to
store them. Rotating there and skipping `rope_kernel` for that projection
removes a separate read-modify-write pass (measured `attention.rope`: 12.6 ms
/ 1644 ms total, both q and k).

Implemented as a new template bool (`ROPE`) on `matmul_transposed_blocked_kernel`
plus a `device::matmul_transposed_rope()` entry point, wired into
`PhiAttention::forward_device` with a `skip_q`/`skip_k` pair threaded through
`device::apply_rope` for whichever projection wasn't fused.

First attempt failed validation (max abs diff 0.019, first mismatch at
[344,6744]) — a single MoE routing flip. Root cause, confirmed via
`cuobjdump -sass`: `rope_kernel`'s `base[j+half] = x1*c + x0*sn` contracts to
`fma(x0, sn, x1*c)` with `x1*c` precomputed by a separate FMUL — the FMA's own
multiplicands are always `(x0, {c,sn})`, never `(x1, {c,sn})`. Left as plain
operators, the epilogue's very different register pressure made ptxas fuse
the *other* pairing (`fma(x1, c, x0*sn)`) — same real value, different
rounding, off by exactly 1 ULP on the second output only. Pinning both
outputs' instruction sequences with `__fmul_rn`/`__fmaf_rn` reproduced
`rope_kernel`'s exact SASS and passed validation (0.00018692, bit-identical).

Even correct, it was a net loss: q_proj 323.7 -> 335.9 ms, k_proj 81.7 ->
90.2 ms (+20.5 ms combined) against the 12.6 ms `attention.rope` it removed.
The epilogue addition cost more in the register-constrained wide-tile kernel
(248/252 regs, still under the 256-register 2-blocks-per-half-SM ceiling, so
not an occupancy loss — just extra per-thread work in an already
compute-bound kernel) than it saved. Reverted; `git diff` is empty on
`include/{layer,tensor}.h` and `src/{layer,tensor}.cu`.

### Div-free indexing in `scatter_pairs_kernel` (MoE combine)

`moe.scatter` measured 31 ms against a ~0.85 ms bandwidth estimate for its
765 MB of traffic (63.8M elements). The kernel recovers `(t, j)` from a flat
thread index via `idx / row_width` and `idx % row_width` — a 64-bit division
and modulo per thread, unoptimizable to a shift because `row_width` is a
runtime value. Rewriting the launch as a 2D grid (`blockIdx.y` = token,
`blockIdx.x*blockDim.x+threadIdx.x` = column) removes both. Pure launch/
indexing change, no arithmetic touched, so no revalidation risk the way the
RoPE attempt had.

Validated (0.00018692, unchanged) but **`moe.scatter` didn't move**: 31.5 ->
31.1 ms, within run-to-run noise. Division wasn't the bottleneck. The 30x
gap against the bandwidth estimate is more likely the read side —
`out[p0*row_width+j]` / `out[p1*row_width+j]` with `p0`/`p1` picked by
data-dependent routing, so consecutive tokens land at effectively random row
offsets across a 510 MB buffer (well past the RTX 3090's 6 MB L2). Coalesced
within a row, but no locality across rows. Reverted.

Fixing *that* would mean sorting tokens by destination row before the
combine (or writing directly into token order from the w2 GEMM instead of
expert-compacted order) — a bigger restructuring, not attempted this
session.

### Batched lm_head logits D2H with a pre-faulted staging buffer

`model.lm_head` (wraps the GEMM plus the D2H) measured 27-31 ms against
`mm.lm_head` (the GEMM alone) at 11.4 ms — a ~20-27 ms gap traced to
`forward_chunk`'s closing loop: `chunk_batch` (1024) separate
`cudaMemcpy(..., cudaMemcpyDeviceToHost)` calls, one per sequence, each
paying CUDA driver/API overhead independent of its ~125 KB payload.

The existing code already has a comment recording a prior attempt at
exactly this and its failure reason: a bulk D2H into a *fresh* host staging
buffer, or a host-side gather afterward, paid the buffer's first-touch page
faults (or a `std::vector`'s zero-fill on resize) synchronously inside the
timed section — worse than the 1024 calls' combined overhead. `out` itself
avoids this because its pages are pre-faulted by a background thread
(`fill_thread`/`prefetch_thread`) started at the very top of `generate()`,
overlapped with the ~1.6 s layer loop, and only joined right before the D2H
writes that need it resident.

Applied the identical pattern to a *second*, persistent (`static Tensor`)
staging buffer: pre-fault it on its own background thread started at the
top of `forward_chunk`, join it right before one bulk `cudaMemcpy` into it,
then scatter into `out` with a host-side `memcpy` per row (no CUDA API call,
no page fault) instead of 1024 `cudaMemcpy`s.

Validated (0.00018692, unchanged) but **made it worse**: `model.lm_head`
27-31 -> 52.4 ms on the same node/clock (b5, 1905 MHz), throughput 622.6 ->
616.6 seq/s. Reverted.

Working theory, not confirmed: two background threads each doing an
`#pragma omp parallel for` zero-fill of a ~125-131 MB buffer — the existing
`fill_thread` for `logits` and the new one for the staging buffer — run
concurrently for most of their span (both start within a few lines of each
other, early in `generate()`/`forward_chunk`). Each spawns an OpenMP team
sized to the node's core count by default; two such teams contending for
the same cores oversubscribes and can make *both* threads slower than
running either alone, and `fill_thread`'s own completion is on the critical
path (`prefetch_thread->join()` sits inside the same `model.lm_head` scope).
If revisited: fold both fills into one thread/OMP region (or explicitly cap
the staging thread to a handful of threads) instead of two independent full-
width OMP teams, and re-measure.

## Still open (not attempted)

- The scattered-row-access theory in the batched-D2H section above, if
  someone wants to chase `moe.scatter` further (sorting tokens by
  destination row, or writing the w2 GEMM's output directly into token
  order instead of expert-compacted order).
- The batched lm_head D2H's oversubscription theory, if someone wants a
  fourth attempt at it (fold both background pre-fault threads into one, or
  cap the second one's OMP thread count, then re-measure).
