# Prefill 최적화 이슈 목록

`docs/README.md` 분석 결과를 이슈 단위로 정리. 범위는 `main.cpp`에서 `-d` 없이
호출되는 prefill 경로(`PhiTinyMoEModel::generate`)로 한정. `generate_decode`
(greedy decode, KV 캐시)는 범위 밖.

우선순위는 영향도(예상 처리량 개선폭) 기준.

---

## #1 [P0] 배치 시퀀스가 완전히 순차 처리됨 — 배치 차원 병렬화 없음

- **위치**: `src/model.cu` `generate()` (`for (b = 0; b < batch; ++b) forward(...)`)
- **현상**: 기본 `-n 1024`개의 독립적인 시퀀스를 하나씩 `forward()`로 순차 처리.
  시퀀스 간 의존성이 전혀 없는데도(KV 캐시 없는 순수 prefill) 배치 차원
  병렬화가 전혀 없음.
- **영향**: GPU 이식 시 가장 큰 개선 여지. 시퀀스 하나만 놓고 보면 GEMM의
  M(=seq_len)이 작아 GPU 활용률이 낮음.
- **제안**: 여러 시퀀스의 토큰을 이어붙여(packed) GEMM의 M을 키운다.
  attention을 제외한 모든 Linear/MoE 연산은 토큰 단위로 시퀀스 경계와 무관하게
  독립이라 packing이 자연스럽다. attention만 시퀀스 경계를 넘지 않도록 블록
  마스킹 필요(→ #2).

---

## #2 [P0] Attention에서 동일 score를 3번 재계산

- **위치**: `src/layer.cu:117-141` (`PhiAttention::forward`)
- **현상**: `score = Σ q·k` 내적을 max 계산, softmax denom 계산, weighted-sum
  계산 세 단계에서 **각각 독립적으로 다시 계산**(`layer.cu:123`, `129`, `136`).
  CPU 싱글스레드 관점에서도 3배 낭비.
- **영향**: attention은 §1(모델 사양)상 GQA(16 q-head / 4 kv-head) ×
  causal+sliding-window(2047)까지 겹쳐 있어 연산 비중이 큼. 3배 재계산 그대로
  GPU에 옮기면 개선 효과가 절반 이상 깎임.
- **제안**: online softmax(FlashAttention 스타일 — score를 한 번만 계산하며
  running max/denom을 갱신) 또는 최소한 score를 버퍼에 캐싱해 1회만 계산.
  GQA(4 kv head를 16 q head가 공유)와 sliding-window 마스크를 그대로 반영.

---

## #3 [P1] GEMM이 CUDA 커널 없이 OpenMP CPU 루프로만 구현됨

- **위치**: `src/tensor.cu:58` (`tensor_ops::matmul_transposed`)
- **현상**: `.cu` 확장자이지만 `__global__` 커널이 아예 없음. `#pragma omp
  parallel for`로 행(M) 단위 병렬화한 3중 루프뿐.
- **영향**: attention의 q/k/v/o projection(`layer.cu:105-143`), MoE의
  router/전문가 FFN(`layer.cu:27-45`, `80-103`), lm_head(`model.cu:30`)까지
  전체 연산량의 대부분이 이 함수를 통과함.
- **제안**: cuBLAS(gemm / batched-gemm) 또는 커스텀 타일드 CUDA 커널로 이식.
  #1의 packing과 결합하면 batched-gemm보다 큰 단일 GEMM으로 합쳐질 수 있음.

---

## #4 [P1] MoE 전문가별 가변 토큰 배치 처리 미최적화

- **위치**: `src/layer.cu:47-70`(`route`), `80-103`(`PhiMoE::forward`)
- **현상**: 토큰마다 top-2 전문가를 결정론적으로 선택한 뒤(`ROUTER_SCORE_QUANTUM`
  양자화 + `ROUTER_TIE_EPS` 타이브레이크), `assignments[e]`로 CPU에서
  모아 전문가별로 순차 forward.
- **영향**: 시퀀스/배치마다 라우팅 결과가 달라 전문가별 토큰 수가 가변적 —
  GPU에서는 gather/scatter 또는 정렬(sort-by-expert) + segment GEMM 방식이
  필요한 전형적인 동적 배치 문제.
- **제약**: 라우팅 결과(양자화·타이브레이크 로직)의 수치적 재현성은 검증
  통과를 위해 반드시 그대로 보존해야 함 — 순서만 바꾸고 판정 로직은 건드리지
  않을 것.
- **제안**: 정렬 기반 gather + segment GEMM, 또는 padding 기반 배치 처리.

---

## #5 [P2] LayerNorm / RoPE / SiLU 등 elementwise 연산이 커널화되지 않음

- **위치**: `src/tensor.cu` (`layer_norm`, `apply_rope`, `silu`, `mul`,
  `add_inplace`, `add_bias_inplace`)
- **현상**: 전부 O(seq_len × hidden) memory-bound 연산이지만 GPU 커널이
  없음. FLOPs 비중 자체는 GEMM/attention보다 작음.
- **영향**: fuse하지 않고 개별 커널 launch + 글로벌 메모리 왕복으로 처리하면
  launch overhead가 누적되어 병목이 될 수 있음.
- **제안**: 가능하면 인접 연산(예: residual add + layer_norm, SiLU + mul)과
  fuse한 커널로 구현.

---

## #6 [P3] Tensor Core 경로(`USE_TC`)가 매크로만 있고 커널 미구현

- **위치**: `Makefile` (`USE_TC=1` → `-DUSE_TC`), `src/main.cpp`
  (`validate_decode`의 `USE_TC` 분기 — 단 이건 decode 전용이라 prefill
  검증(`validate()`)에는 정밀도 완화 분기가 없음)
- **현상**: opt-in 빌드 플래그와 decode 검증 임계값은 이미 준비돼 있으나
  실제 FP16/TF32 등 저정밀도 GEMM 커널은 없음.
- **영향**: #3(GEMM 커널화)이 끝난 뒤 마지막 단계로 적합. prefill
  검증(`validate()`)은 `USE_TC` 여부와 무관하게 절대/상대오차 `3e-3`
  고정이므로, 정밀도 손실이 이 임계값을 넘지 않는지 확인 필요.
- **제안**: #3 완료 후 텐서 코어 GEMM으로 교체, 정밀도 여유 확인.

---

## #7 [Info / P4] ModelLoader I/O 비효율 — 우선순위 낮음 (참고용)

- **위치**: `src/model_loader.cu:32-50` (`ModelLoader::load`)
- **현상**: 텐서를 로드할 때마다 파일을 재오픈(`std::ifstream f(path_, ...)`)
  하고 `seekg`+`read`. 32개 레이어 × (16 experts×3 + attention 4 + norm 4)
  텐서를 개별 파일 I/O로 읽음.
- **영향 없음 이유**: `main.cpp:585-594`의 측정 구간(`synchronize_devices()` →
  `run_start` → `generate()` → `run_end`)에 모델 로딩(`load_start~load_end`)이
  포함되지 않음. 즉 이 비효율은 `Throughput` 측정치에 전혀 반영되지 않는다.
- **제안**: 이번 최적화 사이클에서는 건드리지 않음. 시간 여유가 남을 때만
  고려.

---

## 범위 밖 (참고용으로만 남김)

- `generate_decode` / greedy decode / KV 캐시 (`src/model.cu:53-96`)
- decode 전용 검증 로직(`validate_decode`)의 `USE_TC` step0 완화 임계값
