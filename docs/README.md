# APSS26 프로젝트 — Phi-tiny-MoE Prefill 최적화

`microsoft/Phi-tiny-MoE-instruct` 모델의 **prefill(=main.cpp에서 `-d` 옵션 없이
호출되는 `PhiTinyMoEModel::generate`)** 경로를 CUDA로 최적화하는 것이 목표.
`generate_decode`(greedy decode, KV 캐시 등)는 **이번 최적화 범위 밖**이며 이
문서에서도 다루지 않는다.

현재 코드는 `.cu` 확장자를 갖지만 실제 CUDA 커널은 없고, `#pragma omp parallel for`
기반의 **CPU 참조 구현**이다 (정답 검증 기준). 이걸 GPU로 이식하는 것이 과제.

## 1. Prefill 호출 경로

```
main()                                              [src/main.cpp]
 └─ model_ref->generate(ids, logits)                [model.cu: generate()]
      └─ for each sequence b in batch:               ← ★순차 루프, 시퀀스 간 병렬화 없음
           forward(input_ids[b], one_logits)          [model.cu: forward()]
             ├─ 임베딩 lookup                          embeddings_.at(token, h)
             ├─ for layer in 0..31:                    PhiDecoderLayer::forward  [layer.cu]
             │     ├─ input layer_norm                 tensor_ops::layer_norm
             │     ├─ PhiAttention::forward             ← 연산량 큰 지점
             │     │     ├─ q/k/v proj (Linear×3)         tensor_ops::matmul_transposed
             │     │     ├─ apply_rope
             │     │     ├─ causal+sliding-window attn    (커스텀 삼중 루프, 커널 없음)
             │     │     └─ o_proj (Linear)
             │     ├─ residual add
             │     ├─ post-attn layer_norm
             │     ├─ PhiMoE::forward                    ← 연산량 큰 지점
             │     │     ├─ router matmul (gate)
             │     │     ├─ route() : top-2 결정론적 선택 (CPU, 토큰당 O(experts))
             │     │     └─ 전문가별 배치 forward (PhiMLP×assigned experts)
             │     └─ residual add
             ├─ 최종 layer_norm
             └─ lm_head (Linear) — 마지막 토큰 위치만
```

`batch = 1024`(기본, `-n`으로 조절)개 시퀀스를 **완전히 독립적으로**, 그것도
**하나씩 순차 처리**한다. 시퀀스마다 길이가 다르고(`max_seq_len`까지 가변) KV
캐시가 필요 없는 순수 prefill이므로, 이 배치 차원이 GPU 병렬화의 가장 큰
1차 타겟이다.

## 2. 모델 사양 (`include/config.h`, prefill에 관련된 것만)

| 항목 | 값 |
|---|---|
| VOCAB_SIZE | 32064 |
| HIDDEN_SIZE | 4096 |
| NUM_LAYERS | 32 |
| NUM_ATTENTION_HEADS / NUM_KV_HEADS | 16 / 4 (GQA, group=4) |
| HEAD_DIM | 128 |
| NUM_EXPERTS / TOP_K | 16 / 2 |
| EXPERT_INTERMEDIATE_SIZE | 448 |
| MAX_POSITION_EMBEDDINGS | 4096 |
| SLIDING_WINDOW | 2047 |
| ROPE_THETA | 10000.0 |
| NORM_EPS | 1e-5 |

(`MAX_DECODE_TOKENS`, `EOS_TOKEN_ID`는 decode 전용이라 범위 밖.)

## 3. 연산별 특성과 병목 지점

### 3.1 GEMM (`tensor_ops::matmul_transposed`) — 연산량 대부분을 차지
`[M,K] x [N,K]ᵀ → [M,N]`, 현재는 `#pragma omp parallel for`로 행(M) 병렬화한
3중 루프(`src/tensor.cu:58`)뿐. 이게 불리는 곳:
- Attention의 q/k/v/o 4개 projection (`layer.cu:105-143`)
- MoE의 router(gate) projection + 전문가별 w1/w2/w3 (`layer.cu:80-103`, `27-45`)
- lm_head (`model.cu:30`, 단 M=1이라 이건 시퀀스당으로는 저렴)

시퀀스 하나만 놓고 보면 M(=seq_len)이 작아 GEMM 효율이 낮다. **배치의 여러
시퀀스를 이어붙여 M을 키우면**(토큰을 packed하게 concat) GPU GEMM 효율이 크게
개선됨 — attention을 제외한 모든 Linear/MoE 연산은 시퀀스 경계와 무관하게
토큰 단위로 독립이라 이 packing이 자연스럽다.

### 3.2 Attention (`layer.cu:111-143`) — 현재 구현의 명백한 비효율
같은 `score = Σ q·k`를 **max, denom, weighted-sum 세 단계에서 각각 다시
계산**한다 (`layer.cu:123`, `129`, `136`에서 동일 내적을 3번 반복). CPU
싱글스레드 관점에서도 3배 낭비고, GPU 이식 시 이 구조를 그대로 가져가면 안
됨 — online softmax(FlashAttention 스타일, score를 한 번만 계산하고 running
max/denom을 갱신) 또는 최소한 score를 버퍼에 캐싱해 1회만 계산해야 한다.
causal + sliding-window(2047) 마스크, GQA(4개 kv head를 16개 q head가 공유)도
그대로 반영해야 함.

### 3.3 MoE 라우팅 (`layer.cu:47-70, 80-103`)
- `route()`: 토큰당 16개 전문가 점수를 양자화(`ROUTER_SCORE_QUANTUM`)한 뒤
  top-2를 결정론적으로 선택. 연산량은 작지만(O(experts)) **결과가 정확히
  재현되어야** 검증을 통과하므로, GPU로 옮기더라도 이 양자화/타이브레이크
  로직(`ROUTER_TIE_EPS`)을 정확히 보존해야 한다.
- 전문가별 forward: 토큰들을 `assignments[e]`로 모아 배치 matmul. 시퀀스별로
  라우팅 결과가 다르므로 **가변 크기 gather/scatter**가 필요 — GPU에서는
  보통 정렬(sort-by-expert) + segment GEMM 또는 padding 방식으로 처리.

### 3.4 LayerNorm / RoPE / SiLU / elementwise — 상대적으로 비중은 작지만 커널 필요
전부 O(seq_len × hidden) 수준의 memory-bound 연산. FLOPs 자체는 GEMM/Attention에
비해 작지만, CPU→GPU 전송 없이 GPU 상주 데이터에 대해 fuse된 커널로 처리하지
않으면 launch overhead와 글로벌 메모리 왕복이 병목이 될 수 있다.

### 3.5 임베딩 lookup / ModelLoader
`forward()` 시작의 임베딩 lookup(`model.cu:20-24`)은 단순 gather라 GPU에서
저렴하게 처리 가능. `ModelLoader::load`는 텐서마다 파일을 재오픈하는 등
비효율적이지만, **모델 로딩은 측정 구간 밖**이므로(§5 참고) 우선순위 낮음.

## 4. 출력 바이너리 포맷 (prefill, `outputs.bin`)

```
uint32 batch
uint32 vocab_size
float  logits[batch][vocab_size]
```
입력 파일(`inputs.bin`) 내 시퀀스 순서를 그대로 출력 순서로 유지해야 한다
(내부적으로 재배치/재정렬해 처리해도 최종 출력은 원래 순서로 복원).

## 5. 검증 & 성능 측정 범위 (`main.cpp`)

- 검증(`validate()`, `main.cpp:321`): 요소별 절대오차 또는 상대오차가
  `3e-3` 이하면 통과.
- **측정 구간**: `synchronize_devices()` → `run_start` → `generate()` →
  `synchronize_devices()` → `run_end` (`main.cpp:585-594`)만이 `Elapsed
  time`/`Throughput`에 들어간다. **모델 로딩(`load_start~load_end`)과
  웜업(`-w`)은 타이밍에서 완전히 제외**된다 — 즉 GPU로 가중치를 올리는
  시점이나 최초 커널 컴파일/워밍업 비용은 최적화 우선순위가 낮고, 오직
  `generate()` 호출 1회의 벽시계 시간만 줄이면 된다.

## 6. 빌드 & 실행

```bash
make                      # 기본 FP32 경로
make USE_TC=1               # Tensor Core opt-in 경로 (매크로만 있고 커널 미구현)
./run.sh -n 1024 -v -w       # prefill만 (-d 없음), 검증 + 웜업
```

`-d`를 주지 않는 것이 이번 최적화 범위(prefill-only)에 해당하는 실행 방식.

## 7. 성능 제출 (`submit.sh`)

`-d` 없이 실행하면 "기본반" 트랙으로 판정되어 `sequences_per_s`(=`Throughput`
값 그대로)가 리더보드에 올라간다. `-v`는 항상 자동 첨부되지만 측정 구간
이후에 실행되므로 기록에는 영향 없음.

## 8. 최적화 로드맵 (prefill 전용, 영향도 순)

1. **배치 차원 병렬화** — `generate()`의 시퀀스 순차 루프(§1)를 제거하고
   여러 시퀀스를 동시에 GPU에서 처리. 토큰을 이어붙여(packed) GEMM의 M을
   키우는 방식이 attention 외 모든 연산(§3.1)에 바로 적용 가능.
2. **Attention 커널** — 3중 재계산 제거(§3.2), online softmax, causal +
   sliding-window + GQA를 반영한 타일링 커널. packed 배치에서는 시퀀스 경계를
   넘는 attention이 일어나지 않도록 블록 단위 마스킹 필요.
3. **GEMM 커널/cuBLAS 도입** — attention proj, MoE 전문가 FFN, lm_head가
   전체 FLOPs의 대부분(§3.1). cuBLAS(gemm/batched-gemm) 또는 커스텀 타일드
   커널로 이식.
4. **MoE gather/scatter** — 전문가별 가변 토큰 수를 GPU에서 처리 (정렬 후
   segment GEMM 등). 라우팅 결정 로직 자체(§3.3)의 수치적 재현성은 유지해야 함.
5. **LayerNorm/RoPE/SiLU 등 elementwise 커널** — 가능하면 인접 연산과 fuse해
   글로벌 메모리 왕복 최소화.
6. **Tensor Core(FP16/TF32) 경로 (`USE_TC`)** — GEMM을 텐서 코어로 대체.
   검증 임계값이 이미 이 경로를 고려해 별도로 존재하므로(`main.cpp`의
   `USE_TC` 분기), 정밀도 손실 허용 범위 내에서 속도를 얻는 마지막 단계로
   적합.

## 9. 범위 밖 (참고용으로만 남김)

- `generate_decode` / greedy decode / KV 캐시: 이번 최적화 대상 아님.
- `ModelLoader` I/O 최적화: 측정 구간 밖이라 우선순위 낮음.
