# 2026-08-25 CUDA 최적화 보고서

## 기준선과 프로파일

- 직전 제출은 n=1024에서 6.766990초, 151.322827 seq/s였다.
- Nsight Systems 기준 전체 GPU kernel 합계는 약 5.47초였다.
- 비중은 GEMM 63.5%, LayerNorm 16.2%, attention 13.8% 순이었다.

## 실험

### 폐기: 병렬 LayerNorm reduction

- 256 threads가 평균과 분산을 tree reduction으로 계산하도록 구현했다.
- n=64에서는 0.719304초로 빨라졌고 검증도 통과했다.
- n=1024에서는 max abs diff 0.211973으로 검증에 실패했다.
- 합산 순서 변화가 일부 MoE routing에 영향을 준 것으로 보고 즉시 폐기했다.

### 채택: exact-order shared staging

- 각 행을 여러 threads가 shared memory에 coalesced load한다.
- 평균과 분산은 기존과 같은 단일 thread, 같은 인덱스 순서로 누적한다.
- fused residual add 결과도 shared memory에 동시에 보관해 재사용한다.
- attention query 128개도 기존 예약 shared memory에 한 번만 적재한다.

## 변경 후 프로파일

- 일반 LayerNorm 33회: 약 0.129초
- fused add + LayerNorm 32회: 약 0.124초
- GEMM은 약 3.536초, attention은 약 0.754초로 다음 주요 병목이다.

## 검증과 성능

| 버전 | n | 시간(초) | 처리량(seq/s) | 검증 |
|---|---:|---:|---:|---|
| 직전 제출 | 1024 | 6.766990 | 151.322827 | PASSED |
| shared-staged LayerNorm | 1024 | 6.390456 | 160.238964 | PASSED |
| + attention query reuse | 1024 | 6.328247 | 161.814155 | PASSED |

LayerNorm/attention 단계 제출 결과:

- elapsed: 6.330615초
- throughput: 161.753638 seq/s
- validation max abs diff: 0.0045929
- validation mean abs diff: 1.3264e-05
- 직전 개인 최고 대비: +6.9%
- 제출 시점 순위: 2위 / 8명

## Attention 구조적 병렬화 실험

### 채택: cooperative K staging

- query와 각 key의 dot product 누적은 기존 `d=0..127` 순서를 유지했다.
- 128 threads가 key vector를 shared memory에 coalesced load한 뒤 thread 0이
  같은 순서로 FP32 FMA를 수행했다.
- n=1024 반복 결과는 6.094168초와 6.072704초였으며 모두 검증을 통과했다.

### 폐기한 수치 병렬화

| 실험 | n | 시간(초) | 검증 결과 |
|---|---:|---:|---|
| warp dot-product reduction | 64 | 0.696079 | FAILED, max diff 9.22422 |
| 병렬 product + 순차 sum | 64 | 0.708780 | PASSED |
| 병렬 product + 순차 sum | 1024 | 6.005037 | FAILED, max diff 0.0191498 |

두 방식 모두 FP32 연산 순서를 바꿔 일부 MoE routing 결과에 영향을 줬으므로
성능과 관계없이 폐기했다.

### key 방향 병렬화

서로 다른 key score는 softmax 전까지 독립이므로 warp별로 key를 배정하고,
각 warp의 lane 0은 자기 score의 `d=0..127` 누적 순서를 그대로 유지했다.

| 구성 | n=1024 시간(초) | attention kernel 합계 | 검증 | 판정 |
|---|---:|---:|---|---|
| 단순 K staging | 6.072704 | 0.484초 | PASSED | 채택 |
| 4 warps / 4 keys | 6.071644 | 0.470초 | PASSED | 전체 이득 없음 |
| 8 warps / 8 keys | 6.096911 | - | PASSED | 느림 |

4-way는 attention kernel만 약 14ms 줄였지만 전체 시간에서는 측정 잡음 수준이었고
코드와 shared-memory 요구량이 증가해 남기지 않았다.

Attention 단계 최종 제출 결과:

- elapsed: 6.092199초
- throughput: 168.083798 seq/s
- validation max abs diff: 0.0045929
- validation mean abs diff: 1.3264e-05
- 직전 개인 최고 대비: +3.0%
- 제출 시점 순위: 2위 / 8명

## MoE W1/W3 및 QKV fusion 실험

### 채택: MoE W1/W3 + SiLU fusion

Expert의 독립적인 W1 gate와 W3 up projection이 input shared-memory tile을
공유하도록 구현했다. 두 accumulator는 기존 FP32 K 순서를 유지하고, 최종 store에서
`SiLU(gate) * up`을 바로 계산한다.

| 실행 | n | 시간(초) | 처리량(seq/s) | 검증 |
|---|---:|---:|---:|---|
| 1차 | 64 | 0.683109 | 93.689361 | PASSED |
| 1차 | 1024 | 6.040400 | 169.525203 | PASSED |
| 반복 | 1024 | 6.030337 | 169.808101 | PASSED |
| 제출 | 1024 | 6.124946 | 167.185135 | PASSED |

제출 측정은 변동으로 개인 최고 168.083798 seq/s를 넘지 못했다. 하지만 Nsight에서
기존 register GEMM 3.550초 + SiLU 0.041초가 fused 버전의 기존 GEMM 2.580초 +
pair-SiLU GEMM 0.967초로 줄어 약 44.6ms의 kernel 개선을 확인했다.

### 폐기: QKV projection + bias fusion

- Q 앞 512 columns와 K/V를 하나의 triple-accumulator kernel로 계산했다.
- Q 나머지 1536 columns는 single-accumulator tail kernel로 계산했다.
- input A tile을 공유하고 bias를 store에 합쳤으며 검증 오차는 기존과 동일했다.
- n=64는 0.695918초, n=1024는 6.259094초로 MoE-only보다 느렸다.
- Register pressure와 추가 synchronization 비용 때문에 코드를 제거했다.

## 변경 파일과 안전성

- 구현 변경은 `src/tensor.cu`에 한정했다.
- 모델 구조, routing 규칙, causal mask, 측정 구간은 변경하지 않았다.

## 재현 명령

```bash
make -B
./run.sh -n 1024 -v -o /tmp/project_final.bin
./submit.sh --no-update -n 1024
```

## GEMM blockDim/gridDim sweep

- register-blocked GEMM을 block/tile 상수로 일반화해 같은 FP32 K 누적을 유지했다.
- K tile은 기존 최선인 32로 고정하고 block shape와 output tile만 비교했다.
- n=64는 후보 선별용이며 최종 판단은 반드시 n=1024 결과로 했다.

| blockDim | output tile | n=64 시간(초) | 판정 |
|---|---|---:|---|
| 16x16 | 64x64 | 0.719589 | 기준 |
| 32x8 | 64x64 | 0.725352 | n=1024 재검증 |
| 8x32 | 64x64 | 1.002306 | 폐기 |
| 64x4 | 64x64 | 0.831598 | 폐기 |
| 32x16 | 64x64 | 0.708689 | n=1024 재검증 |
| 16x16 | 32x64 | 0.721336 | 폐기 |
| 16x16 | 64x32 | 0.682106 | n=1024 재검증 |
| 16x16 | 64x128 | 1.029087 | 폐기 |
| 16x16 | 128x64 | 0.866013 | 폐기 |

n=1024 finalist 결과:

| blockDim | output tile | 시간(초) | 처리량(seq/s) | 검증 |
|---|---|---:|---:|---|
| 16x16 | 64x64 | 6.330615 | 161.753638 | PASSED |
| 32x8 | 64x64 | 6.273929 | 163.215117 | PASSED |
| 32x8 반복 | 64x64 | 6.245893 | 163.947731 | PASSED |
| 32x16 | 64x64 | 6.705722 | 152.705403 | PASSED |
| 16x16 | 64x32 | 7.200540 | 142.211552 | PASSED |

선택한 geometry:

- blockDim: `(32, 8)` = 256 threads
- output tile: `64x64`, K tile: `32`
- gridDim: `(ceil(N/64), ceil(M/64))`
- thread당 output: `8x2`, accumulator 16개

최종 제출 결과:

- elapsed: 6.273657초
- throughput: 163.222190 seq/s
- validation max abs diff: 0.0045929
- validation mean abs diff: 1.3264e-05
- 직전 개인 최고 대비: +0.9%
- 제출 시점 순위: 2위 / 8명

## BF16 Tensor Core 및 weight pre-packing 실험

RTX 3090(sm_86)의 native BF16 Tensor Core를 외부 라이브러리 없이 사용하는
`USE_TC=1` opt-in 경로를 추가했다. 기본 FP32 빌드와 연산 경로는 그대로 유지한다.

### 구현 구조

- FP32 weight를 모델 초기화 중 16x16 tile-major BF16으로 pre-pack한다.
- 정확도 보정을 위해 `hi = BF16(x)`, `lo = BF16(x - FP32(hi))` 두 배열을 둔다.
- 두 BF16 배열의 합계는 FP32와 같은 4 bytes/weight이며, packing 완료 후 선택된
  weight의 기존 FP32 GPU 복사본은 비동기로 해제한다. 따라서 순수 BF16의 2배
  대역폭 이득 대신 정확도와 Tensor Core 처리량을 택한 경로다.
- custom WMMA kernel이 `Ahi*Bhi + Ahi*Blo + Alo*Bhi` 세 항을 FP32
  accumulator에 누적한다.
- CUDA block은 256 threads(8 warps), output 64x64, K tile 16이다.
- 한 warp가 16x16 accumulator 두 개를 담당하며 64x16 activation tile을
  16개 output tile이 공유한다.
- 수치 오차가 이후 routing으로 증폭되지 않도록 최종 LM head와 layer 31의
  Q/K/V/O projection에만 적용한다.

### 정확도 경계 ablation

| 구성 | n | 시간(초) | 최대 절대 오차 | 검증 |
|---|---:|---:|---:|---|
| 전체 projection/W2 순수 BF16 | 64 | 1.029111 | 1.32071 | FAILED |
| 전체 projection/W2 split-BF16 3항 | 64 | 1.085304 | 0.0235329 | FAILED |
| router FP32 + split-BF16 4항 | 64 | 1.065877 | 0.299095 | FAILED |
| LM head만 split-BF16 | 64 | 0.683977 | 0.00119019 | PASSED |
| LM head만 split-BF16 | 1024 | 6.030575 | 0.00396729 | PASSED |
| LM head + layer 31 attention | 1024 | 6.013399 | 0.00369644 | PASSED |
| 위 구성, 3항 보정 1차 | 1024 | 6.004792 | 0.00369644 | PASSED |
| 위 구성, 3항 보정 반복 | 1024 | 6.035852 | 0.00369644 | PASSED |
| 최종 재빌드 검증 | 1024 | 6.039755 | 0.00369644 | PASSED |
| 공식 제출 | 1024 | 6.037589 | 0.00369644 | PASSED |
| FP32 GPU copy 해제 후 검증 | 1024 | 6.009348 | 0.00369644 | PASSED |
| 최종 공식 제출 | 1024 | 6.029168 | 0.00369644 | PASSED |

마지막 2개 layer attention까지 확대하면 n=64 최대 오차가 0.0221062로
실패했고, 마지막 4개 layer는 한 token의 expert route가 바뀌어 최대 오차가
0.175797까지 커졌다. 192-column WMMA block도 activation 재사용은 늘었지만
warp당 accumulator 6개의 register pressure로 n=1024가 6.080052초로 느려져
64-column block으로 복구했다.

최종 공식 제출 처리량은 169.841024 seq/s로 기존 개인 최고 168.083798 seq/s보다
약 1.0% 높았으며, 제출 시점 순위는 5위 / 10명이었다.

재현 명령:

```bash
make clean
make -j4 USE_TC=1
./run.sh -n 1024 -v
```

## 루프 순서 및 pre-pack 순서 ablation

### WMMA weight traversal

기존 WMMA 계산은 activation tile을 재사용하기 위해 `K tile → local N tile`
순서로 돈다. 이에 맞춰 weight도 기존 `[N tile][K tile]`에서
`[N block 64][K tile][local N tile 4]`로 바꿔 같은 K 단계의 warp들이
연속 weight를 읽도록 실험했다.

| 실행 | n | 시간(초) | 처리량(seq/s) | 검증 |
|---|---:|---:|---:|---|
| K-major pre-pack 1차 | 64 | 0.663440 | 96.466926 | PASSED |
| K-major pre-pack 1차 | 1024 | 6.005273 | 170.516806 | PASSED |
| K-major pre-pack 반복 1 | 1024 | 6.036054 | 169.647256 | PASSED |
| K-major pre-pack 반복 2 | 1024 | 6.036712 | 169.628757 | PASSED |

첫 실행은 빨랐지만 반복 중앙값 6.036054초로 기존 공식 6.029168초보다
개선되지 않아 원래 N-major pre-pack으로 복구했다. Weight tile 하나가 이미
연속 512 bytes이고 L2가 간격 접근을 충분히 흡수해 배치 순서 효과가 작았다.

### FP32 register GEMM inner loop

기존 `K → thread-row → thread-col`에서 B shared 값 두 개를 먼저 레지스터에
올리고 8개 row에 재사용하는 순서를 명시했다. 출력별 K 누적 순서는 유지되어
검증 오차는 동일했다.

| 실행 | n | 시간(초) | 처리량(seq/s) | 검증 |
|---|---:|---:|---:|---|
| B register hoist 1차 | 1024 | 6.018018 | 170.155694 | PASSED |
| B register hoist 반복 1 | 1024 | 6.055185 | 169.111263 | PASSED |
| B register hoist 반복 2 | 1024 | 6.057044 | 169.059376 | PASSED |

반복 중앙값이 6.055185초로 느려졌다. 명시적 배열이 register lifetime을 늘렸고
컴파일러가 기존 루프에서 수행하던 hoist보다 불리해 원복했다. 현재 FP32 GEMM의
`K → row → col`, attention의 `key → dim`, MoE의 `expert → assigned token`
순서는 각각 shared tile 재사용, 연속 reduction, expert GEMM batching에 맞는
구조로 유지한다.

## Grouped-GQA K/V memory reuse

GQA에서 query head 4개가 하나의 KV head를 공유하지만 기존 attention kernel은
`(token, query head)`마다 block을 만들어 동일한 K/V를 네 번 읽었다. 이를
`(token, KV head)`마다 한 block으로 묶고 다음 순서로 변경했다.

```text
query 4개 shared staging
for key:
    K vector 1회 load
    query 4개의 score를 원래 dimension 순서로 계산
softmax는 head별 원래 key 순서 유지
for key:
    V vector 1회 load
    query head 4개의 value accumulator 갱신
```

Block 수는 `tokens*16`에서 `tokens*4`로 줄고 K/V global load는 이론상 4분의
1이 된다. 각 query head 내부의 QK dimension reduction과 value key reduction
순서는 바꾸지 않아 검증 오차가 기존과 완전히 동일했다.

| 구성 | n | 시간(초) | 처리량(seq/s) | 검증 |
|---|---:|---:|---:|---|
| grouped GQA + split-BF16 1차 | 64 | 0.672233 | 95.205058 | PASSED |
| grouped GQA + split-BF16 1차 | 1024 | 5.936697 | 172.486495 | PASSED |
| 반복 1 | 1024 | 5.940850 | 172.365916 | PASSED |
| 반복 2 | 1024 | 5.966226 | 171.632781 | PASSED |
| grouped GQA + 기본 FP32 빌드 | 1024 | 6.031837 | 169.765858 | PASSED |

split-BF16 빌드의 반복 중앙값은 5.940850초로 직전 공식 결과 6.029168초보다
약 88ms, 1.46% 개선됐다. Nsight Systems에서 grouped attention 32회 합계는
0.399138초였고, 기존 K-staging kernel의 약 0.484초보다 약 85ms 감소해 전체
개선량과 일치했다. 사용자 요청에 따라 이 결과는 대회에 제출하지 않았다.

## Host-free deterministic MoE routing 및 grouped expert GEMM

기존 GPU MoE는 router GEMM 이후 logits 전체를 D2H 복사하고 CPU에서 top-2와
expert별 row list를 만든 다음 각 expert row index를 다시 H2D 복사했다. 이를
다음 device-only pipeline으로 교체했다.

```text
GPU router logits
→ GPU quantized deterministic top-2 + integer expert count
→ GPU expert offsets 및 64-row work queue
→ GPU stable token-order compaction
→ compact expert input gather
→ grouped W1/W3 + SiLU
→ grouped W2
→ token별 deterministic expert-order combine
```

- 모델 초기화 때 expert W1/W2/W3 device pointer table만 등록한다.
- 측정 구간에서는 router logits D2H와 row-index H2D가 없다.
- top-2는 기존 `ROUTER_SCORE_QUANTUM`, `ROUTER_TIE_EPS`, 낮은 expert index
  우선 규칙을 그대로 구현한다.
- Integer atomic은 expert count에만 사용한다. 출력 합산에는 atomic을 쓰지 않는다.
- Expert별 token은 ballot/prefix compaction으로 원래 token 순서를 유지한다.
- Combine은 두 expert 결과를 expert 번호 오름차순으로 더해 기존 16개 expert
  순차 scatter-add의 FP32 순서를 재현한다.
- 각 work item은 최대 64 rows이며 grouped GEMM이 device의 expert pointer와
  offset을 직접 읽으므로 host가 동적 expert count를 알 필요가 없다.

### 성능 및 정확도

| 구성 | n | 시간(초) | 처리량(seq/s) | 검증 |
|---|---:|---:|---:|---|
| 1-thread stable plan 1차 | 64 | 0.407979 | 156.870912 | PASSED |
| 1-thread stable plan 1차 | 1024 | 5.699157 | 179.675705 | PASSED |
| 반복 1 | 1024 | 5.680974 | 180.250789 | PASSED |
| 반복 2 | 1024 | 5.704151 | 179.518385 | PASSED |
| ballot/prefix stable plan | 64 | 0.404579 | 158.189281 | PASSED |
| ballot/prefix 1차 | 1024 | 5.644809 | 181.405609 | PASSED |
| ballot/prefix 반복 1 | 1024 | 5.632597 | 181.798919 | PASSED |
| ballot/prefix 반복 2 | 1024 | 5.674278 | 180.463493 | PASSED |
| 기본 FP32 빌드 | 64 | 0.405022 | 158.016000 | PASSED |
| 기본 FP32 빌드 | 1024 | 5.667703 | 180.672833 | PASSED |

split-BF16 빌드의 최종 중앙값은 5.644809초다. grouped-GQA만 적용한
5.940850초보다 약 296ms(4.98%), 이전 공식 결과 6.029168초보다 약
384ms(6.38%) 줄었다. 초기 device plan은 Nsight에서 32회 합계 0.075037초였고,
integer count와 256-thread ballot/prefix stable compaction으로 바꾼 뒤 전체
시간이 추가로 약 54ms 감소했다. 사용자 요청에 따라 제출하지 않았다.

## K/V projection fusion 및 FP32 cp.async pipeline

### K/V A-tile 공유 fusion

Layer 0–30의 K/V projection은 같은 normalized input을 읽고 출력 크기도 각각
512로 같다. 한 block이 K 64 columns와 V 64 columns의 accumulator를 함께
유지하도록 만들어 다음 접근을 공유했다.

```text
for K tile:
    A[64,32] 1회 shared load
    K weight[64,32] load → K accumulator
    V weight[64,32] load → V accumulator
store: K/V bias addition까지 fusion
```

Layer 31 K/V는 검증된 split-BF16 Tensor Core 경로를 유지했다. 이에 따라
기존 FP32 projection kernel 124회는 Q/O 62회 + fused K/V 31회, 총 93회로
줄었고 K/V bias kernel 62회도 제거됐다.

| 구성 | n | 시간(초) | 처리량(seq/s) | 검증 |
|---|---:|---:|---:|---|
| K/V fusion | 64 | 0.393299 | 162.725885 | PASSED |
| K/V fusion 1차 | 1024 | 5.607393 | 182.616068 | PASSED |
| 반복 1 | 1024 | 5.623026 | 182.108357 | PASSED |
| 반복 2 | 1024 | 5.644586 | 181.412775 | PASSED |

반복 중앙값은 5.623026초로 직전 host-free MoE 중앙값 5.644809초보다 약
22ms 개선됐다. Nsight에서 Q/O single GEMM 62회는 1.699727초, fused K/V
31회는 0.451835초였다. Dual accumulator의 register pressure 때문에 projection
kernel 합계 자체는 소폭 늘었지만 A load, launch, bias read/write 제거가 전체
시간을 줄였다.

### 채택: Q/O FP32 GEMM cp.async

Q/O FP32 register GEMM에 2-stage `cp.async` pipeline을 적용했다.

- 현재 tile을 계산하는 동안 다음 A/B `[64,32]` tile을 비동기 적재한다.
- 16-byte vector copy를 위해 shared stride를 32로 두고, B는 4-float chunk
  단위 XOR swizzle로 bank conflict를 줄였다.
- Boundary M row는 `cp.async`의 zero-fill source size로 처리한다.
- Output별 K 누적 순서는 기존과 동일하다.
- K가 32의 배수가 아닌 일반 입력은 기존 synchronous kernel로 fallback한다.

| 구성 | n | 시간(초) | 처리량(seq/s) | 검증 |
|---|---:|---:|---:|---|
| Q/O cp.async | 64 | 0.378402 | 169.132288 | PASSED |
| Q/O cp.async 1차 | 1024 | 5.456281 | 187.673605 | PASSED |
| 반복 1 | 1024 | 5.440426 | 188.220561 | PASSED |
| 반복 2 | 1024 | 5.415901 | 189.072881 | PASSED |
| 정리 후 재검증 | 1024 | 5.398133 | 189.695219 | PASSED |
| 최종 USE_TC 재빌드 | 1024 | 5.402435 | 189.544147 | PASSED |
| 기본 FP32 빌드 | 64 | 0.387846 | 165.014156 | PASSED |
| 기본 FP32 빌드 | 1024 | 5.472922 | 187.102987 | PASSED |

Nsight에서 Q/O kernel 62회 합계가 synchronous 1.699727초에서 asynchronous
1.512656초로 약 187ms 감소했다.

### 폐기: fused K/V cp.async

K/V fusion은 A와 두 weight의 double buffer가 필요해 shared memory를 정확히
48KB 사용했다. 이로 인한 occupancy 하락이 overlap 이득보다 커 n=64가
0.378402초에서 0.454527초로 느려졌다. K/V는 16KB shared memory를 사용하는
synchronous fused kernel로 복구했다. 사용자 요청에 따라 이 조합 역시 대회에
제출하지 않았다.

## 전체 attention Tensor Core 확대 실험

`USE_TC_ALL` 실험 빌드에서 layer 0–31의 Q/K/V/O weight까지 split-BF16
pre-pack해 attention projection 124개를 Tensor Core로 실행했다. 기존 채택
경로의 마지막 layer 제한을 제거한 정확도 경계 실험이다.

| 구성 | n | 시간(초) | 최대 절대 오차 | 검증 |
|---|---:|---:|---:|---|
| 전체 attention, split 3항 | 64 | 0.368914 | 0.29916 | FAILED |
| 전체 attention, split 4항 | 64 | 0.393465 | 0.29916 | FAILED |

네 번째 `Alo*Blo` 항은 표현 절삭 오차를 줄여도 결과를 바꾸지 못했다. 따라서
주된 원인은 Tensor Core FP32 accumulator의 연산 순서 차이가 여러 layer를 거쳐
MoE top-2 routing을 바꾸는 것이다. 이 실험은 n=1024 성능 측정과 제출을 하지
않고 폐기했으며, 기존의 LM head + layer 31 attention 제한 경로를 유지한다.

## Grouped attention Q-head 4-way 병렬화

### 측정 방법 정정: 노드 간 편차

이 실험 도중 `aps` partition의 노드마다 같은 binary의 실행 시간이 크게 다르다는
사실을 확인했다. 같은 baseline binary가 n=64에서 0.381초(빠른 노드)와
0.590초(느린 노드)로 측정됐다. 따라서 이번 실험부터 baseline과 변경본을
**하나의 `srun` 할당 안에서 번갈아 실행**해 비교했다.

같은 이유로 validation 최대 절대 오차도 노드에 따라 달라진다. 동일 binary가
n=64에서 노드에 따라 0.00174713과 0.000102997을 보고했다. 오차 값 변화만으로
연산 순서 변경을 판단하면 안 되며, 실제 판정은 출력 binary 비교로 해야 한다.

### 채택: head별 score/softmax thread 배정

`k_attention_grouped_gqa4`는 K/V를 4개 query head가 공유하지만, score와 softmax는
thread 0이 head 4개를 연달아 계산했다. GQA group의 4개 head는 value combine
전까지 서로 독립이므로 thread 0–3에 head를 하나씩 배정했다.

- 각 head의 QK 누적은 여전히 `d=0..127` 단일 thread 순차 FP32 FMA다.
- softmax의 max, exp, denom, 정규화도 head별 단일 thread 순차 계산이다.
- `queries`와 `scores`를 head-minor(`[d][g]`, `[w][g]`)로 저장해 4개 thread가
  서로 다른 shared memory bank에 접근하게 했다.
- `q` 적재는 coalesced 순서를 유지하고 shared 저장만 transpose한다.
- shared memory 사용량은 `4*max_seq + 5*head_dim` floats로 동일하다.

### 비트 동일성 검증

출력 binary를 baseline과 직접 비교했다.

| 비교 | 결과 |
|---|---|
| n=8 출력 binary | IDENTICAL |
| n=64 출력 binary | IDENTICAL |
| n=1024 출력 binary | IDENTICAL |

연산 순서를 바꾸지 않았으므로 MoE routing에도 영향이 없다.

### 성능

같은 노드(b5) 안에서 번갈아 3회씩 실행했다.

| 구성 | n | 시간(초) | 처리량(seq/s) | 최대 절대 오차 | 검증 |
|---|---:|---:|---:|---:|---|
| baseline | 1024 | 5.593363 | 183.074110 | 0.0045929 | PASSED |
| head 4-way | 1024 | 5.336384 | 191.890234 | 0.0045929 | PASSED |
| baseline | 1024 | 5.592794 | 183.092742 | 0.0045929 | PASSED |
| head 4-way | 1024 | 5.355142 | 191.218098 | 0.0045929 | PASSED |
| baseline | 1024 | 5.603171 | 182.753652 | 0.0045929 | PASSED |
| head 4-way | 1024 | 5.350063 | 191.399628 | 0.0045929 | PASSED |

중앙값 기준 5.592794초에서 5.350063초로 약 243ms, 4.3% 개선됐다.

Nsight 기준 n=1024 kernel 합계(노드 b6):

| kernel | baseline | head 4-way |
|---|---:|---:|
| `k_matmul_transposed_register_blocked_async` | 1574.1 ms | 1590.5 ms |
| `k_moe_pair_silu_grouped` | 790.2 ms | 798.1 ms |
| `k_matmul_pair_bias_register_blocked` | 441.7 ms | 442.2 ms |
| `k_moe_matmul_grouped` | 385.8 ms | 386.6 ms |
| `k_attention_grouped_gqa4` | 390.6 ms | 112.0 ms |
| 전체 kernel 합계 | 4070.1 ms | 3821.4 ms |

attention kernel은 390.6ms에서 112.0ms로 3.49배 빨라졌고 278.6ms를 줄였다.
전체 GPU kernel 시간에서 attention 비중은 9.6%에서 2.9%로 내려가, 다음 병목은
GEMM(전체의 약 84%)과 kernel 사이 공백 시간이다.

사용자 요청에 따라 이 버전은 대회에 제출하지 않았다.

## CUDA Graph 검토와 host embedding gather 병목

### CUDA Graph는 이득이 없다

launch overhead를 nsys로 직접 측정했다.

- `cudaLaunchKernel` 620회 합계 11.0 ms
- inference kernel 615개 사이의 GPU idle 합계 210 ms
- 그중 200.2 ms는 첫 GEMM 앞의 단일 gap(layer 0 attention weight H2D + memory
  pool 확장)이고, 3.5/3.2/0.8/0.8 ms 4개를 빼면 나머지 약 610개 gap의 합은 2 ms다.

CUDA Graph가 없앨 수 있는 것은 이 마지막 2 ms뿐이므로 전체의 0.05%다. MoE가
device-side work queue를 쓰고 grid 크기가 batch에 대해 고정이라 capture 자체는
가능하지만, 실익이 없어 착수하지 않았다.

### 정정: kernel 밖 공백은 1.5초가 아니었다

이전 기록에서 "kernel 합계 3821 ms vs 실측 5350 ms이므로 약 1.5초가 kernel 밖
공백"이라고 적었으나 두 값이 서로 다른 노드에서 측정된 잘못된 비교였다. 같은
노드에서 다시 측정한 결과는 다음과 같다.

| 구성 | n=1024 시간(초) |
|---|---:|
| cold | 5.353665 |
| warm (`-w`) | 5.155582 |

차이는 198 ms이며 위 gap 분석의 210 ms와 일치한다. GPU는 이미 포화 상태였다.

또한 MoE expert weight 약 11.3 GB는 `PhiMoE` 생성자의
`register_moe_weights_gpu()`가 `cuda_data()`를 호출하면서 이미 timed region
밖에서 업로드되고 있다. timed region 안에 남은 H2D는 attention weight와
lm_head 약 3.3 GB뿐이고 그것이 위 200 ms gap의 실체다.

### 채택: embedding gather를 row memcpy로

`generate()`의 embedding gather가 `Tensor::at()`를 원소마다 호출하고 있었다.
`at()`은 원소마다 `ensure_host()`와 rank 검사·범위 검사가 있는
`offset(initializer_list)`를 수행한다. total_tokens × HIDDEN_SIZE 약 8500만
원소를 읽기와 쓰기 양쪽에서 그렇게 접근했다.

- 두 base pointer를 한 번만 해석하고 행 단위 `std::memcpy`로 교체했다.
- token 범위 검사는 행 단위로 그대로 유지했다.
- 행끼리 독립이므로 OpenMP로 병렬화했다.
- 기록되는 바이트는 완전히 동일하다.

이 구간은 GPU kernel 프로파일에 전혀 나타나지 않는 순수 host 시간이라 지금까지
발견되지 않았다.

| 구성 | n=1024 시간(초) | 처리량(seq/s) | 최대 절대 오차 | 검증 |
|---|---:|---:|---:|---|
| baseline | 1024 | 5.354890 | 191.227075 | 0.0045929 | PASSED |
| row memcpy | 1024 | 4.300023 | 238.138264 | 0.0045929 | PASSED |
| baseline | 1024 | 5.348384 | 191.459696 | 0.0045929 | PASSED |
| row memcpy | 1024 | 4.307282 | 237.736918 | 0.0045929 | PASSED |
| baseline | 1024 | 5.343864 | 191.621657 | 0.0045929 | PASSED |
| row memcpy | 1024 | 4.305678 | 237.825489 | 0.0045929 | PASSED |

출력 binary는 baseline과 IDENTICAL이다. 중앙값 5.348384초에서 4.305678초로
약 1.04초, 19.5% 단축됐다.

### 변경 후 예산

노드 b6, n=1024 기준이다.

| 항목 | 시간 |
|---|---:|
| 실측 elapsed | 4321 ms |
| inference kernel span | 4006 ms |
| GPU busy | 3796 ms (87.9%) |
| kernel 사이 gap | 210 ms (그중 200 ms는 attention weight H2D 단일 gap) |
| kernel span 밖 host 구간 | 315 ms |

kernel 밖 시간을 전부 없애도 상한은 약 3.80초, 253 seq/s다. 그 이상은 GEMM
kernel 자체를 빠르게 하는 방법밖에 없다.

사용자 요청에 따라 이 버전도 대회에 제출하지 않았다.

## Tensor Core 적용 현황 점검

### 빌드 위생 문제

`make`는 `USE_TC` 여부와 무관하게 같은 `obj/*.o` 경로를 쓴다. 앞선 attention과
embedding 실험에서 `obj/layer.o`는 이전 `make USE_TC=1` 빌드가 남긴 것이었고
`obj/tensor.o`와 `obj/model.o`만 `-DUSE_TC` 없이 재컴파일됐다. 그 결과:

- `layer.cu`의 `Linear` 생성자는 `USE_TC`가 켜진 상태로 BF16 pre-pack을 수행하고
  `prepare_cuda_bf16_weight()`가 FP32 device 사본을 `cudaFreeAsync` 한다.
- `tensor.cu` 1730행의 WMMA dispatch는 `#ifdef USE_TC`라 컴파일에서 빠졌다.
- 따라서 해당 weight는 BF16으로 pack된 뒤 사용되지 않고, FP32 kernel이
  `cuda_data()`를 호출해 host에서 FP32를 다시 업로드했다.

결과는 정확했지만 pack 작업과 재업로드가 낭비였다. `USE_TC`를 바꿀 때는 반드시
`make clean`을 먼저 해야 한다. 두 실험의 A/B는 양쪽이 동일한 빌드 상태였으므로
delta는 유효하고, 절대값만 이 낭비를 포함한 값이었다.

### Clean build 비교

노드 b0, n=1024, 번갈아 3회.

| 빌드 | 시간(초) | 처리량(seq/s) | 최대 절대 오차 | 검증 |
|---|---:|---:|---:|---|
| `make` (FP32) | 4.151432 | 246.661870 | 0.0045929 | PASSED |
| `make USE_TC=1` | 4.128762 | 248.016261 | 0.00369644 | PASSED |
| `make` (FP32) | 4.148596 | 246.830477 | 0.0045929 | PASSED |
| `make USE_TC=1` | 4.140954 | 247.286009 | 0.00369644 | PASSED |
| `make` (FP32) | 4.149021 | 246.805219 | 0.0045929 | PASSED |
| `make USE_TC=1` | 4.133062 | 247.758178 | 0.00369644 | PASSED |

### 적용 범위

현재 Tensor Core로 실행되는 weight는 5개뿐이다.

- `lm_head.weight`
- `model.layers.31.self_attn.{q,k,v,o}_proj.weight`

전체 GEMM weight는 1697개(layer당 attention 4 + router 1 + expert 16×3, 그리고
lm_head)이므로 적용률은 0.3%다. 이득이 약 15ms, 0.35%인 것은 이 범위 때문이지
Tensor Core 자체의 한계가 아니다.

layer 31 attention과 lm_head로 범위를 제한한 이유는 앞선 `USE_TC_ALL` 실험에서
attention 전체로 확대하면 최대 절대 오차가 0.29916으로 검증에 실패했기
때문이다. 남은 1692개 GEMM을 Tensor Core로 옮기는 것이 유일하게 남은 큰
레버이며, 병목은 속도가 아니라 MoE top-2 routing을 보존하는 정확도다.

## FP32 GEMM 잔여 최적화 3종

Tensor Core 확대 전에, 검증 리스크가 없는 FP32 경로만 먼저 깎았다. 세 실험 모두
K 누적 순서를 바꾸지 않으므로 출력이 bit 단위로 같아야 하고, 실제로 세 경우 모두
n=1024 출력 binary가 IDENTICAL이었다.

### 채택: MoE W2 grouped GEMM에 cp.async

expert output projection은 cp.async가 걸리지 않은 마지막 단일 weight GEMM이었다.
Q/O가 쓰던 2-stage double buffer와 4-float XOR swizzle을 그대로 적용했다.
`EXPERT_INTERMEDIATE_SIZE`가 448로 `BIG_BK`의 배수라 모든 tile이 꽉 차고
16-byte 정렬도 유지된다. 다른 K는 기존 synchronous kernel로 fallback한다.

| 구성 | n=1024 시간(초) | 처리량(seq/s) |
|---|---:|---:|
| baseline | 4.112822 | 248.977464 |
| W2 cp.async | 4.065078 | 251.901701 |

### 폐기: synchronous kernel의 global load 벡터화

`[BIG_BK + 1]` padding을 유지한 채 global 쪽만 `float4`로 읽고 shared 저장은
scalar로 두는 staging helper를 만들어 pair kernel 6곳에 적용했다. 결과는 오히려
368ms 느렸다.

| 구성 | n=1024 시간(초) | 처리량(seq/s) |
|---|---:|---:|
| W2 cp.async | 4.119617 | 248.566785 |
| + global load 벡터화 | 4.487814 | 228.173436 |

shared 배열을 `float dst[][BIG_BK + 1]` 형태로 `__device__` 함수에 넘기면서
shared state space가 소실돼 generic addressing으로 컴파일된 것이 유력한 원인이다.
출력은 IDENTICAL이었으나 성능 회귀라 폐기했다.

### 채택: pair kernel에 weight별 shared tile 분리

`k_moe_pair_silu_grouped`와 `k_matmul_pair_bias_register_blocked`는 shared `Bs`
하나를 두 weight가 번갈아 쓰고 있었다. 그래서 K tile마다 `__syncthreads()`가
4번 필요했고, A 값도 weight마다 한 번씩 두 번 읽었다.

weight마다 shared tile을 따로 두고 두 개를 모두 적재한 뒤 하나의 barrier로
넘어가게 바꿨다.

- K tile당 barrier 4회 → 2회
- A 값은 `(p, r)`마다 shared에서 1회만 읽음
- shared 사용량 16.9KB → 25.3KB (SM당 block 5개 → 3개)
- 각 accumulator의 K 누적 순서는 그대로

occupancy가 내려가는데도 barrier 절감이 더 컸다.

| 구성 | n=1024 시간(초) | 처리량(seq/s) |
|---|---:|---:|
| W2 cp.async | 4.120179 | 248.532871 |
| + dual B tile | 3.912432 | 261.729773 |

### 누적 결과

| 단계 | n=1024 시간(초) | 처리량(seq/s) |
|---|---:|---:|
| 세션 시작 | 5.402435 | 189.544147 |
| attention Q-head 4-way | — | — |
| embedding row memcpy | — | — |
| clean USE_TC 재빌드 | 4.133062 | 247.758178 |
| W2 cp.async | 4.065078 | 251.901701 |
| dual B tile | 3.912432 | 261.729773 |

세션 시작 대비 1.38배다. 사용자 요청에 따라 제출하지 않았다.

## Thread block 스케줄링 분석과 occupancy 개선

### ncu 실측

| kernel | regs/thread | occupancy limiter (block/SM) | 달성 occupancy | SM tput | issue | shared load conflict |
|---|---:|---|---:|---:|---:|---:|
| Q/O async | 130 | shared 1 · regs 1 · warps 6 | 16.66% | 33.9% | 55.3% | 0 |
| MoE W2 async | 148 | shared 1 · regs 1 · warps 6 | 16.65% | 55.1% | 55.4% | 0 |
| MoE W1/W3 | 110 | shared 2 · regs 2 · warps 6 | 22.89% | 41.8% | 56.4% | 458,752 |
| K/V pair | 96 | shared 2 · regs 2 · warps 6 | 16.66% | 14.1% | 48.4% | 98,304 |
| attention | 40 | shared 16 · regs 12 · warps 12 | 39.93% | 38.8% | 24.5% | 0 |

### grid와 wave 구조는 문제가 없다

n=1024, SM 82개 기준이다.

| kernel | grid | wave 수 |
|---|---:|---:|
| Q proj | 9,920 | 121.0 |
| O proj | 19,840 | 242.0 |
| K/V pair | 2,480 | 15.1 |
| MoE W1/W3 | 4,445 | 27.1 |
| MoE W2 | 40,640 | 495.6 |
| attention | 79,212 | 80.5 |

모든 kernel이 wave를 충분히 많이 돌아 tail effect와 wave quantization은 무시할
수준이다. MoE work queue에서 `work_expert[wi] >= NUM_EXPERTS`로 즉시 return하는
빈 block은 635개 중 최대 16개(2.5%)다. 병목은 grid 배분이 아니라 SM당 block
점유율이었다.

### 채택: __launch_bounds__ + shared carveout

Q/O와 MoE W2가 SM당 block 1개로 돌고 있었다. 제약이 두 개 동시에 걸려 있었다.

- 레지스터: 130 regs × 256 threads = 33,280 regs. SM 정원 65,536 → 1 block
- shared: static 32,768 B. SM에 100 KB를 쓸 수 있는데도 driver가 작은 carveout을
  골라 1 block

둘 중 하나만 풀면 효과가 없다. 실제로 첫 시도에서 `__launch_bounds__(256, 2)`만
적용했더니 register limiter는 1 → 2가 됐지만 shared limiter가 1에 머물러 달성
occupancy가 16.66%로 그대로였고, n=1024도 4.709946초 → 4.705134초로 변화가
없었다.

`cudaFuncAttributePreferredSharedMemoryCarveout`에 50(%)을 요청한 것도 조용히
무시됐다. `cudaSharedmemCarveoutMaxShared`로 바꾸자 비로소 적용됐다.

| 지표 | 이전 | 이후 |
|---|---:|---:|
| register limiter | 1 | 2 |
| shared limiter | 1 | 3 |
| 실질 block/SM | 1 | 2 |
| 달성 occupancy (W2) | 16.65% | 32.38% |

`cuobjdump -res-usage` 기준 두 kernel 모두 `REG:128 STACK:0 LOCAL:0`으로 spill이
없다.

| 구성 | n=1024 시간(초) | 처리량(seq/s) |
|---|---:|---:|
| baseline | 3.847106 | 266.174115 |
| launch_bounds + carveout | 3.695074 | 277.125700 |
| baseline | 3.845897 | 266.257769 |
| launch_bounds + carveout | 3.704769 | 276.400466 |
| baseline | 3.856764 | 265.507534 |
| launch_bounds + carveout | 3.698225 | 276.889572 |

중앙값 3.847106초 → 3.698225초로 약 149ms, 3.9% 개선됐다. 출력 binary는
IDENTICAL이다.

### 남은 항목

- padded kernel 두 개의 2-way shared load bank conflict. `Bs[tx * BIG_TN + cc][p]`는
  stride 33에서 `bank = (2·tx + cc + p) mod 32`가 되어 짝수 뱅크 16개만 쓴다.
  padding 값으로는 못 고치고, 열 매핑을 `cc * BIG_BX + tx`로 바꾸면 conflict가
  사라지고 global store도 연속이 된다. 미착수.
- attention은 occupancy가 아니라 issue rate 24.5%가 문제다. limiter가 이미
  warps(12)라 occupancy로는 더 얻을 게 없다.

## Attention key tiling

`k_attention_grouped_gqa4`의 score 구간은 key를 하나씩 처리하면서 key마다
`__syncthreads()`를 2회 걸었고, 그 사이에 128 threads 중 4개만 계산했다. ncu
기준 warp이 활성 시간의 63.2%를 barrier에서 대기했다.

서로 다른 key의 score는 softmax 전까지 독립이므로 key를 tile 단위로 묶어
한꺼번에 계산하도록 바꿨다. thread `t`가 key slot `t / 4`의 head `t % 4`를
맡는다.

- 각 score는 여전히 단일 thread가 `d = 0..127` 순차 FP32 FMA로 누적한다
- barrier가 key당 2회에서 tile당 2회로 줄어든다
- 이 데이터는 시퀀스가 평균 약 19 토큰이라 대부분의 row가 tile 하나로 끝난다
- key tile은 행마다 float 하나를 덧대 bank가 갈리게 했다

### tile 크기 선택

tile을 키우면 barrier는 줄지만 shared 사용량이 늘어 occupancy가 떨어진다.

| tile | n=1024 시간(초) | 판정 |
|---|---:|---|
| 없음 (baseline) | 3.715407 | |
| 32 | 3.707923 | shared 16.5KB, occupancy 90% → 40% |
| **16** | **3.690873** | 채택 |
| 8 | 3.691629 | 16과 동등 |

tile 32는 n=64 기준 attention kernel을 332.8us에서 266.5us로 20% 줄였고 barrier
stall도 63.2%에서 39.8%로 내렸지만, 달성 occupancy가 90.4%에서 39.9%로
떨어져 전체 이득이 깎였다. 16이 두 효과의 균형점이다.

| 구성 | n=1024 시간(초) | 처리량(seq/s) |
|---|---:|---:|
| baseline | 3.761359 | 272.242060 |
| key tile 16 | 3.726404 | 274.795783 |
| baseline | 3.764890 | 271.986682 |
| key tile 16 | 3.723553 | 275.006185 |

중앙값 3.761488초 → 3.726404초로 약 35ms, 0.9% 개선됐다. 출력 binary는
IDENTICAL이다.

## 폐기: padded GEMM kernel의 bank conflict 제거

`Bs[tx * BIG_TN + cc][p]`는 stride `BIG_BK + 1`에서 `bank = (2·tx + cc + p) mod 32`가
되어 짝수 뱅크 16개만 쓴다. 열 매핑을 `cc * BIG_BX + tx`로 바꾸면 `bank =
(tx + p) mod 32`가 되어 conflict가 사라지고 epilogue store도 연속이 된다.

실제로 ncu 기준 shared load conflict가 `k_moe_pair_silu_grouped` 458,752 → 0,
`k_matmul_pair_bias_register_blocked` 98,304 → 0이 됐다. 그런데 느려졌다.

| 구성 | n=1024 시간(초) | 처리량(seq/s) |
|---|---:|---:|
| baseline | 3.739176 | 273.857096 |
| conflict 0 | 3.775107 | 271.250554 |

출력은 IDENTICAL이었으나 약 36ms 회귀라 폐기했다. stall 분해를 보면
`k_moe_pair_silu_grouped`의 shared stall은 1.6%에 불과했고 실제 병목은 global
15.8%였다. counter 절대값이 크다고 critical path인 것이 아니다.

## GEMM 타일 확장 (thread당 accumulator 16 → 32)

### 진단

n=1024 재프로파일 결과 GEMM이 GPU busy의 84%였고, 커널별 실효 성능이
전부 14 TFLOPS 근처로 균일했다.

| kernel | 연산량(TFLOP) | 시간(ms) | 실효 |
|---|---:|---:|---:|
| Q/O projection | 20.60 | 1381.2 | 14.9 TFLOPS |
| MoE W1/W3 | 9.30 | 692.6 | 13.4 TFLOPS |
| K/V projection | 5.15 | 373.0 | 13.8 TFLOPS |
| MoE W2 | 4.65 | 319.1 | 14.6 TFLOPS |
| 합계 | 39.79 | 2813.8 | 14.1 TFLOPS |

RTX 3090 FP32 peak 35.6 TFLOPS 대비 40%다. 특정 kernel이 느린 것이 아니라
모든 GEMM이 같은 tile 구조를 공유해 같은 한계에 걸려 있었다.

원인은 산술 강도다. thread당 accumulator가 `BIG_TM x BIG_TN = 8 x 2 = 16`이라
K step마다 A를 8개, B를 2개 읽어 16 FMA를 한다. shared load 0.625회/FMA다.

### 채택: N tile을 128로

`WIDE_BN = 128`, `WIDE_TN = 4`로 thread당 accumulator를 32개로 늘렸다.
K step마다 A 8개 + B 4개로 32 FMA, 0.375회/FMA다. B 값은 K step마다 register로
한 번 올려 8개 row가 재사용한다. cp.async 2-stage와 XOR swizzle, K 누적 순서는
그대로다.

| kernel | N | regs | shared | 결과 |
|---|---:|---:|---:|---|
| Q/O projection | 2048 / 4096 | 128, spill 0 | 48.0 KB | 3.688039 → 3.481855초 |
| MoE W2 | 4096 | 128, spill 0 | 48.0 KB | 3.527453 → 3.482493초 |
| K/V projection | 512 | 128, spill 32B | 41.25 KB | 3.481 → 3.457초 |

static shared 상한이 block당 48 KB라 `BIG_BK`를 32로 유지한 채 N만 128까지
늘릴 수 있었다. `BIG_BM`까지 키우면 상한을 넘는다.

K/V는 두 출력이 있어 accumulator가 64개다. `__launch_bounds__` 없이는 ptxas가
158 regs를 잡아 SM당 block이 1개로 떨어지고 오히려 느렸다(3.525초). bound를
걸어 128 regs + stack 32B spill로 block 2개를 유지하는 쪽이 빨랐다.

| 구성 | n=1024 시간(초) |
|---|---:|
| K/V 64-wide (baseline) | 3.481 |
| K/V 128-wide, 158 regs, 1 block/SM | 3.525 |
| K/V 128-wide, 128 regs + 32B spill, 2 block/SM | 3.457 |

### 폐기: MoE W1/W3 타일 확장

`EXPERT_INTERMEDIATE_SIZE`가 448이라 128의 배수가 아니다. tile 4개가 512열을
덮어 12.5%가 낭비되고, spill까지 겹쳐 느려졌다.

| 구성 | n=1024 시간(초) | 처리량(seq/s) |
|---|---:|---:|
| baseline | 3.395747 | 301.553672 |
| W1/W3 128-wide | 3.437604 | 297.881865 |

약 42ms 회귀라 폐기했다. 이 kernel은 64-wide를 유지한다.

### 누적

| 단계 | n=1024 시간(초) | 처리량(seq/s) |
|---|---:|---:|
| 세션 시작 | 5.402435 | 189.544147 |
| clean USE_TC 재빌드 | 4.133062 | 247.758178 |
| W2 cp.async | 4.065078 | 251.901701 |
| dual B tile | 3.912432 | 261.729773 |
| occupancy (launch_bounds + carveout) | 3.698225 | 276.889572 |
| attention key tile | 3.726404 | 274.795783 |
| Q/O 128-wide | 3.481855 | 294.096134 |
| MoE W2 128-wide | 3.482493 | 294.042269 |
| K/V 128-wide | 3.395747 | 301.553672 |

세션 시작 대비 1.59배다. 사용자 요청에 따라 제출하지 않았다.

### Tensor Core 상한 정정

이전 기록에서 "600 seq/s는 compensated TF32로만 가능하다"고 적었으나 잘못된
성능표에 근거한 것이었다. RTX 3090(GA102)의 실제 값은 다음과 같다.

| 경로 | 이론 최대 |
|---|---:|
| FP32 CUDA core | 35.6 TFLOPS |
| TF32 Tensor Core | 35.6 TFLOPS |
| BF16 Tensor Core (FP32 accumulate) | 71 TFLOPS |

TF32는 이 카드에서 FP32와 동일하고, 보정하면 MMA를 3회 해야 하므로 오히려
느리다. BF16은 2배지만 3항 보정이 그 이득을 나눠 상한이 23.7 TFLOPS다.
실제로 기존 `k_matmul_transposed_bf16_wmma` 경로는 1.0996 TFLOP를 70.3ms에
처리해 15.6 TFLOPS로, FP32 경로 14.1 TFLOPS 대비 11%만 빠르다.

600 seq/s(1.707초)는 GEMM에 0.783초만 허용하므로 50.8 TFLOPS가 필요하다.
검증을 유지하는 어떤 경로도 이 값에 도달하지 못한다. 현실적인 상한은
330~350 seq/s다.

## 200 ms H2D gap 재확인과 embedding gather의 GPU 이전

### 문서에 적힌 200 ms gap은 이미 사라졌다

앞의 "CUDA Graph 검토" 절은 첫 GEMM 앞에 layer 0 attention weight H2D로 인한
200.2 ms 단일 gap이 있다고 기록했다. 현재 코드(prefix-dedup, dc338bf)에서
`Tensor::prepare_cuda()`에 계측을 넣어 다시 확인했다.

- 프로그램 전체 `prepare_cuda()` 호출 4,720회.
- 그중 timed region 안에서 실제 H2D 복사가 일어나는 것은 **단 1건**,
  `hidden` embedding 결과 255.3 MB(alloc 0.70 ms + copy 26.05 ms)뿐이다.
- attention weight와 lm_head를 포함한 모든 weight는 `ModelLoader::load()`가
  마지막 줄에서 `out.prepare_cuda()`를 호출하므로 model load 단계에서 이미
  올라가 있다. timed region에는 weight H2D가 남아 있지 않다.

즉 200 ms gap의 원인으로 지목됐던 weight H2D는 이후 커밋에서 제거됐고, 그
기록은 현재 코드에 해당하지 않는다.

### 실제로 남아 있던 것은 host embedding gather였다

`generate()`에 host 타임스탬프를 넣어 측정한 결과(n=1024)다.

| 구간 | host 시간 |
|---|---:|
| generate 진입 → trie 완성 | 1.9 ms |
| trie 완성 → embedding gather 완료 | 165.8 ms |
| gather 완료 → layer 루프 시작 | 0.2 ms |
| layer 루프 32개 enqueue 전체 | 86.6 ms |
| layer 루프 종료 → generate 반환(GPU 대기) | 2,418 ms |

host는 32개 layer를 86.6 ms 만에 전부 enqueue하고 나머지는 GPU를 기다린다.
따라서 layer 루프 앞의 167.7 ms는 GPU가 완전히 idle인 순수 host 구간이다.
`hidden`은 모든 layer의 첫 입력이므로 이 시간은 어떤 GPU 작업과도 겹칠 수
없다.

165.8 ms의 내역은 255.3 MB짜리 host 버퍼의 최초 할당·zero-fill(`data_.assign`)
과 15,583행 × 16 KB의 gather memcpy다. 그 뒤에 layer 0이 같은 255.3 MB를
다시 H2D로 올린다.

### 채택: embedding gather를 GPU에서 수행

embedding table(525 MB)은 이미 device에 상주해 있다. gather를 GPU로 옮기면
host 255 MB 할당·zero-fill, host gather memcpy, 255 MB H2D 업로드 세 패스가
모두 사라지고 62 KB index 업로드와 kernel 하나만 남는다.

```
Tensor hidden({nodes, apss26::HIDDEN_SIZE});
tensor_ops::gather_rows_gpu(embeddings_, node_token, hidden);
```

`gather_rows_gpu`는 prefix expansion이 이미 쓰던 경로다. 같은 행을 같은 순서로
복사하는 순수 copy라 결과는 bit-identical이며, 작업은 여전히 timed region 안에
있다. timed region 밖으로 옮기거나 결과를 미리 캐시하지 않는다.

### 측정

노드 b6, `make clean && make USE_TC=1`, n=1024. 두 바이너리를 번갈아 실행했다.

| 구성 | n=1024 시간(초) | 처리량(seq/s) | 최대 절대 오차 | 평균 절대 오차 |
|---|---:|---:|---:|---:|
| baseline | 2.743767 | 373.209577 | 0.00369644 | 0.000505105 |
| baseline | 2.762259 | 370.711120 | 0.00369644 | 0.000505105 |
| baseline | 3.301946 | — | 0.00369644 | 0.000505105 |
| baseline | 3.303727 | — | 0.00369644 | 0.000505105 |
| baseline | 2.748124 | — | 0.00369644 | 0.000505105 |
| baseline | 2.741028 | — | 0.00369644 | 0.000505105 |
| baseline | 2.744017 | — | 0.00369644 | 0.000505105 |
| baseline | 2.759655 | — | 0.00369644 | 0.000505105 |
| GPU gather | 2.565368 | 399.162967 | 0.00369644 | 0.000505105 |
| GPU gather | 3.139407 | 326.176230 | 0.00369644 | 0.000505105 |
| GPU gather | 2.568856 | — | 0.00369644 | 0.000505105 |
| GPU gather | 2.562972 | — | 0.00369644 | 0.000505105 |
| GPU gather | 2.569810 | — | 0.00369644 | 0.000505105 |
| GPU gather | 2.564727 | — | 0.00369644 | 0.000505105 |
| GPU gather | 2.568114 | — | 0.00369644 | 0.000505105 |
| GPU gather | 3.139274 | — | 0.00369644 | 0.000505105 |

이 노드는 두 개의 mode를 보인다. 빠른 mode는 baseline 2.74~2.76초 / GPU gather
2.563~2.570초, 느린 mode는 baseline 3.30초 / GPU gather 3.14초다. 두 mode
모두에서 이득이 나타난다.

중앙값은 2.753초 → 2.569초로 약 184 ms, 6.7% 단축이다. 처리량은 372 seq/s에서
399 seq/s가 된다.

출력 binary는 baseline과 byte 단위로 IDENTICAL이며(`cmp` 확인), 검증값은
0.00369644 / 0.000505105로 변하지 않았다.
