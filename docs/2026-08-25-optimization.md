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
