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
