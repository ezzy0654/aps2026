# Prefill CUDA 최적화 기록

## 범위와 제약

- 대상은 `PhiTinyMoEModel::generate()`가 수행하는 prefill이다.
- decode 경로, 모델 구조, 라우팅 규칙과 출력 형식은 변경하지 않았다.
- cuBLAS/cuDNN 등 금지 라이브러리를 사용하지 않고 CUDA 커널만 구현했다.
- 모델 연산 결과를 측정 구간 밖에서 계산하거나 warmup 결과를 재사용하지 않는다.
- 모델 로딩 단계의 GPU 전송은 불변 파라미터의 메모리 배치이며 추론 결과 캐싱이 아니다.

## 구현

### Packed batch

1024개 가변 길이 입력을 `[total_tokens, hidden]` 형태로 연결한다. 현재 데이터는
19,803개의 유효 토큰을 가지므로 길이 32로 패딩한 32,768개 토큰보다 연산량이
작다. Attention의 segment offset과 local position을 별도로 유지해 시퀀스 사이의
attention을 차단하고 RoPE position을 각 시퀀스에서 0으로 재시작한다.

### GPU 상주 Tensor

Tensor가 host와 device mirror의 최신 상태를 추적한다. 모델 파라미터는 모델 로딩
중 GPU에 배치하고, prefill의 중간 Tensor는 다음 CUDA 연산까지 device에 유지한다.
기존 GEMM별 H2D/D2H 복사와 `cudaDeviceSynchronize()`를 제거했다.

### CUDA 연산

- FP32 projection 및 expert GEMM
- layer norm
- bias와 residual add
- SiLU와 up projection 결과의 곱셈 fusion
- packed RoPE
- packed causal GQA attention
- MoE expert gather/scatter
- 마지막 유효 토큰 gather

MoE top-2 판정은 기존 CPU 양자화와 tie-break 규칙을 유지한다. Router logits만
CPU에서 판정하며 expert 입력과 출력은 GPU에서 gather/scatter한다.

### Register-blocked GEMM

Router처럼 작은 행렬에는 기존 16x16 커널을 사용한다. M과 N이 64 이상인
projection, expert, LM head에는 다음 구성을 사용한다.

- output tile: 64x64
- K tile: 32
- CUDA block: 16x16 threads
- thread당 4x4 output accumulator
- A/B tile을 shared memory에서 재사용
- FP32 K 누적 순서를 유지해 기존 검증 오차를 보존

### Lazy host storage

GPU 커널이 결과 전체를 덮어쓰는 임시 Tensor는 생성 시 CPU `vector<float>`를
할당하거나 0으로 채우지 않는다. 논리 원소 수를 host storage 크기와 분리하고,
CPU 접근이 실제로 발생할 때만 host storage를 생성한다. 따라서 GPU prefill의
projection 및 expert 중간 결과 때문에 매 layer마다 대용량 CPU 메모리를
할당·초기화하는 비용이 사라진다.

MoE 출력은 여러 expert 결과가 누적되므로 예외적으로 GPU에서
`cudaMemsetAsync`로 0 초기화한다. CPU/decode 경로는 최초 host 접근 시 기존과
동일한 0 초기화 의미를 유지한다. 모델 연산이나 측정 범위는 바뀌지 않았다.

## 측정 결과

RTX 3090, warmup 없음, 기존 `main.cpp` 측정 구간과 검증을 그대로 사용했다.

| 버전 | n | 시간(초) | 처리량(seq/s) | 검증 |
|---|---:|---:|---:|---|
| GPU 상주 packed baseline | 64 | 2.052632 | 31.179476 | PASSED |
| register block, K=16 | 64 | 1.646162 | 38.878302 | PASSED |
| register block, K=32 | 64 | 1.593983 | 40.150984 | PASSED |
| GPU 상주 packed baseline | 1024 | 58.462671 | 17.515450 | PASSED |
| register block, K=16 | 1024 | 56.493838 | 18.125871 | PASSED |
| register block, K=32 | 1024 | 56.405452 | 18.154273 | PASSED |
| lazy host storage | 64 | 0.758166 | 84.414252 | PASSED |
| lazy host storage | 1024 | 7.072075 | 144.794849 | PASSED |
| lazy host storage (repeat) | 1024 | 6.988849 | 146.519119 | PASSED |

최종 n=1024 검증 결과:

- max absolute difference: `0.0045929`
- mean absolute difference: `1.3264e-05`
- verdict: `Validation: PASSED!`

## 2026-08-24 활동 기록

### 병목 분석

- n=1024 실행을 Nsight Systems로 분석했다.
- 당시 전체 측정 시간은 약 56.3초였지만 GPU kernel 합계는 약 5.32초였다.
- GPU kernel 중 register-blocked GEMM은 약 3.43초, layer norm은 약 0.85초,
  attention은 약 0.70초였다.
- 약 49초의 host-side 공백을 추적해, GPU가 결과 전체를 덮어쓰는 임시 Tensor도
  생성 시 CPU `vector<float>`를 할당하고 0으로 채우는 것이 주 병목임을 확인했다.

### 구현 및 검증

- Tensor의 논리 원소 수(`numel_`)와 실제 host storage 크기를 분리했다.
- GPU 출력 Tensor는 host 접근이 발생하기 전까지 CPU 메모리를 할당하지 않도록
  지연 할당했다.
- `cuda_data_write()`, `reshape()`, `fill()`을 논리 원소 수 기준으로 수정했다.
- 누적이 필요한 MoE 출력만 `cudaMemsetAsync`로 명시적으로 0 초기화했다.
- 전체 재빌드 후 n=64와 n=1024 정확도 검증을 모두 통과했다.
- n=1024를 반복 측정해 7.072초와 6.989초를 얻어 개선의 재현성을 확인했다.

### 점수 개선 및 제출

- 기존 개인 최고: 18.17 seq/s
- 최종 제출: 7.048475초, 145.279645 seq/s, n=1024
- 개선율: 약 699.4%
- 제출 시점 순위: 1위 / 7명
- 제출 명령은 기본 n=1 대신 `./submit.sh --no-update -n 1024`를 사용해야 한다.

## 앞으로 해볼 활동

모든 실험은 기존 warmup 없는 n=1024 측정과 정확도 검증을 유지하고, 성능이
나빠지거나 검증 오차를 넘으면 독립적으로 되돌릴 수 있게 진행한다.

| 우선순위 | 활동 | 기대 효과 | 주요 확인 사항 |
|---:|---|---|---|
| 1 | 최적화 후 Nsight Systems/Compute 재프로파일링 | 현재 7초 실행의 실제 병목 확정 | kernel 시간, CPU-GPU gap, occupancy, memory throughput |
| 2 | Q/K/V와 MoE w1/w3 projection fusion | 공통 입력 재사용과 kernel launch 감소 | FP32 누적 순서와 출력 오차 유지 |
| 3 | deterministic top-2 routing 및 dispatch의 GPU 이전 | layer별 D2H 동기화와 CPU routing 제거 | 기존 양자화 및 tie-break 규칙 완전 일치 |
| 4 | shape별 GPU temporary workspace 재사용 | 반복되는 `cudaMallocAsync`/`cudaFreeAsync` 감소 | 동시 사용 Tensor의 lifetime과 aliasing |
| 5 | LayerNorm, bias, residual 등 인접 elementwise fusion | global memory 왕복과 작은 kernel launch 감소 | residual 적용 순서 유지 |
| 6 | packed GQA attention 개선 | K/V 재사용 및 softmax 중간 저장 감소 | causal mask와 sequence 경계 보존 |
| 7 | 직접 구현한 TF32 WMMA GEMM 선택 적용 | RTX 3090 tensor core 활용 | projection별 정확도 검증, 실패 시 FP32 유지 |

첫 활동은 반드시 최신 구현의 재프로파일링으로 한다. 그 결과 CPU-GPU 동기화가
크면 GPU routing을, GEMM 비중이 계속 가장 크면 projection fusion 또는 WMMA를,
allocation 비중이 크면 reusable workspace를 먼저 진행한다.

## 2026-08-25 업데이트

- residual add와 post-attention LayerNorm을 하나의 CUDA kernel로 결합했다.
- LayerNorm 입력을 shared memory에 coalesced staging하되 평균과 분산의 누적
  순서는 그대로 유지했다.
- attention query를 기존 shared-memory 예약 공간에 한 번만 적재해 재사용했다.
- 최종 제출은 n=1024에서 6.330615초, 161.753638 seq/s, 검증 통과였다.
- 상세 실험과 폐기한 병렬 reduction 결과는 `2026-08-25-optimization.md`에 있다.
