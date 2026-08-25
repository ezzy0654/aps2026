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

최종 제출 결과:

- elapsed: 6.330615초
- throughput: 161.753638 seq/s
- validation max abs diff: 0.0045929
- validation mean abs diff: 1.3264e-05
- 직전 개인 최고 대비: +6.9%
- 제출 시점 순위: 2위 / 8명

## 변경 파일과 안전성

- 구현 변경은 `src/tensor.cu`에 한정했다.
- 모델 구조, routing 규칙, causal mask, 측정 구간은 변경하지 않았다.

## 재현 명령

```bash
make -B
./run.sh -n 1024 -v -o /tmp/project_final.bin
./submit.sh --no-update -n 1024
```
