# Prefill 구조적 independence 분석

## 레이어 dependency DAG

```text
layer input x
  ├──────────────────────── residual ───────────────────────┐
  └─ input LayerNorm ─┬─ Q projection ─ RoPE(Q) ─┐         │
                      ├─ K projection ─ RoPE(K) ─┼─ attention ─ O projection
                      └─ V projection ───────────┘         │
                                                           │
                         residual add + post LayerNorm ◀────┘
                                      │
                                   router
                                      │
                 ┌──────── expert 0 ... expert 15 ────────┐
                 └────────── deterministic combine ───────┘
                                      │
                                residual add
                                      │
                                  next layer
```

## 독립적으로 계산 가능한 영역

- Batch sequence는 서로 독립이며 현재 `[total_tokens, hidden]` packing으로 활용한다.
- Linear의 output row/column은 독립이다. 단, 각 output 내부 K reduction 순서는
  FP32 결과를 보존하기 위해 유지해야 한다.
- Q/K/V projection은 동일한 normalized input만 읽으므로 attention join 전까지
  서로 독립이다.
- Q RoPE와 K RoPE는 서로 독립이다.
- Attention의 sequence, query row, query head와 softmax 전 key score는 독립이다.
- Softmax 이후 value의 head dimension 128개는 서로 독립이다.
- GQA query head 네 개는 계산은 독립이지만 동일한 K/V head를 공유하므로 데이터
  재사용이 가능하다.
- Routing 이후 expert 16개의 계산은 독립이다. 다만 한 token의 두 expert 결과를
  output에 합산하는 순서는 FP32 재현성을 위해 고정해야 한다.
- 각 expert의 W1 gate projection과 W3 up projection은 동일한 input만 읽으므로
  SiLU multiplication 전까지 서로 독립이다.

## 반드시 순차적인 경계

- Decoder layer 사이에는 `layer[n+1] input = layer[n] output` 의존성이 있다.
- Q/K/V와 attention, 모든 score와 softmax, attention과 O projection은 join
  dependency를 가진다.
- Router 결과가 있어야 expert dispatch를 만들 수 있다.
- W1/W3 결과가 모두 있어야 `SiLU(W1(x)) * W3(x)`를 계산할 수 있다.
- Expert 계산은 병렬화할 수 있지만 최종 scatter-add는 같은 token에 쓰므로
  무조건적인 concurrent write가 안전하지 않다.

## FP32에서 수학적 independence와 수치 independence의 차이

Dot product dimension과 reduction 항목은 수학적으로 순서를 바꿀 수 있지만,
FP32 덧셈은 결합법칙을 만족하지 않는다. 실제 실험에서 LayerNorm tree reduction,
attention warp reduction, product/sum 분리가 MoE routing을 바꿔 검증에 실패했다.
따라서 fusion은 output별 K 누적 순서를 유지하고, 독립 branch 사이에서 input tile과
kernel launch를 공유하는 방향으로 구현한다.

## 이번 fusion 대상

1. MoE W1/W3: input tile을 한 번 load해 두 projection을 계산하고 SiLU×up까지
   fused output으로 저장한다.
2. Attention Q/K/V: input tile을 공유해 세 projection을 계산하고 각 bias를
   output store에 결합한다.

## 실험 결과

### MoE W1/W3 fusion: 채택

- W1/W3가 A shared-memory tile을 공유하고 각 accumulator의 K 순서는 유지했다.
- `SiLU(W1(x)) * W3(x)`를 accumulator store에 결합해 gate/up Tensor와 별도
  elementwise kernel을 제거했다.
- n=1024 측정은 6.040400초와 6.030337초였고 기존과 동일한 검증 오차로 통과했다.
- Nsight 기준 관련 register GEMM+SiLU 합계가 약 3.591초에서 3.547초로 줄어
  약 44.6ms의 kernel 시간을 절감했다.

### QKV projection fusion: 폐기

- Q의 앞 512 columns와 K/V를 triple-accumulator kernel에서 계산하고, Q의
  나머지 1536 columns는 tail kernel로 계산했다.
- 각 output의 K 누적 순서와 bias 적용은 유지해 검증 오차는 변하지 않았다.
- 그러나 n=64는 MoE-only 0.683109초보다 느린 0.695918초, n=1024는
  6.259094초였다.
- Triple accumulator의 register pressure와 projection마다 필요한 synchronization이
  A tile 재사용 이득보다 커서 구현을 제거했다.
