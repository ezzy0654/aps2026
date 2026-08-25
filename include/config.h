#pragma once

#include <cstddef>

// Phi-tiny-MoE-instruct (microsoft/Phi-tiny-MoE-instruct).
namespace apss26 {
constexpr std::size_t VOCAB_SIZE = 32064;
constexpr std::size_t HIDDEN_SIZE = 4096;
constexpr std::size_t NUM_LAYERS = 32;
constexpr std::size_t NUM_ATTENTION_HEADS = 16;
constexpr std::size_t NUM_KV_HEADS = 4;
constexpr std::size_t HEAD_DIM = 128;
constexpr std::size_t NUM_EXPERTS = 16;
constexpr std::size_t TOP_K = 2;
constexpr std::size_t EXPERT_INTERMEDIATE_SIZE = 448;
constexpr std::size_t MAX_POSITION_EMBEDDINGS = 4096;
constexpr std::size_t SLIDING_WINDOW = 2047;
constexpr float ROPE_THETA = 10000.0f;
constexpr float NORM_EPS = 1.0e-5f;
constexpr float ROUTER_JITTER_EPS = 0.01f;
// Canonicalize router scores before deterministic dynamic top-2 routing.
constexpr float ROUTER_SCORE_QUANTUM = 1.0e-3f;
constexpr float ROUTER_TIE_EPS = 2.0e-2f;
constexpr std::size_t MAX_DECODE_TOKENS = 8;
constexpr int EOS_TOKEN_ID = 32000;
}
