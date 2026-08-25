#include "model.h"
#include "config.h"
#include <cuda_runtime.h>
#include <getopt.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static std::string input_fname = "/apss26/project-data/inputs.bin";
static std::string model_fname = "/apss26/project-data/model.bin";
static std::string output_fname = "./outputs.bin";
static std::string answer_fname = "/apss26/project-data/answers.bin";
static std::size_t num_sequences = 1024;
static bool run_validation = false;
static bool run_warmup = false;
static bool run_decode = false;

static double get_time() {
    struct timespec tv{};
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return static_cast<double>(tv.tv_sec) +
           static_cast<double>(tv.tv_nsec) * 1.0e-9;
}

static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

static int get_visible_device_count() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice ||
        status == cudaErrorInsufficientDriver) {
        cudaGetLastError();
        return 0;
    }
    check_cuda(status, "cudaGetDeviceCount");
    return device_count;
}

static void synchronize_devices() {
    const int device_count = get_visible_device_count();
    if (device_count == 0) return;

    int current_device = 0;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");

    for (int device = 0; device < device_count; ++device) {
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    check_cuda(cudaSetDevice(current_device), "cudaSetDevice(restore)");
}

static void print_help(const char* prog) {
    std::fprintf(stdout,
                 " Usage: %s [-o path] [-n count] [-d] [-v] [-w] [-h]\n",
                 prog);
    std::fprintf(stdout, " Options:\n");
    std::fprintf(stdout,
                 "  -o: Output binary path (default: ./data/outputs.bin)\n");
    std::fprintf(stdout,
                 "  -n: Number of leading sequences to run (default: 1024)\n");
    std::fprintf(stdout,
                 "  -d: Run prefill plus greedy decode (default: OFF)\n");
    std::fprintf(stdout,
                 "  -v: Validate output against the answer binary (default: OFF)\n");
    std::fprintf(stdout,
                 "  -w: Enable three warm-up iterations (default: OFF)\n");
    std::fprintf(stdout, "  -h: Print this help message\n");
}

static void parse_args(int argc, char** argv) {
    static const option long_options[] = {
        {"output", required_argument, nullptr, 'o'},
        {"num-sequences", required_argument, nullptr, 'n'},
        {"decode", no_argument, nullptr, 'd'},
        {"validate", no_argument, nullptr, 'v'},
        {"warmup", no_argument, nullptr, 'w'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int option = 0;
    while ((option = getopt_long(argc, argv, "i:p:o:a:n:dvwh", long_options,
                                 nullptr)) != -1) {
        switch (option) {
            case 'o': output_fname = optarg; break;
            case 'a': answer_fname = optarg; break;
            case 'n': {
                char* end = nullptr;
                const unsigned long long parsed =
                    std::strtoull(optarg, &end, 10);
                if (end == optarg || *end != '\0' || parsed == 0) {
                    std::fprintf(stderr,
                                 "-n/--num-sequences must be a positive integer\n");
                    print_help(argv[0]);
                    std::exit(2);
                }
                num_sequences = static_cast<std::size_t>(parsed);
                break;
            }
            case 'd': run_decode = true; answer_fname = "/apss26/project-data/decode_answers.bin"; break;
            case 'v': run_validation = true; break;
            case 'w': run_warmup = true; break;
            case 'h':
                print_help(argv[0]);
                std::exit(0);
            default:
                print_help(argv[0]);
                std::exit(2);
        }
    }

    std::fprintf(stdout, "\n=============================================\n");
    std::fprintf(stdout, " Model\n");
    std::fprintf(stdout, "---------------------------------------------\n");
    std::fprintf(stdout, " Validation: %s\n", run_validation ? "ON" : "OFF");
    std::fprintf(stdout, " Warm-up: %s\n", run_warmup ? "ON" : "OFF");
    std::fprintf(stdout, " Decode: %s\n", run_decode ? "ON" : "OFF");
    if (run_decode) {
        std::fprintf(stdout, " Max decode tokens: %zu\n",
                     apss26::MAX_DECODE_TOKENS);
    }
    std::fprintf(stdout, " Number of sequences: %zu\n", num_sequences);
    std::fprintf(stdout, " Output binary path: %s\n", output_fname.c_str());
    std::fprintf(stdout, "=============================================\n\n");
}

// Variable-length input format:
//   uint32 batch
//   uint32 max_seq_len
//   repeated batch times:
//     uint32 actual_seq_len
//     int32  input_ids[actual_seq_len]
// The sequence order in the file is the order required in the output. The
// model implementation may reorder internally, but must restore this order.
static std::vector<std::vector<int>> read_inputs(
    const std::string& path, std::size_t& max_seq_len) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open input: " + path);
    std::uint32_t batch = 0, max_seq = 0;
    f.read(reinterpret_cast<char*>(&batch), sizeof(batch));
    f.read(reinterpret_cast<char*>(&max_seq), sizeof(max_seq));
    if (!f || batch == 0 || max_seq == 0 ||
        max_seq > apss26::MAX_POSITION_EMBEDDINGS) {
        throw std::runtime_error("invalid input header");
    }

    max_seq_len = max_seq;
    std::vector<std::vector<int>> ids(batch);
    for (auto& row : ids) {
        std::uint32_t actual_seq = 0;
        f.read(reinterpret_cast<char*>(&actual_seq), sizeof(actual_seq));
        if (!f || actual_seq == 0 || actual_seq > max_seq) {
            throw std::runtime_error("invalid sequence length in input");
        }
        row.resize(actual_seq);
        for (int& token : row) {
            std::int32_t value = 0;
            f.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!f) throw std::runtime_error("truncated input");
            token = value;
        }
    }
    return ids;
}

static void write_outputs(const std::string& path, const Tensor& logits) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output: " + path);
    const std::uint32_t batch = static_cast<std::uint32_t>(logits.size(0));
    const std::uint32_t vocab = static_cast<std::uint32_t>(logits.size(1));
    f.write(reinterpret_cast<const char*>(&batch), sizeof(batch));
    f.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));
    f.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
    if (!f) throw std::runtime_error("failed while writing output: " + path);
}

// Decode output/answer format:
//   uint32 batch
//   uint32 max_decode_tokens
//   uint32 vocab_size
//   uint32 lengths[batch]
//   int32  generated_ids[batch][max_decode_tokens]  (-1 after length)
//   float  logits[batch][max_decode_tokens][vocab_size]
//
// The logits at decode step s are the logits used to greedily select the
// s-th generated token. Step 0 is the final-position prefill logits.
static void write_decode_outputs(
    const std::string& path,
    const Tensor& logits,
    const std::vector<std::vector<int>>& generated_ids) {
    if (logits.ndim() != 3 || logits.size(0) != generated_ids.size()) {
        throw std::runtime_error("invalid decode output shape");
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output: " + path);
    const std::uint32_t batch = static_cast<std::uint32_t>(logits.size(0));
    const std::uint32_t max_steps = static_cast<std::uint32_t>(logits.size(1));
    const std::uint32_t vocab = static_cast<std::uint32_t>(logits.size(2));
    f.write(reinterpret_cast<const char*>(&batch), sizeof(batch));
    f.write(reinterpret_cast<const char*>(&max_steps), sizeof(max_steps));
    f.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));

    for (const auto& row : generated_ids) {
        if (row.size() > max_steps) {
            throw std::runtime_error("generated sequence exceeds decode limit");
        }
        const std::uint32_t length = static_cast<std::uint32_t>(row.size());
        f.write(reinterpret_cast<const char*>(&length), sizeof(length));
    }
    for (const auto& row : generated_ids) {
        for (std::size_t step = 0; step < max_steps; ++step) {
            const std::int32_t token =
                step < row.size() ? static_cast<std::int32_t>(row[step]) : -1;
            f.write(reinterpret_cast<const char*>(&token), sizeof(token));
        }
    }
    f.write(reinterpret_cast<const char*>(logits.data()),
            static_cast<std::streamsize>(logits.size() * sizeof(float)));
    if (!f) throw std::runtime_error("failed while writing output: " + path);
}

static Tensor read_answers(const std::string& path,
                           std::size_t batch,
                           std::size_t vocab) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open answer: " + path);

    const std::streamsize expected_bytes = static_cast<std::streamsize>(
        batch * vocab * sizeof(float));
    const std::streamsize answer_file_bytes = f.tellg();
    if (answer_file_bytes < expected_bytes) {
        throw std::runtime_error(
            "invalid answer size: expected at least " +
            std::to_string(expected_bytes) +
            " bytes, got " + std::to_string(static_cast<long long>(f.tellg())));
    }

    f.seekg(0, std::ios::beg);
    Tensor answers({batch, vocab});
    f.read(reinterpret_cast<char*>(answers.data()), expected_bytes);
    if (!f) throw std::runtime_error("failed while reading answer: " + path);
    return answers;
}

struct DecodeAnswers {
    std::vector<std::uint32_t> lengths;
    std::vector<std::int32_t> generated_ids;
    Tensor logits;
};

static DecodeAnswers read_decode_answers(
    const std::string& path,
    std::size_t batch,
    std::size_t max_steps,
    std::size_t vocab) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open decode answer: " + path);
    const std::streamsize file_bytes = f.tellg();
    constexpr std::streamsize header_bytes = 3 * sizeof(std::uint32_t);
    if (file_bytes < header_bytes) {
        throw std::runtime_error("decode answer is too small");
    }

    f.seekg(0, std::ios::beg);
    std::uint32_t file_batch = 0, file_steps = 0, file_vocab = 0;
    f.read(reinterpret_cast<char*>(&file_batch), sizeof(file_batch));
    f.read(reinterpret_cast<char*>(&file_steps), sizeof(file_steps));
    f.read(reinterpret_cast<char*>(&file_vocab), sizeof(file_vocab));
    if (!f || file_batch < batch || file_steps != max_steps ||
        file_vocab != vocab) {
        throw std::runtime_error("decode answer header mismatch");
    }

    const std::streamoff lengths_offset = header_bytes;
    const std::streamoff ids_offset =
        lengths_offset + static_cast<std::streamoff>(file_batch) * sizeof(std::uint32_t);
    const std::streamoff logits_offset =
        ids_offset + static_cast<std::streamoff>(file_batch) * file_steps * sizeof(std::int32_t);
    const std::streamoff expected_file_bytes =
        logits_offset + static_cast<std::streamoff>(file_batch) * file_steps * file_vocab * sizeof(float);
    if (file_bytes < expected_file_bytes) {
        throw std::runtime_error("decode answer is truncated");
    }

    DecodeAnswers answer;
    answer.lengths.resize(batch);
    answer.generated_ids.resize(batch * max_steps);
    answer.logits = Tensor({batch, max_steps, vocab});

    f.seekg(lengths_offset, std::ios::beg);
    f.read(reinterpret_cast<char*>(answer.lengths.data()),
           static_cast<std::streamsize>(batch * sizeof(std::uint32_t)));
    f.seekg(ids_offset, std::ios::beg);
    f.read(reinterpret_cast<char*>(answer.generated_ids.data()),
           static_cast<std::streamsize>(batch * max_steps * sizeof(std::int32_t)));
    f.seekg(logits_offset, std::ios::beg);
    f.read(reinterpret_cast<char*>(answer.logits.data()),
           static_cast<std::streamsize>(batch * max_steps * vocab * sizeof(float)));
    if (!f) throw std::runtime_error("failed while reading decode answer");
    return answer;
}

static bool validate(const Tensor& output, const Tensor& answer) {
    if (output.shape() != answer.shape()) {
        throw std::runtime_error("output/answer shape mismatch");
    }

    constexpr float threshold = 3.0e-3f;
    float max_abs_error = 0.0f;
    float mean_abs_error = 0.0f;
    std::size_t first_bad = output.size();

    for (std::size_t i = 0; i < output.size(); ++i) {
        const float actual = output[i];
        const float expected = answer[i];
        const float abs_error = std::fabs(actual - expected);
        const float rel_error = (std::fabs(expected) > 1.0e-8f)
                                    ? abs_error / std::fabs(expected)
                                    : abs_error;
        max_abs_error = std::max(max_abs_error, abs_error);
        mean_abs_error += abs_error;

        if (first_bad == output.size() &&
            (std::isnan(actual) ||
             (abs_error > threshold && rel_error > threshold))) {
            first_bad = i;
        }
    }
    mean_abs_error /= static_cast<float>(output.size());

    std::fprintf(stdout, "Validation max abs diff: %g\n", max_abs_error);
    std::fprintf(stdout, "Validation mean abs diff: %g\n", mean_abs_error);
    if (first_bad == output.size()) {
        std::fprintf(stdout, "Validation: PASSED!\n");
        return true;
    }

    const std::size_t vocab = output.size(1);
    std::fprintf(stdout,
                 "Validation: FAILED! First mismatch at [%zu, %zu] "
                 "(output=%g, answer=%g)\n",
                 first_bad / vocab, first_bad % vocab,
                 output[first_bad], answer[first_bad]);
    return false;
}

static bool validate_decode(
    const Tensor& output,
    const std::vector<std::vector<int>>& generated_ids,
    const DecodeAnswers& answer) {
    if (output.shape() != answer.logits.shape() ||
        generated_ids.size() != answer.lengths.size()) {
        throw std::runtime_error("decode output/answer shape mismatch");
    }

#ifdef USE_TC
    constexpr double step0_normalized_mae_threshold = 2.0e-2;
    constexpr double step0_huber_mean_threshold = 1.0e-4;
#else
    constexpr double step0_threshold = 3.0e-3;
#endif
    constexpr double normalized_mae_threshold = 7.5e-2;
    constexpr double huber_delta = 1.0e-2;
    constexpr double huber_mean_threshold = 7.5e-4;

    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    std::size_t compared = 0;

    for (std::size_t b = 0; b < generated_ids.size(); ++b) {
        if (generated_ids[b].size() != answer.lengths[b]) {
            std::fprintf(stdout,
                         "Validation: FAILED! Generated length mismatch at "
                         "sequence %zu (output=%zu, answer=%u)\n",
                         b, generated_ids[b].size(), answer.lengths[b]);
            return false;
        }

        for (std::size_t step = 0; step < generated_ids[b].size(); ++step) {
            const std::size_t flat_id = b * answer.logits.size(1) + step;
            const std::int32_t expected_token =
                answer.generated_ids[flat_id];
            if (generated_ids[b][step] != expected_token) {
                std::fprintf(stdout,
                             "Validation: FAILED! Token mismatch at [%zu, %zu] "
                             "(output=%d, answer=%d)\n",
                             b, step, generated_ids[b][step], expected_token);
                return false;
            }

            double normalized_abs_sum = 0.0;
            double huber_sum = 0.0;

            for (std::size_t v = 0; v < output.size(2); ++v) {
                const double actual = output.at(b, step, v);
                const double expected = answer.logits.at(b, step, v);
                if (!std::isfinite(actual) || !std::isfinite(expected)) {
                    std::fprintf(stdout,
                                 "Validation: FAILED! Non-finite decode logit "
                                 "at [%zu, %zu, %zu]\n", b, step, v);
                    return false;
                }

                const double abs_error = std::fabs(actual - expected);
#ifndef USE_TC
                const double rel_error =
                    (std::fabs(expected) > 1.0e-8)
                        ? abs_error / std::fabs(expected)
                        : abs_error;
#endif
                max_abs_error = std::max(max_abs_error, abs_error);
                mean_abs_error += abs_error;
                ++compared;

#ifndef USE_TC
                if (step == 0 &&
                    abs_error > step0_threshold &&
                    rel_error > step0_threshold) {
                    std::fprintf(stdout,
                                 "Validation max abs diff: %g\n"
                                 "Validation mean abs diff: %g\n"
                                 "Validation: FAILED! Prefill-logit mismatch "
                                 "at [%zu, %zu, %zu] (output=%g, answer=%g)\n",
                                 max_abs_error,
                                 mean_abs_error /
                                     static_cast<double>(compared),
                                 b, step, v, actual, expected);
                    return false;
                }

#endif

#ifdef USE_TC
                if (step == 0) {
                    const double normalized =
                        abs_error / (1.0 + std::fabs(expected));
                    normalized_abs_sum += normalized;
                    huber_sum += normalized <= huber_delta
                        ? 0.5 * normalized * normalized
                        : huber_delta *
                              (normalized - 0.5 * huber_delta);
                }
#endif

                if (step > 0) {
                    const double normalized =
                        abs_error / (1.0 + std::fabs(expected));
                    normalized_abs_sum += normalized;
                    huber_sum += normalized <= huber_delta
                        ? 0.5 * normalized * normalized
                        : huber_delta *
                              (normalized - 0.5 * huber_delta);
                }
            }

            if (step > 0) {
                const double vocab = static_cast<double>(output.size(2));
                const double normalized_mae =
                    normalized_abs_sum / vocab;
                const double huber_mean = huber_sum / vocab;
                if (normalized_mae > normalized_mae_threshold ||
                    huber_mean > huber_mean_threshold) {
                    std::fprintf(stdout,
                                 "Validation max abs diff: %g\n"
                                 "Validation mean abs diff: %g\n"
                                 "Validation: FAILED! Decode vector mismatch "
                                 "at [%zu, %zu] (normalized MAE=%g, Huber=%g)\n",
                                 max_abs_error,
                                 mean_abs_error /
                                     static_cast<double>(compared),
                                 b, step, normalized_mae, huber_mean);
                    return false;
                }
            }

#ifdef USE_TC
            if (step == 0) {
                const double vocab = static_cast<double>(output.size(2));
                const double normalized_mae = normalized_abs_sum / vocab;
                const double huber_mean = huber_sum / vocab;
                if (normalized_mae > step0_normalized_mae_threshold ||
                    huber_mean > step0_huber_mean_threshold) {
                    std::fprintf(stdout,
                                 "Validation max abs diff: %g\n"
                                 "Validation mean abs diff: %g\n"
                                 "Validation: FAILED! Prefill vector mismatch "
                                 "at [%zu, %zu] (normalized MAE=%g, "
                                 "Huber=%g)\n",
                                 max_abs_error,
                                 mean_abs_error /
                                     static_cast<double>(compared),
                                 b, step, normalized_mae,
                                 huber_mean);
                    return false;
                }
            }
#endif
        }
    }

    std::fprintf(stdout, "Validation max abs diff: %g\n", max_abs_error);
    std::fprintf(stdout, "Validation mean abs diff: %g\n",
                 compared == 0
                     ? 0.0
                     : mean_abs_error / static_cast<double>(compared));
    std::fprintf(stdout, "Validation: PASSED!\n");
    return true;
}
int main(int argc, char** argv) {
    parse_args(argc, argv);

    try {
        ////////////////////////////////////////////////////////////////////
        // INITIALIZATION                                                 //
        ////////////////////////////////////////////////////////////////////

        std::fprintf(stdout, "Initializing inputs and parameters...");
        std::fflush(stdout);

        std::size_t max_seq_len = 0;
        auto ids = read_inputs(input_fname, max_seq_len);
        if (num_sequences < ids.size()) {
            ids.resize(num_sequences);
        }
        const std::size_t batch = ids.size();

        // Keep the model lifetime explicit so that the finalization phase
        // below releases all model-owned tensors before writing outputs.
        const double load_start = get_time();
        auto model_ref = std::make_unique<PhiTinyMoEModel>(model_fname);
        const double load_end = get_time();

        Tensor logits;
        std::vector<std::vector<int>> generated_ids;
        std::fprintf(stdout, "Done!\n");
        std::fflush(stdout);

        ////////////////////////////////////////////////////////////////////
        // WARM-UP                                                        //
        ////////////////////////////////////////////////////////////////////

        if (run_warmup) {
            std::fprintf(stdout, "Warm-up...");
            std::fflush(stdout);
            for (std::size_t i = 0; i < 3; ++i) {
                if (run_decode) {
                    model_ref->generate_decode(ids, apss26::MAX_DECODE_TOKENS,
                                               logits, generated_ids);
                } else {
                    model_ref->generate(ids, logits);
                }
            }
            std::fprintf(stdout, "Done!\n");
            std::fflush(stdout);
        }

        ////////////////////////////////////////////////////////////////////
        // MODEL COMPUTATION                                              //
        ////////////////////////////////////////////////////////////////////

        std::fprintf(stdout, run_decode
                             ? "Generating sequences (prefill + decode)..."
                             : "Generating sequences (prefill only)...");
        std::fflush(stdout);

        // Synchronize all CUDA-visible devices immediately before and after the timed region
        synchronize_devices();
        const double run_start = get_time();
        if (run_decode) {
            model_ref->generate_decode(ids, apss26::MAX_DECODE_TOKENS,
                                       logits, generated_ids);
        } else {
            model_ref->generate(ids, logits);
        }
        synchronize_devices();
        const double run_end = get_time();

        std::fprintf(stdout, "Done!\n");
        std::fflush(stdout);
        const double elapsed = run_end - run_start;
        std::fprintf(stdout, "Elapsed time: %lf (sec)\n", elapsed);
        std::fprintf(stdout, "Throughput: %lf (sequences/sec)\n",
                     static_cast<double>(batch) / elapsed);

        ////////////////////////////////////////////////////////////////////
        // FINALIZATION                                                   //
        ////////////////////////////////////////////////////////////////////

        std::fprintf(stdout, "Finalizing...");
        std::fflush(stdout);
        model_ref.reset();
        std::fprintf(stdout, "Done!\n");
        std::fflush(stdout);

        std::fprintf(stdout, "Saving outputs to %s...", output_fname.c_str());
        std::fflush(stdout);
        if (run_decode) {
            write_decode_outputs(output_fname, logits, generated_ids);
        } else {
            write_outputs(output_fname, logits);
        }
        std::fprintf(stdout, "Done!\n");
        std::fflush(stdout);

        if (run_validation) {
            std::fprintf(stdout, "Validating...\n");
            if (run_decode) {
                const DecodeAnswers answers = read_decode_answers(
                    answer_fname, logits.size(0), logits.size(1), logits.size(2));
                if (!validate_decode(logits, generated_ids, answers)) return 1;
            } else {
                const Tensor answers = read_answers(
                    answer_fname, logits.size(0), logits.size(1));
                if (!validate(logits, answers)) return 1;
            }
        }

        std::fprintf(stdout, "Model load time: %lf (sec)\n",
                     load_end - load_start);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
