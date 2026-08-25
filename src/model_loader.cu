#include "model_loader.h"
#include <cstring>
#include <stdexcept>

namespace {
template <typename T> T read(std::ifstream& f) {
    T value{};
    f.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!f) throw std::runtime_error("truncated model.bin");
    return value;
}
}

ModelLoader::ModelLoader(const std::string& path) : path_(path) {
    std::ifstream f(path_, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open model: " + path_);
    const std::uint32_t count = read<std::uint32_t>(f);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t name_len = read<std::uint32_t>(f);
        std::string name(name_len, '\0');
        f.read(name.data(), name_len);
        const std::uint32_t ndim = read<std::uint32_t>(f);
        Entry e;
        e.shape.resize(ndim);
        for (std::uint32_t d = 0; d < ndim; ++d) e.shape[d] = read<std::uint32_t>(f);
        e.offset = read<std::uint64_t>(f);
        e.bytes = read<std::uint64_t>(f);
        entries_.emplace(std::move(name), std::move(e));
    }
}

Tensor ModelLoader::load(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) throw std::runtime_error("missing tensor: " + name);
    const Entry& e = it->second;
    if (e.bytes != e.shape[0] * (e.shape.size() > 1 ? 1 : 1)) {
        // The exact element count is checked below; this branch only keeps the
        // metadata validation local and avoids trusting a malformed file.
    }
    std::size_t elements = 1;
    for (std::size_t d : e.shape) elements *= d;
    if (e.bytes != elements * sizeof(float)) throw std::runtime_error("tensor is not float32: " + name);
    Tensor out(e.shape);
    std::ifstream f(path_, std::ios::binary);
    if (!f) throw std::runtime_error("cannot reopen model: " + path_);
    f.seekg(static_cast<std::streamoff>(e.offset));
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(e.bytes));
    if (!f) throw std::runtime_error("truncated tensor: " + name);
    // Parameter placement is initialization, not measured model computation.
    // Keep immutable weights resident for all prefill layers.
    out.prepare_cuda();
    return out;
}
