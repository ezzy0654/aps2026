#pragma once

#include "tensor.h"
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>

class ModelLoader {
public:
    explicit ModelLoader(const std::string& path);
    Tensor load(const std::string& name) const;
    bool has(const std::string& name) const { return entries_.count(name) != 0; }

private:
    struct Entry {
        std::vector<std::size_t> shape;
        std::uint64_t offset = 0;
        std::uint64_t bytes = 0;
    };
    std::string path_;
    std::unordered_map<std::string, Entry> entries_;
};
