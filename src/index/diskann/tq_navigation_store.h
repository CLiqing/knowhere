// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#pragma once
#include <memory>
#include <string>

#include "diskann/pq_flash_index.h"

namespace faiss {
struct Index;
}
namespace knowhere {
class TQNavigationStore {
 public:
    static std::string
    Filename(const std::string& prefix) {
        return prefix + "_tq_navigation.index";
    }
    static void
    Build(const std::string& source, const std::string& file, bool mse, int bits, const std::string& metric,
          double budget_gb);
    TQNavigationStore(const std::string& file, bool mse, int bits, const std::string& metric);
    ~TQNavigationStore();
    std::unique_ptr<diskann::NavigationDistanceComputer>
    CreateDistanceComputer(int qb = 0, bool int_qjl = false) const;
    int64_t
    Count() const;
    int64_t
    Dimension() const;
    size_t
    MemorySize() const;
    bool
    IsMse() const {
        return mse_;
    }

 private:
    std::unique_ptr<faiss::Index> index_;
    bool mse_;
};
}  // namespace knowhere
