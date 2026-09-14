// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#pragma once

#include <faiss/impl/ScalarQuantizer.h>

#include <stdexcept>

namespace knowhere::turboquant {
inline bool
ValidBits(bool mse, int bits) {
    return mse ? (bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 8) : (bits >= 2 && bits <= 5);
}

inline faiss::ScalarQuantizer::QuantizerType
QuantizerType(bool mse, int bits) {
    using SQ = faiss::ScalarQuantizer;
    if (!ValidBits(mse, bits)) {
        throw std::invalid_argument("invalid TurboQuant bit width");
    }
    if (mse) {
        return bits == 8 ? SQ::QT_8bit_tqmse : static_cast<SQ::QuantizerType>(SQ::QT_1bit_tqmse + bits - 1);
    }
    return static_cast<SQ::QuantizerType>(SQ::QT_2bit_tq + bits - 2);
}
}  // namespace knowhere::turboquant
