// Copyright (C) 2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/cppcontrib/knowhere/IndexCosine.h>

#include <memory>

#include "index/hnsw/impl/IndexHNSWWrapper.h"

namespace knowhere {

struct SearchParametersHNSWTurboQuant : SearchParametersHNSWWrapper {
    uint8_t query_bits = 0;
    bool int_qjl = false;

    faiss::DistanceComputer*
    storage_distance_computer(const faiss::Index* index) const override {
        auto dc = std::unique_ptr<faiss::DistanceComputer>(index->get_distance_computer());
        auto* inner = dc.get();
        if (auto* cosine = dynamic_cast<faiss::cppcontrib::knowhere::WithCosineNormDistanceComputer*>(inner)) {
            inner = cosine->basedis.get();
        }
        auto* tq = dynamic_cast<faiss::ScalarQuantizer::TurboQuantRefine::DistanceComputer*>(inner);
        FAISS_THROW_IF_NOT_MSG(tq, "Full TurboQuant search parameters require Full TurboQuant storage");
        tq->configure(query_bits, int_qjl);
        return dc.release();
    }
};

}  // namespace knowhere
