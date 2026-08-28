/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>

namespace faiss::cppcontrib::knowhere {

// Private Knowhere serialization tag. Upstream Faiss reserves "IHNr" for
// its incompatible direct-build/staged-search IndexHNSWRaBitQ format.
inline constexpr char kHnswRaBitQFourcc[] = "IHRK";

/** HNSW graph backed by a randomly-rotated standalone RaBitQ index.
 *
 * The storage layout is deliberately strict:
 *
 *   IndexPreTransform
 *     -> RandomRotationMatrix
 *     -> faiss::IndexRaBitQ
 *
 * RaBitQ does not implement code-to-code symmetric distances, so this index
 * is immutable after its graph and storage have been assembled. Build the
 * graph with exact storage first, then attach the trained/populated RaBitQ
 * storage to this runtime type.
 */
struct IndexHNSWRaBitQ : IndexHNSW {
    IndexHNSWRaBitQ();

    explicit IndexHNSWRaBitQ(faiss::IndexPreTransform* storage, int M = 32);

    void add(idx_t n, const float* x) override;

    faiss::IndexPreTransform* pretransform_index();
    const faiss::IndexPreTransform* pretransform_index() const;

    faiss::IndexRaBitQ* rabitq_index();
    const faiss::IndexRaBitQ* rabitq_index() const;

    /** Validate the complete runtime/storage shape and serialized invariants.
     * Throws FaissException on malformed state. */
    void validate_storage() const;
};

} // namespace faiss::cppcontrib::knowhere
