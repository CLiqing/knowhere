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

    /** Native (non-negated for inner product) distance computer using the
     * leaf index's default query quantization settings. */
    DistanceComputer* get_distance_computer() const override;

    /** Native request-local distance computer. This does not mutate the
     * shared IndexRaBitQ::qb or IndexRaBitQ::centered fields. */
    DistanceComputer* get_distance_computer(uint8_t qb, bool centered = false)
            const;

    faiss::IndexPreTransform* pretransform_index();
    const faiss::IndexPreTransform* pretransform_index() const;

    faiss::IndexRaBitQ* rabitq_index();
    const faiss::IndexRaBitQ* rabitq_index() const;

    /** Validate the complete runtime/storage shape and serialized invariants.
     * Throws FaissException on malformed state. */
    void validate_storage() const;
};

} // namespace faiss::cppcontrib::knowhere
