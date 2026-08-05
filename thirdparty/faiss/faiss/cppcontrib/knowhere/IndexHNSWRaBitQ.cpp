/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/knowhere/IndexHNSWRaBitQ.h>

#include <memory>

#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss::cppcontrib::knowhere {

namespace {

struct RaBitQPreTransformDistanceComputer : DistanceComputer {
    const faiss::IndexPreTransform* index;
    std::unique_ptr<DistanceComputer> sub_dc;
    std::unique_ptr<const float[]> transformed_query;

    RaBitQPreTransformDistanceComputer(
            const faiss::IndexPreTransform* index_in,
            const faiss::IndexRaBitQ* rabitq,
            uint8_t qb,
            bool centered)
            : index(index_in),
              sub_dc(rabitq->get_quantized_distance_computer(qb, centered)) {}

    void set_query(const float* x) override {
        const float* xt = index->apply_chain(1, x);
        if (xt == x) {
            transformed_query.reset();
            sub_dc->set_query(x);
        } else {
            transformed_query.reset(xt);
            sub_dc->set_query(xt);
        }
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return sub_dc->symmetric_dis(i, j);
    }

    float operator()(idx_t i) override {
        return (*sub_dc)(i);
    }
};

} // namespace

IndexHNSWRaBitQ::IndexHNSWRaBitQ() = default;

IndexHNSWRaBitQ::IndexHNSWRaBitQ(faiss::IndexPreTransform* storage_in, int M)
        : IndexHNSW(storage_in, M) {
    own_fields = true;
    ntotal = storage_in ? storage_in->ntotal : 0;
    is_trained = storage_in && storage_in->is_trained;
    validate_storage();
}

void IndexHNSWRaBitQ::add(idx_t, const float*) {
    FAISS_THROW_MSG(
            "IndexHNSWRaBitQ does not support incremental add: build the "
            "HNSW graph with exact storage before attaching RaBitQ storage");
}

faiss::IndexPreTransform* IndexHNSWRaBitQ::pretransform_index() {
    return dynamic_cast<faiss::IndexPreTransform*>(storage);
}

const faiss::IndexPreTransform* IndexHNSWRaBitQ::pretransform_index() const {
    return dynamic_cast<const faiss::IndexPreTransform*>(storage);
}

faiss::IndexRaBitQ* IndexHNSWRaBitQ::rabitq_index() {
    auto* pretransform = pretransform_index();
    return pretransform ? dynamic_cast<faiss::IndexRaBitQ*>(pretransform->index)
                        : nullptr;
}

const faiss::IndexRaBitQ* IndexHNSWRaBitQ::rabitq_index() const {
    const auto* pretransform = pretransform_index();
    return pretransform
            ? dynamic_cast<const faiss::IndexRaBitQ*>(pretransform->index)
            : nullptr;
}

void IndexHNSWRaBitQ::validate_storage() const {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT,
            "IndexHNSWRaBitQ only supports L2 and inner product metrics");
    FAISS_THROW_IF_NOT_MSG(
            storage != nullptr, "IndexHNSWRaBitQ requires non-null storage");

    const auto* pretransform = pretransform_index();
    FAISS_THROW_IF_NOT_MSG(
            pretransform != nullptr,
            "IndexHNSWRaBitQ storage must be IndexPreTransform");
    FAISS_THROW_IF_NOT_MSG(
            pretransform->chain.size() == 1,
            "IndexHNSWRaBitQ storage must contain exactly one transform");

    const auto* rotation = dynamic_cast<const faiss::RandomRotationMatrix*>(
            pretransform->chain[0]);
    FAISS_THROW_IF_NOT_MSG(
            rotation != nullptr,
            "IndexHNSWRaBitQ transform must be RandomRotationMatrix");

    const auto* rabitq = rabitq_index();
    FAISS_THROW_IF_NOT_MSG(
            rabitq != nullptr,
            "IndexHNSWRaBitQ pretransform leaf must be IndexRaBitQ");

    FAISS_THROW_IF_NOT_MSG(
            d == pretransform->d && metric_type == pretransform->metric_type &&
                    ntotal == pretransform->ntotal,
            "IndexHNSWRaBitQ outer index and pretransform metadata mismatch");
    FAISS_THROW_IF_NOT_MSG(
            is_trained && pretransform->is_trained && rotation->is_trained &&
                    rabitq->is_trained,
            "IndexHNSWRaBitQ requires fully trained storage");
    FAISS_THROW_IF_NOT_MSG(
            pretransform->index != nullptr &&
                    pretransform->ntotal == rabitq->ntotal &&
                    pretransform->metric_type == rabitq->metric_type,
            "IndexHNSWRaBitQ pretransform and RaBitQ metadata mismatch");
    FAISS_THROW_IF_NOT_MSG(
            rotation->d_in == d && rotation->d_out == rabitq->d &&
                    rotation->d_in == rotation->d_out,
            "IndexHNSWRaBitQ requires a square rotation matching index dimensions");
    FAISS_THROW_IF_NOT_MSG(
            rotation->is_orthonormal && !rotation->have_bias &&
                    rotation->b.empty() &&
                    rotation->A.size() ==
                            static_cast<size_t>(rotation->d_in) *
                                    rotation->d_out,
            "IndexHNSWRaBitQ rotation matrix has invalid storage");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->rabitq.d == static_cast<size_t>(rabitq->d) &&
                    rabitq->rabitq.metric_type == rabitq->metric_type,
            "IndexHNSWRaBitQ RaBitQ quantizer metadata mismatch");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->rabitq.nb_bits >= 1 && rabitq->rabitq.nb_bits <= 9,
            "IndexHNSWRaBitQ RaBitQ nb_bits must be in [1, 9]");

    const size_t expected_code_size =
            rabitq->rabitq.compute_code_size(rabitq->d, rabitq->rabitq.nb_bits);
    FAISS_THROW_IF_NOT_MSG(
            rabitq->rabitq.code_size == expected_code_size &&
                    rabitq->code_size == expected_code_size,
            "IndexHNSWRaBitQ RaBitQ code size mismatch");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->codes.size() ==
                    static_cast<size_t>(rabitq->ntotal) * expected_code_size,
            "IndexHNSWRaBitQ RaBitQ codes size mismatch");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->center.size() == static_cast<size_t>(rabitq->d),
            "IndexHNSWRaBitQ RaBitQ center size mismatch");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->qb <= 8, "IndexHNSWRaBitQ RaBitQ qb must be in [0, 8]");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->rabitq.nb_bits == 1 || rabitq->qb == 0,
            "IndexHNSWRaBitQ requires qb=0 when nb_bits > 1");
    FAISS_THROW_IF_NOT_MSG(
            !rabitq->centered, "IndexHNSWRaBitQ V1 requires centered=false");
}

DistanceComputer* IndexHNSWRaBitQ::get_distance_computer() const {
    validate_storage();
    const auto* rabitq = rabitq_index();
    return get_distance_computer(rabitq->qb, rabitq->centered);
}

DistanceComputer* IndexHNSWRaBitQ::get_distance_computer(
        uint8_t qb,
        bool centered) const {
    validate_storage();
    FAISS_THROW_IF_NOT_FMT(
            qb <= 8, "invalid HNSW RaBitQ qb=%d (must be in [0, 8])", qb);
    FAISS_THROW_IF_NOT_MSG(
            !centered, "IndexHNSWRaBitQ V1 requires centered=false");

    const auto* rabitq = rabitq_index();
    FAISS_THROW_IF_NOT_MSG(
            rabitq->rabitq.nb_bits == 1 || qb == 0,
            "HNSW RaBitQ query quantization is only supported for 1-bit "
            "database codes; use qb=0 when nb_bits > 1");

    return new RaBitQPreTransformDistanceComputer(
            pretransform_index(), rabitq, qb, centered);
}

} // namespace faiss::cppcontrib::knowhere
