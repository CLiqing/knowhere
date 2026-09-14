// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <faiss/IndexFlat.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <memory>
#include <shared_mutex>

#include "index/turboquant/turboquant_utils.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/context.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/thread_pool.h"

namespace knowhere {

template <bool Mse>
class IvfTurboQuantConfig : public BaseConfig {
 public:
    CFG_INT nlist;
    CFG_INT nprobe;
    CFG_INT tq_bits;
    CFG_INT tq_query_bits;
    CFG_BOOL tq_int_qjl;
    CFG_BOOL by_residual;
    CFG_BOOL refine;

    KNOWHERE_DECLARE_CONFIG(IvfTurboQuantConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(nlist).set_default(128).set_range(1, 65536).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(nprobe).set_default(8).set_range(1, 65536).for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(tq_bits).set_default(4).set_range(1, 8).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(tq_query_bits).set_default(0).set_range(0, 8).for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(tq_int_qjl).set_default(false).for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(by_residual).set_default(false).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine).set_default(false).for_train();
    }

    Status
    CheckAndAdjust(PARAM_TYPE type, std::string* msg) override {
        if (type == PARAM_TYPE::TRAIN) {
            const auto metric = metric_type.value();
            if (metric != "IP" && metric != "COSINE" && (Mse || metric != "L2")) {
                return HandleError(msg, "unsupported TurboQuant metric", Status::invalid_metric_type);
            }
            if (!turboquant::ValidBits(Mse, tq_bits.value())) {
                return HandleError(msg, "invalid TurboQuant variant/bit width", Status::invalid_args);
            }
            if (by_residual.value() || refine.value()) {
                return HandleError(msg, "IVF TurboQuant residual/refine is not supported", Status::not_implemented);
            }
        }
        if (type == PARAM_TYPE::SEARCH) {
            if ((tq_query_bits.value() == 0 && tq_int_qjl.value()) ||
                (Mse && (tq_query_bits.value() != 0 || tq_int_qjl.value()))) {
                return HandleError(msg, "invalid TurboQuant query parameters", Status::invalid_args);
            }
        }
        return Status::success;
    }
};

// Compose standard Faiss objects. No TQ-specific branch is needed in the
// common IVF search or IO code, and the transform chain travels with the index.
template <typename DataType, bool Mse>
class IvfTurboQuantNode : public IndexNode {
 public:
    IvfTurboQuantNode(int32_t version, const Object&) : IndexNode(version) {
        static_assert(std::is_same_v<DataType, fp32>);
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<IvfTurboQuantConfig<Mse>>();
    }
    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }
    static bool
    StaticHasRawData(const BaseConfig&, const IndexVersion&) {
        return false;
    }
    bool
    HasRawData(const std::string&) const override {
        return false;
    }
    static Status
    StaticConfigCheck(const Config&, PARAM_TYPE, std::string&) {
        return Status::success;
    }

    Status
    Train(const DataSetPtr data, std::shared_ptr<Config> config, bool use_pool) override {
        if (use_pool) {
            return ThreadPool::GetGlobalBuildThreadPool()->push([&] { return Train(data, config, false); }).get();
        }
        std::unique_lock lock(mutex_);
        const auto& cfg = static_cast<const IvfTurboQuantConfig<Mse>&>(*config);
        if (!data || !data->GetTensor() || data->GetDim() <= 0 || data->GetRows() < cfg.nlist.value()) {
            return Status::invalid_args;
        }
        try {
            const bool cosine = cfg.metric_type.value() == "COSINE";
            const auto metric = cfg.metric_type.value() == "L2" ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
            auto coarse = std::make_unique<faiss::IndexFlat>(data->GetDim(), metric);
            auto ivf = std::make_unique<faiss::IndexIVFScalarQuantizer>(
                coarse.get(), data->GetDim(), cfg.nlist.value(), turboquant::QuantizerType(Mse, cfg.tq_bits.value()),
                metric, false);
            ivf->own_fields = true;
            coarse.release();
            auto rr = std::make_unique<faiss::RandomRotationMatrix>(data->GetDim(), data->GetDim());
            rr->init(12345);
            auto chain = std::make_unique<faiss::IndexPreTransform>(rr.get(), ivf.get());
            chain->own_fields = true;
            rr.release();
            ivf.release();
            if (cosine) {
                auto norm = std::make_unique<faiss::NormalizationTransform>(data->GetDim());
                chain->prepend_transform(norm.get());
                norm.release();
            }
            ThreadPool::ScopedBuildOmpSetter omp(
                cfg.num_build_thread.value_or(ThreadPool::GetGlobalBuildThreadPool()->size()));
            chain->train(data->GetRows(), static_cast<const float*>(data->GetTensor()));
            index_ = std::move(chain);
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << e.what();
            return Status::faiss_inner_error;
        }
    }

    Status
    Add(const DataSetPtr data, std::shared_ptr<Config> cfg, bool use_pool) override {
        if (use_pool) {
            return ThreadPool::GetGlobalBuildThreadPool()->push([&] { return Add(data, cfg, false); }).get();
        }
        std::unique_lock lock(mutex_);
        if (!index_)
            return Status::empty_index;
        if (!index_->is_trained)
            return Status::index_not_trained;
        if (!data || !data->GetTensor() || data->GetDim() != index_->d || data->GetRows() < 0) {
            return Status::invalid_args;
        }
        try {
            ThreadPool::ScopedBuildOmpSetter omp(static_cast<const BaseConfig&>(*cfg).num_build_thread.value_or(
                ThreadPool::GetGlobalBuildThreadPool()->size()));
            index_->add(data->GetRows(), static_cast<const float*>(data->GetTensor()));
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << e.what();
            return Status::faiss_inner_error;
        }
    }

    expected<DataSetPtr>
    Search(const DataSetPtr data, std::unique_ptr<Config> config, const BitsetView& bitset,
           milvus::OpContext* context) const override {
        std::shared_lock lock(mutex_);
        if (!index_)
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        if (!data || !data->GetTensor() || data->GetDim() != index_->d || data->GetRows() < 0) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "query shape mismatch");
        }
        const auto& cfg = static_cast<const IvfTurboQuantConfig<Mse>&>(*config);
        const auto nq = data->GetRows(), k = cfg.k.value();
        auto ids = std::make_unique<int64_t[]>(nq * k);
        auto distances = std::make_unique<float[]>(nq * k);
        BitsetViewIDSelector selector(bitset);
        faiss::IVFSQTurboQSearchParameters params;
        params.nprobe = cfg.nprobe.value();
        params.qb = cfg.tq_query_bits.value();
        params.int_qjl = cfg.tq_int_qjl.value();
        params.sel = bitset.empty() ? nullptr : &selector;
        try {
            std::vector<folly::Future<folly::Unit>> futures;
            auto pool = ThreadPool::GetGlobalSearchThreadPool();
            for (int64_t q = 0; q < nq; ++q) {
                futures.emplace_back(pool->push([&, q] {
                    checkCancellation(context);
                    ThreadPool::ScopedSearchOmpSetter omp(1);
                    index_->search(1, static_cast<const float*>(data->GetTensor()) + q * index_->d, k,
                                   distances.get() + q * k, ids.get() + q * k, &params);
                }));
            }
            WaitAllSuccess(futures);
            auto result = GenResultDataSet(nq, k, std::move(ids), std::move(distances));
            MapSearchResultIdsToOutIds(result);
            return result;
        } catch (const std::exception& e) {
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr, milvus::OpContext*) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "TurboQuant does not store original vectors");
    }
    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config>) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "IVF TurboQuant metadata is not implemented");
    }
    Status
    Serialize(BinarySet& binary) const override {
        std::shared_lock lock(mutex_);
        if (!index_)
            return Status::empty_index;
        try {
            faiss::VectorIOWriter writer;
            faiss::write_index(index_.get(), &writer);
            auto bytes = std::shared_ptr<uint8_t[]>(new uint8_t[writer.data.size()]);
            std::copy(writer.data.begin(), writer.data.end(), bytes.get());
            binary.Append(Type(), std::move(bytes), writer.data.size());
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << e.what();
            return Status::faiss_inner_error;
        }
    }

    Status
    Deserialize(const BinarySet& binary, std::shared_ptr<Config>) override {
        std::unique_lock lock(mutex_);
        auto data = binary.GetByName(Type());
        if (!data)
            return Status::invalid_binary_set;
        try {
            MemoryIOReader reader(data->data.get(), data->size);
            return Install(std::unique_ptr<faiss::Index>(faiss::read_index(&reader)));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << e.what();
            return Status::faiss_inner_error;
        }
    }
    Status
    DeserializeFromFile(const std::string& file, std::shared_ptr<Config> cfg) override {
        if (static_cast<const BaseConfig&>(*cfg).enable_mmap.value_or(false))
            return Status::not_implemented;
        std::unique_lock lock(mutex_);
        try {
            return Install(std::unique_ptr<faiss::Index>(faiss::read_index(file.c_str())));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << e.what();
            return Status::faiss_inner_error;
        }
    }
    int64_t
    Dim() const override {
        std::shared_lock lock(mutex_);
        return index_ ? index_->d : 0;
    }
    int64_t
    Count() const override {
        std::shared_lock lock(mutex_);
        return index_ ? index_->ntotal : 0;
    }
    int64_t
    Size() const override {
        std::shared_lock lock(mutex_);
        if (!index_)
            return 0;
        auto* ivf = static_cast<const faiss::IndexIVFScalarQuantizer*>(index_->index);
        return ivf->ntotal * (ivf->code_size + sizeof(faiss::idx_t)) +
               (int64_t(index_->d) * index_->d + ivf->nlist * index_->d + ivf->sq.trained.size()) * sizeof(float);
    }
    std::string
    Type() const override {
        return Mse ? "IVF_TQMSE" : "IVF_TURBOQUANT";
    }

 private:
    Status
    Install(std::unique_ptr<faiss::Index> index) {
        auto* chain = dynamic_cast<faiss::IndexPreTransform*>(index.get());
        if (!chain || !chain->is_trained || chain->chain.empty() || chain->chain.size() > 2) {
            return Status::invalid_serialized_index_type;
        }
        const auto* ivf = dynamic_cast<faiss::IndexIVFScalarQuantizer*>(chain->index);
        const auto* rr = dynamic_cast<faiss::RandomRotationMatrix*>(chain->chain.back());
        bool type_ok = false;
        if (ivf)
            for (int bits = 1; bits <= 8; ++bits) {
                if (turboquant::ValidBits(Mse, bits) && ivf->sq.qtype == turboquant::QuantizerType(Mse, bits))
                    type_ok = true;
            }
        if (!type_ok || !rr || rr->have_bias || rr->d_in != chain->d || rr->d_out != chain->d ||
            rr->A.size() != size_t(chain->d) * chain->d || ivf->by_residual || ivf->d != chain->d ||
            ivf->ntotal != chain->ntotal || ivf->metric_type != chain->metric_type ||
            (Mse && ivf->metric_type != faiss::METRIC_INNER_PRODUCT))
            return Status::invalid_serialized_index_type;
        if (chain->chain.size() == 2) {
            const auto* norm = dynamic_cast<faiss::NormalizationTransform*>(chain->chain.front());
            if (!norm || norm->norm != 2 || norm->d_in != chain->d || norm->d_out != chain->d ||
                ivf->metric_type != faiss::METRIC_INNER_PRODUCT)
                return Status::invalid_serialized_index_type;
        }
        index.release();
        index_.reset(chain);
        return Status::success;
    }
    mutable std::shared_mutex mutex_;
    std::unique_ptr<faiss::IndexPreTransform> index_;
};

template <typename T>
using IvfFullTurboQuantNode = IvfTurboQuantNode<T, false>;
template <typename T>
using IvfTurboQuantMseNode = IvfTurboQuantNode<T, true>;
KNOWHERE_MOCK_REGISTER_DENSE_FLOAT_ALL_GLOBAL(IVF_TURBOQUANT, IvfFullTurboQuantNode, knowhere::feature::NONE)
KNOWHERE_MOCK_REGISTER_DENSE_FLOAT_ALL_GLOBAL(IVF_TQMSE, IvfTurboQuantMseNode, knowhere::feature::NONE)
}  // namespace knowhere
