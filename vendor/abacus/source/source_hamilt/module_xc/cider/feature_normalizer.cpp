#include "feature_normalizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace hamilt
{
namespace cider
{
namespace
{

std::size_t checked_size(std::size_t nspin, std::size_t nfeature, std::size_t ngrids)
{
    if (nspin == 0 || nfeature == 0 || ngrids == 0)
    {
        throw std::invalid_argument("native CIDER feature dimensions must be nonzero");
    }
    if (nspin > std::numeric_limits<std::size_t>::max() / nfeature
        || nspin * nfeature > std::numeric_limits<std::size_t>::max() / ngrids)
    {
        throw std::overflow_error("native CIDER feature dimensions overflow");
    }
    return nspin * nfeature * ngrids;
}

void validate_contract(const std::vector<Normalizer>& normalizers,
                       SemilocalMode semilocal_mode,
                       std::size_t nspin,
                       std::size_t nfeature,
                       std::size_t ngrids,
                       const double* first,
                       std::size_t first_size,
                       const double* second,
                       std::size_t second_size)
{
    if (nspin != 1 && nspin != 2)
    {
        throw std::invalid_argument("native CIDER feature spin count must be one or two");
    }
    const std::size_t total = checked_size(nspin, nfeature, ngrids);
    std::size_t semilocal_count = 0;
    switch (semilocal_mode)
    {
        case SemilocalMode::Np:
            semilocal_count = 2;
            break;
        case SemilocalMode::Npa:
            semilocal_count = 3;
            break;
        default:
            throw std::invalid_argument("native CIDER semilocal mode is unsupported");
    }
    if (nfeature < semilocal_count)
    {
        throw std::invalid_argument("native CIDER feature count is incompatible with semilocal mode");
    }
    if (normalizers.size() != nfeature)
    {
        throw std::invalid_argument("native CIDER normalizer count does not match feature count");
    }
    if (first_size != total || second_size != total)
    {
        throw std::invalid_argument("native CIDER feature buffer size does not match dimensions");
    }
    if (first == nullptr || second == nullptr)
    {
        throw std::invalid_argument("native CIDER feature buffer is null");
    }
    for (std::size_t feature = 0; feature < normalizers.size(); ++feature)
    {
        if (!std::isfinite(normalizers[feature].constant()))
        {
            throw std::invalid_argument("native CIDER normalizer constant is not finite");
        }
        switch (normalizers[feature].kind())
        {
            case NormalizerKind::Identity:
            case NormalizerKind::Constant:
                break;
            default:
                throw std::invalid_argument("native CIDER normalizer kind is unsupported");
        }
    }
}

double factor(const Normalizer& normalizer)
{
    switch (normalizer.kind())
    {
        case NormalizerKind::Identity:
            return 1.0;
        case NormalizerKind::Constant:
            return normalizer.constant();
    }
    throw std::invalid_argument("native CIDER normalizer kind is unsupported");
}

void validate_finite(const double* values, std::size_t size, const char* label)
{
    for (std::size_t i = 0; i < size; ++i)
    {
        if (!std::isfinite(values[i]))
        {
            throw std::invalid_argument(std::string("native CIDER ") + label
                                        + " contains a nonfinite value");
        }
    }
}

} // namespace

void normalize_features(const std::vector<Normalizer>& normalizers,
                        SemilocalMode semilocal_mode,
                        std::size_t nspin,
                        std::size_t nfeature,
                        std::size_t ngrids,
                        const double* raw_features,
                        std::size_t raw_size,
                        double* normalized_features,
                        std::size_t normalized_size)
{
    validate_contract(normalizers,
                      semilocal_mode,
                      nspin,
                      nfeature,
                      ngrids,
                      raw_features,
                      raw_size,
                      normalized_features,
                      normalized_size);
    validate_finite(raw_features, raw_size, "raw feature buffer");
    std::vector<double> computed(normalized_size);
    for (std::size_t spin = 0; spin < nspin; ++spin)
    {
        for (std::size_t feature = 0; feature < nfeature; ++feature)
        {
            const double normalizer_factor = factor(normalizers[feature]);
            const std::size_t offset = (spin * nfeature + feature) * ngrids;
            for (std::size_t grid = 0; grid < ngrids; ++grid)
            {
                const double value = raw_features[offset + grid] * normalizer_factor;
                if (!std::isfinite(value))
                {
                    throw std::overflow_error("native CIDER normalized feature is not finite");
                }
                computed[offset + grid] = value;
            }
        }
    }
    std::copy(computed.begin(), computed.end(), normalized_features);
}

void accumulate_normalization_reverse(const std::vector<Normalizer>& normalizers,
                                      SemilocalMode semilocal_mode,
                                      std::size_t nspin,
                                      std::size_t nfeature,
                                      std::size_t ngrids,
                                      const double* raw_features,
                                      std::size_t raw_size,
                                      const double* normalized_adjoint,
                                      std::size_t normalized_adjoint_size,
                                      double* raw_adjoint,
                                      std::size_t raw_adjoint_size)
{
    validate_contract(normalizers,
                      semilocal_mode,
                      nspin,
                      nfeature,
                      ngrids,
                      raw_features,
                      raw_size,
                      raw_adjoint,
                      raw_adjoint_size);
    const std::size_t total = checked_size(nspin, nfeature, ngrids);
    if (normalized_adjoint_size != total || normalized_adjoint == nullptr)
    {
        throw std::invalid_argument("native CIDER normalized-adjoint buffer size does not match dimensions");
    }
    validate_finite(raw_features, raw_size, "raw feature buffer");
    validate_finite(normalized_adjoint,
                    normalized_adjoint_size,
                    "normalized-adjoint buffer");
    validate_finite(raw_adjoint, raw_adjoint_size, "raw-adjoint buffer");
    std::vector<double> computed(raw_adjoint_size);
    for (std::size_t spin = 0; spin < nspin; ++spin)
    {
        for (std::size_t feature = 0; feature < nfeature; ++feature)
        {
            const double normalizer_factor = factor(normalizers[feature]);
            const std::size_t offset = (spin * nfeature + feature) * ngrids;
            for (std::size_t grid = 0; grid < ngrids; ++grid)
            {
                const std::size_t index = offset + grid;
                const double value
                    = raw_adjoint[index] + normalized_adjoint[index] * normalizer_factor;
                if (!std::isfinite(value))
                {
                    throw std::overflow_error("native CIDER raw-feature adjoint is not finite");
                }
                computed[index] = value;
            }
        }
    }
    std::copy(computed.begin(), computed.end(), raw_adjoint);
}

} // namespace cider
} // namespace hamilt
