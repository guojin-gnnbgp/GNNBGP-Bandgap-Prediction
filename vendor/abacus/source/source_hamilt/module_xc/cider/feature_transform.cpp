#include "feature_transform.h"

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

std::size_t checked_size(std::size_t first, std::size_t second, std::size_t third)
{
    if (first == 0 || second == 0 || third == 0)
    {
        throw std::invalid_argument("native CIDER transform dimensions must be nonzero");
    }
    if (first > std::numeric_limits<std::size_t>::max() / second
        || first * second > std::numeric_limits<std::size_t>::max() / third)
    {
        throw std::overflow_error("native CIDER transform dimensions overflow");
    }
    return first * second * third;
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

void validate_maps(const std::vector<FeatureMap>& maps, std::size_t raw_feature_count)
{
    if (maps.empty())
    {
        throw std::invalid_argument("native CIDER transform map list must be nonempty");
    }
    for (std::size_t i = 0; i < maps.size(); ++i)
    {
        const FeatureMap& map = maps[i];
        if (map.raw_feature_index() >= raw_feature_count)
        {
            throw std::invalid_argument("native CIDER transform feature index is out of bounds");
        }
        if (!std::isfinite(map.gamma()) || !(map.gamma() > 0.0)
            || !std::isfinite(map.scale()) || !(map.scale() > 0.0)
            || !std::isfinite(map.center()) || !std::isfinite(map.lower_bound())
            || !std::isfinite(map.upper_bound())
            || !(map.lower_bound() < map.upper_bound()))
        {
            throw std::invalid_argument("native CIDER transform parameters are invalid");
        }
        switch (map.kind())
        {
            case FeatureMapKind::U:
            case FeatureMapKind::V:
                break;
            case FeatureMapKind::Z:
                if (map.exponent() != 1)
                {
                    throw std::invalid_argument("native CIDER Z transform exponent must equal one");
                }
                break;
            default:
                throw std::invalid_argument("native CIDER transform map kind is unsupported");
        }
    }
}

double map_value(const FeatureMap& map, double raw)
{
    double value = 0.0;
    switch (map.kind())
    {
        case FeatureMapKind::U:
            value = map.gamma() * raw / (1.0 + map.gamma() * raw);
            break;
        case FeatureMapKind::V:
            value = -map.center()
                    + map.scale() * map.gamma() * raw / (1.0 + map.gamma() * raw);
            break;
        case FeatureMapKind::Z:
            value = -map.center()
                    + map.scale() / (1.0 + map.gamma() * raw * raw);
            break;
        default:
            throw std::invalid_argument("native CIDER transform map kind is unsupported");
    }
    if (!std::isfinite(value))
    {
        throw std::domain_error("native CIDER transformed feature is not finite");
    }
    return value;
}

double map_derivative(const FeatureMap& map, double raw)
{
    double derivative = 0.0;
    switch (map.kind())
    {
        case FeatureMapKind::U:
        {
            const double denominator = 1.0 + map.gamma() * raw;
            derivative = map.gamma() / (denominator * denominator);
            break;
        }
        case FeatureMapKind::V:
        {
            const double denominator = 1.0 + map.gamma() * raw;
            derivative = map.scale() * map.gamma() / (denominator * denominator);
            break;
        }
        case FeatureMapKind::Z:
        {
            const double denominator = 1.0 + map.gamma() * raw * raw;
            derivative = -2.0 * map.scale() * map.gamma() * raw
                         / (denominator * denominator);
            break;
        }
        default:
            throw std::invalid_argument("native CIDER transform map kind is unsupported");
    }
    if (!std::isfinite(derivative))
    {
        throw std::domain_error("native CIDER transform derivative is not finite");
    }
    return derivative;
}

} // namespace

void transform_features(const std::vector<FeatureMap>& maps,
                        std::size_t nspin,
                        std::size_t raw_feature_count,
                        std::size_t ngrids,
                        const double* raw_features,
                        std::size_t raw_size,
                        double* transformed_features,
                        std::size_t transformed_size)
{
    if (nspin != 1 && nspin != 2)
    {
        throw std::invalid_argument("native CIDER transform spin count must be one or two");
    }
    validate_maps(maps, raw_feature_count);
    const std::size_t expected_raw = checked_size(nspin, raw_feature_count, ngrids);
    const std::size_t expected_transformed = checked_size(nspin, maps.size(), ngrids);
    if (raw_features == nullptr || transformed_features == nullptr)
    {
        throw std::invalid_argument("native CIDER transform buffer is null");
    }
    if (raw_size != expected_raw || transformed_size != expected_transformed)
    {
        throw std::invalid_argument("native CIDER transform buffer size does not match dimensions");
    }
    validate_finite(raw_features, raw_size, "raw feature buffer");
    std::vector<double> computed(transformed_size);
    for (std::size_t spin = 0; spin < nspin; ++spin)
    {
        for (std::size_t transformed = 0; transformed < maps.size(); ++transformed)
        {
            const FeatureMap& map = maps[transformed];
            const std::size_t raw_offset
                = (spin * raw_feature_count + map.raw_feature_index()) * ngrids;
            const std::size_t transformed_offset
                = (spin * maps.size() + transformed) * ngrids;
            for (std::size_t grid = 0; grid < ngrids; ++grid)
            {
                computed[transformed_offset + grid]
                    = map_value(map, raw_features[raw_offset + grid]);
            }
        }
    }
    std::copy(computed.begin(), computed.end(), transformed_features);
}

void accumulate_transform_reverse(const std::vector<FeatureMap>& maps,
                                  std::size_t nspin,
                                  std::size_t raw_feature_count,
                                  std::size_t ngrids,
                                  const double* raw_features,
                                  std::size_t raw_size,
                                  const double* transformed_adjoint,
                                  std::size_t transformed_adjoint_size,
                                  double* raw_adjoint,
                                  std::size_t raw_adjoint_size)
{
    if (nspin != 1 && nspin != 2)
    {
        throw std::invalid_argument("native CIDER transform spin count must be one or two");
    }
    validate_maps(maps, raw_feature_count);
    const std::size_t expected_raw = checked_size(nspin, raw_feature_count, ngrids);
    const std::size_t expected_transformed = checked_size(nspin, maps.size(), ngrids);
    if (raw_features == nullptr || transformed_adjoint == nullptr || raw_adjoint == nullptr)
    {
        throw std::invalid_argument("native CIDER transform reverse buffer is null");
    }
    if (raw_size != expected_raw || raw_adjoint_size != expected_raw
        || transformed_adjoint_size != expected_transformed)
    {
        throw std::invalid_argument("native CIDER transform reverse buffer size does not match dimensions");
    }
    validate_finite(raw_features, raw_size, "raw feature buffer");
    validate_finite(transformed_adjoint,
                    transformed_adjoint_size,
                    "transformed-adjoint buffer");
    validate_finite(raw_adjoint, raw_adjoint_size, "raw-adjoint buffer");
    std::vector<double> computed(raw_adjoint, raw_adjoint + raw_adjoint_size);
    for (std::size_t spin = 0; spin < nspin; ++spin)
    {
        for (std::size_t transformed = 0; transformed < maps.size(); ++transformed)
        {
            const FeatureMap& map = maps[transformed];
            const std::size_t raw_offset
                = (spin * raw_feature_count + map.raw_feature_index()) * ngrids;
            const std::size_t transformed_offset
                = (spin * maps.size() + transformed) * ngrids;
            for (std::size_t grid = 0; grid < ngrids; ++grid)
            {
                const std::size_t raw_index = raw_offset + grid;
                const double contribution
                    = transformed_adjoint[transformed_offset + grid]
                      * map_derivative(map, raw_features[raw_index]);
                const double value = computed[raw_index] + contribution;
                if (!std::isfinite(value))
                {
                    throw std::overflow_error("native CIDER raw-feature adjoint is not finite");
                }
                computed[raw_index] = value;
            }
        }
    }
    std::copy(computed.begin(), computed.end(), raw_adjoint);
}

} // namespace cider
} // namespace hamilt
