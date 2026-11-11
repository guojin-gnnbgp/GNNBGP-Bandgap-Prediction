#ifndef MODULE_HAMILT_CIDER_FEATURE_TRANSFORM_H
#define MODULE_HAMILT_CIDER_FEATURE_TRANSFORM_H

#include "model.h"

#include <cstddef>
#include <vector>

namespace hamilt
{
namespace cider
{

// Raw and transformed buffers use contiguous spin-major
// (nspin, nfeature, ngrids) order.
// Every read-only input range must be disjoint from the writable output range;
// read-only input ranges may overlap each other.
// On any exception, the output range remains byte-for-byte unchanged.
void transform_features(const std::vector<FeatureMap>& maps,
                        std::size_t nspin,
                        std::size_t raw_feature_count,
                        std::size_t ngrids,
                        const double* raw_features,
                        std::size_t raw_size,
                        double* transformed_features,
                        std::size_t transformed_size);

void accumulate_transform_reverse(const std::vector<FeatureMap>& maps,
                                  std::size_t nspin,
                                  std::size_t raw_feature_count,
                                  std::size_t ngrids,
                                  const double* raw_features,
                                  std::size_t raw_size,
                                  const double* transformed_adjoint,
                                  std::size_t transformed_adjoint_size,
                                  double* raw_adjoint,
                                  std::size_t raw_adjoint_size);

} // namespace cider
} // namespace hamilt

#endif // MODULE_HAMILT_CIDER_FEATURE_TRANSFORM_H
