#ifndef MODULE_HAMILT_CIDER_SPLINE_EVALUATOR_H
#define MODULE_HAMILT_CIDER_SPLINE_EVALUATOR_H

#include "model.h"

#include <cstddef>
#include <vector>

namespace hamilt
{
namespace cider
{

// Descriptors and descriptor derivatives use contiguous spin-major
// (nspin, nfeature, ngrids) order. Values use (nspin, ngrids) order.
// The descriptor input range and both writable output ranges must be mutually
// disjoint. On any exception, both output ranges remain byte-for-byte unchanged.
// Values are replaced by constant plus the scaled term sum; descriptor
// derivatives are replaced by the accumulated derivative of every term.
void evaluate_spline_set(double constant,
                         const std::vector<SplineTerm>& terms,
                         std::size_t nspin,
                         std::size_t nfeature,
                         std::size_t ngrids,
                         const double* descriptors,
                         std::size_t descriptor_size,
                         double* values,
                         std::size_t value_size,
                         double* descriptor_derivatives,
                         std::size_t derivative_size);

} // namespace cider
} // namespace hamilt

#endif // MODULE_HAMILT_CIDER_SPLINE_EVALUATOR_H
