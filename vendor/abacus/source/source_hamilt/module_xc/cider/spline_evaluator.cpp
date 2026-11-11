#include "spline_evaluator.h"

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

const double basis_matrix[4][4] = {
    {-1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0},
    {3.0 / 6.0, -6.0 / 6.0, 0.0, 4.0 / 6.0},
    {-3.0 / 6.0, 3.0 / 6.0, 3.0 / 6.0, 1.0 / 6.0},
    {1.0 / 6.0, 0.0, 0.0, 0.0}};

std::size_t checked_product(std::size_t first,
                            std::size_t second,
                            const char* label)
{
    if (first == 0 || second == 0)
    {
        throw std::invalid_argument(std::string("native CIDER ") + label
                                    + " dimensions must be nonzero");
    }
    if (first > std::numeric_limits<std::size_t>::max() / second)
    {
        throw std::overflow_error(std::string("native CIDER ") + label
                                  + " dimensions overflow");
    }
    return first * second;
}

void validate_finite(const double* data, std::size_t size, const char* label)
{
    for (std::size_t i = 0; i < size; ++i)
    {
        if (!std::isfinite(data[i]))
        {
            throw std::invalid_argument(std::string("native CIDER ") + label
                                        + " contains a nonfinite value");
        }
    }
}

void validate_terms(const std::vector<SplineTerm>& terms, std::size_t nfeature)
{
    if (terms.empty())
    {
        throw std::invalid_argument("native CIDER spline term list must be nonempty");
    }
    for (std::size_t term_index = 0; term_index < terms.size(); ++term_index)
    {
        const SplineTerm& term = terms[term_index];
        const std::size_t dimension = term.dimension();
        if (dimension < 1 || dimension > 3)
        {
            throw std::invalid_argument(
                "native CIDER spline dimension must be between one and three");
        }
        if (!std::isfinite(term.scale()))
        {
            throw std::invalid_argument("native CIDER spline scale is not finite");
        }
        if (term.grids().size() != dimension
            || term.coefficient_extents().size() != dimension)
        {
            throw std::invalid_argument("native CIDER spline term cardinality is inconsistent");
        }
        for (std::size_t d = 0; d < dimension; ++d)
        {
            const std::size_t descriptor = term.descriptor_indices()[d];
            if (descriptor >= nfeature)
            {
                throw std::invalid_argument(
                    "native CIDER spline descriptor index is out of bounds");
            }
            if (std::find(term.descriptor_indices().begin(),
                          term.descriptor_indices().begin() + d,
                          descriptor)
                != term.descriptor_indices().begin() + d)
            {
                throw std::invalid_argument(
                    "native CIDER spline descriptor indices must be distinct");
            }
            const SplineGrid& grid = term.grids()[d];
            if (!std::isfinite(grid.lower()) || !std::isfinite(grid.upper())
                || !(grid.lower() < grid.upper()) || grid.order() < 2
                || grid.order() > std::numeric_limits<std::size_t>::max() - 2)
            {
                throw std::invalid_argument("native CIDER spline grid is invalid");
            }
            if (term.coefficient_extents()[d] != grid.order() + 2)
            {
                throw std::invalid_argument(
                    "native CIDER spline coefficient extent differs from grid order plus two");
            }
        }
        std::size_t coefficient_count = term.coefficient_extents()[0];
        for (std::size_t d = 1; d < dimension; ++d)
        {
            coefficient_count = checked_product(coefficient_count,
                                                term.coefficient_extents()[d],
                                                "spline coefficient");
        }
        if (term.coefficients().size() != coefficient_count)
        {
            throw std::invalid_argument(
                "native CIDER spline coefficient count differs from extents");
        }
        validate_finite(term.coefficients().data(),
                        term.coefficients().size(),
                        "spline coefficient table");
    }
}

struct AxisBasis
{
    std::size_t index;
    double value[4];
    double derivative[4];
};

AxisBasis evaluate_axis(const SplineGrid& grid, double coordinate)
{
    const double inverse_spacing
        = (static_cast<double>(grid.order()) - 1.0) / (grid.upper() - grid.lower());
    const double u = (coordinate - grid.lower()) * inverse_spacing;
    if (!std::isfinite(inverse_spacing) || !std::isfinite(u))
    {
        throw std::overflow_error("native CIDER spline coordinate scaling is not finite");
    }
    AxisBasis result;
    if (u < 0.0)
    {
        result.index = 0;
    }
    else if (u >= static_cast<double>(grid.order() - 1))
    {
        result.index = grid.order() - 2;
    }
    else
    {
        result.index = static_cast<std::size_t>(std::floor(u));
    }
    const double t = u - static_cast<double>(result.index);
    const double t2 = t * t;
    const double t3 = t2 * t;
    for (std::size_t basis = 0; basis < 4; ++basis)
    {
        if (t < 0.0)
        {
            result.value[basis]
                = basis_matrix[basis][2] * t + basis_matrix[basis][3];
        }
        else if (t > 1.0)
        {
            const double at_one = basis_matrix[basis][0] + basis_matrix[basis][1]
                                  + basis_matrix[basis][2] + basis_matrix[basis][3];
            const double slope = 3.0 * basis_matrix[basis][0]
                                 + 2.0 * basis_matrix[basis][1]
                                 + basis_matrix[basis][2];
            result.value[basis] = slope * (t - 1.0) + at_one;
        }
        else
        {
            result.value[basis] = basis_matrix[basis][0] * t3
                                  + basis_matrix[basis][1] * t2
                                  + basis_matrix[basis][2] * t
                                  + basis_matrix[basis][3];
        }
        // Match interpolation.splines.eval_cubic_numba exactly: unlike the
        // continued value basis, its derivative path always evaluates the
        // derivative of the cubic polynomial at t, including outside [0, 1].
        result.derivative[basis]
            = (3.0 * basis_matrix[basis][0] * t2
               + 2.0 * basis_matrix[basis][1] * t
               + basis_matrix[basis][2])
              * inverse_spacing;
    }
    return result;
}

void evaluate_term(const SplineTerm& term,
                   const AxisBasis axes[3],
                   double& value,
                   double derivative[3])
{
    value = 0.0;
    derivative[0] = derivative[1] = derivative[2] = 0.0;
    const std::size_t dimension = term.dimension();
    std::size_t combinations = 1;
    for (std::size_t d = 0; d < dimension; ++d)
    {
        combinations *= 4;
    }
    for (std::size_t combination = 0; combination < combinations; ++combination)
    {
        std::size_t digits = combination;
        std::size_t basis_index[3] = {0, 0, 0};
        for (std::size_t reverse = dimension; reverse > 0; --reverse)
        {
            basis_index[reverse - 1] = digits % 4;
            digits /= 4;
        }
        std::size_t coefficient_index = 0;
        double weight = 1.0;
        for (std::size_t d = 0; d < dimension; ++d)
        {
            coefficient_index
                = coefficient_index * term.coefficient_extents()[d]
                  + axes[d].index + basis_index[d];
            weight *= axes[d].value[basis_index[d]];
        }
        const double coefficient = term.coefficients()[coefficient_index];
        value += weight * coefficient;
        for (std::size_t target = 0; target < dimension; ++target)
        {
            double derivative_weight = axes[target].derivative[basis_index[target]];
            for (std::size_t d = 0; d < dimension; ++d)
            {
                if (d != target)
                {
                    derivative_weight *= axes[d].value[basis_index[d]];
                }
            }
            derivative[target] += derivative_weight * coefficient;
        }
    }
}

} // namespace

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
                         std::size_t derivative_size)
{
    if (nspin != 1 && nspin != 2)
    {
        throw std::invalid_argument("native CIDER spline spin count must be one or two");
    }
    const std::size_t expected_values = checked_product(nspin, ngrids, "spline value");
    const std::size_t spin_features = checked_product(nspin, nfeature, "spline descriptor");
    const std::size_t expected_descriptors
        = checked_product(spin_features, ngrids, "spline descriptor");
    if (descriptors == nullptr || values == nullptr || descriptor_derivatives == nullptr)
    {
        throw std::invalid_argument("native CIDER spline buffer is null");
    }
    if (descriptor_size != expected_descriptors || value_size != expected_values
        || derivative_size != expected_descriptors)
    {
        throw std::invalid_argument("native CIDER spline buffer size does not match dimensions");
    }
    if (!std::isfinite(constant))
    {
        throw std::invalid_argument("native CIDER spline constant is not finite");
    }
    validate_terms(terms, nfeature);
    validate_finite(descriptors, descriptor_size, "transformed descriptor buffer");

    std::vector<double> computed_values(expected_values, constant);
    std::vector<double> computed_derivatives(expected_descriptors, 0.0);
    for (std::size_t spin = 0; spin < nspin; ++spin)
    {
        for (std::size_t grid_point = 0; grid_point < ngrids; ++grid_point)
        {
            const std::size_t value_index = spin * ngrids + grid_point;
            for (std::size_t term_index = 0; term_index < terms.size(); ++term_index)
            {
                const SplineTerm& term = terms[term_index];
                AxisBasis axes[3];
                for (std::size_t d = 0; d < term.dimension(); ++d)
                {
                    const std::size_t descriptor = term.descriptor_indices()[d];
                    const std::size_t descriptor_index
                        = (spin * nfeature + descriptor) * ngrids + grid_point;
                    axes[d] = evaluate_axis(term.grids()[d], descriptors[descriptor_index]);
                }
                double term_value = 0.0;
                double term_derivative[3];
                evaluate_term(term, axes, term_value, term_derivative);
                computed_values[value_index] += term.scale() * term_value;
                if (!std::isfinite(computed_values[value_index]))
                {
                    throw std::overflow_error("native CIDER spline value is not finite");
                }
                for (std::size_t d = 0; d < term.dimension(); ++d)
                {
                    const std::size_t descriptor = term.descriptor_indices()[d];
                    const std::size_t derivative_index
                        = (spin * nfeature + descriptor) * ngrids + grid_point;
                    computed_derivatives[derivative_index]
                        += term.scale() * term_derivative[d];
                    if (!std::isfinite(computed_derivatives[derivative_index]))
                    {
                        throw std::overflow_error("native CIDER spline derivative is not finite");
                    }
                }
            }
        }
    }
    std::copy(computed_values.begin(), computed_values.end(), values);
    std::copy(computed_derivatives.begin(),
              computed_derivatives.end(),
              descriptor_derivatives);
}

} // namespace cider
} // namespace hamilt
