#include "mapped_evaluator.h"

#include "feature_normalizer.h"
#include "feature_transform.h"
#include "model.h"
#include "spline_evaluator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hamilt
{
namespace cider
{

MappedEvaluationInput::MappedEvaluationInput(std::size_t nspin,
                                             std::size_t ngrids,
                                             const double* raw_features,
                                             std::size_t raw_feature_size,
                                             const double* rho,
                                             std::size_t rho_size,
                                             double density_cutoff)
    : nspin_(nspin),
      ngrids_(ngrids),
      raw_features_(raw_features),
      raw_feature_size_(raw_feature_size),
      rho_(rho),
      rho_size_(rho_size),
      density_cutoff_(density_cutoff)
{
}

std::size_t MappedEvaluationInput::nspin() const { return nspin_; }
std::size_t MappedEvaluationInput::ngrids() const { return ngrids_; }
const double* MappedEvaluationInput::raw_features() const { return raw_features_; }
std::size_t MappedEvaluationInput::raw_feature_size() const { return raw_feature_size_; }
const double* MappedEvaluationInput::rho() const { return rho_; }
std::size_t MappedEvaluationInput::rho_size() const { return rho_size_; }
double MappedEvaluationInput::density_cutoff() const { return density_cutoff_; }

MappedEvaluationOutput::MappedEvaluationOutput(double* energy_density,
                                               std::size_t energy_density_size,
                                               double* raw_feature_derivatives,
                                               std::size_t raw_feature_derivative_size)
    : energy_density_(energy_density),
      energy_density_size_(energy_density_size),
      raw_feature_derivatives_(raw_feature_derivatives),
      raw_feature_derivative_size_(raw_feature_derivative_size)
{
}

double* MappedEvaluationOutput::energy_density() const { return energy_density_; }
std::size_t MappedEvaluationOutput::energy_density_size() const
{
    return energy_density_size_;
}
double* MappedEvaluationOutput::raw_feature_derivatives() const
{
    return raw_feature_derivatives_;
}
std::size_t MappedEvaluationOutput::raw_feature_derivative_size() const
{
    return raw_feature_derivative_size_;
}

MappedEvaluator::~MappedEvaluator() = default;

namespace
{

const double pi = 3.141592653589793238462643383279502884;
const double lda_factor = -0.75 * std::pow(3.0 / pi, 1.0 / 3.0);

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

struct BufferRange
{
    std::uintptr_t begin;
    std::uintptr_t end;
};

BufferRange buffer_range(const double* pointer, std::size_t size, const char* label)
{
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        throw std::overflow_error(std::string("native CIDER ") + label
                                  + " byte size overflows");
    }
    const std::size_t bytes = size * sizeof(double);
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    {
        throw std::overflow_error(std::string("native CIDER ") + label
                                  + " address range overflows");
    }
    return {begin, begin + bytes};
}

bool overlap(const BufferRange& first, const BufferRange& second)
{
    return first.begin < second.end && second.begin < first.end;
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

struct EvaluationShape
{
    std::size_t raw_feature_count;
    std::size_t transformed_feature_count;
    std::size_t rho_size;
    std::size_t raw_size;
    std::size_t transformed_size;
};

EvaluationShape validate_contract(const Model& model,
                                  const MappedEvaluationInput& input,
                                  const MappedEvaluationOutput& output)
{
    if (input.nspin() != 1 && input.nspin() != 2)
    {
        throw std::invalid_argument("native CIDER mapped spin count must be one or two");
    }
    const ModelMetadata& metadata = model.metadata();
    const std::size_t raw_feature_count = metadata.raw_feature_count();
    const std::size_t transformed_feature_count = metadata.transformed_feature_count();
    if (raw_feature_count < 2 || transformed_feature_count == 0)
    {
        throw std::invalid_argument("native CIDER mapped feature counts are invalid");
    }
    if (model.normalizers().size() != raw_feature_count
        || model.feature_maps().size() != transformed_feature_count)
    {
        throw std::invalid_argument("native CIDER mapped model cardinality is inconsistent");
    }
    const std::size_t rho_size
        = checked_product(input.nspin(), input.ngrids(), "mapped density");
    const std::size_t spin_features
        = checked_product(input.nspin(), raw_feature_count, "mapped raw feature");
    const std::size_t raw_size
        = checked_product(spin_features, input.ngrids(), "mapped raw feature");
    const std::size_t spin_transformed
        = checked_product(input.nspin(), transformed_feature_count, "mapped descriptor");
    const std::size_t transformed_size
        = checked_product(spin_transformed, input.ngrids(), "mapped descriptor");
    if (input.raw_features() == nullptr || input.rho() == nullptr
        || output.energy_density() == nullptr || output.raw_feature_derivatives() == nullptr)
    {
        throw std::invalid_argument("native CIDER mapped buffer is null");
    }
    if (input.rho_size() != rho_size || input.raw_feature_size() != raw_size
        || output.energy_density_size() != input.ngrids()
        || output.raw_feature_derivative_size() != raw_size)
    {
        throw std::invalid_argument("native CIDER mapped buffer size does not match dimensions");
    }
    if (!std::isfinite(input.density_cutoff()) || input.density_cutoff() < 0.0)
    {
        throw std::invalid_argument("native CIDER mapped density cutoff is invalid");
    }
    if (!std::isfinite(metadata.kernel_output_scale()))
    {
        throw std::invalid_argument("native CIDER mapped output scale is not finite");
    }
    validate_finite(input.rho(), input.rho_size(), "physical density buffer");
    validate_finite(input.raw_features(), input.raw_feature_size(), "raw feature buffer");
    for (std::size_t spin = 0; spin < input.nspin(); ++spin)
    {
        for (std::size_t grid = 0; grid < input.ngrids(); ++grid)
        {
            const std::size_t rho_index = spin * input.ngrids() + grid;
            const double rho = input.rho()[rho_index];
            if (rho < 0.0)
            {
                throw std::domain_error("native CIDER physical density is negative");
            }
            const double descriptor_density = static_cast<double>(input.nspin()) * rho;
            if (!std::isfinite(descriptor_density))
            {
                throw std::overflow_error("native CIDER SEP density descriptor is not finite");
            }
            const std::size_t raw_density_index
                = (spin * raw_feature_count) * input.ngrids() + grid;
            if (input.raw_features()[raw_density_index] != descriptor_density)
            {
                throw std::invalid_argument(
                    "native CIDER raw density descriptor differs from nspin times rho");
            }
        }
    }

    const BufferRange raw_range
        = buffer_range(input.raw_features(), input.raw_feature_size(), "raw feature buffer");
    const BufferRange rho_range = buffer_range(input.rho(), input.rho_size(), "density buffer");
    const BufferRange energy_range
        = buffer_range(output.energy_density(), output.energy_density_size(), "energy buffer");
    const BufferRange derivative_range = buffer_range(output.raw_feature_derivatives(),
                                                       output.raw_feature_derivative_size(),
                                                       "derivative buffer");
    if (overlap(energy_range, derivative_range) || overlap(raw_range, energy_range)
        || overlap(raw_range, derivative_range) || overlap(rho_range, energy_range)
        || overlap(rho_range, derivative_range))
    {
        throw std::invalid_argument(
            "native CIDER mapped input and output buffers must be disjoint");
    }
    return {raw_feature_count,
            transformed_feature_count,
            rho_size,
            raw_size,
            transformed_size};
}

struct BaselineValue
{
    double energy;
    double density_derivative;
    double reduced_gradient_derivative;
};

BaselineValue lda_baseline(double density)
{
    const double rho_one_third = std::pow(density, 1.0 / 3.0);
    const double rho_four_thirds = density * rho_one_third;
    return {lda_factor * rho_four_thirds,
            (4.0 / 3.0) * lda_factor * rho_one_third,
            0.0};
}

BaselineValue pbe_baseline(double density, double reduced_gradient_squared)
{
    const double kappa = 0.804;
    const double mu = 0.2195149727645171;
    const double factor = 1.0 / (1.0 + (mu / kappa) * reduced_gradient_squared);
    const double enhancement = 1.0 + kappa - kappa * factor;
    const double enhancement_derivative = mu * factor * factor;
    const double rho_one_third = std::pow(density, 1.0 / 3.0);
    const double rho_four_thirds = density * rho_one_third;
    return {lda_factor * rho_four_thirds * enhancement,
            (4.0 / 3.0) * lda_factor * rho_one_third * enhancement,
            lda_factor * rho_four_thirds * enhancement_derivative};
}

BaselineValue chachiyo_baseline(double density, double reduced_gradient_squared)
{
    double enhancement = 0.0;
    double enhancement_derivative = 0.0;
    if (reduced_gradient_squared < 1.0e-8)
    {
        enhancement = 1.0 + (8.0 / 27.0) * reduced_gradient_squared;
        enhancement_derivative = 8.0 / 27.0;
    }
    else
    {
        const double c = 4.0 * pi / 9.0;
        const double square_root = std::sqrt(reduced_gradient_squared);
        const double x = c * square_root;
        const double logarithm = std::log1p(x);
        const double pi_squared = pi * pi;
        const double denominator = (pi_squared + 3.0 * x) * logarithm;
        enhancement = (3.0 * x * x + pi_squared * logarithm) / denominator;
        const double numerator_derivative
            = -3.0 * x * x * (pi_squared + 3.0 * x)
              + 3.0 * x * (1.0 + x) * (2.0 * pi_squared + 3.0 * x) * logarithm
              - 3.0 * pi_squared * (1.0 + x) * logarithm * logarithm;
        const double denominator_derivative
            = (1.0 + x) * (pi_squared + 3.0 * x) * (pi_squared + 3.0 * x)
              * logarithm * logarithm;
        enhancement_derivative
            = numerator_derivative / denominator_derivative * c / (2.0 * square_root);
    }
    const double rho_one_third = std::pow(density, 1.0 / 3.0);
    const double rho_four_thirds = density * rho_one_third;
    return {lda_factor * rho_four_thirds * enhancement,
            (4.0 / 3.0) * lda_factor * rho_one_third * enhancement,
            lda_factor * rho_four_thirds * enhancement_derivative};
}

BaselineValue additive_baseline(AdditiveBaseline baseline,
                                double density,
                                double reduced_gradient_squared)
{
    switch (baseline)
    {
        case AdditiveBaseline::GgaXChachiyo:
            return chachiyo_baseline(density, reduced_gradient_squared);
        case AdditiveBaseline::GgaXPbe:
            return pbe_baseline(density, reduced_gradient_squared);
    }
    throw std::invalid_argument("native CIDER additive baseline is unsupported");
}

void validate_computed(const std::vector<double>& values, const char* label)
{
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (!std::isfinite(values[i]))
        {
            throw std::overflow_error(std::string("native CIDER mapped ") + label
                                      + " is not finite");
        }
    }
}

} // namespace

void MappedDFTKernel::evaluate(const Model& model,
                               const MappedEvaluationInput& input,
                               MappedEvaluationOutput& output) const
{
    const EvaluationShape shape = validate_contract(model, input, output);
    std::vector<double> normalized(shape.raw_size);
    normalize_features(model.normalizers(),
                       model.semilocal_mode(),
                       input.nspin(),
                       shape.raw_feature_count,
                       input.ngrids(),
                       input.raw_features(),
                       input.raw_feature_size(),
                       normalized.data(),
                       normalized.size());

    std::vector<double> descriptors(shape.transformed_size);
    transform_features(model.feature_maps(),
                       input.nspin(),
                       shape.raw_feature_count,
                       input.ngrids(),
                       normalized.data(),
                       normalized.size(),
                       descriptors.data(),
                       descriptors.size());

    std::vector<double> spline_values(shape.rho_size);
    std::vector<double> descriptor_derivatives(shape.transformed_size);
    evaluate_spline_set(model.spline_constant(),
                        model.spline_terms(),
                        input.nspin(),
                        shape.transformed_feature_count,
                        input.ngrids(),
                        descriptors.data(),
                        descriptors.size(),
                        spline_values.data(),
                        spline_values.size(),
                        descriptor_derivatives.data(),
                        descriptor_derivatives.size());

    std::vector<double> normalized_derivatives(shape.raw_size, 0.0);
    accumulate_transform_reverse(model.feature_maps(),
                                 input.nspin(),
                                 shape.raw_feature_count,
                                 input.ngrids(),
                                 normalized.data(),
                                 normalized.size(),
                                 descriptor_derivatives.data(),
                                 descriptor_derivatives.size(),
                                 normalized_derivatives.data(),
                                 normalized_derivatives.size());

    if (model.multiplicative_baseline() != MultiplicativeBaseline::LdaX)
    {
        throw std::invalid_argument("native CIDER multiplicative baseline is unsupported");
    }
    const double inverse_spin = 1.0 / static_cast<double>(input.nspin());
    std::vector<double> spin_energy(shape.rho_size, 0.0);
    for (std::size_t spin = 0; spin < input.nspin(); ++spin)
    {
        for (std::size_t grid = 0; grid < input.ngrids(); ++grid)
        {
            const std::size_t spin_grid = spin * input.ngrids() + grid;
            const std::size_t density_index
                = (spin * shape.raw_feature_count) * input.ngrids() + grid;
            const std::size_t gradient_index
                = (spin * shape.raw_feature_count + 1) * input.ngrids() + grid;
            const double density = normalized[density_index];
            if (density < 0.0)
            {
                throw std::domain_error("native CIDER normalized density is negative");
            }
            const BaselineValue multiplicative = lda_baseline(density);
            const BaselineValue additive
                = additive_baseline(model.additive_baseline(), density, normalized[gradient_index]);
            const double multiplier = multiplicative.energy * inverse_spin;
            spin_energy[spin_grid]
                = spline_values[spin_grid] * multiplier + additive.energy * inverse_spin;
            for (std::size_t feature = 0; feature < shape.raw_feature_count; ++feature)
            {
                const std::size_t index
                    = (spin * shape.raw_feature_count + feature) * input.ngrids() + grid;
                normalized_derivatives[index] *= multiplier;
            }
            normalized_derivatives[density_index]
                += spline_values[spin_grid] * multiplicative.density_derivative * inverse_spin
                   + additive.density_derivative * inverse_spin;
            normalized_derivatives[gradient_index]
                += additive.reduced_gradient_derivative * inverse_spin;
        }
    }

    const double output_scale = model.metadata().kernel_output_scale();
    for (std::size_t spin = 0; spin < input.nspin(); ++spin)
    {
        for (std::size_t grid = 0; grid < input.ngrids(); ++grid)
        {
            const std::size_t spin_grid = spin * input.ngrids() + grid;
            const std::size_t density_index
                = (spin * shape.raw_feature_count) * input.ngrids() + grid;
            spin_energy[spin_grid] *= output_scale;
            for (std::size_t feature = 0; feature < shape.raw_feature_count; ++feature)
            {
                const std::size_t index
                    = (spin * shape.raw_feature_count + feature) * input.ngrids() + grid;
                normalized_derivatives[index] *= output_scale;
            }
            if (input.density_cutoff() > 0.0
                && normalized[density_index] < input.density_cutoff())
            {
                spin_energy[spin_grid] = 0.0;
                for (std::size_t feature = 0; feature < shape.raw_feature_count; ++feature)
                {
                    const std::size_t index
                        = (spin * shape.raw_feature_count + feature) * input.ngrids() + grid;
                    normalized_derivatives[index] = 0.0;
                }
            }
        }
    }
    validate_computed(spin_energy, "spin energy density");
    validate_computed(normalized_derivatives, "normalized-feature derivative");

    std::vector<double> energy(input.ngrids(), 0.0);
    for (std::size_t spin = 0; spin < input.nspin(); ++spin)
    {
        for (std::size_t grid = 0; grid < input.ngrids(); ++grid)
        {
            energy[grid] += spin_energy[spin * input.ngrids() + grid];
        }
    }
    validate_computed(energy, "energy density");

    std::vector<double> raw_derivatives(shape.raw_size, 0.0);
    accumulate_normalization_reverse(model.normalizers(),
                                     model.semilocal_mode(),
                                     input.nspin(),
                                     shape.raw_feature_count,
                                     input.ngrids(),
                                     input.raw_features(),
                                     input.raw_feature_size(),
                                     normalized_derivatives.data(),
                                     normalized_derivatives.size(),
                                     raw_derivatives.data(),
                                     raw_derivatives.size());
    validate_computed(raw_derivatives, "raw-feature derivative");
    std::copy(energy.begin(), energy.end(), output.energy_density());
    std::copy(raw_derivatives.begin(),
              raw_derivatives.end(),
              output.raw_feature_derivatives());
}

void MappedXC::evaluate(const Model& model,
                        const MappedEvaluationInput& input,
                        MappedEvaluationOutput& output) const
{
    const MappedDFTKernel kernel;
    kernel.evaluate(model, input, output);
}

} // namespace cider
} // namespace hamilt
