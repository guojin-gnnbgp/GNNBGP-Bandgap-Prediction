#include "semilocal_plan.h"

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

SemilocalForwardInput::SemilocalForwardInput(std::size_t nspin,
                                             std::size_t ngrids,
                                             const double* rho,
                                             std::size_t rho_size,
                                             const double* sigma,
                                             std::size_t sigma_size,
                                             const double* tau,
                                             std::size_t tau_size)
    : nspin_(nspin),
      ngrids_(ngrids),
      rho_(rho),
      rho_size_(rho_size),
      sigma_(sigma),
      sigma_size_(sigma_size),
      tau_(tau),
      tau_size_(tau_size)
{
}

std::size_t SemilocalForwardInput::nspin() const { return nspin_; }
std::size_t SemilocalForwardInput::ngrids() const { return ngrids_; }
const double* SemilocalForwardInput::rho() const { return rho_; }
std::size_t SemilocalForwardInput::rho_size() const { return rho_size_; }
const double* SemilocalForwardInput::sigma() const { return sigma_; }
std::size_t SemilocalForwardInput::sigma_size() const { return sigma_size_; }
const double* SemilocalForwardInput::tau() const { return tau_; }
std::size_t SemilocalForwardInput::tau_size() const { return tau_size_; }

SemilocalForwardOutput::SemilocalForwardOutput(double* features,
                                               std::size_t feature_size)
    : features_(features), feature_size_(feature_size)
{
}

double* SemilocalForwardOutput::features() const { return features_; }
std::size_t SemilocalForwardOutput::feature_size() const { return feature_size_; }

SemilocalReverseInput::SemilocalReverseInput(std::size_t nspin,
                                             std::size_t ngrids,
                                             const double* rho,
                                             std::size_t rho_size,
                                             const double* sigma,
                                             std::size_t sigma_size,
                                             const double* tau,
                                             std::size_t tau_size,
                                             const double* feature_adjoint,
                                             std::size_t feature_adjoint_size)
    : nspin_(nspin),
      ngrids_(ngrids),
      rho_(rho),
      rho_size_(rho_size),
      sigma_(sigma),
      sigma_size_(sigma_size),
      tau_(tau),
      tau_size_(tau_size),
      feature_adjoint_(feature_adjoint),
      feature_adjoint_size_(feature_adjoint_size)
{
}

std::size_t SemilocalReverseInput::nspin() const { return nspin_; }
std::size_t SemilocalReverseInput::ngrids() const { return ngrids_; }
const double* SemilocalReverseInput::rho() const { return rho_; }
std::size_t SemilocalReverseInput::rho_size() const { return rho_size_; }
const double* SemilocalReverseInput::sigma() const { return sigma_; }
std::size_t SemilocalReverseInput::sigma_size() const { return sigma_size_; }
const double* SemilocalReverseInput::tau() const { return tau_; }
std::size_t SemilocalReverseInput::tau_size() const { return tau_size_; }
const double* SemilocalReverseInput::feature_adjoint() const
{
    return feature_adjoint_;
}
std::size_t SemilocalReverseInput::feature_adjoint_size() const
{
    return feature_adjoint_size_;
}

SemilocalReverseOutput::SemilocalReverseOutput(double* vrho,
                                               std::size_t vrho_size,
                                               double* vsigma,
                                               std::size_t vsigma_size,
                                               double* vtau,
                                               std::size_t vtau_size)
    : vrho_(vrho),
      vrho_size_(vrho_size),
      vsigma_(vsigma),
      vsigma_size_(vsigma_size),
      vtau_(vtau),
      vtau_size_(vtau_size)
{
}

double* SemilocalReverseOutput::vrho() const { return vrho_; }
std::size_t SemilocalReverseOutput::vrho_size() const { return vrho_size_; }
double* SemilocalReverseOutput::vsigma() const { return vsigma_; }
std::size_t SemilocalReverseOutput::vsigma_size() const { return vsigma_size_; }
double* SemilocalReverseOutput::vtau() const { return vtau_; }
std::size_t SemilocalReverseOutput::vtau_size() const { return vtau_size_; }

namespace
{

const double pi = 3.141592653589793238462643383279502884;
const double alpha_tolerance = 1.0e-10;
const double denominator_epsilon = 1.0e-16;
const double gradient_factor = 2.0 * std::pow(3.0 * pi * pi, 1.0 / 3.0);
const double uniform_tau_factor = 0.3 * std::pow(3.0 * pi * pi, 2.0 / 3.0);

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

struct ForwardShape
{
    std::size_t rho_size;
    std::size_t sigma_size;
    std::size_t feature_size;
};

ForwardShape validate_forward(const SemilocalPlan& plan,
                              const SemilocalForwardInput& input,
                              const SemilocalForwardOutput& output)
{
    if (input.nspin() != plan.nspin())
    {
        throw std::invalid_argument("native CIDER semilocal spin count differs from plan");
    }
    const std::size_t rho_size
        = checked_product(plan.nspin(), input.ngrids(), "semilocal density");
    const std::size_t sigma_channels = 2 * plan.nspin() - 1;
    const std::size_t sigma_size
        = checked_product(sigma_channels, input.ngrids(), "semilocal sigma");
    const std::size_t spin_features
        = checked_product(plan.nspin(), plan.feature_count(), "semilocal feature");
    const std::size_t feature_size
        = checked_product(spin_features, input.ngrids(), "semilocal feature");
    if (input.rho() == nullptr || input.sigma() == nullptr || output.features() == nullptr)
    {
        throw std::invalid_argument("native CIDER semilocal buffer is null");
    }
    if (input.rho_size() != rho_size || input.sigma_size() != sigma_size
        || output.feature_size() != feature_size)
    {
        throw std::invalid_argument(
            "native CIDER semilocal buffer size does not match dimensions");
    }
    if (plan.mode() == SemilocalMode::Np)
    {
        if (input.tau() != nullptr || input.tau_size() != 0)
        {
            throw std::invalid_argument("native CIDER GGA semilocal input must omit tau");
        }
    }
    else if (input.tau() == nullptr || input.tau_size() != rho_size)
    {
        throw std::invalid_argument("native CIDER MGGA semilocal input requires tau");
    }
    const BufferRange feature_range
        = buffer_range(output.features(), output.feature_size(), "semilocal feature buffer");
    const BufferRange rho_range
        = buffer_range(input.rho(), input.rho_size(), "semilocal density buffer");
    const BufferRange sigma_range
        = buffer_range(input.sigma(), input.sigma_size(), "semilocal sigma buffer");
    if (overlap(feature_range, rho_range) || overlap(feature_range, sigma_range))
    {
        throw std::invalid_argument(
            "native CIDER semilocal input and output buffers must be disjoint");
    }
    if (input.tau() != nullptr
        && overlap(feature_range,
                   buffer_range(input.tau(), input.tau_size(), "semilocal tau buffer")))
    {
        throw std::invalid_argument(
            "native CIDER semilocal input and output buffers must be disjoint");
    }
    validate_finite(input.rho(), input.rho_size(), "semilocal density buffer");
    validate_finite(input.sigma(), input.sigma_size(), "semilocal sigma buffer");
    if (input.tau() != nullptr)
    {
        validate_finite(input.tau(), input.tau_size(), "semilocal tau buffer");
    }
    return {rho_size, sigma_size, feature_size};
}

struct ReverseShape
{
    std::size_t rho_size;
    std::size_t sigma_size;
    std::size_t feature_size;
};

ReverseShape validate_reverse(const SemilocalPlan& plan,
                              const SemilocalReverseInput& input,
                              const SemilocalReverseOutput& output)
{
    if (input.nspin() != plan.nspin())
    {
        throw std::invalid_argument("native CIDER semilocal spin count differs from plan");
    }
    const std::size_t rho_size
        = checked_product(plan.nspin(), input.ngrids(), "semilocal density");
    const std::size_t sigma_size
        = checked_product(2 * plan.nspin() - 1, input.ngrids(), "semilocal sigma");
    const std::size_t spin_features
        = checked_product(plan.nspin(), plan.feature_count(), "semilocal feature");
    const std::size_t feature_size
        = checked_product(spin_features, input.ngrids(), "semilocal feature");
    if (input.rho() == nullptr || input.sigma() == nullptr
        || input.feature_adjoint() == nullptr || output.vrho() == nullptr
        || output.vsigma() == nullptr)
    {
        throw std::invalid_argument("native CIDER semilocal reverse buffer is null");
    }
    if (input.rho_size() != rho_size || input.sigma_size() != sigma_size
        || input.feature_adjoint_size() != feature_size || output.vrho_size() != rho_size
        || output.vsigma_size() != sigma_size)
    {
        throw std::invalid_argument(
            "native CIDER semilocal reverse buffer size does not match dimensions");
    }
    if (plan.mode() == SemilocalMode::Np)
    {
        if (input.tau() != nullptr || input.tau_size() != 0 || output.vtau() != nullptr
            || output.vtau_size() != 0)
        {
            throw std::invalid_argument("native CIDER GGA semilocal reverse must omit tau");
        }
    }
    else if (input.tau() == nullptr || input.tau_size() != rho_size
             || output.vtau() == nullptr || output.vtau_size() != rho_size)
    {
        throw std::invalid_argument("native CIDER MGGA semilocal reverse requires tau");
    }
    const BufferRange input_ranges[] = {
        buffer_range(input.rho(), input.rho_size(), "semilocal density buffer"),
        buffer_range(input.sigma(), input.sigma_size(), "semilocal sigma buffer"),
        buffer_range(input.feature_adjoint(),
                     input.feature_adjoint_size(),
                     "semilocal feature-adjoint buffer"),
    };
    const BufferRange output_ranges[] = {
        buffer_range(output.vrho(), output.vrho_size(), "semilocal vrho buffer"),
        buffer_range(output.vsigma(), output.vsigma_size(), "semilocal vsigma buffer"),
    };
    for (const BufferRange& writable : output_ranges)
    {
        for (const BufferRange& readonly : input_ranges)
        {
            if (overlap(writable, readonly))
            {
                throw std::invalid_argument(
                    "native CIDER semilocal reverse inputs and outputs must be disjoint");
            }
        }
    }
    if (overlap(output_ranges[0], output_ranges[1]))
    {
        throw std::invalid_argument(
            "native CIDER semilocal reverse output buffers must be disjoint");
    }
    if (input.tau() != nullptr)
    {
        const BufferRange tau_range
            = buffer_range(input.tau(), input.tau_size(), "semilocal tau buffer");
        const BufferRange vtau_range
            = buffer_range(output.vtau(), output.vtau_size(), "semilocal vtau buffer");
        for (const BufferRange& writable : output_ranges)
        {
            if (overlap(writable, tau_range) || overlap(writable, vtau_range))
            {
                throw std::invalid_argument(
                    "native CIDER semilocal reverse buffers must be disjoint");
            }
        }
        for (const BufferRange& readonly : input_ranges)
        {
            if (overlap(vtau_range, readonly))
            {
                throw std::invalid_argument(
                    "native CIDER semilocal reverse buffers must be disjoint");
            }
        }
        if (overlap(vtau_range, tau_range))
        {
            throw std::invalid_argument(
                "native CIDER semilocal reverse buffers must be disjoint");
        }
    }
    validate_finite(input.rho(), input.rho_size(), "semilocal density buffer");
    validate_finite(input.sigma(), input.sigma_size(), "semilocal sigma buffer");
    validate_finite(input.feature_adjoint(),
                    input.feature_adjoint_size(),
                    "semilocal feature-adjoint buffer");
    validate_finite(output.vrho(), output.vrho_size(), "semilocal vrho buffer");
    validate_finite(output.vsigma(), output.vsigma_size(), "semilocal vsigma buffer");
    if (input.tau() != nullptr)
    {
        validate_finite(input.tau(), input.tau_size(), "semilocal tau buffer");
        validate_finite(output.vtau(), output.vtau_size(), "semilocal vtau buffer");
    }
    return {rho_size, sigma_size, feature_size};
}

} // namespace

SemilocalPlan::SemilocalPlan(SemilocalMode mode, std::size_t nspin)
    : mode_(mode), nspin_(nspin), feature_count_(0)
{
    if (nspin != 1 && nspin != 2)
    {
        throw std::invalid_argument("native CIDER semilocal spin count must be one or two");
    }
    switch (mode)
    {
        case SemilocalMode::Np:
            feature_count_ = 2;
            break;
        case SemilocalMode::Npa:
            feature_count_ = 3;
            break;
        default:
            throw std::invalid_argument("native CIDER semilocal mode is unsupported");
    }
}

SemilocalMode SemilocalPlan::mode() const { return mode_; }
std::size_t SemilocalPlan::nspin() const { return nspin_; }
std::size_t SemilocalPlan::feature_count() const { return feature_count_; }

void SemilocalPlan::forward(const SemilocalForwardInput& input,
                            SemilocalForwardOutput& output) const
{
    const ForwardShape shape = validate_forward(*this, input, output);
    std::vector<double> computed(shape.feature_size);
    const double inverse_spin_two_thirds
        = std::pow(static_cast<double>(nspin_), -2.0 / 3.0);
    for (std::size_t spin = 0; spin < nspin_; ++spin)
    {
        const std::size_t spin_grid_offset = spin * input.ngrids();
        const std::size_t sigma_offset = 2 * spin * input.ngrids();
        const std::size_t feature_offset = spin * feature_count_ * input.ngrids();
        for (std::size_t grid = 0; grid < input.ngrids(); ++grid)
        {
            const double rho = input.rho()[spin_grid_offset + grid];
            const double sigma = input.sigma()[sigma_offset + grid];
            computed[feature_offset + grid] = static_cast<double>(nspin_) * rho;
            if (rho < alpha_tolerance)
            {
                computed[feature_offset + input.ngrids() + grid] = 0.0;
                if (mode_ == SemilocalMode::Npa)
                {
                    computed[feature_offset + 2 * input.ngrids() + grid] = 0.0;
                }
                continue;
            }
            if (sigma < 0.0)
            {
                throw std::domain_error(
                    "native CIDER active same-spin sigma is negative");
            }
            const double density_four_thirds = std::pow(rho, 4.0 / 3.0);
            const double denominator
                = gradient_factor * density_four_thirds + denominator_epsilon;
            const double reduced_gradient = std::sqrt(sigma) / denominator;
            computed[feature_offset + input.ngrids() + grid]
                = inverse_spin_two_thirds * reduced_gradient * reduced_gradient;
            if (mode_ == SemilocalMode::Npa)
            {
                const double tau0 = uniform_tau_factor * std::pow(rho, 5.0 / 3.0);
                const double tauw = sigma / (8.0 * rho);
                computed[feature_offset + 2 * input.ngrids() + grid]
                    = inverse_spin_two_thirds
                      * (input.tau()[spin_grid_offset + grid] - tauw) / tau0;
            }
        }
    }
    validate_finite(computed.data(), computed.size(), "semilocal computed feature");
    std::copy(computed.begin(), computed.end(), output.features());
}

void SemilocalPlan::reverse(const SemilocalReverseInput& input,
                            SemilocalReverseOutput& output) const
{
    const ReverseShape shape = validate_reverse(*this, input, output);
    std::vector<double> vrho(output.vrho(), output.vrho() + shape.rho_size);
    std::vector<double> vsigma(output.vsigma(), output.vsigma() + shape.sigma_size);
    std::vector<double> vtau;
    if (mode_ == SemilocalMode::Npa)
    {
        vtau.assign(output.vtau(), output.vtau() + shape.rho_size);
    }
    const double spin_factor = static_cast<double>(nspin_);
    const double inverse_spin_two_thirds = std::pow(spin_factor, -2.0 / 3.0);
    for (std::size_t spin = 0; spin < nspin_; ++spin)
    {
        const std::size_t spin_grid_offset = spin * input.ngrids();
        const std::size_t sigma_offset = 2 * spin * input.ngrids();
        const std::size_t feature_offset = spin * feature_count_ * input.ngrids();
        for (std::size_t grid = 0; grid < input.ngrids(); ++grid)
        {
            const std::size_t spin_grid = spin_grid_offset + grid;
            const std::size_t same_spin_sigma = sigma_offset + grid;
            const double rho = input.rho()[spin_grid];
            const double sigma = input.sigma()[same_spin_sigma];
            const double density_adjoint = input.feature_adjoint()[feature_offset + grid];
            vrho[spin_grid] += spin_factor * density_adjoint;
            if (rho < alpha_tolerance)
            {
                continue;
            }
            if (sigma < 0.0)
            {
                throw std::domain_error(
                    "native CIDER active same-spin sigma is negative");
            }
            const double denominator
                = gradient_factor * std::pow(rho, 4.0 / 3.0) + denominator_epsilon;
            const double reduced_gradient = std::sqrt(sigma) / denominator;
            const double reduced_gradient_squared
                = reduced_gradient * reduced_gradient;
            const double dpdn
                = -8.0 * reduced_gradient_squared / (3.0 * rho + denominator_epsilon);
            const double dpdsigma = 1.0 / (denominator * denominator);
            const double gradient_adjoint
                = input.feature_adjoint()[feature_offset + input.ngrids() + grid];
            vrho[spin_grid] += inverse_spin_two_thirds * gradient_adjoint * dpdn;
            vsigma[same_spin_sigma]
                += inverse_spin_two_thirds * gradient_adjoint * dpdsigma;
            if (mode_ == SemilocalMode::Npa)
            {
                const double tau = input.tau()[spin_grid];
                const double tau0 = uniform_tau_factor * std::pow(rho, 5.0 / 3.0);
                const double tauw = sigma / (8.0 * rho);
                const double dwdn = -sigma / (8.0 * rho * rho);
                const double dadn
                    = 5.0 * (tauw - tau) / (3.0 * tau0 * rho) - dwdn / tau0;
                const double dadsigma = -1.0 / (8.0 * rho * tau0);
                const double dadtau = 1.0 / tau0;
                const double alpha_adjoint
                    = input.feature_adjoint()[feature_offset + 2 * input.ngrids() + grid];
                vrho[spin_grid] += inverse_spin_two_thirds * alpha_adjoint * dadn;
                vsigma[same_spin_sigma]
                    += inverse_spin_two_thirds * alpha_adjoint * dadsigma;
                vtau[spin_grid]
                    += inverse_spin_two_thirds * alpha_adjoint * dadtau;
            }
        }
    }
    validate_finite(vrho.data(), vrho.size(), "semilocal computed vrho");
    validate_finite(vsigma.data(), vsigma.size(), "semilocal computed vsigma");
    if (!vtau.empty())
    {
        validate_finite(vtau.data(), vtau.size(), "semilocal computed vtau");
    }
    std::copy(vrho.begin(), vrho.end(), output.vrho());
    std::copy(vsigma.begin(), vsigma.end(), output.vsigma());
    if (!vtau.empty())
    {
        std::copy(vtau.begin(), vtau.end(), output.vtau());
    }
}

} // namespace cider
} // namespace hamilt
