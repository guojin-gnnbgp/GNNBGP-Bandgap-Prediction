#include "nldf_plan.h"

#include "nldf_interpolation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hamilt
{
namespace cider
{
namespace
{

const double pi = 3.141592653589793238462643383279502884;
const int smooth_order = 12;

struct BufferRange
{
    std::uintptr_t begin;
    std::uintptr_t end;
};

BufferRange buffer_range(const double* pointer, std::size_t count)
{
    if (pointer == nullptr || count == 0)
    {
        return BufferRange{0, 0};
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        throw std::invalid_argument("NLDF buffer byte size overflows");
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    const std::size_t bytes = count * sizeof(double);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    {
        throw std::invalid_argument("NLDF buffer address range overflows");
    }
    return BufferRange{begin, begin + bytes};
}

bool overlaps(const BufferRange& first, const BufferRange& second)
{
    return first.begin != first.end && second.begin != second.end
           && first.begin < second.end && second.begin < first.end;
}

std::size_t checked_product(std::size_t first,
                            std::size_t second,
                            const char* message)
{
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
    {
        throw std::invalid_argument(message);
    }
    return first * second;
}

double smooth_exponent(double value, double cutoff, double& derivative)
{
    const double scaled = value / cutoff;
    double total = 0.0;
    double derivative_total = 0.0;
    double power = 1.0;
    for (int order = 1; order <= smooth_order; ++order)
    {
        derivative_total += power;
        power *= scaled;
        total += power / order;
    }
    derivative = std::exp(-total) * derivative_total;
    return cutoff * (1.0 - std::exp(-total));
}

void construct_natural_spline_table(const std::vector<double>& norms,
                                    std::vector<double>& table)
{
    const std::size_t count = norms.size();
    const std::size_t square
        = checked_product(count, count, "NLDF spline table size overflows");
    table.assign(checked_product(square, 4, "NLDF spline table size overflows"),
                 0.0);
    std::vector<double> a(count), b(count), c(count), d(count), mu(count), z(count);
    for (std::size_t alpha = 0; alpha < count; ++alpha)
    {
        std::fill(a.begin(), a.end(), 0.0);
        a[alpha] = 1.0 / norms[alpha];
        std::fill(mu.begin(), mu.end(), 0.0);
        std::fill(z.begin(), z.end(), 0.0);
        for (std::size_t i = 1; i + 1 < count; ++i)
        {
            const double l = 4.0 - mu[i - 1];
            mu[i] = 1.0 / l;
            const double rhs = 3.0 * (a[i + 1] - 2.0 * a[i] + a[i - 1]);
            z[i] = (rhs - z[i - 1]) / l;
        }
        std::fill(b.begin(), b.end(), 0.0);
        std::fill(c.begin(), c.end(), 0.0);
        std::fill(d.begin(), d.end(), 0.0);
        for (std::size_t reverse = count - 1; reverse > 0; --reverse)
        {
            const std::size_t i = reverse - 1;
            c[i] = z[i] - mu[i] * c[i + 1];
            b[i] = a[i + 1] - a[i] - (c[i + 1] + 2.0 * c[i]) / 3.0;
            d[i] = (c[i + 1] - c[i]) / 3.0;
        }
        for (std::size_t interval = 0; interval + 1 < count; ++interval)
        {
            const std::size_t base = (interval * count + alpha) * 4;
            table[base] = a[interval];
            table[base + 1] = b[interval];
            table[base + 2] = c[interval];
            table[base + 3] = d[interval];
        }
    }
    // Preserve the upstream endpoint sentinel, though clipping means it is not
    // selected for finite arguments.
    table[((count - 1) * count + (count - 1)) * 4] = 1.0;
}

} // namespace

NldfAlphaSelection::NldfAlphaSelection(double cutoff,
                                       double lambda,
                                       std::size_t alpha_count)
    : cutoff_(cutoff), lambda_(lambda), alpha_count_(alpha_count)
{
    if (!(cutoff > 0.0) || !std::isfinite(cutoff)
        || !(lambda > 1.0) || !std::isfinite(lambda)
        || alpha_count == 1)
    {
        throw std::invalid_argument("invalid NLDF alpha selection");
    }
}

double NldfAlphaSelection::cutoff() const { return cutoff_; }
double NldfAlphaSelection::lambda() const { return lambda_; }
std::size_t NldfAlphaSelection::alpha_count() const { return alpha_count_; }

NldfPlan::NldfPlan(const NldfSettings& settings,
                   std::size_t nspin,
                   const NldfGrid& grid,
                   const NldfAlphaSelection& selection)
    : settings_(settings),
      nspin_(nspin),
      grid_(grid),
      alpha0_(0.0),
      lambda_(selection.lambda()),
      rho_cutoff_(0.0),
      backend_()
{
    if (nspin != 1 && nspin != 2)
    {
        throw std::invalid_argument("NLDF spin count must be one or two");
    }
    if (settings.feature_count() == 0)
    {
        throw std::invalid_argument("NLDF plan requires at least one feature");
    }
    double minimum = settings.theta_parameters().a0();
    for (std::size_t i = 0; i < settings.feature_parameters().size(); ++i)
    {
        minimum = std::min(minimum, settings.feature_parameters()[i].a0());
    }
    if (!(minimum > 0.0) || !std::isfinite(minimum))
    {
        throw std::invalid_argument("NLDF exponent a0 values must be positive");
    }
    std::size_t count = selection.alpha_count();
    if (count == 0)
    {
        const double smallest = minimum / 64.0;
        const double selected
            = std::ceil(std::log(selection.cutoff() / smallest)
                        / std::log(lambda_))
              + 1.0;
        if (!(selected >= 2.0)
            || selected > static_cast<double>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument("automatic NLDF alpha count is invalid");
        }
        count = static_cast<std::size_t>(selected);
        lambda_ = std::exp(std::log(selection.cutoff() / smallest)
                           / static_cast<double>(count - 1));
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("NLDF alpha count exceeds backend limits");
    }
    alpha0_ = selection.cutoff()
              / std::pow(lambda_, static_cast<double>(count - 1));
    alphas_.resize(count);
    alpha_norms_.resize(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        alphas_[i] = alpha0_ * std::pow(lambda_, static_cast<double>(i));
        alpha_norms_[i] = std::pow(pi / (2.0 * alphas_[i]), -0.75);
        if (!std::isfinite(alphas_[i]) || !std::isfinite(alpha_norms_[i]))
        {
            throw std::invalid_argument("NLDF alpha grid is not finite");
        }
    }
    rho_cutoff_ = 1.0e-10 / static_cast<double>(nspin_);
    construct_natural_spline_table(alpha_norms_, spline_table_);
    backend_.reset(new NldfBackend(grid_, alphas_, alpha_norms_));
}

NldfPlan::~NldfPlan() = default;

std::size_t NldfPlan::spin_count() const { return nspin_; }
std::size_t NldfPlan::feature_count() const { return settings_.feature_count(); }
std::size_t NldfPlan::alpha_count() const { return alphas_.size(); }
double NldfPlan::alpha0() const { return alpha0_; }
double NldfPlan::lambda() const { return lambda_; }
const std::vector<double>& NldfPlan::alphas() const { return alphas_; }
const std::vector<double>& NldfPlan::alpha_norms() const { return alpha_norms_; }

NldfExponentValue NldfPlan::exponent(double rho,
                                     double sigma,
                                     double tau,
                                     int parameter_index) const
{
    if (!std::isfinite(rho) || !std::isfinite(sigma) || !std::isfinite(tau))
    {
        throw std::invalid_argument("NLDF exponent inputs must be finite");
    }
    const ParameterTriple* parameters = &settings_.theta_parameters();
    if (parameter_index >= 0)
    {
        if (static_cast<std::size_t>(parameter_index)
            >= settings_.feature_parameters().size())
        {
            throw std::out_of_range("NLDF feature exponent index is out of range");
        }
        parameters = &settings_.feature_parameters()[parameter_index];
    }
    else if (parameter_index != -1)
    {
        throw std::out_of_range("NLDF feature exponent index is out of range");
    }

    const bool below_cutoff = rho < rho_cutoff_;
    const double active_rho = below_cutoff ? rho_cutoff_ : rho;
    const double active_sigma = below_cutoff ? 0.0 : sigma;
    const double active_tau = below_cutoff ? 0.0 : tau;
    const double spin_factor = nspin_ == 1 ? std::pow(2.0, -2.0 / 3.0) : 1.0;
    const double a0 = parameters->a0();
    const double gradient_multiplier = parameters->gradient_multiplier();
    const double tau_multiplier = parameters->tau_multiplier();
    const double common = 1.2 * std::pow(6.0 * pi * pi, 2.0 / 3.0) / pi;
    const double cfc = 0.3 * std::pow(3.0 * pi * pi, 2.0 / 3.0);
    const double tau_factor = tau_multiplier * common;
    const double b = pi * spin_factor * (a0 - tau_factor);
    const double c = pi * std::pow(2.0, -2.0 / 3.0) * tau_factor / cfc;
    double value = b * std::pow(active_rho, 2.0 / 3.0);
    double drho = 2.0 * b / (3.0 * std::pow(active_rho, 1.0 / 3.0));
    double dsigma = 0.0;
    double dtau = 0.0;
    if (settings_.semilocal_level() == SemilocalLevel::Mgga)
    {
        value += c * active_tau / active_rho;
        drho -= c * active_tau / (active_rho * active_rho);
        dtau = c / active_rho;
    }
    if (gradient_multiplier != 0.0)
    {
        double gradient_factor = gradient_multiplier * common;
        gradient_factor *= pi * std::pow(2.0, -2.0 / 3.0) * 0.125 / cfc;
        value += gradient_factor * active_sigma / (active_rho * active_rho);
        dsigma = gradient_factor / (active_rho * active_rho);
        drho -= 2.0 * gradient_factor * active_sigma
                 / (active_rho * active_rho * active_rho);
    }
    if (below_cutoff)
    {
        drho = dsigma = dtau = 0.0;
    }
    double smooth_derivative = 0.0;
    value = smooth_exponent(value, alphas_.back(), smooth_derivative);
    drho *= smooth_derivative;
    dsigma *= smooth_derivative;
    dtau *= smooth_derivative;
    if (!std::isfinite(value) || !std::isfinite(drho)
        || !std::isfinite(dsigma) || !std::isfinite(dtau))
    {
        throw std::domain_error("NLDF exponent evaluation is not finite");
    }
    return NldfExponentValue{value, drho, dsigma, dtau};
}

NldfInterpolationArgument NldfPlan::interpolation_argument(
    const NldfExponentValue& exponent_value) const
{
    double exponent = exponent_value.value;
    double index = 0.0;
    double derivative = 0.0;
    if (!(exponent > 0.0) || !std::isfinite(exponent))
    {
        throw std::domain_error("NLDF interpolation exponent must be positive");
    }
    abacus_cider_nldf_ind_etb(
        &index, &derivative, &exponent, 1, alpha0_, lambda_);
    abacus_cider_nldf_ind_clip(
        &index, &derivative, static_cast<int>(alphas_.size() - 1), 1);
    return NldfInterpolationArgument{index,
                                     exponent_value.drho * derivative,
                                     exponent_value.dsigma * derivative,
                                     exponent_value.dtau * derivative};
}

void NldfPlan::interpolation_coefficients(
    double argument,
    std::vector<double>& coefficients,
    std::vector<double>& derivatives) const
{
    if (!std::isfinite(argument) || argument < 0.0
        || argument >= static_cast<double>(alphas_.size() - 1))
    {
        throw std::invalid_argument("NLDF interpolation argument is out of range");
    }
    coefficients.resize(alphas_.size());
    derivatives.resize(alphas_.size());
    abacus_cider_nldf_coefs_spline_gq(coefficients.data(),
                                      derivatives.data(),
                                      &argument,
                                      spline_table_.data(),
                                      1,
                                      static_cast<int>(alphas_.size()));
}

NldfForwardInput::NldfForwardInput(std::size_t input_nspin,
                                   std::size_t input_ngrids,
                                   const double* input_rho,
                                   std::size_t input_rho_size,
                                   const double* input_sigma,
                                   std::size_t input_sigma_size,
                                   const double* input_tau,
                                   std::size_t input_tau_size)
    : nspin(input_nspin), ngrids(input_ngrids), rho(input_rho),
      rho_size(input_rho_size), sigma(input_sigma), sigma_size(input_sigma_size),
      tau(input_tau), tau_size(input_tau_size)
{
}

NldfForwardOutput::NldfForwardOutput(double* output_features,
                                     std::size_t output_feature_size,
                                     double* output_arguments,
                                     std::size_t output_argument_size,
                                     double* output_function,
                                     std::size_t output_function_size)
    : features(output_features), feature_size(output_feature_size),
      interpolation_arguments(output_arguments), argument_size(output_argument_size),
      convolution_function(output_function), function_size(output_function_size)
{
}

NldfReverseInput::NldfReverseInput(std::size_t input_nspin,
                                   std::size_t input_ngrids,
                                   const double* input_rho,
                                   std::size_t input_rho_size,
                                   const double* input_sigma,
                                   std::size_t input_sigma_size,
                                   const double* input_tau,
                                   std::size_t input_tau_size,
                                   const double* input_feature_adjoint,
                                   std::size_t input_feature_adjoint_size)
    : nspin(input_nspin), ngrids(input_ngrids), rho(input_rho),
      rho_size(input_rho_size), sigma(input_sigma), sigma_size(input_sigma_size),
      tau(input_tau), tau_size(input_tau_size),
      feature_adjoint(input_feature_adjoint),
      feature_adjoint_size(input_feature_adjoint_size)
{
}

NldfReverseOutput::NldfReverseOutput(double* output_vrho,
                                     std::size_t output_vrho_size,
                                     double* output_vsigma,
                                     std::size_t output_vsigma_size,
                                     double* output_vtau,
                                     std::size_t output_vtau_size)
    : vrho(output_vrho), vrho_size(output_vrho_size),
      vsigma(output_vsigma), vsigma_size(output_vsigma_size),
      vtau(output_vtau), vtau_size(output_vtau_size)
{
}

void NldfPlan::forward(const NldfForwardInput& input,
                       NldfForwardOutput& output) const
{
    const std::size_t grids = grid_.grid_count();
    const std::size_t density_size
        = checked_product(nspin_, grids, "NLDF density size overflows");
    const std::size_t sigma_size
        = checked_product(2 * nspin_ - 1, grids, "NLDF sigma size overflows");
    const std::size_t feature_size = checked_product(
        checked_product(nspin_, feature_count(), "NLDF feature size overflows"),
        grids,
        "NLDF feature size overflows");
    const bool mgga = settings_.semilocal_level() == SemilocalLevel::Mgga;
    if (input.nspin != nspin_ || input.ngrids != grids
        || input.rho == nullptr || input.rho_size != density_size
        || input.sigma == nullptr || input.sigma_size != sigma_size
        || (mgga && (input.tau == nullptr || input.tau_size != density_size))
        || (!mgga && (input.tau != nullptr || input.tau_size != 0))
        || output.features == nullptr || output.feature_size != feature_size
        || ((output.interpolation_arguments == nullptr) != (output.argument_size == 0))
        || (output.interpolation_arguments != nullptr && output.argument_size != density_size)
        || ((output.convolution_function == nullptr) != (output.function_size == 0))
        || (output.convolution_function != nullptr && output.function_size != density_size))
    {
        throw std::invalid_argument("invalid NLDF forward shape or buffer");
    }
    const BufferRange inputs[] = {buffer_range(input.rho, input.rho_size),
                                  buffer_range(input.sigma, input.sigma_size),
                                  buffer_range(input.tau, input.tau_size)};
    const BufferRange outputs[] = {
        buffer_range(output.features, output.feature_size),
        buffer_range(output.interpolation_arguments, output.argument_size),
        buffer_range(output.convolution_function, output.function_size)};
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            if (overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("NLDF forward inputs and outputs must be disjoint");
        }
        for (std::size_t j = i + 1; j < 3; ++j)
        {
            if (overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("NLDF forward outputs must be disjoint");
        }
    }
    for (std::size_t i = 0; i < density_size; ++i)
    {
        if (!std::isfinite(input.rho[i]) || (mgga && !std::isfinite(input.tau[i])))
        {
            throw std::invalid_argument("NLDF forward density inputs must be finite");
        }
    }
    for (std::size_t i = 0; i < sigma_size; ++i)
    {
        if (!std::isfinite(input.sigma[i]))
        {
            throw std::invalid_argument("NLDF forward sigma must be finite");
        }
    }
    std::vector<double> arguments(density_size);
    std::vector<double> function(density_size);
    std::vector<double> features(feature_size);
    std::vector<double> source(
        checked_product(grids, alpha_count(), "NLDF work size overflows"));
    std::vector<double> convolved(source.size());
    std::vector<double> coefficients, derivatives;
    for (std::size_t spin = 0; spin < nspin_; ++spin)
    {
        for (std::size_t grid = 0; grid < grids; ++grid)
        {
            const std::size_t density_index = spin * grids + grid;
            const double rho = input.rho[density_index];
            const double sigma = input.sigma[2 * spin * grids + grid];
            const double tau = mgga ? input.tau[density_index] : 0.0;
            const NldfInterpolationArgument argument
                = interpolation_argument(exponent(rho, sigma, tau, -1));
            arguments[density_index] = argument.value;
            function[density_index] = rho < rho_cutoff_ ? 0.0 : rho;
            interpolation_coefficients(argument.value, coefficients, derivatives);
            for (std::size_t alpha = 0; alpha < alpha_count(); ++alpha)
            {
                source[grid * alpha_count() + alpha]
                    = function[density_index] * coefficients[alpha];
            }
        }
        backend_->convolve(source.data(), source.size(), convolved.data(), convolved.size());
        for (std::size_t feature = 0; feature < feature_count(); ++feature)
        {
            for (std::size_t grid = 0; grid < grids; ++grid)
            {
                const std::size_t density_index = spin * grids + grid;
                const double rho = input.rho[density_index];
                const double sigma = input.sigma[2 * spin * grids + grid];
                const double tau = mgga ? input.tau[density_index] : 0.0;
                const NldfInterpolationArgument argument = interpolation_argument(
                    exponent(rho, sigma, tau, static_cast<int>(feature)));
                interpolation_coefficients(argument.value, coefficients, derivatives);
                double value = 0.0;
                for (std::size_t alpha = 0; alpha < alpha_count(); ++alpha)
                {
                    value += convolved[grid * alpha_count() + alpha]
                             * coefficients[alpha];
                }
                features[(spin * feature_count() + feature) * grids + grid]
                    = static_cast<double>(nspin_) * value;
            }
        }
    }
    std::copy(features.begin(), features.end(), output.features);
    if (output.interpolation_arguments != nullptr)
    {
        std::copy(arguments.begin(), arguments.end(), output.interpolation_arguments);
    }
    if (output.convolution_function != nullptr)
    {
        std::copy(function.begin(), function.end(), output.convolution_function);
    }
}

void NldfPlan::reverse(const NldfReverseInput& input,
                       NldfReverseOutput& output) const
{
    const std::size_t grids = grid_.grid_count();
    const std::size_t density_size
        = checked_product(nspin_, grids, "NLDF density size overflows");
    const std::size_t sigma_size
        = checked_product(2 * nspin_ - 1, grids, "NLDF sigma size overflows");
    const std::size_t feature_size = checked_product(
        checked_product(nspin_, feature_count(), "NLDF feature size overflows"),
        grids,
        "NLDF feature size overflows");
    const bool mgga = settings_.semilocal_level() == SemilocalLevel::Mgga;
    if (input.nspin != nspin_ || input.ngrids != grids
        || input.rho == nullptr || input.rho_size != density_size
        || input.sigma == nullptr || input.sigma_size != sigma_size
        || (mgga && (input.tau == nullptr || input.tau_size != density_size))
        || (!mgga && (input.tau != nullptr || input.tau_size != 0))
        || input.feature_adjoint == nullptr
        || input.feature_adjoint_size != feature_size
        || output.vrho == nullptr || output.vrho_size != density_size
        || output.vsigma == nullptr || output.vsigma_size != sigma_size
        || (mgga && (output.vtau == nullptr || output.vtau_size != density_size))
        || (!mgga && (output.vtau != nullptr || output.vtau_size != 0)))
    {
        throw std::invalid_argument("invalid NLDF reverse shape or buffer");
    }
    const BufferRange inputs[] = {
        buffer_range(input.rho, input.rho_size),
        buffer_range(input.sigma, input.sigma_size),
        buffer_range(input.tau, input.tau_size),
        buffer_range(input.feature_adjoint, input.feature_adjoint_size)};
    const BufferRange outputs[] = {buffer_range(output.vrho, output.vrho_size),
                                   buffer_range(output.vsigma, output.vsigma_size),
                                   buffer_range(output.vtau, output.vtau_size)};
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            if (overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("NLDF reverse inputs and outputs must be disjoint");
        }
        for (std::size_t j = i + 1; j < 3; ++j)
        {
            if (overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("NLDF reverse outputs must be disjoint");
        }
    }
    for (std::size_t i = 0; i < density_size; ++i)
    {
        if (!std::isfinite(input.rho[i]) || !std::isfinite(output.vrho[i])
            || (mgga && (!std::isfinite(input.tau[i]) || !std::isfinite(output.vtau[i]))))
        {
            throw std::invalid_argument("NLDF reverse density buffers must be finite");
        }
    }
    for (std::size_t i = 0; i < sigma_size; ++i)
    {
        if (!std::isfinite(input.sigma[i]) || !std::isfinite(output.vsigma[i]))
        {
            throw std::invalid_argument("NLDF reverse sigma buffers must be finite");
        }
    }
    for (std::size_t i = 0; i < feature_size; ++i)
    {
        if (!std::isfinite(input.feature_adjoint[i]))
        {
            throw std::invalid_argument("NLDF feature adjoint must be finite");
        }
    }

    std::vector<double> vrho(output.vrho, output.vrho + density_size);
    std::vector<double> vsigma(output.vsigma, output.vsigma + sigma_size);
    std::vector<double> vtau;
    if (mgga) vtau.assign(output.vtau, output.vtau + density_size);
    std::vector<double> source(
        checked_product(grids, alpha_count(), "NLDF work size overflows"));
    std::vector<double> convolved(source.size());
    std::vector<double> potential_source(source.size());
    std::vector<double> potential_convolved(source.size());
    std::vector<double> coefficients, derivatives;
    std::vector<NldfInterpolationArgument> theta_arguments(grids);
    std::vector<double> function(grids);
    for (std::size_t spin = 0; spin < nspin_; ++spin)
    {
        for (std::size_t grid = 0; grid < grids; ++grid)
        {
            const std::size_t density_index = spin * grids + grid;
            const double rho = input.rho[density_index];
            const double sigma = input.sigma[2 * spin * grids + grid];
            const double tau = mgga ? input.tau[density_index] : 0.0;
            theta_arguments[grid]
                = interpolation_argument(exponent(rho, sigma, tau, -1));
            function[grid] = rho < rho_cutoff_ ? 0.0 : rho;
            interpolation_coefficients(
                theta_arguments[grid].value, coefficients, derivatives);
            for (std::size_t alpha = 0; alpha < alpha_count(); ++alpha)
            {
                source[grid * alpha_count() + alpha]
                    = function[grid] * coefficients[alpha];
            }
        }
        backend_->convolve(source.data(), source.size(), convolved.data(), convolved.size());
        std::fill(potential_source.begin(), potential_source.end(), 0.0);
        for (std::size_t feature = 0; feature < feature_count(); ++feature)
        {
            for (std::size_t grid = 0; grid < grids; ++grid)
            {
                const std::size_t density_index = spin * grids + grid;
                const double rho = input.rho[density_index];
                const double sigma = input.sigma[2 * spin * grids + grid];
                const double tau = mgga ? input.tau[density_index] : 0.0;
                const NldfInterpolationArgument argument = interpolation_argument(
                    exponent(rho, sigma, tau, static_cast<int>(feature)));
                interpolation_coefficients(argument.value, coefficients, derivatives);
                double feature_exponent_derivative = 0.0;
                for (std::size_t alpha = 0; alpha < alpha_count(); ++alpha)
                {
                    feature_exponent_derivative
                        += convolved[grid * alpha_count() + alpha] * derivatives[alpha];
                }
                const double adjoint
                    = static_cast<double>(nspin_)
                      * input.feature_adjoint[(spin * feature_count() + feature) * grids + grid];
                const double local = feature_exponent_derivative * adjoint;
                vrho[density_index] += argument.drho * local;
                vsigma[2 * spin * grids + grid] += argument.dsigma * local;
                if (mgga) vtau[density_index] += argument.dtau * local;
                for (std::size_t alpha = 0; alpha < alpha_count(); ++alpha)
                {
                    potential_source[grid * alpha_count() + alpha]
                        += adjoint * coefficients[alpha];
                }
            }
        }
        backend_->convolve(potential_source.data(), potential_source.size(),
                           potential_convolved.data(), potential_convolved.size());
        for (std::size_t grid = 0; grid < grids; ++grid)
        {
            const std::size_t density_index = spin * grids + grid;
            interpolation_coefficients(
                theta_arguments[grid].value, coefficients, derivatives);
            double density_adjoint = 0.0;
            double argument_adjoint = 0.0;
            for (std::size_t alpha = 0; alpha < alpha_count(); ++alpha)
            {
                density_adjoint += potential_convolved[grid * alpha_count() + alpha]
                                   * coefficients[alpha];
                argument_adjoint += potential_convolved[grid * alpha_count() + alpha]
                                    * derivatives[alpha];
            }
            argument_adjoint *= function[grid];
            vrho[density_index]
                += theta_arguments[grid].drho * argument_adjoint
                   + (input.rho[density_index] < rho_cutoff_ ? 0.0 : density_adjoint);
            vsigma[2 * spin * grids + grid]
                += theta_arguments[grid].dsigma * argument_adjoint;
            if (mgga)
            {
                vtau[density_index]
                    += theta_arguments[grid].dtau * argument_adjoint;
            }
        }
    }
    for (std::size_t i = 0; i < density_size; ++i)
    {
        if (!std::isfinite(vrho[i]) || (mgga && !std::isfinite(vtau[i])))
        {
            throw std::runtime_error("NLDF reverse result is not finite");
        }
    }
    for (std::size_t i = 0; i < sigma_size; ++i)
    {
        if (!std::isfinite(vsigma[i]))
        {
            throw std::runtime_error("NLDF reverse result is not finite");
        }
    }
    std::copy(vrho.begin(), vrho.end(), output.vrho);
    std::copy(vsigma.begin(), vsigma.end(), output.vsigma);
    if (mgga) std::copy(vtau.begin(), vtau.end(), output.vtau);
}

} // namespace cider
} // namespace hamilt
