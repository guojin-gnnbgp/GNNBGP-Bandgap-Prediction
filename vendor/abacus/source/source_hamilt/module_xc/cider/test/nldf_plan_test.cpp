#include "source_hamilt/module_xc/cider/nldf_convolution.h"
#include "source_hamilt/module_xc/cider/nldf_plan.h"
#include "source_hamilt/module_xc/cider/semilocal_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

hamilt::cider::NldfSettings gga_settings()
{
    typedef hamilt::cider::ParameterTriple P;
    return hamilt::cider::NldfSettings(
        hamilt::cider::SemilocalLevel::Gga,
        P(0.25, 0.0078125, 0.0),
        std::vector<P>{P(0.25, 0.0078125, 0.0),
                       P(0.5, 0.015625, 0.0),
                       P(1.0, 0.03125, 0.0)});
}

hamilt::cider::NldfGrid fixture_grid()
{
    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    return hamilt::cider::NldfGrid(shape, cell);
}

std::string fixture_path()
{
    const std::string source(__FILE__);
    return source.substr(0, source.find_last_of('/')) + "/fixtures/v1/arrays.bin";
}

std::vector<double> read_fixture_array(std::size_t offset, std::size_t count)
{
    std::ifstream input(fixture_path().c_str(), std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("failed to open native CIDER v1 fixture");
    }
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<double> values(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(double)));
    if (!input)
    {
        throw std::runtime_error("failed to read native CIDER v1 fixture array");
    }
    return values;
}

hamilt::cider::NldfSettings mgga_settings()
{
    typedef hamilt::cider::ParameterTriple P;
    return hamilt::cider::NldfSettings(
        hamilt::cider::SemilocalLevel::Mgga,
        P(0.5, 0.0, 0.015625),
        std::vector<P>{P(0.25, 0.0, 0.0078125),
                       P(0.5, 0.0, 0.015625),
                       P(1.0, 0.0, 0.03125)});
}

std::vector<double> nldf_features(const hamilt::cider::NldfPlan& plan,
                                  std::size_t nspin,
                                  const std::vector<double>& rho,
                                  const std::vector<double>& sigma,
                                  const std::vector<double>& tau)
{
    const std::size_t grids = rho.size() / nspin;
    std::vector<double> result(nspin * plan.feature_count() * grids);
    const hamilt::cider::NldfForwardInput input(
        nspin, grids, rho.data(), rho.size(), sigma.data(), sigma.size(),
        tau.empty() ? nullptr : tau.data(), tau.size());
    hamilt::cider::NldfForwardOutput output(result.data(), result.size());
    plan.forward(input, output);
    return result;
}

double nldf_contraction(const hamilt::cider::NldfPlan& plan,
                        std::size_t nspin,
                        const std::vector<double>& rho,
                        const std::vector<double>& sigma,
                        const std::vector<double>& tau,
                        const std::vector<double>& adjoint)
{
    const std::vector<double> features
        = nldf_features(plan, nspin, rho, sigma, tau);
    double result = 0.0;
    for (std::size_t i = 0; i < features.size(); ++i) result += features[i] * adjoint[i];
    return result;
}

TEST(NativeCiderNldfConvolution, SymmetricKernelMatchesFullKernel)
{
    const int nalpha = 3;
    const double norms[] = {1.0, 2.0, 3.0,
                            2.0, 4.0, 5.0,
                            3.0, 5.0, 6.0};
    const double exponents[] = {0.1, 0.2, 0.3,
                                0.2, 0.4, 0.5,
                                0.3, 0.5, 0.6};
    std::vector<double> full(9);
    std::vector<double> symmetric(9, -1.0);

    abacus_cider_nldf_compute_kernel(
        0.75, full.data(), norms, exponents, 9);
    abacus_cider_nldf_compute_kernel_symmetric(
        0.75, symmetric.data(), norms, exponents, nalpha);

    EXPECT_EQ(full, symmetric);
}

TEST(NativeCiderNldfPlan, SelectsExactAlphaGridAndNormalization)
{
    const hamilt::cider::NldfPlan plan(
        gga_settings(), 2, fixture_grid(),
        hamilt::cider::NldfAlphaSelection(20.0, 1.8, 8));
    const double expected_alpha[] = {
        0.32667993457620148, 0.58802388223716273,
        1.0584429880268929, 1.9051973784484073,
        3.4293552812071328, 6.1728395061728394,
        11.111111111111111, 20.0};
    const double expected_norm[] = {
        0.30796540783572557, 0.47858195120805863,
        0.74372211357025908, 1.1557531177621634,
        1.7960542584979513, 2.7910899394453983,
        4.3373873663417504, 6.7403521827889366};
    ASSERT_EQ(8U, plan.alpha_count());
    for (std::size_t i = 0; i < plan.alpha_count(); ++i)
    {
        EXPECT_NEAR(expected_alpha[i], plan.alphas()[i], 1.0e-14);
        EXPECT_NEAR(expected_norm[i], plan.alpha_norms()[i], 1.0e-14);
    }

    const hamilt::cider::NldfPlan automatic(
        gga_settings(), 2, fixture_grid(),
        hamilt::cider::NldfAlphaSelection(20.0, 1.8, 0));
    EXPECT_EQ(16U, automatic.alpha_count());
    EXPECT_NEAR(1.7671957730013215, automatic.lambda(), 1.0e-15);
    EXPECT_NEAR(0.00390625, automatic.alpha0(), 1.0e-17);
}

TEST(NativeCiderNldfPlan, MatchesPythonScalarExponentArgumentAndSplineCoefficients)
{
    const hamilt::cider::NldfPlan plan(
        gga_settings(), 2, fixture_grid(),
        hamilt::cider::NldfAlphaSelection(20.0, 1.8, 8));
    const hamilt::cider::NldfExponentValue exponent
        = plan.exponent(0.0565, 0.00327, 0.0, 2);
    EXPECT_NEAR(0.47858057497914386, exponent.value, 1.0e-14);
    EXPECT_NEAR(4.8915437594493802, exponent.drho, 1.0e-13);
    EXPECT_NEAR(4.8946667710862251, exponent.dsigma, 1.0e-13);
    EXPECT_DOUBLE_EQ(0.0, exponent.dtau);

    const hamilt::cider::NldfInterpolationArgument argument
        = plan.interpolation_argument(exponent);
    EXPECT_NEAR(0.64962972462156643, argument.value, 1.0e-14);
    EXPECT_NEAR(17.388861439907874, argument.drho, 1.0e-12);
    EXPECT_NEAR(17.399963378130018, argument.dsigma, 1.0e-12);

    std::vector<double> coefficients, derivatives;
    plan.interpolation_coefficients(argument.value, coefficients, derivatives);
    const double expected[] = {
        0.8110079421896037, 1.8341754077181447,
        -0.21748246794445009, 0.037498303060092303,
        -0.0064633915767970396, 0.0011091105116832643,
        -0.00017842694976441653, 0.000019136161255272791};
    const double expected_derivative[] = {
        -3.0156322908469946, 1.7516726345806006,
        0.1541055373990049, -0.026570859707653688,
        0.0045798838029414007, -0.0007859027582926433,
        0.00012643125323984233, -0.000013559660426288096};
    ASSERT_EQ(8U, coefficients.size());
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
        EXPECT_NEAR(expected[i], coefficients[i], 2.0e-13);
        EXPECT_NEAR(expected_derivative[i], derivatives[i], 2.0e-13);
    }
}

TEST(NativeCiderNldfPlan, ReusesInterpolationCoefficientBuffers)
{
    const hamilt::cider::NldfPlan plan(
        gga_settings(), 2, fixture_grid(),
        hamilt::cider::NldfAlphaSelection(20.0, 1.8, 8));
    std::vector<double> coefficients, derivatives;
    plan.interpolation_coefficients(0.25, coefficients, derivatives);
    const double* coefficient_data = coefficients.data();
    const double* derivative_data = derivatives.data();

    plan.interpolation_coefficients(0.75, coefficients, derivatives);

    EXPECT_EQ(coefficient_data, coefficients.data());
    EXPECT_EQ(derivative_data, derivatives.data());
    EXPECT_EQ(8U, coefficients.size());
    EXPECT_EQ(8U, derivatives.size());
}

TEST(NativeCiderNldfPlan, MatchesAllThreeV1ForwardOraclesAndIntermediates)
{
    struct Case
    {
        bool mgga;
        std::size_t nspin;
        std::size_t rho_offset;
        std::size_t sigma_offset;
        std::size_t tau_offset;
        std::size_t argument_offset;
        std::size_t function_offset;
        std::size_t feature_offset;
        std::size_t argument_derivative_offset;
        std::size_t function_derivative_offset;
    };
    const Case cases[] = {
        {false, 2, 9140, 9268, 0, 14132, 14260, 14388, 14772, 15028},
        {true, 1, 18524, 18588, 18652, 21596, 21660, 21724, 21916, 22108},
        {true, 2, 25412, 25540, 25732, 31556, 31684, 31812, 32196, 32580},
    };
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    for (const Case& item : cases)
    {
        const std::size_t grids = 8;
        const std::size_t rho_size = item.nspin * grids;
        const std::size_t sigma_size = (2 * item.nspin - 1) * grids;
        const std::vector<double> rho = read_fixture_array(item.rho_offset, rho_size);
        const std::vector<double> sigma
            = read_fixture_array(item.sigma_offset, sigma_size);
        const std::vector<double> tau
            = item.mgga ? read_fixture_array(item.tau_offset, rho_size)
                        : std::vector<double>();
        const std::vector<double> expected_arguments
            = read_fixture_array(item.argument_offset, rho_size);
        const std::vector<double> expected_function
            = read_fixture_array(item.function_offset, rho_size);
        const std::vector<double> expected_features
            = read_fixture_array(item.feature_offset, item.nspin * 3 * grids);
        const std::size_t component_count = item.mgga ? 3 : 2;
        const std::vector<double> expected_argument_derivatives
            = read_fixture_array(item.argument_derivative_offset,
                                 component_count * rho_size);
        const std::vector<double> expected_function_derivatives
            = read_fixture_array(item.function_derivative_offset,
                                 component_count * rho_size);
        std::vector<double> arguments(rho_size, -1.0);
        std::vector<double> function(rho_size, -1.0);
        std::vector<double> features(expected_features.size(), -1.0);

        const hamilt::cider::NldfPlan plan(
            item.mgga ? mgga_settings() : gga_settings(),
            item.nspin,
            fixture_grid(),
            hamilt::cider::NldfAlphaSelection(20.0, 1.8, 8));
        const hamilt::cider::NldfForwardInput input(
            item.nspin, grids,
            rho.data(), rho.size(), sigma.data(), sigma.size(),
            tau.empty() ? nullptr : tau.data(), tau.size());
        hamilt::cider::NldfForwardOutput output(
            features.data(), features.size(),
            arguments.data(), arguments.size(), function.data(), function.size());
        plan.forward(input, output);

        for (std::size_t i = 0; i < rho_size; ++i)
        {
            EXPECT_NEAR(expected_arguments[i], arguments[i], 1.0e-12);
            EXPECT_NEAR(expected_function[i], function[i], 1.0e-12);
            const std::size_t spin = i / grids;
            const std::size_t grid = i % grids;
            const double tau_value = item.mgga ? tau[i] : 0.0;
            const hamilt::cider::NldfInterpolationArgument derivative
                = plan.interpolation_argument(plan.exponent(
                    rho[i], sigma[2 * spin * grids + grid], tau_value, -1));
            const double actual_derivatives[]
                = {derivative.drho, derivative.dsigma, derivative.dtau};
            for (std::size_t component = 0; component < component_count; ++component)
            {
                const std::size_t index = component * rho_size + i;
                EXPECT_NEAR(expected_argument_derivatives[index],
                            actual_derivatives[component], 1.0e-12);
                const double expected_function_derivative
                    = component == 0 ? 1.0 : 0.0;
                EXPECT_DOUBLE_EQ(expected_function_derivative,
                                 expected_function_derivatives[index]);
            }
        }
        for (std::size_t i = 0; i < features.size(); ++i)
        {
            const double error = std::abs(features[i] - expected_features[i]);
            max_absolute_error = std::max(max_absolute_error, error);
            EXPECT_NEAR(expected_features[i], features[i], 1.0e-12);
            if (expected_features[i] != 0.0)
            {
                const double relative = error / std::abs(expected_features[i]);
                max_relative_error = std::max(max_relative_error, relative);
                EXPECT_LE(relative, 1.0e-10);
            }
        }
    }
    RecordProperty("max_forward_absolute_error", max_absolute_error);
    RecordProperty("max_forward_relative_error", max_relative_error);
}

TEST(NativeCiderNldfPlan, MatchesAllThreeV1ReverseOracles)
{
    struct Case
    {
        bool mgga;
        std::size_t nspin;
        std::size_t rho_offset, sigma_offset, tau_offset;
        std::size_t adjoint_offset;
        std::size_t vrho_offset, vsigma_offset, vtau_offset;
    };
    const Case cases[] = {
        {false, 2, 9140, 9268, 0, 13492, 15732, 15860, 0},
        {true, 1, 18524, 18588, 18652, 21212, 22620, 22684, 22748},
        {true, 2, 25412, 25540, 25732, 30788, 33540, 33668, 33860},
    };
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    for (const Case& item : cases)
    {
        const std::size_t grids = 8;
        const std::size_t semilocal_count = item.mgga ? 3 : 2;
        const std::size_t total_count = semilocal_count + 3;
        const std::size_t rho_size = item.nspin * grids;
        const std::size_t sigma_size = (2 * item.nspin - 1) * grids;
        const std::vector<double> rho = read_fixture_array(item.rho_offset, rho_size);
        const std::vector<double> sigma
            = read_fixture_array(item.sigma_offset, sigma_size);
        const std::vector<double> tau
            = item.mgga ? read_fixture_array(item.tau_offset, rho_size)
                        : std::vector<double>();
        std::vector<double> full_adjoint
            = read_fixture_array(item.adjoint_offset, item.nspin * total_count * grids);
        for (std::size_t i = 0; i < full_adjoint.size(); ++i) full_adjoint[i] *= 0.25;
        std::vector<double> semilocal_adjoint(item.nspin * semilocal_count * grids);
        std::vector<double> nldf_adjoint(item.nspin * 3 * grids);
        for (std::size_t spin = 0; spin < item.nspin; ++spin)
        {
            std::copy(full_adjoint.begin() + spin * total_count * grids,
                      full_adjoint.begin() + (spin * total_count + semilocal_count) * grids,
                      semilocal_adjoint.begin() + spin * semilocal_count * grids);
            std::copy(full_adjoint.begin() + (spin * total_count + semilocal_count) * grids,
                      full_adjoint.begin() + (spin + 1) * total_count * grids,
                      nldf_adjoint.begin() + spin * 3 * grids);
        }
        std::vector<double> vrho(rho_size, 0.0), vsigma(sigma_size, 0.0);
        std::vector<double> vtau(item.mgga ? rho_size : 0, 0.0);
        const hamilt::cider::SemilocalMode mode
            = item.mgga ? hamilt::cider::SemilocalMode::Npa
                        : hamilt::cider::SemilocalMode::Np;
        const hamilt::cider::SemilocalPlan semilocal(mode, item.nspin);
        const hamilt::cider::SemilocalReverseInput semilocal_input(
            item.nspin, grids, rho.data(), rho.size(), sigma.data(), sigma.size(),
            tau.empty() ? nullptr : tau.data(), tau.size(),
            semilocal_adjoint.data(), semilocal_adjoint.size());
        hamilt::cider::SemilocalReverseOutput semilocal_output(
            vrho.data(), vrho.size(), vsigma.data(), vsigma.size(),
            vtau.empty() ? nullptr : vtau.data(), vtau.size());
        semilocal.reverse(semilocal_input, semilocal_output);

        const hamilt::cider::NldfPlan plan(
            item.mgga ? mgga_settings() : gga_settings(), item.nspin,
            fixture_grid(), hamilt::cider::NldfAlphaSelection(20.0, 1.8, 8));
        const hamilt::cider::NldfReverseInput reverse_input(
            item.nspin, grids, rho.data(), rho.size(), sigma.data(), sigma.size(),
            tau.empty() ? nullptr : tau.data(), tau.size(),
            nldf_adjoint.data(), nldf_adjoint.size());
        hamilt::cider::NldfReverseOutput reverse_output(
            vrho.data(), vrho.size(), vsigma.data(), vsigma.size(),
            vtau.empty() ? nullptr : vtau.data(), vtau.size());
        plan.reverse(reverse_input, reverse_output);

        const std::vector<double> expected_rho
            = read_fixture_array(item.vrho_offset, rho_size);
        const std::vector<double> expected_sigma
            = read_fixture_array(item.vsigma_offset, sigma_size);
        const std::vector<double> expected_tau
            = item.mgga ? read_fixture_array(item.vtau_offset, rho_size)
                        : std::vector<double>();
        const std::vector<const std::vector<double>*> actuals
            = {&vrho, &vsigma, &vtau};
        const std::vector<const std::vector<double>*> expecteds
            = {&expected_rho, &expected_sigma, &expected_tau};
        for (std::size_t array = 0; array < actuals.size(); ++array)
        {
            for (std::size_t i = 0; i < actuals[array]->size(); ++i)
            {
                const double expected = (*expecteds[array])[i];
                const double error = std::abs((*actuals[array])[i] - expected);
                max_absolute_error = std::max(max_absolute_error, error);
                EXPECT_NEAR(expected, (*actuals[array])[i], 1.0e-12);
                if (expected != 0.0)
                {
                    const double relative = error / std::abs(expected);
                    max_relative_error = std::max(max_relative_error, relative);
                    EXPECT_LE(relative, 1.0e-10);
                }
            }
        }
    }
    RecordProperty("max_reverse_absolute_error", max_absolute_error);
    RecordProperty("max_reverse_relative_error", max_relative_error);
}

TEST(NativeCiderNldfPlan, ReverseMatchesFiniteDifferencesAndAdjointDotProduct)
{
    const std::size_t nspin = 2, grids = 8;
    std::vector<double> rho = read_fixture_array(25412, nspin * grids);
    std::vector<double> sigma = read_fixture_array(25540, 3 * grids);
    std::vector<double> tau = read_fixture_array(25732, nspin * grids);
    std::vector<double> adjoint(nspin * 3 * grids);
    for (std::size_t i = 0; i < adjoint.size(); ++i)
        adjoint[i] = 0.015 * static_cast<double>(static_cast<int>(i % 11) - 5);
    const hamilt::cider::NldfPlan plan(
        mgga_settings(), nspin, fixture_grid(),
        hamilt::cider::NldfAlphaSelection(20.0, 1.8, 8));
    std::vector<double> vrho(rho.size(), 0.0), vsigma(sigma.size(), 0.0),
        vtau(tau.size(), 0.0);
    const hamilt::cider::NldfReverseInput input(
        nspin, grids, rho.data(), rho.size(), sigma.data(), sigma.size(),
        tau.data(), tau.size(), adjoint.data(), adjoint.size());
    hamilt::cider::NldfReverseOutput output(
        vrho.data(), vrho.size(), vsigma.data(), vsigma.size(),
        vtau.data(), vtau.size());
    plan.reverse(input, output);

    const double step = 1.0e-6;
    double max_fd_error = 0.0;
    struct Probe { std::vector<double>* values; const std::vector<double>* gradient; std::size_t index; };
    Probe probes[] = {{&rho, &vrho, 3}, {&sigma, &vsigma, 2}, {&tau, &vtau, 11}};
    for (Probe& probe : probes)
    {
        (*probe.values)[probe.index] += step;
        const double plus = nldf_contraction(plan, nspin, rho, sigma, tau, adjoint);
        (*probe.values)[probe.index] -= 2.0 * step;
        const double minus = nldf_contraction(plan, nspin, rho, sigma, tau, adjoint);
        (*probe.values)[probe.index] += step;
        const double fd = (plus - minus) / (2.0 * step);
        const double error = std::abs(fd - (*probe.gradient)[probe.index]);
        max_fd_error = std::max(max_fd_error, error);
        EXPECT_NEAR(fd, (*probe.gradient)[probe.index], 2.0e-9);
    }
    EXPECT_DOUBLE_EQ(0.0, vsigma[grids + 2]);

    std::vector<double> rho_direction(rho.size()), sigma_direction(sigma.size()),
        tau_direction(tau.size());
    for (std::size_t i = 0; i < rho.size(); ++i)
    {
        rho_direction[i] = 0.003 * static_cast<double>(static_cast<int>(i % 5) - 2);
        tau_direction[i] = 0.004 * static_cast<double>(static_cast<int>(i % 7) - 3);
    }
    for (std::size_t i = 0; i < sigma.size(); ++i)
        sigma_direction[i] = (i / grids == 1) ? 0.0
                              : 0.002 * static_cast<double>(static_cast<int>(i % 3) - 1);
    std::vector<double> rho_plus(rho), rho_minus(rho), sigma_plus(sigma),
        sigma_minus(sigma), tau_plus(tau), tau_minus(tau);
    for (std::size_t i = 0; i < rho.size(); ++i)
    {
        rho_plus[i] += step * rho_direction[i]; rho_minus[i] -= step * rho_direction[i];
        tau_plus[i] += step * tau_direction[i]; tau_minus[i] -= step * tau_direction[i];
    }
    for (std::size_t i = 0; i < sigma.size(); ++i)
    {
        sigma_plus[i] += step * sigma_direction[i];
        sigma_minus[i] -= step * sigma_direction[i];
    }
    const double finite_dot
        = (nldf_contraction(plan, nspin, rho_plus, sigma_plus, tau_plus, adjoint)
           - nldf_contraction(plan, nspin, rho_minus, sigma_minus, tau_minus, adjoint))
          / (2.0 * step);
    double analytic_dot = 0.0;
    for (std::size_t i = 0; i < rho.size(); ++i)
        analytic_dot += vrho[i] * rho_direction[i] + vtau[i] * tau_direction[i];
    for (std::size_t i = 0; i < sigma.size(); ++i)
        analytic_dot += vsigma[i] * sigma_direction[i];
    EXPECT_NEAR(finite_dot, analytic_dot, 2.0e-10);
    RecordProperty("max_fd_absolute_error", max_fd_error);
    RecordProperty("adjoint_dot_absolute_error", std::abs(finite_dot - analytic_dot));
}

TEST(NativeCiderNldfPlan, RejectsAliasesAndPreservesLateOutputs)
{
    const hamilt::cider::NldfPlan plan(
        gga_settings(), 2, fixture_grid(),
        hamilt::cider::NldfAlphaSelection(20.0, 1.8, 8));
    const std::vector<double> rho_fixture = read_fixture_array(9140, 16);
    std::vector<double> rho(48, 0.0);
    std::copy(rho_fixture.begin(), rho_fixture.end(), rho.begin());
    const std::vector<double> sigma = read_fixture_array(9268, 24);
    const hamilt::cider::NldfForwardInput forward_input(
        2, 8, rho.data(), 16, sigma.data(), sigma.size(), nullptr, 0);
    hamilt::cider::NldfForwardOutput aliased_output(rho.data(), 48);
    EXPECT_THROW(plan.forward(forward_input, aliased_output), std::invalid_argument);

    std::vector<double> adjoint(48, 0.1);
    adjoint.back() = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> vrho(16, 7.0), vsigma(24, 9.0);
    const hamilt::cider::NldfReverseInput reverse_input(
        2, 8, rho.data(), 16, sigma.data(), sigma.size(), nullptr, 0,
        adjoint.data(), adjoint.size());
    hamilt::cider::NldfReverseOutput reverse_output(
        vrho.data(), vrho.size(), vsigma.data(), vsigma.size(), nullptr, 0);
    EXPECT_THROW(plan.reverse(reverse_input, reverse_output), std::invalid_argument);
    EXPECT_EQ(std::vector<double>(16, 7.0), vrho);
    EXPECT_EQ(std::vector<double>(24, 9.0), vsigma);
}

TEST(NativeCiderNldfBackend, RejectsPartialOverlapWithoutMutation)
{
    const std::vector<double> alphas = {0.5, 1.0};
    const std::vector<double> norms = {0.75, 1.25};
    const hamilt::cider::NldfBackend backend(fixture_grid(), alphas, norms);
    std::vector<double> storage(17);
    for (std::size_t i = 0; i < storage.size(); ++i)
        storage[i] = 0.1 * static_cast<double>(i + 1);
    const std::vector<double> original(storage);
    EXPECT_THROW(backend.convolve(storage.data(), 16, storage.data() + 1, 16),
                 std::invalid_argument);
    EXPECT_EQ(original, storage);
}

TEST(NativeCiderNldfBackend, RejectsInvalidAlphaAndNormData)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(hamilt::cider::NldfBackend(
                     fixture_grid(), std::vector<double>{0.0, 1.0},
                     std::vector<double>{1.0, 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::NldfBackend(
                     fixture_grid(), std::vector<double>{-0.5, 1.0},
                     std::vector<double>{1.0, 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::NldfBackend(
                     fixture_grid(), std::vector<double>{nan, 1.0},
                     std::vector<double>{1.0, 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::NldfBackend(
                     fixture_grid(), std::vector<double>{0.5, inf},
                     std::vector<double>{1.0, 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::NldfBackend(
                     fixture_grid(), std::vector<double>{0.5, 1.0},
                     std::vector<double>{0.0, 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::NldfBackend(
                     fixture_grid(), std::vector<double>{0.5, 1.0},
                     std::vector<double>{-0.25, 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::NldfBackend(
                     fixture_grid(), std::vector<double>{0.5, 1.0},
                     std::vector<double>{1.0, nan}),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::NldfBackend(
                     fixture_grid(), std::vector<double>{0.5, 1.0},
                     std::vector<double>{1.0, inf}),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::NldfBackend(
                     fixture_grid(), std::vector<double>{1.0e-300, 2.0e-300},
                     std::vector<double>{1.0e300, 1.0e300}),
                 std::invalid_argument);
}

TEST(NativeCiderNldfBackend, RejectsAddressAndByteSizeOverflowWithoutMutation)
{
    const hamilt::cider::NldfBackend backend(
        fixture_grid(), std::vector<double>{0.5, 1.0},
        std::vector<double>{0.75, 1.25});
    const std::vector<double> input(16, 0.25);
    std::vector<double> output(16, 4.5);
    EXPECT_THROW(backend.convolve(input.data(),
                                  std::numeric_limits<std::size_t>::max(),
                                  output.data(),
                                  output.size()),
                 std::invalid_argument);
    EXPECT_EQ(std::vector<double>(16, 4.5), output);

    const std::uintptr_t invalid_address
        = std::numeric_limits<std::uintptr_t>::max() - sizeof(double) + 1;
    double* invalid_output = reinterpret_cast<double*>(invalid_address);
    EXPECT_THROW(backend.convolve(input.data(), input.size(), invalid_output, 16),
                 std::invalid_argument);
    EXPECT_EQ(std::vector<double>(16, 4.5), output);
}

TEST(NativeCiderNldfBackend, ConcurrentInstancesCreateExecuteAndDestroyPlans)
{
    const std::vector<double> alphas = {0.5, 1.0};
    const std::vector<double> norms = {0.75, 1.25};
    const hamilt::cider::NldfBackend first(fixture_grid(), alphas, norms);
    const hamilt::cider::NldfBackend second(fixture_grid(), alphas, norms);
    std::vector<double> input(16);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.03 * static_cast<double>(i + 1);
    std::vector<double> expected(16);
    first.convolve(input.data(), input.size(), expected.data(), expected.size());

    std::exception_ptr first_error, second_error;
    const auto exercise = [&](const hamilt::cider::NldfBackend& backend,
                              std::exception_ptr& error) {
        try
        {
            for (int repetition = 0; repetition < 32; ++repetition)
            {
                std::vector<double> actual(16, -1.0);
                backend.convolve(
                    input.data(), input.size(), actual.data(), actual.size());
                if (actual != expected)
                    throw std::runtime_error("concurrent NLDF FFT result changed");
            }
        }
        catch (...)
        {
            error = std::current_exception();
        }
    };
    std::thread first_thread(exercise, std::cref(first), std::ref(first_error));
    std::thread second_thread(exercise, std::cref(second), std::ref(second_error));
    first_thread.join();
    second_thread.join();
    EXPECT_EQ(nullptr, first_error);
    EXPECT_EQ(nullptr, second_error);
}

} // namespace
