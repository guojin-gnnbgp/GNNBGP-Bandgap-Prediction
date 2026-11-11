#include "source_hamilt/module_xc/cider/semilocal_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{

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

std::vector<double> forward_features(hamilt::cider::SemilocalMode mode,
                                     std::size_t nspin,
                                     std::size_t ngrids,
                                     const std::vector<double>& rho,
                                     const std::vector<double>& sigma,
                                     const std::vector<double>& tau)
{
    const std::size_t nfeature
        = mode == hamilt::cider::SemilocalMode::Np ? 2 : 3;
    std::vector<double> features(nspin * nfeature * ngrids, -1.0);
    const hamilt::cider::SemilocalPlan plan(mode, nspin);
    const hamilt::cider::SemilocalForwardInput input(nspin,
                                                     ngrids,
                                                     rho.data(),
                                                     rho.size(),
                                                     sigma.data(),
                                                     sigma.size(),
                                                     tau.empty() ? nullptr : tau.data(),
                                                     tau.size());
    hamilt::cider::SemilocalForwardOutput output(features.data(), features.size());
    plan.forward(input, output);
    return features;
}

double feature_contraction(hamilt::cider::SemilocalMode mode,
                           std::size_t nspin,
                           std::size_t ngrids,
                           const std::vector<double>& rho,
                           const std::vector<double>& sigma,
                           const std::vector<double>& tau,
                           const std::vector<double>& adjoint)
{
    const std::vector<double> features
        = forward_features(mode, nspin, ngrids, rho, sigma, tau);
    double value = 0.0;
    for (std::size_t i = 0; i < features.size(); ++i)
    {
        value += features[i] * adjoint[i];
    }
    return value;
}

TEST(NativeCiderSemilocalPlan, ExposesConcreteForwardPlan)
{
    EXPECT_FALSE(std::is_abstract<hamilt::cider::SemilocalPlan>::value);
    const hamilt::cider::SemilocalPlan plan(hamilt::cider::SemilocalMode::Np, 1);
    EXPECT_EQ(2U, plan.feature_count());
}

TEST(NativeCiderSemilocalPlan, MatchesAllFiveV1RawFeatureOracles)
{
    struct Case
    {
        hamilt::cider::SemilocalMode mode;
        std::size_t nspin;
        std::size_t rho_offset;
        std::size_t sigma_offset;
        std::size_t tau_offset;
        std::size_t expected_offset;
    };
    const Case cases[] = {
        {hamilt::cider::SemilocalMode::Np, 1, 240, 304, 0, 368},
        {hamilt::cider::SemilocalMode::Npa, 2, 2604, 2732, 2924, 3052},
        {hamilt::cider::SemilocalMode::Np, 2, 9140, 9268, 0, 9460},
        {hamilt::cider::SemilocalMode::Npa, 1, 18524, 18588, 18652, 18716},
        {hamilt::cider::SemilocalMode::Npa, 2, 25412, 25540, 25732, 25860},
    };
    const std::size_t ngrids = 8;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    for (const Case& item : cases)
    {
        const std::size_t nfeature
            = item.mode == hamilt::cider::SemilocalMode::Np ? 2 : 3;
        const std::size_t rho_size = item.nspin * ngrids;
        const std::size_t sigma_size = (2 * item.nspin - 1) * ngrids;
        const std::vector<double> rho = read_fixture_array(item.rho_offset, rho_size);
        const std::vector<double> sigma
            = read_fixture_array(item.sigma_offset, sigma_size);
        const std::vector<double> tau
            = item.tau_offset == 0
                  ? std::vector<double>()
                  : read_fixture_array(item.tau_offset, rho_size);
        const std::vector<double> expected
            = read_fixture_array(item.expected_offset, item.nspin * nfeature * ngrids);
        std::vector<double> actual(expected.size(), -91.0);

        const hamilt::cider::SemilocalPlan plan(item.mode, item.nspin);
        const hamilt::cider::SemilocalForwardInput input(item.nspin,
                                                         ngrids,
                                                         rho.data(),
                                                         rho.size(),
                                                         sigma.data(),
                                                         sigma.size(),
                                                         tau.empty() ? nullptr : tau.data(),
                                                         tau.size());
        hamilt::cider::SemilocalForwardOutput output(actual.data(), actual.size());
        plan.forward(input, output);

        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            const double error = std::abs(actual[i] - expected[i]);
            max_absolute_error = std::max(max_absolute_error, error);
            EXPECT_NEAR(expected[i], actual[i], 1.0e-12)
                << "spin=" << item.nspin << " feature index=" << i;
            if (expected[i] != 0.0)
            {
                const double relative_error = error / std::abs(expected[i]);
                max_relative_error = std::max(max_relative_error, relative_error);
                EXPECT_LE(relative_error, 1.0e-11)
                    << "spin=" << item.nspin << " feature index=" << i;
            }
        }
    }
    RecordProperty("max_oracle_absolute_error", max_absolute_error);
    RecordProperty("max_oracle_relative_error", max_relative_error);
}

TEST(NativeCiderSemilocalPlan, GgaReverseAccumulatesAndLeavesCrossSigmaUntouched)
{
    const std::size_t nspin = 2;
    const std::size_t ngrids = 1;
    const std::vector<double> rho = {0.7, 1.1};
    const std::vector<double> sigma = {0.21, 93.0, 0.37};
    const std::vector<double> feature_adjoint = {1.2, -0.4, -0.8, 0.9};
    std::vector<double> vrho = {10.0, 20.0};
    std::vector<double> vsigma = {30.0, 40.0, 50.0};

    const hamilt::cider::SemilocalPlan plan(hamilt::cider::SemilocalMode::Np,
                                            nspin);
    const hamilt::cider::SemilocalReverseInput input(nspin,
                                                     ngrids,
                                                     rho.data(),
                                                     rho.size(),
                                                     sigma.data(),
                                                     sigma.size(),
                                                     nullptr,
                                                     0,
                                                     feature_adjoint.data(),
                                                     feature_adjoint.size());
    hamilt::cider::SemilocalReverseOutput output(vrho.data(),
                                                 vrho.size(),
                                                 vsigma.data(),
                                                 vsigma.size(),
                                                 nullptr,
                                                 0);
    plan.reverse(input, output);

    const double pi = 3.141592653589793238462643383279502884;
    const double b = 2.0 * std::pow(3.0 * pi * pi, 1.0 / 3.0);
    const double sm23 = std::pow(2.0, -2.0 / 3.0);
    for (std::size_t spin = 0; spin < nspin; ++spin)
    {
        const double denominator = b * std::pow(rho[spin], 4.0 / 3.0) + 1.0e-16;
        const double s = std::sqrt(sigma[2 * spin]) / denominator;
        const double dpdn = -8.0 * s * s / (3.0 * rho[spin] + 1.0e-16);
        const double dpdsigma = 1.0 / (denominator * denominator);
        const double expected_rho
            = (spin == 0 ? 10.0 : 20.0) + 2.0 * feature_adjoint[2 * spin]
              + sm23 * feature_adjoint[2 * spin + 1] * dpdn;
        const double expected_sigma
            = (spin == 0 ? 30.0 : 50.0)
              + sm23 * feature_adjoint[2 * spin + 1] * dpdsigma;
        EXPECT_NEAR(expected_rho, vrho[spin], 1.0e-14);
        EXPECT_NEAR(expected_sigma, vsigma[2 * spin], 1.0e-14);
    }
    EXPECT_DOUBLE_EQ(40.0, vsigma[1]);
}

TEST(NativeCiderSemilocalPlan, MatchesStrictDensityCutAndPythonNegativeBehavior)
{
    const double cutoff = 1.0e-10;
    const std::vector<double> rho = {0.0, std::nextafter(cutoff, 0.0), cutoff};
    const std::vector<double> sigma = {-3.0, -7.0, 0.4};
    const std::vector<double> tau = {-0.2, -0.3, -0.4};
    const std::vector<double> features = forward_features(
        hamilt::cider::SemilocalMode::Npa, 1, 3, rho, sigma, tau);
    EXPECT_DOUBLE_EQ(0.0, features[3]);
    EXPECT_DOUBLE_EQ(0.0, features[4]);
    EXPECT_GT(features[5], 0.0);
    EXPECT_DOUBLE_EQ(0.0, features[6]);
    EXPECT_DOUBLE_EQ(0.0, features[7]);
    EXPECT_LT(features[8], 0.0);

    const std::vector<double> adjoint(9, 1.0);
    std::vector<double> vrho(3, 2.0), vsigma(3, 3.0), vtau(3, 4.0);
    const hamilt::cider::SemilocalPlan plan(hamilt::cider::SemilocalMode::Npa, 1);
    const hamilt::cider::SemilocalReverseInput input(1,
                                                     3,
                                                     rho.data(),
                                                     rho.size(),
                                                     sigma.data(),
                                                     sigma.size(),
                                                     tau.data(),
                                                     tau.size(),
                                                     adjoint.data(),
                                                     adjoint.size());
    hamilt::cider::SemilocalReverseOutput output(vrho.data(),
                                                 vrho.size(),
                                                 vsigma.data(),
                                                 vsigma.size(),
                                                 vtau.data(),
                                                 vtau.size());
    plan.reverse(input, output);
    EXPECT_DOUBLE_EQ(3.0, vrho[0]);
    EXPECT_DOUBLE_EQ(3.0, vrho[1]);
    EXPECT_DOUBLE_EQ(3.0, vsigma[0]);
    EXPECT_DOUBLE_EQ(3.0, vsigma[1]);
    EXPECT_DOUBLE_EQ(4.0, vtau[0]);
    EXPECT_DOUBLE_EQ(4.0, vtau[1]);
    EXPECT_TRUE(std::isfinite(vrho[2]));
    EXPECT_TRUE(std::isfinite(vsigma[2]));
    EXPECT_TRUE(std::isfinite(vtau[2]));
}

TEST(NativeCiderSemilocalPlan, EnforcesTauModeAndPreservesOutputsOnErrors)
{
    const std::vector<double> rho = {0.4, 0.5};
    const std::vector<double> sigma = {0.2, -0.1};
    const std::vector<double> tau = {-0.3, -0.4};
    std::vector<double> features(4, 17.0);
    const hamilt::cider::SemilocalPlan gga(hamilt::cider::SemilocalMode::Np, 1);
    const hamilt::cider::SemilocalForwardInput bad_sigma_input(
        1, 2, rho.data(), rho.size(), sigma.data(), sigma.size());
    hamilt::cider::SemilocalForwardOutput feature_output(features.data(), features.size());
    EXPECT_THROW(gga.forward(bad_sigma_input, feature_output), std::domain_error);
    EXPECT_EQ(std::vector<double>(4, 17.0), features);

    const std::vector<double> positive_sigma = {0.2, 0.1};
    const hamilt::cider::SemilocalForwardInput gga_with_tau(1,
                                                           2,
                                                           rho.data(),
                                                           rho.size(),
                                                           positive_sigma.data(),
                                                           positive_sigma.size(),
                                                           tau.data(),
                                                           tau.size());
    EXPECT_THROW(gga.forward(gga_with_tau, feature_output), std::invalid_argument);
    EXPECT_EQ(std::vector<double>(4, 17.0), features);

    const hamilt::cider::SemilocalPlan mgga(hamilt::cider::SemilocalMode::Npa, 1);
    std::vector<double> mgga_features(6, 19.0);
    const hamilt::cider::SemilocalForwardInput mgga_without_tau(
        1, 2, rho.data(), rho.size(), positive_sigma.data(), positive_sigma.size());
    hamilt::cider::SemilocalForwardOutput mgga_output(mgga_features.data(),
                                                      mgga_features.size());
    EXPECT_THROW(mgga.forward(mgga_without_tau, mgga_output), std::invalid_argument);
    EXPECT_EQ(std::vector<double>(6, 19.0), mgga_features);

    const std::vector<double> adjoint(6, 0.7);
    std::vector<double> vrho(2, 23.0), vsigma(2, 29.0);
    const hamilt::cider::SemilocalReverseInput reverse_input(1,
                                                             2,
                                                             rho.data(),
                                                             rho.size(),
                                                             positive_sigma.data(),
                                                             positive_sigma.size(),
                                                             tau.data(),
                                                             tau.size(),
                                                             adjoint.data(),
                                                             adjoint.size());
    hamilt::cider::SemilocalReverseOutput reverse_without_vtau(
        vrho.data(), vrho.size(), vsigma.data(), vsigma.size(), nullptr, 0);
    EXPECT_THROW(mgga.reverse(reverse_input, reverse_without_vtau), std::invalid_argument);
    EXPECT_EQ(std::vector<double>(2, 23.0), vrho);
    EXPECT_EQ(std::vector<double>(2, 29.0), vsigma);

    const hamilt::cider::SemilocalReverseInput computational_error_input(
        1,
        2,
        rho.data(),
        rho.size(),
        sigma.data(),
        sigma.size(),
        tau.data(),
        tau.size(),
        adjoint.data(),
        adjoint.size());
    std::vector<double> vtau(2, 31.0);
    hamilt::cider::SemilocalReverseOutput computational_error_output(
        vrho.data(), vrho.size(), vsigma.data(), vsigma.size(), vtau.data(), vtau.size());
    EXPECT_THROW(mgga.reverse(computational_error_input, computational_error_output),
                 std::domain_error);
    EXPECT_EQ(std::vector<double>(2, 23.0), vrho);
    EXPECT_EQ(std::vector<double>(2, 29.0), vsigma);
    EXPECT_EQ(std::vector<double>(2, 31.0), vtau);
}

TEST(NativeCiderSemilocalPlan, RejectsOverlappingWritableRanges)
{
    const std::vector<double> rho = {0.4};
    const std::vector<double> sigma = {0.2};
    std::vector<double> storage = {1.0, 2.0};
    const std::vector<double> adjoint = {0.3, 0.4};
    const hamilt::cider::SemilocalPlan plan(hamilt::cider::SemilocalMode::Np, 1);
    const hamilt::cider::SemilocalReverseInput input(1,
                                                     1,
                                                     rho.data(),
                                                     rho.size(),
                                                     sigma.data(),
                                                     sigma.size(),
                                                     nullptr,
                                                     0,
                                                     adjoint.data(),
                                                     adjoint.size());
    hamilt::cider::SemilocalReverseOutput output(
        storage.data(), 1, storage.data(), 1, nullptr, 0);
    EXPECT_THROW(plan.reverse(input, output), std::invalid_argument);
    EXPECT_EQ((std::vector<double>{1.0, 2.0}), storage);
}

TEST(NativeCiderSemilocalPlan, RejectsForwardAliasesAndNonPrefixSizes)
{
    std::vector<double> shared = {0.4, 71.0};
    const std::vector<double> sigma = {0.2};
    const hamilt::cider::SemilocalPlan plan(hamilt::cider::SemilocalMode::Np, 1);
    const hamilt::cider::SemilocalForwardInput input(
        1, 1, shared.data(), 1, sigma.data(), sigma.size());
    hamilt::cider::SemilocalForwardOutput aliased(shared.data(), shared.size());
    EXPECT_THROW(plan.forward(input, aliased), std::invalid_argument);
    EXPECT_EQ((std::vector<double>{0.4, 71.0}), shared);

    std::vector<double> oversized(3, 73.0);
    hamilt::cider::SemilocalForwardOutput nonprefix(oversized.data(), oversized.size());
    EXPECT_THROW(plan.forward(input, nonprefix), std::invalid_argument);
    EXPECT_EQ(std::vector<double>(3, 73.0), oversized);
}

TEST(NativeCiderSemilocalPlan, RejectsUnsupportedSpinAndMode)
{
    EXPECT_THROW((hamilt::cider::SemilocalPlan(hamilt::cider::SemilocalMode::Np, 0)),
                 std::invalid_argument);
    EXPECT_THROW((hamilt::cider::SemilocalPlan(hamilt::cider::SemilocalMode::Np, 3)),
                 std::invalid_argument);
    EXPECT_THROW((hamilt::cider::SemilocalPlan(
                     static_cast<hamilt::cider::SemilocalMode>(99), 1)),
                 std::invalid_argument);
}

TEST(NativeCiderSemilocalPlan, RejectsByteOverflowBeforeReadingBuffers)
{
    const std::size_t ngrids
        = std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
    const std::size_t feature_size = 2 * ngrids;
    double rho = 0.4;
    double sigma = 0.2;
    double features = 0.0;
    const hamilt::cider::SemilocalPlan plan(hamilt::cider::SemilocalMode::Np, 1);
    const hamilt::cider::SemilocalForwardInput input(
        1, ngrids, &rho, ngrids, &sigma, ngrids);
    hamilt::cider::SemilocalForwardOutput output(&features, feature_size);
    EXPECT_THROW(plan.forward(input, output), std::overflow_error);
    EXPECT_DOUBLE_EQ(0.0, features);

    double feature_adjoint = 0.3;
    double vrho = 5.0;
    double vsigma = 7.0;
    const hamilt::cider::SemilocalReverseInput reverse_input(1,
                                                             ngrids,
                                                             &rho,
                                                             ngrids,
                                                             &sigma,
                                                             ngrids,
                                                             nullptr,
                                                             0,
                                                             &feature_adjoint,
                                                             feature_size);
    hamilt::cider::SemilocalReverseOutput reverse_output(
        &vrho, ngrids, &vsigma, ngrids, nullptr, 0);
    EXPECT_THROW(plan.reverse(reverse_input, reverse_output), std::overflow_error);
    EXPECT_DOUBLE_EQ(5.0, vrho);
    EXPECT_DOUBLE_EQ(7.0, vsigma);
}

TEST(NativeCiderSemilocalPlan, RejectsHighAddressRangeOverflowBeforeReadingBuffers)
{
    const std::vector<double> rho = {0.4};
    const std::vector<double> sigma = {0.2};
    const std::uintptr_t invalid_address
        = std::numeric_limits<std::uintptr_t>::max() - sizeof(double) + 1;
    double* invalid_features = reinterpret_cast<double*>(invalid_address);
    const hamilt::cider::SemilocalPlan plan(hamilt::cider::SemilocalMode::Np, 1);
    const hamilt::cider::SemilocalForwardInput input(
        1, 1, rho.data(), rho.size(), sigma.data(), sigma.size());
    hamilt::cider::SemilocalForwardOutput output(invalid_features, 2);

    EXPECT_THROW(plan.forward(input, output), std::overflow_error);
    EXPECT_EQ((std::vector<double>{0.4}), rho);
    EXPECT_EQ((std::vector<double>{0.2}), sigma);
}

TEST(NativeCiderSemilocalPlan, MatchesSemilocalOnlyMappedReturnAdjointOracles)
{
    struct Case
    {
        const char* model;
        hamilt::cider::SemilocalMode mode;
        std::size_t nspin;
        std::size_t rho_offset;
        std::size_t sigma_offset;
        std::size_t tau_offset;
        std::size_t adjoint_offset;
        std::size_t vrho_offset;
        std::size_t vsigma_offset;
        std::size_t vtau_offset;
    };
    const Case cases[] = {
        {"functionals/CIDER23X_SL_GGA.yaml",
         hamilt::cider::SemilocalMode::Np,
         1,
         240,
         304,
         0,
         1264,
         1648,
         1712,
         0},
        {"functionals/CIDER23X_SL_MGGA.yaml",
         hamilt::cider::SemilocalMode::Npa,
         2,
         2604,
         2732,
         2924,
         5676,
         6636,
         6764,
         6956},
    };
    const std::size_t ngrids = 8;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    for (const Case& item : cases)
    {
        const std::size_t nfeature
            = item.mode == hamilt::cider::SemilocalMode::Np ? 2 : 3;
        const std::size_t rho_size = item.nspin * ngrids;
        const std::size_t sigma_size = (2 * item.nspin - 1) * ngrids;
        const std::vector<double> rho = read_fixture_array(item.rho_offset, rho_size);
        const std::vector<double> sigma
            = read_fixture_array(item.sigma_offset, sigma_size);
        const std::vector<double> tau
            = item.tau_offset == 0 ? std::vector<double>()
                                   : read_fixture_array(item.tau_offset, rho_size);
        std::vector<double> adjoint
            = read_fixture_array(item.adjoint_offset, item.nspin * nfeature * ngrids);
        // v1 stores the mapped return before bridge runtime scaling.  Both
        // stock semilocal models use the default trained scale 1.0, while the
        // fixture generation xmix is 0.25.
        const double runtime_scale = 0.25;
        for (double& value : adjoint)
        {
            value *= runtime_scale;
        }
        const std::vector<double> expected_vrho
            = read_fixture_array(item.vrho_offset, rho_size);
        const std::vector<double> expected_vsigma
            = read_fixture_array(item.vsigma_offset, sigma_size);
        const std::vector<double> expected_vtau
            = item.vtau_offset == 0 ? std::vector<double>()
                                    : read_fixture_array(item.vtau_offset, rho_size);
        std::vector<double> vrho(rho_size, 0.0), vsigma(sigma_size, 0.0);
        std::vector<double> vtau(tau.empty() ? 0 : rho_size, 0.0);
        const hamilt::cider::SemilocalPlan plan(item.mode, item.nspin);
        const hamilt::cider::SemilocalReverseInput input(item.nspin,
                                                         ngrids,
                                                         rho.data(),
                                                         rho.size(),
                                                         sigma.data(),
                                                         sigma.size(),
                                                         tau.empty() ? nullptr : tau.data(),
                                                         tau.size(),
                                                         adjoint.data(),
                                                         adjoint.size());
        hamilt::cider::SemilocalReverseOutput output(vrho.data(),
                                                     vrho.size(),
                                                     vsigma.data(),
                                                     vsigma.size(),
                                                     vtau.empty() ? nullptr : vtau.data(),
                                                     vtau.size());
        plan.reverse(input, output);
        for (std::size_t i = 0; i < vrho.size(); ++i)
        {
            const double error = std::abs(expected_vrho[i] - vrho[i]);
            max_absolute_error = std::max(max_absolute_error, error);
            if (expected_vrho[i] != 0.0)
            {
                max_relative_error
                    = std::max(max_relative_error, error / std::abs(expected_vrho[i]));
            }
            EXPECT_NEAR(expected_vrho[i], vrho[i], 1.0e-12) << item.model << " vrho " << i;
        }
        for (std::size_t i = 0; i < vsigma.size(); ++i)
        {
            const double error = std::abs(expected_vsigma[i] - vsigma[i]);
            max_absolute_error = std::max(max_absolute_error, error);
            if (expected_vsigma[i] != 0.0)
            {
                max_relative_error
                    = std::max(max_relative_error, error / std::abs(expected_vsigma[i]));
            }
            EXPECT_NEAR(expected_vsigma[i], vsigma[i], 1.0e-12)
                << item.model << " vsigma " << i;
        }
        for (std::size_t i = 0; i < vtau.size(); ++i)
        {
            const double error = std::abs(expected_vtau[i] - vtau[i]);
            max_absolute_error = std::max(max_absolute_error, error);
            if (expected_vtau[i] != 0.0)
            {
                max_relative_error
                    = std::max(max_relative_error, error / std::abs(expected_vtau[i]));
            }
            EXPECT_NEAR(expected_vtau[i], vtau[i], 1.0e-12) << item.model << " vtau " << i;
        }
    }
    RecordProperty("max_reverse_oracle_absolute_error", max_absolute_error);
    RecordProperty("max_reverse_oracle_relative_error", max_relative_error);
    EXPECT_LE(max_absolute_error, 1.0e-12);
    EXPECT_LE(max_relative_error, 1.0e-11);
}

TEST(NativeCiderSemilocalPlan, GgaReverseMatchesCentralFiniteDifferences)
{
    const std::size_t nspin = 1;
    const std::size_t ngrids = 2;
    std::vector<double> rho = {0.37, 0.81};
    std::vector<double> sigma = {0.05, 0.14};
    const std::vector<double> tau;
    const std::vector<double> adjoint = {0.2, -0.7, 0.4, 0.8};
    std::vector<double> vrho(rho.size(), 0.0), vsigma(sigma.size(), 0.0);
    const hamilt::cider::SemilocalPlan plan(hamilt::cider::SemilocalMode::Np,
                                            nspin);
    const hamilt::cider::SemilocalReverseInput input(nspin,
                                                     ngrids,
                                                     rho.data(),
                                                     rho.size(),
                                                     sigma.data(),
                                                     sigma.size(),
                                                     nullptr,
                                                     0,
                                                     adjoint.data(),
                                                     adjoint.size());
    hamilt::cider::SemilocalReverseOutput output(
        vrho.data(), vrho.size(), vsigma.data(), vsigma.size(), nullptr, 0);
    plan.reverse(input, output);
    double max_error = 0.0;
    const auto check_fd = [&](std::vector<double>& variable,
                              std::size_t index,
                              double analytic) {
        const double original = variable[index];
        const double step = 1.0e-6;
        variable[index] = original + step;
        const double plus = feature_contraction(hamilt::cider::SemilocalMode::Np,
                                                nspin,
                                                ngrids,
                                                rho,
                                                sigma,
                                                tau,
                                                adjoint);
        variable[index] = original - step;
        const double minus = feature_contraction(hamilt::cider::SemilocalMode::Np,
                                                 nspin,
                                                 ngrids,
                                                 rho,
                                                 sigma,
                                                 tau,
                                                 adjoint);
        variable[index] = original;
        max_error = std::max(max_error, std::abs((plus - minus) / (2.0 * step) - analytic));
    };
    for (std::size_t i = 0; i < rho.size(); ++i)
    {
        check_fd(rho, i, vrho[i]);
        check_fd(sigma, i, vsigma[i]);
    }
    RecordProperty("max_gga_fd_absolute_error", max_error);
    EXPECT_LE(max_error, 2.0e-9);
}

TEST(NativeCiderSemilocalPlan, MggaReverseMatchesCentralFiniteDifferences)
{
    const std::size_t nspin = 2;
    const std::size_t ngrids = 2;
    std::vector<double> rho = {0.31, 0.47, 0.59, 0.83};
    std::vector<double> sigma = {0.04, 0.07, 0.13, -0.21, 0.09, 0.12};
    std::vector<double> tau = {0.12, -0.18, 0.25, 0.32};
    const std::vector<double> adjoint = {0.2, -0.7, 0.4, 0.8, -0.3, 0.9,
                                         0.6, -0.5, 0.1, -0.2, 0.7, -0.4};
    std::vector<double> vrho(rho.size(), 0.0), vsigma(sigma.size(), 0.0),
        vtau(tau.size(), 0.0);
    const hamilt::cider::SemilocalPlan plan(hamilt::cider::SemilocalMode::Npa,
                                            nspin);
    const hamilt::cider::SemilocalReverseInput input(nspin,
                                                     ngrids,
                                                     rho.data(),
                                                     rho.size(),
                                                     sigma.data(),
                                                     sigma.size(),
                                                     tau.data(),
                                                     tau.size(),
                                                     adjoint.data(),
                                                     adjoint.size());
    hamilt::cider::SemilocalReverseOutput output(vrho.data(),
                                                 vrho.size(),
                                                 vsigma.data(),
                                                 vsigma.size(),
                                                 vtau.data(),
                                                 vtau.size());
    plan.reverse(input, output);

    double max_absolute_error = 0.0;
    double max_scaled_error = 0.0;
    const auto check_fd = [&](std::vector<double>& variable,
                              std::size_t index,
                              double analytic) {
        const double original = variable[index];
        const double step = 1.0e-6 * std::max(1.0, std::abs(original));
        variable[index] = original + step;
        const double plus = feature_contraction(hamilt::cider::SemilocalMode::Npa,
                                                nspin,
                                                ngrids,
                                                rho,
                                                sigma,
                                                tau,
                                                adjoint);
        variable[index] = original - step;
        const double minus = feature_contraction(hamilt::cider::SemilocalMode::Npa,
                                                 nspin,
                                                 ngrids,
                                                 rho,
                                                 sigma,
                                                 tau,
                                                 adjoint);
        variable[index] = original;
        const double finite_difference = (plus - minus) / (2.0 * step);
        const double error = std::abs(finite_difference - analytic);
        max_absolute_error = std::max(max_absolute_error, error);
        max_scaled_error
            = std::max(max_scaled_error, error / std::max(1.0, std::abs(analytic)));
    };
    for (std::size_t i = 0; i < rho.size(); ++i)
    {
        check_fd(rho, i, vrho[i]);
    }
    for (std::size_t i = 0; i < sigma.size(); ++i)
    {
        check_fd(sigma, i, vsigma[i]);
    }
    for (std::size_t i = 0; i < tau.size(); ++i)
    {
        check_fd(tau, i, vtau[i]);
    }
    RecordProperty("max_fd_absolute_error", max_absolute_error);
    RecordProperty("max_fd_scaled_error", max_scaled_error);
    EXPECT_LE(max_absolute_error, 2.0e-8);
    EXPECT_LE(max_scaled_error, 2.0e-9);
    EXPECT_DOUBLE_EQ(0.0, vsigma[2]);
    EXPECT_DOUBLE_EQ(0.0, vsigma[3]);
}

} // namespace
