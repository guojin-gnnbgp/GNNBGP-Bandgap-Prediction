#include "source_hamilt/module_xc/cider/spline_evaluator.h"
#include "source_hamilt/module_xc/cider/model_loader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

TEST(NativeCiderSplineEvaluator, MatchesPythonOneDimensionalBoundaryOracle)
{
    const hamilt::cider::SplineTerm term(
        1.0,
        {0},
        {{0.0, 1.0, 5}},
        {7},
        {0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0});
    const std::vector<double> descriptors = {
        -0.25, -1.0e-14, 0.0, 1.0e-14, 0.37,
        1.0 - 1.0e-14, 1.0, 1.0 + 1.0e-14, 1.25};
    const std::vector<double> expected = {
        -0.6666666666666669,
        1.333333333333253,
        1.3333333333333333,
        1.3333333333334134,
        6.483733333333333,
        25.33333333333293,
        25.333333333333332,
        25.333333333333734,
        35.33333333333333};
    const std::vector<double> expected_derivatives = {
        0.0,
        7.999999999999679,
        8.0,
        8.00000000000032,
        19.840000000000003,
        39.99999999999968,
        40.0,
        40.00000000000032,
        48.0};
    std::vector<double> values(descriptors.size(), -99.0);
    std::vector<double> derivatives(descriptors.size(), -99.0);

    hamilt::cider::evaluate_spline_set(0.0,
                                       {term},
                                       1,
                                       1,
                                       descriptors.size(),
                                       descriptors.data(),
                                       descriptors.size(),
                                       values.data(),
                                       values.size(),
                                       derivatives.data(),
                                       derivatives.size());

    for (std::size_t grid = 0; grid < descriptors.size(); ++grid)
    {
        EXPECT_LE(std::abs(values[grid] - expected[grid]), 1.0e-14) << grid;
        EXPECT_LE(std::abs(derivatives[grid] - expected_derivatives[grid]), 1.0e-13)
            << grid;
    }
}

TEST(NativeCiderSplineEvaluator, UsesCOrderTensorCoefficientsInTwoAndThreeDimensions)
{
    std::vector<double> coefficients2(30);
    std::vector<double> coefficients3(125);
    for (std::size_t i = 0; i < coefficients2.size(); ++i)
    {
        coefficients2[i] = static_cast<double>(i) / 7.0;
    }
    for (std::size_t i = 0; i < coefficients3.size(); ++i)
    {
        coefficients3[i] = static_cast<double>(i) / 11.0;
    }
    const std::vector<hamilt::cider::SplineTerm> terms = {
        {1.0, {0, 1}, {{0.0, 1.0, 3}, {-1.0, 1.0, 4}}, {5, 6}, coefficients2},
        {1.0,
         {0, 1, 2},
         {{0.0, 1.0, 3}, {-1.0, 1.0, 3}, {2.0, 4.0, 3}},
         {5, 5, 5},
         coefficients3}};
    const std::vector<double> descriptor2 = {0.23, -0.41};
    const std::vector<double> descriptor3 = {0.23, -0.41, 3.2};
    double value2 = 0.0;
    double value3 = 0.0;
    std::vector<double> derivative2(2);
    std::vector<double> derivative3(3);

    hamilt::cider::evaluate_spline_set(0.0,
                                       {terms[0]},
                                       1,
                                       2,
                                       1,
                                       descriptor2.data(),
                                       descriptor2.size(),
                                       &value2,
                                       1,
                                       derivative2.data(),
                                       derivative2.size());
    hamilt::cider::evaluate_spline_set(0.0,
                                       {terms[1]},
                                       1,
                                       3,
                                       1,
                                       descriptor3.data(),
                                       descriptor3.size(),
                                       &value3,
                                       1,
                                       derivative3.data(),
                                       derivative3.size());

    EXPECT_NEAR(1.5207142857142857, value2, 1.0e-14);
    EXPECT_NEAR(1.7142857142857149, derivative2[0], 1.0e-14);
    EXPECT_NEAR(0.21428571428571414, derivative2[1], 1.0e-14);
    EXPECT_NEAR(4.240909090909091, value3, 1.0e-14);
    EXPECT_NEAR(4.545454545454547, derivative3[0], 1.0e-14);
    EXPECT_NEAR(0.4545454545454549, derivative3[1], 1.0e-14);
    EXPECT_NEAR(0.0909090909090912, derivative3[2], 1.0e-14);
}

TEST(NativeCiderSplineEvaluator, BatchesSpinMajorPointsAndAccumulatesSharedDescriptors)
{
    std::vector<double> coefficients2(30);
    for (std::size_t i = 0; i < coefficients2.size(); ++i)
    {
        coefficients2[i] = static_cast<double>(i) / 7.0;
    }
    const std::vector<hamilt::cider::SplineTerm> terms = {
        {2.0,
         {0},
         {{0.0, 1.0, 5}},
         {7},
         {0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0}},
        {-0.5, {0, 2}, {{0.0, 1.0, 3}, {-1.0, 1.0, 4}}, {5, 6}, coefficients2}};
    // (nspin=2, nfeature=3, ngrids=2), contiguous spin-major storage.
    const std::vector<double> descriptors = {
        0.2, 0.4, 0.9, 0.8, -0.4, 0.1,
        0.6, 0.8, 0.7, 0.6, 0.5, 0.9};
    const std::vector<double> expected_values = {
        7.160952380952382, 13.975952380952382,
        23.36166666666666, 35.30738095238096};
    const std::vector<double> expected_derivatives = {
        27.942857142857147, 40.74285714285714, 0.0, 0.0,
        -0.10714285714285712, -0.10714285714285723,
        53.542857142857144, 66.34285714285716, 0.0, 0.0,
        -0.10714285714285722, -0.1071428571428572};
    std::vector<double> values(4, 123.0);
    std::vector<double> derivatives(12, 123.0);

    hamilt::cider::evaluate_spline_set(0.75,
                                       terms,
                                       2,
                                       3,
                                       2,
                                       descriptors.data(),
                                       descriptors.size(),
                                       values.data(),
                                       values.size(),
                                       derivatives.data(),
                                       derivatives.size());

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_NEAR(expected_values[i], values[i], 1.0e-13) << i;
    }
    for (std::size_t i = 0; i < derivatives.size(); ++i)
    {
        EXPECT_NEAR(expected_derivatives[i], derivatives[i], 1.0e-13) << i;
    }
}

TEST(NativeCiderSplineEvaluator, DescriptorDerivativesMatchFiniteDifferences)
{
    std::vector<double> coefficients(125);
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
        coefficients[i] = std::sin(0.17 * static_cast<double>(i));
    }
    const std::vector<hamilt::cider::SplineTerm> terms = {{
        1.7,
        {0, 1, 2},
        {{0.0, 1.0, 3}, {-1.0, 1.0, 3}, {2.0, 4.0, 3}},
        {5, 5, 5},
        coefficients}};
    std::vector<double> descriptors = {0.23, -0.41, 3.2};
    double value = 0.0;
    std::vector<double> derivative(3);
    hamilt::cider::evaluate_spline_set(0.2,
                                       terms,
                                       1, 3, 1,
                                       descriptors.data(), descriptors.size(),
                                       &value, 1,
                                       derivative.data(), derivative.size());
    const double step = 1.0e-6;
    for (std::size_t descriptor = 0; descriptor < descriptors.size(); ++descriptor)
    {
        double displaced_value[2] = {0.0, 0.0};
        for (int sign = 0; sign < 2; ++sign)
        {
            std::vector<double> displaced = descriptors;
            displaced[descriptor] += sign == 0 ? step : -step;
            std::vector<double> ignored_derivative(3);
            hamilt::cider::evaluate_spline_set(0.2,
                                               terms,
                                               1, 3, 1,
                                               displaced.data(), displaced.size(),
                                               &displaced_value[sign], 1,
                                               ignored_derivative.data(),
                                               ignored_derivative.size());
        }
        EXPECT_NEAR(derivative[descriptor],
                    (displaced_value[0] - displaced_value[1]) / (2.0 * step),
                    2.0e-9)
            << descriptor;
    }
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

std::vector<double> sample_major_to_spin_feature(const std::vector<double>& values,
                                                 std::size_t nspin,
                                                 std::size_t nfeature,
                                                 std::size_t ngrids)
{
    std::vector<double> result(values.size());
    for (std::size_t spin = 0; spin < nspin; ++spin)
    {
        for (std::size_t feature = 0; feature < nfeature; ++feature)
        {
            for (std::size_t grid = 0; grid < ngrids; ++grid)
            {
                result[(spin * nfeature + feature) * ngrids + grid]
                    = values[(spin * ngrids + grid) * nfeature + feature];
            }
        }
    }
    return result;
}

TEST(NativeCiderSplineEvaluator, MatchesAllFiveV1SplineForwardAndDerivativeOracles)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    struct Case
    {
        const char* name;
        const char* model;
        std::size_t nspin;
        std::size_t nfeature;
        std::size_t transformed_offset;
        std::size_t value_offset;
        std::size_t derivative_offset;
    };
    const Case cases[] = {
        {"sl_gga_spin1", "functionals/CIDER23X_SL_GGA.yaml", 1, 1, 752, 816, 880},
        {"sl_mgga_spin2", "functionals/CIDER23X_SL_MGGA.yaml", 2, 2, 4204, 4460, 4588},
        {"nl_gga_spin2", "functionals/CIDER23X_NL_GGA.yaml", 2, 4, 10996, 11508, 11636},
        {"nl_mgga_spin1", "functionals/CIDER23X_NL_MGGA_DTR.yaml", 1, 5, 19676, 19996, 20060},
        {"trained_nl_mgga_spin2",
         "abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/full/"
         "nldf_mgga_cand00_n96_noise0p01.yaml",
         2, 5, 27780, 28420, 28548}};
    const hamilt::cider::ModelLoader loader;
    double max_value_error = 0.0;
    double max_derivative_error = 0.0;
    for (const Case& item : cases)
    {
        SCOPED_TRACE(item.name);
        const std::unique_ptr<const hamilt::cider::Model> model
            = loader.load(std::string(parent) + "/" + item.model);
        const std::size_t value_size = item.nspin * 8;
        const std::size_t descriptor_size = value_size * item.nfeature;
        const std::vector<double> descriptors = sample_major_to_spin_feature(
            read_fixture_array(item.transformed_offset, descriptor_size),
            item.nspin,
            item.nfeature,
            8);
        const std::vector<double> expected_values
            = read_fixture_array(item.value_offset, value_size);
        const std::vector<double> expected_derivatives = sample_major_to_spin_feature(
            read_fixture_array(item.derivative_offset, descriptor_size),
            item.nspin,
            item.nfeature,
            8);
        std::vector<double> values(value_size);
        std::vector<double> derivatives(descriptor_size);
        hamilt::cider::evaluate_spline_set(model->spline_constant(),
                                           model->spline_terms(),
                                           item.nspin,
                                           item.nfeature,
                                           8,
                                           descriptors.data(),
                                           descriptors.size(),
                                           values.data(),
                                           values.size(),
                                           derivatives.data(),
                                           derivatives.size());
        for (std::size_t i = 0; i < value_size; ++i)
        {
            const double error = std::abs(values[i] - expected_values[i]);
            max_value_error = std::max(max_value_error, error);
            EXPECT_LE(error, 1.0e-12) << "value " << i;
        }
        for (std::size_t i = 0; i < descriptor_size; ++i)
        {
            const double error = std::abs(derivatives[i] - expected_derivatives[i]);
            max_derivative_error = std::max(max_derivative_error, error);
            EXPECT_LE(error, 1.0e-12) << "derivative " << i;
        }
    }
    RecordProperty("v1_spline_forward_max_abs_error", max_value_error);
    RecordProperty("v1_spline_derivative_max_abs_error", max_derivative_error);
}

void expect_invalid_term(const hamilt::cider::SplineTerm& term)
{
    const std::vector<double> descriptors = {0.25, 0.5, 0.75, 0.9};
    double value = 17.0;
    std::vector<double> derivative(4, 19.0);
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(0.0,
                                                   {term},
                                                   1, 4, 1,
                                                   descriptors.data(), descriptors.size(),
                                                   &value, 1,
                                                   derivative.data(), derivative.size()),
                 std::invalid_argument);
    EXPECT_DOUBLE_EQ(17.0, value);
    EXPECT_EQ(std::vector<double>(4, 19.0), derivative);
}

TEST(NativeCiderSplineEvaluator, RejectsInvalidTermStructureAndTables)
{
    const std::vector<double> coefficients(5, 1.0);
    expect_invalid_term({1.0, {}, {}, {}, {}});
    expect_invalid_term({1.0,
                         {0, 1, 2, 3},
                         {{0.0, 1.0, 3}, {0.0, 1.0, 3}, {0.0, 1.0, 3}, {0.0, 1.0, 3}},
                         {5, 5, 5, 5},
                         std::vector<double>(625)});
    expect_invalid_term({1.0, {4}, {{0.0, 1.0, 3}}, {5}, coefficients});
    expect_invalid_term({1.0,
                         {0, 0},
                         {{0.0, 1.0, 3}, {0.0, 1.0, 3}},
                         {5, 5},
                         std::vector<double>(25)});
    expect_invalid_term({1.0, {0}, {}, {5}, coefficients});
    expect_invalid_term({1.0, {0}, {{0.0, 1.0, 3}}, {}, coefficients});
    expect_invalid_term({1.0, {0}, {{0.0, 1.0, 3}}, {4}, std::vector<double>(4)});
    expect_invalid_term({1.0, {0}, {{0.0, 1.0, 3}}, {5}, std::vector<double>(4)});
    expect_invalid_term({1.0, {0}, {{0.0, 0.0, 3}}, {5}, coefficients});
    expect_invalid_term({1.0, {0}, {{0.0, 1.0, 1}}, {3}, std::vector<double>(3)});
    expect_invalid_term({std::numeric_limits<double>::infinity(),
                         {0}, {{0.0, 1.0, 3}}, {5}, coefficients});
    std::vector<double> nonfinite = coefficients;
    nonfinite[4] = std::numeric_limits<double>::quiet_NaN();
    expect_invalid_term({1.0, {0}, {{0.0, 1.0, 3}}, {5}, nonfinite});
}

TEST(NativeCiderSplineEvaluator, RejectsInvalidCallContractAndPreservesBothOutputs)
{
    const hamilt::cider::SplineTerm term(
        1.0, {0}, {{0.0, 1.0, 3}}, {5}, std::vector<double>(5, 1.0));
    std::vector<double> descriptors = {0.25};
    double value = 17.0;
    double derivative = 19.0;
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(0.0, {}, 1, 1, 1,
                                                   descriptors.data(), 1,
                                                   &value, 1, &derivative, 1),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(0.0, {term}, 3, 1, 1,
                                                   descriptors.data(), 1,
                                                   &value, 1, &derivative, 1),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(0.0, {term}, 1, 1, 1,
                                                   descriptors.data(), 0,
                                                   &value, 1, &derivative, 1),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(0.0, {term}, 1, 0, 1,
                                                   descriptors.data(), 1,
                                                   &value, 1, &derivative, 1),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(0.0, {term}, 1, 1, 0,
                                                   descriptors.data(), 1,
                                                   &value, 1, &derivative, 1),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(0.0, {term}, 1, 1, 1,
                                                   nullptr, 1,
                                                   &value, 1, &derivative, 1),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(
                     0.0,
                     {term},
                     2,
                     std::numeric_limits<std::size_t>::max(),
                     2,
                     descriptors.data(), 1,
                     &value, 1, &derivative, 1),
                 std::overflow_error);
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(std::numeric_limits<double>::quiet_NaN(),
                                                   {term}, 1, 1, 1,
                                                   descriptors.data(), 1,
                                                   &value, 1, &derivative, 1),
                 std::invalid_argument);
    descriptors[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(0.0, {term}, 1, 1, 1,
                                                   descriptors.data(), 1,
                                                   &value, 1, &derivative, 1),
                 std::invalid_argument);
    EXPECT_DOUBLE_EQ(17.0, value);
    EXPECT_DOUBLE_EQ(19.0, derivative);
}

TEST(NativeCiderSplineEvaluator, LateNonfiniteResultPreservesOutputsByteForByte)
{
    const hamilt::cider::SplineTerm valid(
        1.0, {0}, {{0.0, 1.0, 3}}, {5}, std::vector<double>(5, 1.0));
    const hamilt::cider::SplineTerm overflow(
        std::numeric_limits<double>::max(),
        {0},
        {{0.0, 1.0, 3}},
        {5},
        std::vector<double>(5, std::numeric_limits<double>::max()));
    const std::vector<double> descriptors = {0.25, 0.75};
    const std::vector<double> value_before = {17.0, -0.0};
    const std::vector<double> derivative_before = {19.0, -0.0};
    std::vector<double> values = value_before;
    std::vector<double> derivatives = derivative_before;
    EXPECT_THROW(hamilt::cider::evaluate_spline_set(0.0,
                                                   {valid, overflow},
                                                   1, 1, 2,
                                                   descriptors.data(), descriptors.size(),
                                                   values.data(), values.size(),
                                                   derivatives.data(), derivatives.size()),
                 std::overflow_error);
    EXPECT_EQ(0, std::memcmp(values.data(), value_before.data(), values.size() * sizeof(double)));
    EXPECT_EQ(0,
              std::memcmp(derivatives.data(),
                          derivative_before.data(),
                          derivatives.size() * sizeof(double)));
}

} // namespace
