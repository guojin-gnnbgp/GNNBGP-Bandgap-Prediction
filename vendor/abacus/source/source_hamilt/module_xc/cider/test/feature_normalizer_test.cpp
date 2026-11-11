#include "source_hamilt/module_xc/cider/feature_normalizer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
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

std::vector<hamilt::cider::Normalizer> identity_normalizers(std::size_t count)
{
    return std::vector<hamilt::cider::Normalizer>(
        count, hamilt::cider::Normalizer(hamilt::cider::NormalizerKind::Identity, 1.0));
}

TEST(NativeCiderFeatureNormalizer, MatchesV1NldfGgaForwardOracle)
{
    const std::size_t nspin = 2;
    const std::size_t nfeature = 5;
    const std::size_t ngrids = 8;
    const std::vector<double> raw = read_fixture_array(9716, nspin * nfeature * ngrids);
    const std::vector<double> expected
        = read_fixture_array(10356, nspin * nfeature * ngrids);
    const std::vector<hamilt::cider::Normalizer> normalizers = {
        {hamilt::cider::NormalizerKind::Identity, 1.0},
        {hamilt::cider::NormalizerKind::Identity, 1.0},
        {hamilt::cider::NormalizerKind::Constant, 0.3535533905932738},
        {hamilt::cider::NormalizerKind::Constant, 0.6495190528383291},
        {hamilt::cider::NormalizerKind::Constant, 1.3975424859373686}};
    std::vector<double> actual(expected.size());

    hamilt::cider::normalize_features(normalizers,
                                     hamilt::cider::SemilocalMode::Np,
                                     nspin,
                                     nfeature,
                                     ngrids,
                                     raw.data(),
                                     raw.size(),
                                     actual.data(),
                                     actual.size());

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_LE(std::abs(actual[i] - expected[i]), 1.0e-12) << "flat index " << i;
    }
}

TEST(NativeCiderFeatureNormalizer, MatchesV1NldfGgaReverseOracleAndAccumulates)
{
    const std::size_t nspin = 2;
    const std::size_t nfeature = 5;
    const std::size_t ngrids = 8;
    const std::size_t size = nspin * nfeature * ngrids;
    const std::vector<double> raw = read_fixture_array(9716, size);
    const std::vector<double> normalized_adjoint = read_fixture_array(12852, size);
    const std::vector<double> expected = read_fixture_array(13492, size);
    const std::vector<hamilt::cider::Normalizer> normalizers = {
        {hamilt::cider::NormalizerKind::Identity, 1.0},
        {hamilt::cider::NormalizerKind::Identity, 1.0},
        {hamilt::cider::NormalizerKind::Constant, 0.3535533905932738},
        {hamilt::cider::NormalizerKind::Constant, 0.6495190528383291},
        {hamilt::cider::NormalizerKind::Constant, 1.3975424859373686}};
    std::vector<double> actual(size, 0.375);

    hamilt::cider::accumulate_normalization_reverse(normalizers,
                                                   hamilt::cider::SemilocalMode::Np,
                                                   nspin,
                                                   nfeature,
                                                   ngrids,
                                                   raw.data(),
                                                   raw.size(),
                                                   normalized_adjoint.data(),
                                                   normalized_adjoint.size(),
                                                   actual.data(),
                                                   actual.size());

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_LE(std::abs(actual[i] - (expected[i] + 0.375)), 1.0e-12)
            << "flat index " << i;
    }
}

TEST(NativeCiderFeatureNormalizer, MatchesEveryApplicableV1ForwardAndReverseArray)
{
    struct Case
    {
        const char* name;
        std::size_t nspin;
        std::size_t nfeature;
        hamilt::cider::SemilocalMode mode;
        std::size_t raw_offset;
        std::size_t normalized_offset;
        std::size_t normalized_adjoint_offset;
        std::size_t raw_adjoint_offset;
        std::vector<hamilt::cider::Normalizer> normalizers;
    };
    std::vector<hamilt::cider::Normalizer> nl_gga = identity_normalizers(2);
    nl_gga.push_back({hamilt::cider::NormalizerKind::Constant, 0.3535533905932738});
    nl_gga.push_back({hamilt::cider::NormalizerKind::Constant, 0.6495190528383291});
    nl_gga.push_back({hamilt::cider::NormalizerKind::Constant, 1.3975424859373686});
    std::vector<hamilt::cider::Normalizer> nl_mgga = identity_normalizers(3);
    nl_mgga.push_back({hamilt::cider::NormalizerKind::Constant, 0.6495190528383291});
    nl_mgga.push_back({hamilt::cider::NormalizerKind::Constant, 1.0000000000000004});
    nl_mgga.push_back({hamilt::cider::NormalizerKind::Constant, 1.8371173070873839});
    const std::vector<Case> cases = {
        {"sl_gga_spin1", 1, 2, hamilt::cider::SemilocalMode::Np,
         496, 624, 1136, 1264, identity_normalizers(2)},
        {"sl_mgga_spin2", 2, 3, hamilt::cider::SemilocalMode::Npa,
         3436, 3820, 5292, 5676, identity_normalizers(3)},
        {"nl_gga_spin2", 2, 5, hamilt::cider::SemilocalMode::Np,
         9716, 10356, 12852, 13492, nl_gga},
        {"nl_mgga_spin1", 1, 6, hamilt::cider::SemilocalMode::Npa,
         18908, 19292, 20828, 21212, nl_mgga},
        {"trained_nl_mgga_spin2", 2, 6, hamilt::cider::SemilocalMode::Npa,
         26244, 27012, 30020, 30788, nl_mgga}};

    double max_forward_error = 0.0;
    double max_reverse_error = 0.0;
    for (const Case& item : cases)
    {
        SCOPED_TRACE(item.name);
        const std::size_t size = item.nspin * item.nfeature * 8;
        const std::vector<double> raw = read_fixture_array(item.raw_offset, size);
        const std::vector<double> expected_forward
            = read_fixture_array(item.normalized_offset, size);
        const std::vector<double> normalized_adjoint
            = read_fixture_array(item.normalized_adjoint_offset, size);
        const std::vector<double> expected_reverse
            = read_fixture_array(item.raw_adjoint_offset, size);
        std::vector<double> actual_forward(size);
        std::vector<double> actual_reverse(size, 0.0);
        hamilt::cider::normalize_features(item.normalizers,
                                         item.mode,
                                         item.nspin,
                                         item.nfeature,
                                         8,
                                         raw.data(),
                                         raw.size(),
                                         actual_forward.data(),
                                         actual_forward.size());
        hamilt::cider::accumulate_normalization_reverse(item.normalizers,
                                                       item.mode,
                                                       item.nspin,
                                                       item.nfeature,
                                                       8,
                                                       raw.data(),
                                                       raw.size(),
                                                       normalized_adjoint.data(),
                                                       normalized_adjoint.size(),
                                                       actual_reverse.data(),
                                                       actual_reverse.size());
        for (std::size_t i = 0; i < size; ++i)
        {
            const double forward_error = std::abs(actual_forward[i] - expected_forward[i]);
            const double reverse_error = std::abs(actual_reverse[i] - expected_reverse[i]);
            max_forward_error = std::max(max_forward_error, forward_error);
            max_reverse_error = std::max(max_reverse_error, reverse_error);
            EXPECT_LE(forward_error, 1.0e-12)
                << "forward index " << i;
            EXPECT_LE(reverse_error, 1.0e-12)
                << "reverse index " << i;
        }
    }
    RecordProperty("v1_normalization_forward_max_abs_error", max_forward_error);
    RecordProperty("v1_normalization_reverse_max_abs_error", max_reverse_error);
}

TEST(NativeCiderFeatureNormalizer, ReverseMatchesFiniteDifferenceJacobian)
{
    const std::vector<hamilt::cider::Normalizer> normalizers = {
        {hamilt::cider::NormalizerKind::Identity, 1.0},
        {hamilt::cider::NormalizerKind::Constant, 0.75},
        {hamilt::cider::NormalizerKind::Constant, 1.25}};
    std::vector<double> raw = {0.8, -0.2, 1.1};
    const std::vector<double> adjoint = {0.3, -0.7, 1.4};
    std::vector<double> analytic(3, 0.0);
    hamilt::cider::accumulate_normalization_reverse(normalizers,
                                                   hamilt::cider::SemilocalMode::Npa,
                                                   1,
                                                   3,
                                                   1,
                                                   raw.data(),
                                                   raw.size(),
                                                   adjoint.data(),
                                                   adjoint.size(),
                                                   analytic.data(),
                                                   analytic.size());
    const double step = 1.0e-6;
    for (std::size_t index = 0; index < raw.size(); ++index)
    {
        double objective[2] = {0.0, 0.0};
        for (int sign_index = 0; sign_index < 2; ++sign_index)
        {
            std::vector<double> displaced = raw;
            displaced[index] += sign_index == 0 ? step : -step;
            std::vector<double> normalized(3);
            hamilt::cider::normalize_features(normalizers,
                                             hamilt::cider::SemilocalMode::Npa,
                                             1,
                                             3,
                                             1,
                                             displaced.data(),
                                             displaced.size(),
                                             normalized.data(),
                                             normalized.size());
            for (std::size_t i = 0; i < normalized.size(); ++i)
            {
                objective[sign_index] += normalized[i] * adjoint[i];
            }
        }
        EXPECT_NEAR(analytic[index], (objective[0] - objective[1]) / (2.0 * step), 1.0e-9);
    }
}

TEST(NativeCiderFeatureNormalizer, RejectsInvalidDimensionsOverflowAndNonfiniteValues)
{
    const std::vector<hamilt::cider::Normalizer> normalizers = identity_normalizers(2);
    std::vector<double> raw = {1.0, 2.0};
    std::vector<double> output(2);
    EXPECT_THROW(hamilt::cider::normalize_features(normalizers,
                                                  hamilt::cider::SemilocalMode::Np,
                                                  1, 2, 1,
                                                  raw.data(), 1,
                                                  output.data(), output.size()),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::normalize_features(normalizers,
                                                  hamilt::cider::SemilocalMode::Npa,
                                                  1, 2, 1,
                                                  raw.data(), raw.size(),
                                                  output.data(), output.size()),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::normalize_features(
                     normalizers,
                     static_cast<hamilt::cider::SemilocalMode>(99),
                     1, 2, 1,
                     raw.data(), raw.size(),
                     output.data(), output.size()),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::normalize_features(normalizers,
                                                  hamilt::cider::SemilocalMode::Np,
                                                  2,
                                                  std::numeric_limits<std::size_t>::max(),
                                                  2,
                                                  raw.data(), raw.size(),
                                                  output.data(), output.size()),
                 std::overflow_error);
    raw[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(hamilt::cider::normalize_features(normalizers,
                                                  hamilt::cider::SemilocalMode::Np,
                                                  1, 2, 1,
                                                  raw.data(), raw.size(),
                                                  output.data(), output.size()),
                 std::invalid_argument);
    raw[0] = 1.0;
    std::vector<double> adjoint(2, 1.0);
    output[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(hamilt::cider::accumulate_normalization_reverse(
                     normalizers,
                     hamilt::cider::SemilocalMode::Np,
                     1, 2, 1,
                     raw.data(), raw.size(),
                     adjoint.data(), adjoint.size(),
                     output.data(), output.size()),
                 std::invalid_argument);
}

TEST(NativeCiderFeatureNormalizer, ForwardLateInvalidKindLeavesOutputByteUnchanged)
{
    const std::vector<hamilt::cider::Normalizer> normalizers = {
        {hamilt::cider::NormalizerKind::Identity, 1.0},
        {static_cast<hamilt::cider::NormalizerKind>(99), 1.0}};
    const std::vector<double> raw = {0.5, 1.5, 2.5, 3.5};
    std::vector<double> output = {11.0, 12.0, 13.0, 14.0};
    const std::vector<double> original = output;

    EXPECT_THROW(hamilt::cider::normalize_features(normalizers,
                                                  hamilt::cider::SemilocalMode::Np,
                                                  1, 2, 2,
                                                  raw.data(), raw.size(),
                                                  output.data(), output.size()),
                 std::invalid_argument);
    EXPECT_EQ(0, std::memcmp(output.data(), original.data(), output.size() * sizeof(double)));
}

TEST(NativeCiderFeatureNormalizer, ForwardLateOverflowLeavesOutputByteUnchanged)
{
    const std::vector<hamilt::cider::Normalizer> normalizers = {
        {hamilt::cider::NormalizerKind::Identity, 1.0},
        {hamilt::cider::NormalizerKind::Constant,
         std::numeric_limits<double>::max()}};
    const std::vector<double> raw = {0.5, 1.5, 0.0, 2.0};
    std::vector<double> output = {16.0, 17.0, 18.0, 19.0};
    const std::vector<double> original = output;

    EXPECT_THROW(hamilt::cider::normalize_features(normalizers,
                                                  hamilt::cider::SemilocalMode::Np,
                                                  1, 2, 2,
                                                  raw.data(), raw.size(),
                                                  output.data(), output.size()),
                 std::overflow_error);
    EXPECT_EQ(0, std::memcmp(output.data(), original.data(), output.size() * sizeof(double)));
}

TEST(NativeCiderFeatureNormalizer, ReverseLateOverflowLeavesAdjointByteUnchanged)
{
    const std::vector<hamilt::cider::Normalizer> normalizers = {
        {hamilt::cider::NormalizerKind::Identity, 1.0},
        {hamilt::cider::NormalizerKind::Constant,
         std::numeric_limits<double>::max()}};
    const std::vector<double> raw = {0.5, 1.5, 2.5, 3.5};
    const std::vector<double> normalized_adjoint = {0.5, 1.0, 0.0, 2.0};
    std::vector<double> raw_adjoint = {21.0, 22.0, 23.0, 24.0};
    const std::vector<double> original = raw_adjoint;

    EXPECT_THROW(hamilt::cider::accumulate_normalization_reverse(
                     normalizers,
                     hamilt::cider::SemilocalMode::Np,
                     1, 2, 2,
                     raw.data(), raw.size(),
                     normalized_adjoint.data(), normalized_adjoint.size(),
                     raw_adjoint.data(), raw_adjoint.size()),
                 std::overflow_error);
    EXPECT_EQ(0,
              std::memcmp(raw_adjoint.data(),
                          original.data(),
                          raw_adjoint.size() * sizeof(double)));
}

} // namespace
