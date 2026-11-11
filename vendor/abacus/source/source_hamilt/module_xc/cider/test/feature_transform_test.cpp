#include "source_hamilt/module_xc/cider/feature_transform.h"

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

hamilt::cider::FeatureMap umap()
{
    return {hamilt::cider::FeatureMapKind::U,
            1,
            0.24308265985843178,
            1.0,
            0.0,
            0,
            0.0,
            1.0};
}

hamilt::cider::FeatureMap vmap(std::size_t index)
{
    return {hamilt::cider::FeatureMapKind::V, index, 0.5, 1.0, 0.5, 0, -0.5, 0.5};
}

hamilt::cider::FeatureMap zmap()
{
    return {hamilt::cider::FeatureMapKind::Z, 2, 1.0, 2.0, 1.0, 1, -1.0, 1.0};
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

TEST(NativeCiderFeatureTransform, MatchesV1SemilocalMggaUAndZForwardOracle)
{
    const std::size_t nspin = 2;
    const std::size_t nraw = 3;
    const std::size_t ntransformed = 2;
    const std::size_t ngrids = 8;
    const std::vector<double> raw = read_fixture_array(3820, nspin * nraw * ngrids);
    const std::vector<double> fixture_sample_major
        = read_fixture_array(4204, nspin * ngrids * ntransformed);
    const std::vector<hamilt::cider::FeatureMap> maps = {umap(), zmap()};
    std::vector<double> actual(nspin * ntransformed * ngrids);

    hamilt::cider::transform_features(maps,
                                     nspin,
                                     nraw,
                                     ngrids,
                                     raw.data(),
                                     raw.size(),
                                     actual.data(),
                                     actual.size());

    for (std::size_t spin = 0; spin < nspin; ++spin)
    {
        for (std::size_t transformed = 0; transformed < ntransformed; ++transformed)
        {
            for (std::size_t grid = 0; grid < ngrids; ++grid)
            {
                const std::size_t actual_index
                    = (spin * ntransformed + transformed) * ngrids + grid;
                const std::size_t fixture_index
                    = (spin * ngrids + grid) * ntransformed + transformed;
                EXPECT_LE(std::abs(actual[actual_index]
                                   - fixture_sample_major[fixture_index]),
                          1.0e-12)
                    << "spin " << spin << " map " << transformed << " grid " << grid;
            }
        }
    }
}

TEST(NativeCiderFeatureTransform, MatchesV1SemilocalMggaReverseOracleAndAccumulates)
{
    const std::size_t nspin = 2;
    const std::size_t nraw = 3;
    const std::size_t ntransformed = 2;
    const std::size_t ngrids = 8;
    const std::vector<double> raw = read_fixture_array(3820, nspin * nraw * ngrids);
    const std::vector<double> fixture_adjoint
        = read_fixture_array(4588, nspin * ngrids * ntransformed);
    const std::vector<double> expected
        = read_fixture_array(4844, nspin * nraw * ngrids);
    const std::vector<hamilt::cider::FeatureMap> maps = {umap(), zmap()};
    const std::vector<double> transformed_adjoint
        = sample_major_to_spin_feature(fixture_adjoint,
                                       nspin,
                                       ntransformed,
                                       ngrids);
    std::vector<double> actual(expected.size(), -0.125);

    hamilt::cider::accumulate_transform_reverse(maps,
                                               nspin,
                                               nraw,
                                               ngrids,
                                               raw.data(),
                                               raw.size(),
                                               transformed_adjoint.data(),
                                               transformed_adjoint.size(),
                                               actual.data(),
                                               actual.size());

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_LE(std::abs(actual[i] - (expected[i] - 0.125)), 1.0e-12)
            << "flat index " << i;
    }
}

TEST(NativeCiderFeatureTransform, MatchesEveryApplicableV1ForwardAndReverseArray)
{
    struct Case
    {
        const char* name;
        std::size_t nspin;
        std::size_t nraw;
        std::size_t normalized_offset;
        std::size_t transformed_offset;
        std::size_t transformed_adjoint_offset;
        std::size_t raw_adjoint_offset;
        std::vector<hamilt::cider::FeatureMap> maps;
    };
    const std::vector<Case> cases = {
        {"sl_gga_spin1", 1, 2, 624, 752, 880, 944, {umap()}},
        {"sl_mgga_spin2", 2, 3, 3820, 4204, 4588, 4844, {umap(), zmap()}},
        {"nl_gga_spin2", 2, 5, 10356, 10996, 11636, 12148,
         {umap(), vmap(2), vmap(3), vmap(4)}},
        {"nl_mgga_spin1", 1, 6, 19292, 19676, 20060, 20380,
         {umap(), zmap(), vmap(3), vmap(4), vmap(5)}},
        {"trained_nl_mgga_spin2", 2, 6, 27012, 27780, 28548, 29188,
         {umap(), zmap(), vmap(3), vmap(4), vmap(5)}}};

    double max_forward_error = 0.0;
    double max_reverse_error = 0.0;
    for (const Case& item : cases)
    {
        SCOPED_TRACE(item.name);
        const std::size_t ntransformed = item.maps.size();
        const std::size_t raw_size = item.nspin * item.nraw * 8;
        const std::size_t transformed_size = item.nspin * ntransformed * 8;
        const std::vector<double> raw
            = read_fixture_array(item.normalized_offset, raw_size);
        const std::vector<double> expected_forward = sample_major_to_spin_feature(
            read_fixture_array(item.transformed_offset, transformed_size),
            item.nspin,
            ntransformed,
            8);
        const std::vector<double> transformed_adjoint = sample_major_to_spin_feature(
            read_fixture_array(item.transformed_adjoint_offset, transformed_size),
            item.nspin,
            ntransformed,
            8);
        const std::vector<double> expected_reverse
            = read_fixture_array(item.raw_adjoint_offset, raw_size);
        std::vector<double> actual_forward(transformed_size);
        std::vector<double> actual_reverse(raw_size, 0.0);
        hamilt::cider::transform_features(item.maps,
                                         item.nspin,
                                         item.nraw,
                                         8,
                                         raw.data(),
                                         raw.size(),
                                         actual_forward.data(),
                                         actual_forward.size());
        hamilt::cider::accumulate_transform_reverse(item.maps,
                                                   item.nspin,
                                                   item.nraw,
                                                   8,
                                                   raw.data(),
                                                   raw.size(),
                                                   transformed_adjoint.data(),
                                                   transformed_adjoint.size(),
                                                   actual_reverse.data(),
                                                   actual_reverse.size());
        for (std::size_t i = 0; i < transformed_size; ++i)
        {
            const double error = std::abs(actual_forward[i] - expected_forward[i]);
            max_forward_error = std::max(max_forward_error, error);
            EXPECT_LE(error, 1.0e-12)
                << "forward index " << i;
        }
        for (std::size_t i = 0; i < raw_size; ++i)
        {
            const double error = std::abs(actual_reverse[i] - expected_reverse[i]);
            max_reverse_error = std::max(max_reverse_error, error);
            EXPECT_LE(error, 1.0e-12)
                << "reverse index " << i;
        }
    }
    RecordProperty("v1_transform_forward_max_abs_error", max_forward_error);
    RecordProperty("v1_transform_reverse_max_abs_error", max_reverse_error);
}

TEST(NativeCiderFeatureTransform, MapsBoundaryInteriorAndUnclippedValuesExactly)
{
    const std::vector<hamilt::cider::FeatureMap> maps = {
        {hamilt::cider::FeatureMapKind::U, 0, 0.5, 1.0, 0.0, 0, 0.0, 1.0},
        {hamilt::cider::FeatureMapKind::V, 1, 0.5, 2.0, 0.25, 0, -0.25, 1.75},
        {hamilt::cider::FeatureMapKind::Z, 2, 2.0, 3.0, 1.0, 1, -1.0, 2.0}};
    const std::vector<double> raw = {0.0, 2.0, -1.0,
                                     0.0, 2.0, 0.0,
                                     0.0, 1.0, 2.0};
    std::vector<double> transformed(9);
    hamilt::cider::transform_features(maps,
                                     1, 3, 3,
                                     raw.data(), raw.size(),
                                     transformed.data(), transformed.size());
    EXPECT_DOUBLE_EQ(0.0, transformed[0]);
    EXPECT_DOUBLE_EQ(0.5, transformed[1]);
    EXPECT_DOUBLE_EQ(-1.0, transformed[2]);
    EXPECT_DOUBLE_EQ(-0.25, transformed[3]);
    EXPECT_DOUBLE_EQ(0.75, transformed[4]);
    EXPECT_DOUBLE_EQ(-0.25, transformed[5]);
    EXPECT_DOUBLE_EQ(2.0, transformed[6]);
    EXPECT_DOUBLE_EQ(0.0, transformed[7]);
    EXPECT_NEAR(-2.0 / 3.0, transformed[8], 1.0e-15);
}

TEST(NativeCiderFeatureTransform, ReverseMatchesFiniteDifferenceJacobianForUvz)
{
    const std::vector<hamilt::cider::FeatureMap> maps = {
        {hamilt::cider::FeatureMapKind::U, 0, 0.4, 1.0, 0.0, 0, 0.0, 1.0},
        {hamilt::cider::FeatureMapKind::V, 0, 0.7, 1.3, 0.2, 0, -0.2, 1.1},
        {hamilt::cider::FeatureMapKind::Z, 2, 0.9, 1.7, 0.6, 1, -0.6, 1.1}};
    std::vector<double> raw = {0.3, 1.2, -0.4};
    const std::vector<double> adjoint = {0.6, -1.1, 0.8};
    std::vector<double> analytic(3, 0.0);
    hamilt::cider::accumulate_transform_reverse(maps,
                                               1, 3, 1,
                                               raw.data(), raw.size(),
                                               adjoint.data(), adjoint.size(),
                                               analytic.data(), analytic.size());
    const double step = 1.0e-6;
    for (std::size_t index = 0; index < raw.size(); ++index)
    {
        double objective[2] = {0.0, 0.0};
        for (int sign_index = 0; sign_index < 2; ++sign_index)
        {
            std::vector<double> displaced = raw;
            displaced[index] += sign_index == 0 ? step : -step;
            std::vector<double> transformed(3);
            hamilt::cider::transform_features(maps,
                                             1, 3, 1,
                                             displaced.data(), displaced.size(),
                                             transformed.data(), transformed.size());
            for (std::size_t i = 0; i < transformed.size(); ++i)
            {
                objective[sign_index] += transformed[i] * adjoint[i];
            }
        }
        EXPECT_NEAR(analytic[index], (objective[0] - objective[1]) / (2.0 * step), 2.0e-9);
    }
}

TEST(NativeCiderFeatureTransform, RejectsInvalidShapeIndexParametersAndNonfiniteValues)
{
    const std::vector<hamilt::cider::FeatureMap> maps = {
        {hamilt::cider::FeatureMapKind::U, 0, 0.5, 1.0, 0.0, 0, 0.0, 1.0}};
    std::vector<double> raw = {0.5, 1.0};
    std::vector<double> transformed(1);
    EXPECT_THROW(hamilt::cider::transform_features(maps,
                                                  1, 2, 1,
                                                  raw.data(), 1,
                                                  transformed.data(), transformed.size()),
                 std::invalid_argument);
    const std::vector<hamilt::cider::FeatureMap> bad_index = {
        {hamilt::cider::FeatureMapKind::U, 2, 0.5, 1.0, 0.0, 0, 0.0, 1.0}};
    EXPECT_THROW(hamilt::cider::transform_features(bad_index,
                                                  1, 2, 1,
                                                  raw.data(), raw.size(),
                                                  transformed.data(), transformed.size()),
                 std::invalid_argument);
    const std::vector<hamilt::cider::FeatureMap> bad_parameter = {
        {hamilt::cider::FeatureMapKind::U,
         0,
         std::numeric_limits<double>::quiet_NaN(),
         1.0,
         0.0,
         0,
         0.0,
         1.0}};
    EXPECT_THROW(hamilt::cider::transform_features(bad_parameter,
                                                  1, 2, 1,
                                                  raw.data(), raw.size(),
                                                  transformed.data(), transformed.size()),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::transform_features(maps,
                                                  2,
                                                  std::numeric_limits<std::size_t>::max(),
                                                  2,
                                                  raw.data(), raw.size(),
                                                  transformed.data(), transformed.size()),
                 std::overflow_error);
    raw[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(hamilt::cider::transform_features(maps,
                                                  1, 2, 1,
                                                  raw.data(), raw.size(),
                                                  transformed.data(), transformed.size()),
                 std::invalid_argument);
    raw[0] = -2.0;
    EXPECT_THROW(hamilt::cider::transform_features(maps,
                                                  1, 2, 1,
                                                  raw.data(), raw.size(),
                                                  transformed.data(), transformed.size()),
                 std::domain_error);
    raw[0] = 0.5;
    std::vector<double> seed(2, 0.0);
    seed[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(hamilt::cider::accumulate_transform_reverse(
                     maps,
                     1, 2, 1,
                     raw.data(), raw.size(),
                     transformed.data(), transformed.size(),
                     seed.data(), seed.size()),
                 std::invalid_argument);
}

TEST(NativeCiderFeatureTransform, ForwardLateInvalidKindLeavesOutputByteUnchanged)
{
    const std::vector<hamilt::cider::FeatureMap> maps = {
        {hamilt::cider::FeatureMapKind::U, 0, 0.5, 1.0, 0.0, 0, 0.0, 1.0},
        {static_cast<hamilt::cider::FeatureMapKind>(99),
         1, 0.5, 1.0, 0.0, 0, 0.0, 1.0}};
    const std::vector<double> raw = {0.5, 1.5, 2.5, 3.5};
    std::vector<double> transformed = {31.0, 32.0, 33.0, 34.0};
    const std::vector<double> original = transformed;

    EXPECT_THROW(hamilt::cider::transform_features(maps,
                                                  1, 2, 2,
                                                  raw.data(), raw.size(),
                                                  transformed.data(), transformed.size()),
                 std::invalid_argument);
    EXPECT_EQ(0,
              std::memcmp(transformed.data(),
                          original.data(),
                          transformed.size() * sizeof(double)));
}

TEST(NativeCiderFeatureTransform, ForwardLateSingularityLeavesOutputByteUnchanged)
{
    const std::vector<hamilt::cider::FeatureMap> maps = {
        {hamilt::cider::FeatureMapKind::U, 0, 0.5, 1.0, 0.0, 0, 0.0, 1.0},
        {hamilt::cider::FeatureMapKind::V, 1, 1.0, 1.0, 0.0, 0, 0.0, 1.0}};
    const std::vector<double> raw = {0.5, 1.5, 0.0, -1.0};
    std::vector<double> transformed = {41.0, 42.0, 43.0, 44.0};
    const std::vector<double> original = transformed;

    EXPECT_THROW(hamilt::cider::transform_features(maps,
                                                  1, 2, 2,
                                                  raw.data(), raw.size(),
                                                  transformed.data(), transformed.size()),
                 std::domain_error);
    EXPECT_EQ(0,
              std::memcmp(transformed.data(),
                          original.data(),
                          transformed.size() * sizeof(double)));
}

TEST(NativeCiderFeatureTransform, ReverseLateSingularityLeavesAdjointByteUnchanged)
{
    const std::vector<hamilt::cider::FeatureMap> maps = {
        {hamilt::cider::FeatureMapKind::U, 0, 0.5, 1.0, 0.0, 0, 0.0, 1.0},
        {hamilt::cider::FeatureMapKind::V, 1, 1.0, 1.0, 0.0, 0, 0.0, 1.0}};
    const std::vector<double> raw = {0.5, 1.5, 0.0, -1.0};
    const std::vector<double> transformed_adjoint = {0.5, 1.0, 1.5, 2.0};
    std::vector<double> raw_adjoint = {51.0, 52.0, 53.0, 54.0};
    const std::vector<double> original = raw_adjoint;

    EXPECT_THROW(hamilt::cider::accumulate_transform_reverse(
                     maps,
                     1, 2, 2,
                     raw.data(), raw.size(),
                     transformed_adjoint.data(), transformed_adjoint.size(),
                     raw_adjoint.data(), raw_adjoint.size()),
                 std::domain_error);
    EXPECT_EQ(0,
              std::memcmp(raw_adjoint.data(),
                          original.data(),
                          raw_adjoint.size() * sizeof(double)));
}

} // namespace
