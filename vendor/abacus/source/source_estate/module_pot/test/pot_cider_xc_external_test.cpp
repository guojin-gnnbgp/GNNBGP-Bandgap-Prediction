#include "source_estate/module_pot/pot_cider_xc_external.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::vector<double> read_fixture_array(const std::size_t offset,
                                       const std::size_t count)
{
    const std::string source(__FILE__);
    const std::string test_dir = source.substr(0, source.find_last_of('/'));
    std::ifstream input(
        (test_dir
         + "/../../../source_hamilt/module_xc/cider/test/fixtures/v1/arrays.bin")
            .c_str(),
        std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("failed to open CIDER v1 fixture");
    }
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<double> values(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(double)));
    if (!input)
    {
        throw std::runtime_error("failed to read CIDER v1 fixture array");
    }
    return values;
}

hamilt::cider::DensityTupleInput density_tuple(const std::vector<double>& rho,
                                               const std::vector<double>& sigma)
{
    return hamilt::cider::DensityTupleInput(
        rho.data(), rho.size(), sigma.data(), sigma.size());
}

struct ComponentStorage
{
    explicit ComponentStorage(const std::size_t ngrids)
        : exc(ngrids), vrho(ngrids), vsigma(ngrids)
    {
    }

    hamilt::cider::ComponentOutput output()
    {
        return hamilt::cider::ComponentOutput(exc.data(),
                                              exc.size(),
                                              vrho.data(),
                                              vrho.size(),
                                              vsigma.data(),
                                              vsigma.size());
    }

    std::vector<double> exc;
    std::vector<double> vrho;
    std::vector<double> vsigma;
};

void expect_fixture(const std::vector<double>& actual, const std::size_t offset)
{
    const std::vector<double> expected = read_fixture_array(offset, actual.size());
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(expected[i], actual[i], 2.0e-11) << i;
    }
}

} // namespace

TEST(PotCiderXCExternal, LoadsBridgeMetadata)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for bridge metadata replay";
    }
    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    elecstate::PotCiderExternalRuntime runtime(
        std::string(corpus_root) + "/functionals/CIDER23X_SL_GGA.yaml",
        shape,
        cell,
        1,
        0.25);

    EXPECT_STREQ("external", runtime.backend_name());
    EXPECT_FALSE(runtime.metadata().requires_tau);
    EXPECT_EQ("valence", runtime.metadata().feature_density_policy);
    EXPECT_EQ("energy_density", runtime.metadata().feature_exc_convention);
}

TEST(PotCiderXCExternal, ReplaysRoleSeparatedFrozenArrays)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for bridge replay";
    }
    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    elecstate::PotCiderExternalRuntime runtime(
        std::string(corpus_root) + "/functionals/CIDER23X_SL_GGA.yaml",
        shape,
        cell,
        1,
        0.25);
    const std::size_t ngrids = 8;
    const std::vector<double> rho_baseline = read_fixture_array(112, ngrids);
    const std::vector<double> sigma_baseline = read_fixture_array(176, ngrids);
    const std::vector<double> rho_feature = read_fixture_array(240, ngrids);
    const std::vector<double> sigma_feature = read_fixture_array(304, ngrids);
    const hamilt::cider::EvaluationInput input(
        1,
        ngrids,
        density_tuple(rho_baseline, sigma_baseline),
        density_tuple(rho_feature, sigma_feature));
    ComponentStorage baseline(ngrids);
    ComponentStorage feature(ngrids);

    runtime.evaluate(input, baseline.output(), feature.output());

    expect_fixture(baseline.exc, 1392);
    expect_fixture(baseline.vrho, 1456);
    expect_fixture(baseline.vsigma, 1520);
    expect_fixture(feature.exc, 1584);
    expect_fixture(feature.vrho, 1648);
    expect_fixture(feature.vsigma, 1712);
}
