#include "source_estate/module_pot/pot_cider_xc_native.h"
#include "source_estate/module_pot/pot_cider_xc_runtime.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

std::string fixture_path()
{
    const std::string source(__FILE__);
    const std::string test_dir = source.substr(0, source.find_last_of('/'));
    return test_dir
           + "/../../../source_hamilt/module_xc/cider/test/fixtures/v1/arrays.bin";
}

std::vector<double> read_fixture_array(const std::size_t offset,
                                       const std::size_t count)
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

hamilt::cider::DensityTupleInput density_tuple(const std::vector<double>& rho,
                                               const std::vector<double>& sigma,
                                               const std::vector<double>& tau
                                                   = std::vector<double>())
{
    return hamilt::cider::DensityTupleInput(rho.data(),
                                            rho.size(),
                                            sigma.data(),
                                            sigma.size(),
                                            tau.empty() ? nullptr : tau.data(),
                                            tau.size());
}

struct ComponentStorage
{
    ComponentStorage(const std::size_t ngrids,
                     const std::size_t nspin = 1,
                     const bool has_tau = false)
        : exc(ngrids, 91.0),
          vrho(nspin * ngrids, 92.0),
          vsigma((2 * nspin - 1) * ngrids, 93.0),
          vtau(has_tau ? nspin * ngrids : 0, 94.0)
    {
    }

    hamilt::cider::ComponentOutput output()
    {
        return hamilt::cider::ComponentOutput(exc.data(),
                                              exc.size(),
                                              vrho.data(),
                                              vrho.size(),
                                              vsigma.data(),
                                              vsigma.size(),
                                              vtau.empty() ? nullptr : vtau.data(),
                                              vtau.size());
    }

    std::vector<double> exc;
    std::vector<double> vrho;
    std::vector<double> vsigma;
    std::vector<double> vtau;
};

void expect_fixture(const std::vector<double>& actual, const std::size_t offset)
{
    const std::vector<double> expected = read_fixture_array(offset, actual.size());
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(expected[i], actual[i], 1.0e-12) << i;
    }
}

void copy_file(const std::string& source, const std::string& destination)
{
    std::ifstream input(source.c_str(), std::ios::binary);
    std::ofstream output(destination.c_str(), std::ios::binary);
    if (!input || !output)
    {
        throw std::runtime_error("failed to prepare renamed CIDER model fixture");
    }
    output << input.rdbuf();
    if (!output)
    {
        throw std::runtime_error("failed to write renamed CIDER model fixture");
    }
}

struct TemporaryFile
{
    explicit TemporaryFile(const std::string& path_in) : path(path_in) {}
    ~TemporaryFile() { std::remove(path.c_str()); }
    std::string path;
};

} // namespace

static_assert(std::is_base_of<elecstate::PotCiderRuntime,
                              elecstate::PotCiderNativeRuntime>::value,
              "native CIDER runtime must implement the common backend");

TEST(PotCiderXCNative, ReportsBackendNeutralMetadata)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for metadata replay";
    }
    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    elecstate::PotCiderNativeRuntime runtime(
        std::string(corpus_root) + "/functionals/CIDER23X_SL_MGGA.yaml",
        shape,
        cell,
        2,
        0.25);
    EXPECT_STREQ("native", runtime.backend_name());
    EXPECT_TRUE(runtime.metadata().requires_tau);
    EXPECT_EQ("energy_density", runtime.metadata().feature_exc_convention);
}

TEST(PotCiderXCNative, LoadsMetadataAndReplaysFrozenBridgeV1Arrays)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the frozen replay";
    }

    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    const std::string model_path
        = std::string(corpus_root) + "/functionals/CIDER23X_SL_GGA.yaml";
    elecstate::PotCiderNativeRuntime runtime(model_path, shape, cell, 1, 0.25);

    EXPECT_FALSE(runtime.metadata().requires_tau);
    EXPECT_EQ("valence", runtime.metadata().feature_density_policy);
    EXPECT_EQ("energy_density", runtime.metadata().feature_exc_convention);

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

TEST(PotCiderXCNative, UsesAuthoritativeMetadataForRenamedGgaAndMggaModels)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the metadata replay";
    }

    const TemporaryFile renamed_gga("/tmp/task12_model_without_level_a.yaml");
    const TemporaryFile renamed_mgga("/tmp/task12_model_without_level_b.yaml");
    copy_file(std::string(corpus_root) + "/functionals/CIDER23X_SL_GGA.yaml",
              renamed_gga.path);
    copy_file(std::string(corpus_root) + "/functionals/CIDER23X_SL_MGGA.yaml",
              renamed_mgga.path);
    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};

    const elecstate::PotCiderNativeRuntime gga(
        renamed_gga.path, shape, cell, 1, 0.25);
    const elecstate::PotCiderNativeRuntime mgga(
        renamed_mgga.path, shape, cell, 2, 0.25);
    EXPECT_FALSE(gga.metadata().requires_tau);
    EXPECT_TRUE(mgga.metadata().requires_tau);
}

TEST(PotCiderXCNative, ReplaysFrozenSpinTwoMggaAndNldfBridgeArrays)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr || *corpus_root == '\0')
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the spin/MGGA/NLDF replay";
    }
    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    const std::size_t ngrids = 8;

    {
        const std::size_t nspin = 2;
        const elecstate::PotCiderNativeRuntime runtime(
            std::string(corpus_root) + "/functionals/CIDER23X_SL_MGGA.yaml",
            shape,
            cell,
            nspin,
            0.25);
        const std::vector<double> rho_baseline = read_fixture_array(2156, 16);
        const std::vector<double> sigma_baseline = read_fixture_array(2284, 24);
        const std::vector<double> tau_baseline = read_fixture_array(2476, 16);
        const std::vector<double> rho_feature = read_fixture_array(2604, 16);
        const std::vector<double> sigma_feature = read_fixture_array(2732, 24);
        const std::vector<double> tau_feature = read_fixture_array(2924, 16);
        const hamilt::cider::EvaluationInput input(
            nspin,
            ngrids,
            density_tuple(rho_baseline, sigma_baseline, tau_baseline),
            density_tuple(rho_feature, sigma_feature, tau_feature));
        ComponentStorage baseline(ngrids, nspin, true);
        ComponentStorage feature(ngrids, nspin, true);
        runtime.evaluate(input, baseline.output(), feature.output());
        expect_fixture(baseline.exc, 6060);
        expect_fixture(baseline.vrho, 6124);
        expect_fixture(baseline.vsigma, 6252);
        expect_fixture(baseline.vtau, 6444);
        expect_fixture(feature.exc, 6572);
        expect_fixture(feature.vrho, 6636);
        expect_fixture(feature.vsigma, 6764);
        expect_fixture(feature.vtau, 6956);
    }

    {
        const std::size_t nspin = 2;
        const elecstate::PotCiderNativeRuntime runtime(
            std::string(corpus_root) + "/functionals/CIDER23X_NL_GGA.yaml",
            shape,
            cell,
            nspin,
            0.25,
            20.0,
            1.8,
            8);
        const std::vector<double> rho_baseline = read_fixture_array(8820, 16);
        const std::vector<double> sigma_baseline = read_fixture_array(8948, 24);
        const std::vector<double> rho_feature = read_fixture_array(9140, 16);
        const std::vector<double> sigma_feature = read_fixture_array(9268, 24);
        const hamilt::cider::EvaluationInput input(
            nspin,
            ngrids,
            density_tuple(rho_baseline, sigma_baseline),
            density_tuple(rho_feature, sigma_feature));
        ComponentStorage baseline(ngrids, nspin, false);
        ComponentStorage feature(ngrids, nspin, false);
        runtime.evaluate(input, baseline.output(), feature.output());
        expect_fixture(baseline.exc, 15284);
        expect_fixture(baseline.vrho, 15348);
        expect_fixture(baseline.vsigma, 15476);
        expect_fixture(feature.exc, 15668);
        expect_fixture(feature.vrho, 15732);
        expect_fixture(feature.vsigma, 15860);
    }
}

TEST(PotCiderXCNative, RepeatsEvaluationAcrossRuntimeLifetimes)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the lifetime replay";
    }

    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    const std::string model_path
        = std::string(corpus_root) + "/functionals/CIDER23X_SL_GGA.yaml";
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

    for (int lifetime = 0; lifetime < 3; ++lifetime)
    {
        const elecstate::PotCiderNativeRuntime runtime(
            model_path, shape, cell, 1, 0.25);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            ComponentStorage baseline(ngrids);
            ComponentStorage feature(ngrids);
            runtime.evaluate(input, baseline.output(), feature.output());
            expect_fixture(baseline.exc, 1392);
            expect_fixture(feature.exc, 1584);
        }
    }
}

TEST(PotCiderXCNative, LeavesOutputsUnchangedOnErrorAndRemainsUsable)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the error replay";
    }

    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    const std::string model_path
        = std::string(corpus_root) + "/functionals/CIDER23X_SL_GGA.yaml";
    const elecstate::PotCiderNativeRuntime runtime(
        model_path, shape, cell, 1, 0.25);
    const std::size_t ngrids = 8;
    std::vector<double> rho_baseline = read_fixture_array(112, ngrids);
    const std::vector<double> sigma_baseline = read_fixture_array(176, ngrids);
    const std::vector<double> rho_feature = read_fixture_array(240, ngrids);
    const std::vector<double> sigma_feature = read_fixture_array(304, ngrids);
    rho_baseline[3] = std::numeric_limits<double>::quiet_NaN();
    const hamilt::cider::EvaluationInput invalid_input(
        1,
        ngrids,
        density_tuple(rho_baseline, sigma_baseline),
        density_tuple(rho_feature, sigma_feature));
    ComponentStorage baseline(ngrids);
    ComponentStorage feature(ngrids);

    EXPECT_THROW(runtime.evaluate(invalid_input, baseline.output(), feature.output()),
                 std::invalid_argument);
    EXPECT_EQ(std::vector<double>(ngrids, 91.0), baseline.exc);
    EXPECT_EQ(std::vector<double>(ngrids, 92.0), baseline.vrho);
    EXPECT_EQ(std::vector<double>(ngrids, 93.0), baseline.vsigma);
    EXPECT_EQ(std::vector<double>(ngrids, 91.0), feature.exc);
    EXPECT_EQ(std::vector<double>(ngrids, 92.0), feature.vrho);
    EXPECT_EQ(std::vector<double>(ngrids, 93.0), feature.vsigma);

    rho_baseline[3] = read_fixture_array(112, ngrids)[3];
    const hamilt::cider::EvaluationInput valid_input(
        1,
        ngrids,
        density_tuple(rho_baseline, sigma_baseline),
        density_tuple(rho_feature, sigma_feature));
    runtime.evaluate(valid_input, baseline.output(), feature.output());
    expect_fixture(baseline.exc, 1392);
    expect_fixture(feature.exc, 1584);
}

TEST(PotCiderXCNative, ReportsModelPathOnConstructionErrorWithoutPoisoningNextRuntime)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the construction replay";
    }

    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    const std::string missing = "/tmp/task12_missing_native_model.yaml";
    try
    {
        const elecstate::PotCiderNativeRuntime invalid(
            missing, shape, cell, 1, 0.25);
        FAIL() << "missing model unexpectedly loaded";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string::npos, std::string(error.what()).find(missing));
        EXPECT_NE(std::string::npos,
                  std::string(error.what()).find("native CIDER model"));
    }

    const elecstate::PotCiderNativeRuntime valid(
        std::string(corpus_root) + "/functionals/CIDER23X_SL_GGA.yaml",
        shape,
        cell,
        1,
        0.25);
    EXPECT_FALSE(valid.metadata().requires_tau);
}
