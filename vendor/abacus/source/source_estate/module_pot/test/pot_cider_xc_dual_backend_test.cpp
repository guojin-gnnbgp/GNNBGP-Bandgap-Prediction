#include "source_estate/module_pot/pot_cider_xc_runtime_factory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{

struct ComponentStorage
{
    ComponentStorage(std::size_t ngrids, std::size_t nspin, bool has_tau)
        : exc(ngrids, 0.0),
          vrho(nspin * ngrids, 0.0),
          vsigma((2 * nspin - 1) * ngrids, 0.0),
          vtau(has_tau ? nspin * ngrids : 0U, 0.0)
    {
    }

    hamilt::cider::ComponentOutput output()
    {
        return hamilt::cider::ComponentOutput(
            exc.data(),
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

struct DensityStorage
{
    DensityStorage(std::size_t ngrids, std::size_t nspin, bool has_tau, double shift)
        : rho(nspin * ngrids),
          sigma((2 * nspin - 1) * ngrids),
          tau(has_tau ? nspin * ngrids : 0U)
    {
        for (std::size_t i = 0; i < rho.size(); ++i)
        {
            rho[i] = 0.08 + shift + 0.003 * static_cast<double>(i);
            if (has_tau)
            {
                tau[i] = 0.11 + 0.002 * static_cast<double>(i);
            }
        }
        for (std::size_t i = 0; i < sigma.size(); ++i)
        {
            sigma[i] = 0.006 + 0.0004 * static_cast<double>(i);
        }
    }

    hamilt::cider::DensityTupleInput input() const
    {
        return hamilt::cider::DensityTupleInput(
            rho.data(),
            rho.size(),
            sigma.data(),
            sigma.size(),
            tau.empty() ? nullptr : tau.data(),
            tau.size());
    }

    std::vector<double> rho;
    std::vector<double> sigma;
    std::vector<double> tau;
};

double compare_values(const std::vector<double>& native_values,
                      const std::vector<double>& external_values,
                      double& max_relative)
{
    EXPECT_EQ(native_values.size(), external_values.size());
    double max_absolute = 0.0;
    for (std::size_t i = 0; i < native_values.size(); ++i)
    {
        const double absolute = std::abs(native_values[i] - external_values[i]);
        const double relative = absolute
                                / std::max(1.0e-14, std::abs(external_values[i]));
        max_absolute = std::max(max_absolute, absolute);
        max_relative = std::max(max_relative, relative);
        EXPECT_NEAR(native_values[i],
                    external_values[i],
                    2.0e-11 + 2.0e-10 * std::abs(external_values[i]))
            << i;
    }
    return max_absolute;
}

} // namespace

TEST(PotCiderXCDualBackend, MatchesFiveSupportedModelsInOneExecutable)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for dual-backend parity";
    }
    struct Case
    {
        const char* path;
        std::size_t nspin;
        bool has_tau;
    };
    const Case cases[] = {
        {"functionals/CIDER23X_SL_GGA.yaml", 1, false},
        {"functionals/CIDER23X_SL_MGGA.yaml", 2, true},
        {"functionals/CIDER23X_NL_GGA.yaml", 2, false},
        {"functionals/CIDER23X_NL_MGGA_DTR.yaml", 1, true},
        {"abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/full/nldf_mgga_cand00_n96_noise0p01.yaml", 2, true},
    };
    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    const std::size_t ngrids = 8;
    double max_absolute = 0.0;
    double max_relative = 0.0;

    for (const Case& item : cases)
    {
        SCOPED_TRACE(item.path);
        const std::string model_path = std::string(corpus_root) + "/" + item.path;
        std::unique_ptr<elecstate::PotCiderRuntime> native
            = elecstate::make_pot_cider_runtime(
                "native", model_path, shape, cell, item.nspin, 0.25);
        std::unique_ptr<elecstate::PotCiderRuntime> external
            = elecstate::make_pot_cider_runtime(
                "external", model_path, shape, cell, item.nspin, 0.25);
        EXPECT_EQ(native->metadata().requires_tau,
                  external->metadata().requires_tau);
        EXPECT_EQ(native->metadata().feature_density_policy,
                  external->metadata().feature_density_policy);
        EXPECT_EQ(native->metadata().feature_exc_convention,
                  external->metadata().feature_exc_convention);

        const DensityStorage baseline_density(
            ngrids, item.nspin, item.has_tau, 0.009);
        const DensityStorage feature_density(
            ngrids, item.nspin, item.has_tau, 0.0);
        const hamilt::cider::EvaluationInput input(
            item.nspin,
            ngrids,
            baseline_density.input(),
            feature_density.input());
        ComponentStorage native_baseline(ngrids, item.nspin, item.has_tau);
        ComponentStorage native_feature(ngrids, item.nspin, item.has_tau);
        ComponentStorage external_baseline(ngrids, item.nspin, item.has_tau);
        ComponentStorage external_feature(ngrids, item.nspin, item.has_tau);

        native->evaluate(
            input, native_baseline.output(), native_feature.output());
        external->evaluate(
            input, external_baseline.output(), external_feature.output());

        const std::vector<double>* native_arrays[] = {
            &native_baseline.exc,
            &native_baseline.vrho,
            &native_baseline.vsigma,
            &native_baseline.vtau,
            &native_feature.exc,
            &native_feature.vrho,
            &native_feature.vsigma,
            &native_feature.vtau,
        };
        const std::vector<double>* external_arrays[] = {
            &external_baseline.exc,
            &external_baseline.vrho,
            &external_baseline.vsigma,
            &external_baseline.vtau,
            &external_feature.exc,
            &external_feature.vrho,
            &external_feature.vsigma,
            &external_feature.vtau,
        };
        for (std::size_t i = 0; i < 8; ++i)
        {
            max_absolute = std::max(
                max_absolute,
                compare_values(*native_arrays[i],
                               *external_arrays[i],
                               max_relative));
        }
    }
    RecordProperty("dual_backend_max_absolute_error", max_absolute);
    RecordProperty("dual_backend_max_relative_error", max_relative);
}

TEST(PotCiderXCDualBackend, PreservesExternalOnlyEvaluatorWithoutFallback)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for external-only model";
    }
    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    const std::string model_path
        = std::string(corpus_root) + "/functionals/CIDER24Xe.yaml";

    EXPECT_NO_THROW(elecstate::make_pot_cider_runtime(
        "external", model_path, shape, cell, 1, 0.25));
    try
    {
        elecstate::make_pot_cider_runtime(
            "native", model_path, shape, cell, 1, 0.25);
        FAIL() << "native backend unexpectedly accepted NNEvaluator";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string::npos,
                  std::string(error.what()).find("UNSUPPORTED_EVALUATOR"));
    }
}
