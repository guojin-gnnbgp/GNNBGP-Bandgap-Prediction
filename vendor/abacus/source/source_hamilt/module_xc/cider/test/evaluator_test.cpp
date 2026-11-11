#include "source_hamilt/module_xc/cider/evaluator.h"
#include "source_hamilt/module_xc/cider/model_loader.h"
#include "source_hamilt/module_xc/cider/nldf_plan.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <array>
#include <fstream>
#include <memory>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
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

hamilt::cider::DensityTupleInput density_tuple(const std::vector<double>& rho,
                                               const std::vector<double>& sigma,
                                               const std::vector<double>& tau)
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
    ComponentStorage(std::size_t ngrids, std::size_t nrho, std::size_t nsigma, bool has_tau)
        : exc(ngrids, 91.0),
          vrho(nrho, 92.0),
          vsigma(nsigma, 93.0),
          vtau(has_tau ? nrho : 0, 94.0)
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

struct V1Case
{
    const char* name;
    const char* model_path;
    std::size_t nspin;
    bool mgga;
    bool nldf;
    std::size_t rho_baseline;
    std::size_t sigma_baseline;
    std::size_t tau_baseline;
    std::size_t rho_feature;
    std::size_t sigma_feature;
    std::size_t tau_feature;
    std::size_t exc_baseline;
    std::size_t vrho_baseline;
    std::size_t vsigma_baseline;
    std::size_t vtau_baseline;
    std::size_t exc_feature;
    std::size_t vrho_feature;
    std::size_t vsigma_feature;
    std::size_t vtau_feature;
    std::size_t exc_total;
    std::size_t vrho_total;
    std::size_t vsigma_total;
    std::size_t vtau_total;
    std::size_t integrated;
};

const V1Case v1_cases[] = {
    {"sl_gga_spin1_valence", "functionals/CIDER23X_SL_GGA.yaml", 1, false, false,
     112, 176, 0, 240, 304, 0, 1392, 1456, 1520, 0, 1584, 1648, 1712, 0,
     1776, 1840, 1904, 0, 1968},
    {"sl_mgga_spin2_valence", "functionals/CIDER23X_SL_MGGA.yaml", 2, true, false,
     2156, 2284, 2476, 2604, 2732, 2924, 6060, 6124, 6252, 6444, 6572, 6636,
     6764, 6956, 7084, 7148, 7276, 7468, 7596},
    {"nl_gga_spin2_valence_pseudo_core", "functionals/CIDER23X_NL_GGA.yaml", 2,
     false, true, 8820, 8948, 0, 9140, 9268, 0, 15284, 15348, 15476, 0, 15668,
     15732, 15860, 0, 16052, 16116, 16244, 0, 16436},
    {"nl_mgga_spin1_valence_pseudo_core", "functionals/CIDER23X_NL_MGGA_DTR.yaml",
     1, true, true, 18332, 18396, 18460, 18524, 18588, 18652, 22300, 22364,
     22428, 22492, 22556, 22620, 22684, 22748, 22812, 22876, 22940, 23004, 23068},
    {"trained_nl_mgga_spin2_valence",
     "abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/full/nldf_mgga_cand00_n96_noise0p01.yaml",
     2, true, true, 24964, 25092, 25284, 25412, 25540, 25732, 32964, 33028,
     33156, 33348, 33476, 33540, 33668, 33860, 33988, 34052, 34180, 34372, 34500}};

class SyntheticModel final : public hamilt::cider::Model
{
  public:
    SyntheticModel(hamilt::cider::SemilocalLevel level,
                   double ml_exchange_scale,
                   double kernel_output_scale = 1.0)
        : metadata_(level,
                    false,
                    level == hamilt::cider::SemilocalLevel::Gga ? 2 : 3,
                    0,
                    1,
                    hamilt::cider::FeatureDensityPolicy::Valence,
                    "synthetic_valence",
                    kernel_output_scale,
                    ml_exchange_scale,
                    hamilt::cider::TrainingLabelMetadata()),
          normalizers_(metadata_.raw_feature_count(),
                       {hamilt::cider::NormalizerKind::Identity, 1.0}),
          maps_(1,
                {hamilt::cider::FeatureMapKind::U,
                 metadata_.raw_feature_count() - 1,
                 0.7,
                 1.0,
                 0.0,
                 0,
                 0.0,
                 1.0}),
          terms_(1,
                 {0.6,
                  {0},
                  {{0.0, 1.0, 3}},
                  {5},
                  {0.1, -0.2, 0.35, 0.17, -0.08}})
    {
    }

    const hamilt::cider::ModelMetadata& metadata() const override { return metadata_; }
    hamilt::cider::SemilocalMode semilocal_mode() const override
    {
        return metadata_.is_mgga() ? hamilt::cider::SemilocalMode::Npa
                                   : hamilt::cider::SemilocalMode::Np;
    }
    const hamilt::cider::NldfSettings* nldf_settings() const override { return nullptr; }
    double normalizer_cutoff() const override { return 0.0; }
    const std::vector<hamilt::cider::Normalizer>& normalizers() const override
    {
        return normalizers_;
    }
    const std::vector<hamilt::cider::FeatureMap>& feature_maps() const override
    {
        return maps_;
    }
    hamilt::cider::MultiplicativeBaseline multiplicative_baseline() const override
    {
        return hamilt::cider::MultiplicativeBaseline::LdaX;
    }
    hamilt::cider::AdditiveBaseline additive_baseline() const override
    {
        return hamilt::cider::AdditiveBaseline::GgaXPbe;
    }
    double spline_constant() const override { return 0.13; }
    const std::vector<hamilt::cider::SplineTerm>& spline_terms() const override
    {
        return terms_;
    }

  private:
    const hamilt::cider::ModelMetadata metadata_;
    const std::vector<hamilt::cider::Normalizer> normalizers_;
    const std::vector<hamilt::cider::FeatureMap> maps_;
    const std::vector<hamilt::cider::SplineTerm> terms_;
};

class AdversarialModel final : public hamilt::cider::Model
{
  public:
    AdversarialModel(hamilt::cider::SemilocalLevel level,
                     hamilt::cider::SemilocalMode mode,
                     bool has_nldf,
                     std::size_t semilocal_count,
                     std::size_t nldf_count,
                     std::size_t transformed_count,
                     bool settings_present,
                     hamilt::cider::SemilocalLevel settings_level,
                     std::size_t settings_feature_count,
                     std::size_t normalizer_count,
                     std::size_t map_count)
        : metadata_(level,
                    has_nldf,
                    semilocal_count,
                    nldf_count,
                    transformed_count,
                    hamilt::cider::FeatureDensityPolicy::Valence,
                    "adversarial",
                    1.0,
                    1.0,
                    hamilt::cider::TrainingLabelMetadata()),
          mode_(mode),
          settings_present_(settings_present),
          settings_(settings_level,
                    hamilt::cider::ParameterTriple(1.0, 0.2, 0.1),
                    std::vector<hamilt::cider::ParameterTriple>(
                        settings_feature_count,
                        hamilt::cider::ParameterTriple(1.2, 0.3, 0.1))),
          normalizers_(normalizer_count,
                       {hamilt::cider::NormalizerKind::Identity, 1.0}),
          maps_(map_count,
                {hamilt::cider::FeatureMapKind::U,
                 1,
                 0.7,
                 1.0,
                 0.0,
                 0,
                 0.0,
                 1.0})
    {
    }

    const hamilt::cider::ModelMetadata& metadata() const override { return metadata_; }
    hamilt::cider::SemilocalMode semilocal_mode() const override { return mode_; }
    const hamilt::cider::NldfSettings* nldf_settings() const override
    {
        return settings_present_ ? &settings_ : nullptr;
    }
    double normalizer_cutoff() const override { return 0.0; }
    const std::vector<hamilt::cider::Normalizer>& normalizers() const override
    {
        return normalizers_;
    }
    const std::vector<hamilt::cider::FeatureMap>& feature_maps() const override
    {
        return maps_;
    }
    hamilt::cider::MultiplicativeBaseline multiplicative_baseline() const override
    {
        return hamilt::cider::MultiplicativeBaseline::LdaX;
    }
    hamilt::cider::AdditiveBaseline additive_baseline() const override
    {
        return hamilt::cider::AdditiveBaseline::GgaXPbe;
    }
    double spline_constant() const override { return 0.0; }
    const std::vector<hamilt::cider::SplineTerm>& spline_terms() const override
    {
        return terms_;
    }

  private:
    const hamilt::cider::ModelMetadata metadata_;
    const hamilt::cider::SemilocalMode mode_;
    const bool settings_present_;
    const hamilt::cider::NldfSettings settings_;
    const std::vector<hamilt::cider::Normalizer> normalizers_;
    const std::vector<hamilt::cider::FeatureMap> maps_;
    const std::vector<hamilt::cider::SplineTerm> terms_;
};

void evaluate_into(const hamilt::cider::DualDensityEvaluator& evaluator,
                   std::size_t nspin,
                   std::size_t ngrids,
                   const std::vector<double>& rho_baseline,
                   const std::vector<double>& sigma_baseline,
                   const std::vector<double>& tau_baseline,
                   const std::vector<double>& rho_feature,
                   const std::vector<double>& sigma_feature,
                   const std::vector<double>& tau_feature,
                   ComponentStorage& baseline,
                   ComponentStorage& feature,
                   ComponentStorage& total,
                   hamilt::cider::EvaluationDiagnostics* diagnostics = nullptr)
{
    const hamilt::cider::EvaluationInput input(
        nspin,
        ngrids,
        density_tuple(rho_baseline, sigma_baseline, tau_baseline),
        density_tuple(rho_feature, sigma_feature, tau_feature));
    hamilt::cider::EvaluationOutput output(
        baseline.output(), feature.output(), total.output());
    evaluator.evaluate(input, output);
    if (diagnostics != nullptr)
    {
        *diagnostics = output.diagnostics();
    }
}

double update_errors(const std::vector<double>& expected,
                     const std::vector<double>& actual,
                     double& max_relative)
{
    EXPECT_EQ(expected.size(), actual.size());
    double max_absolute = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const double error = std::abs(expected[i] - actual[i]);
        max_absolute = std::max(max_absolute, error);
        if (expected[i] != 0.0)
        {
            max_relative = std::max(max_relative, error / std::abs(expected[i]));
        }
        EXPECT_LE(error, 1.0e-12) << i;
        EXPECT_LE(error, 1.0e-12 + 1.0e-10 * std::abs(expected[i])) << i;
    }
    return max_absolute;
}

TEST(NativeCiderEvaluator, ExposesConcreteImmutableTopLevelEvaluator)
{
    EXPECT_FALSE(std::is_abstract<hamilt::cider::DualDensityEvaluator>::value);
    EXPECT_TRUE((std::is_base_of<hamilt::cider::Evaluator,
                                 hamilt::cider::DualDensityEvaluator>::value));
    EXPECT_FALSE(std::is_copy_constructible<hamilt::cider::DualDensityEvaluator>::value);
    EXPECT_FALSE(std::is_copy_assignable<hamilt::cider::DualDensityEvaluator>::value);
}

TEST(NativeCiderEvaluator, MatchesV1SpinOneGgaDualDensityOracle)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    const hamilt::cider::ModelLoader loader;
    const std::unique_ptr<const hamilt::cider::Model> model
        = loader.load(std::string(parent) + "/functionals/CIDER23X_SL_GGA.yaml");
    const std::size_t ngrids = 8;
    const std::vector<double> rho_baseline = read_fixture_array(112, 8);
    const std::vector<double> sigma_baseline = read_fixture_array(176, 8);
    const std::vector<double> rho_feature = read_fixture_array(240, 8);
    const std::vector<double> sigma_feature = read_fixture_array(304, 8);
    const std::vector<double> no_tau;
    const hamilt::cider::EvaluationInput input(
        1,
        ngrids,
        density_tuple(rho_baseline, sigma_baseline, no_tau),
        density_tuple(rho_feature, sigma_feature, no_tau));
    ComponentStorage baseline(ngrids, 8, 8, false);
    ComponentStorage feature(ngrids, 8, 8, false);
    ComponentStorage total(ngrids, 8, 8, false);
    hamilt::cider::EvaluationOutput output(
        baseline.output(), feature.output(), total.output());

    const hamilt::cider::DualDensityEvaluator evaluator(*model, 1, 0.25);
    evaluator.evaluate(input, output);

    const std::vector<double> expected_baseline_exc = read_fixture_array(1392, 8);
    const std::vector<double> expected_baseline_vrho = read_fixture_array(1456, 8);
    const std::vector<double> expected_baseline_vsigma = read_fixture_array(1520, 8);
    const std::vector<double> expected_feature_exc = read_fixture_array(1584, 8);
    const std::vector<double> expected_feature_vrho = read_fixture_array(1648, 8);
    const std::vector<double> expected_feature_vsigma = read_fixture_array(1712, 8);
    const std::vector<double> expected_total_exc = read_fixture_array(1776, 8);
    const std::vector<double> expected_total_vrho = read_fixture_array(1840, 8);
    const std::vector<double> expected_total_vsigma = read_fixture_array(1904, 8);
    const std::vector<std::vector<double>*> actual = {
        &baseline.exc,
        &baseline.vrho,
        &baseline.vsigma,
        &feature.exc,
        &feature.vrho,
        &feature.vsigma,
        &total.exc,
        &total.vrho,
        &total.vsigma};
    const std::vector<const std::vector<double>*> expected = {
        &expected_baseline_exc,
        &expected_baseline_vrho,
        &expected_baseline_vsigma,
        &expected_feature_exc,
        &expected_feature_vrho,
        &expected_feature_vsigma,
        &expected_total_exc,
        &expected_total_vrho,
        &expected_total_vsigma};
    for (std::size_t array = 0; array < actual.size(); ++array)
    {
        for (std::size_t i = 0; i < actual[array]->size(); ++i)
        {
            EXPECT_NEAR((*expected[array])[i], (*actual[array])[i], 1.0e-12)
                << "array=" << array << " index=" << i;
        }
    }
    EXPECT_EQ(hamilt::cider::FeatureDensityPolicy::Valence,
              output.diagnostics().feature_density_policy());
    EXPECT_EQ(hamilt::cider::EnergyConvention::EnergyDensity,
              output.diagnostics().feature_energy_convention());
    EXPECT_EQ(hamilt::cider::BaselineExcConvention::LibxcExcPerParticle,
              output.diagnostics().baseline_exc_convention());
    EXPECT_DOUBLE_EQ(0.25, output.diagnostics().cider_xmix());
    EXPECT_DOUBLE_EQ(0.25, output.diagnostics().runtime_scale());
}

TEST(NativeCiderEvaluator, MatchesAllFiveV1DualDensityAndIntegratedOracles)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    const std::size_t ngrids = 8;
    const std::size_t grid_shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
    const hamilt::cider::NldfGrid grid(grid_shape, cell);
    const hamilt::cider::NldfAlphaSelection alpha_selection(20.0, 1.8, 8);
    double max_absolute = 0.0;
    double max_relative = 0.0;
    double max_integrated = 0.0;
    for (const V1Case& item : v1_cases)
    {
        SCOPED_TRACE(item.name);
        const std::size_t nrho = item.nspin * ngrids;
        const std::size_t nsigma = (2 * item.nspin - 1) * ngrids;
        const std::vector<double> rho_baseline
            = read_fixture_array(item.rho_baseline, nrho);
        const std::vector<double> sigma_baseline
            = read_fixture_array(item.sigma_baseline, nsigma);
        const std::vector<double> tau_baseline
            = item.mgga ? read_fixture_array(item.tau_baseline, nrho)
                        : std::vector<double>();
        const std::vector<double> rho_feature
            = read_fixture_array(item.rho_feature, nrho);
        const std::vector<double> sigma_feature
            = read_fixture_array(item.sigma_feature, nsigma);
        const std::vector<double> tau_feature
            = item.mgga ? read_fixture_array(item.tau_feature, nrho)
                        : std::vector<double>();
        const hamilt::cider::EvaluationInput input(
            item.nspin,
            ngrids,
            density_tuple(rho_baseline, sigma_baseline, tau_baseline),
            density_tuple(rho_feature, sigma_feature, tau_feature));
        ComponentStorage baseline(ngrids, nrho, nsigma, item.mgga);
        ComponentStorage feature(ngrids, nrho, nsigma, item.mgga);
        ComponentStorage total(ngrids, nrho, nsigma, item.mgga);
        hamilt::cider::EvaluationOutput output(
            baseline.output(), feature.output(), total.output());
        const hamilt::cider::ModelLoader loader;
        const std::unique_ptr<const hamilt::cider::Model> model
            = loader.load(std::string(parent) + "/" + item.model_path);
        std::unique_ptr<hamilt::cider::DualDensityEvaluator> evaluator;
        if (item.nldf)
        {
            evaluator.reset(new hamilt::cider::DualDensityEvaluator(
                *model, item.nspin, 0.25, grid, alpha_selection));
        }
        else
        {
            evaluator.reset(
                new hamilt::cider::DualDensityEvaluator(*model, item.nspin, 0.25));
        }
        evaluator->evaluate(input, output);

        struct Check
        {
            std::size_t offset;
            const std::vector<double>* actual;
        };
        const Check checks[] = {{item.exc_baseline, &baseline.exc},
                                {item.vrho_baseline, &baseline.vrho},
                                {item.vsigma_baseline, &baseline.vsigma},
                                {item.exc_feature, &feature.exc},
                                {item.vrho_feature, &feature.vrho},
                                {item.vsigma_feature, &feature.vsigma},
                                {item.exc_total, &total.exc},
                                {item.vrho_total, &total.vrho},
                                {item.vsigma_total, &total.vsigma}};
        for (const Check& check : checks)
        {
            SCOPED_TRACE(check.offset);
            const std::vector<double> expected
                = read_fixture_array(check.offset, check.actual->size());
            max_absolute = std::max(
                max_absolute, update_errors(expected, *check.actual, max_relative));
        }
        if (item.mgga)
        {
            const Check tau_checks[] = {{item.vtau_baseline, &baseline.vtau},
                                        {item.vtau_feature, &feature.vtau},
                                        {item.vtau_total, &total.vtau}};
            for (const Check& check : tau_checks)
            {
                SCOPED_TRACE(check.offset);
                const std::vector<double> expected
                    = read_fixture_array(check.offset, check.actual->size());
                max_absolute = std::max(
                    max_absolute, update_errors(expected, *check.actual, max_relative));
            }
        }

        const std::vector<double> expected_integrated
            = read_fixture_array(item.integrated, 6);
        std::vector<double> integrated(6, 0.0);
        integrated[0] = 8.0;
        integrated[1] = output.diagnostics().baseline_exc_sum() * 8.0;
        integrated[2] = output.diagnostics().feature_exc_sum() * 8.0;
        integrated[3] = output.diagnostics().total_exc_sum() * 8.0;
        for (std::size_t i = 0; i < nrho; ++i)
        {
            integrated[4] += baseline.vrho[i] * rho_baseline[i];
            integrated[5] += feature.vrho[i] * rho_feature[i];
            if (item.mgga)
            {
                integrated[4] += baseline.vtau[i] * tau_baseline[i];
                integrated[5] += feature.vtau[i] * tau_feature[i];
            }
        }
        for (std::size_t i = 0; i < nsigma; ++i)
        {
            integrated[4] += baseline.vsigma[i] * sigma_baseline[i];
            integrated[5] += feature.vsigma[i] * sigma_feature[i];
        }
        integrated[4] *= 8.0;
        integrated[5] *= 8.0;
        double integrated_relative = 0.0;
        max_integrated = std::max(max_integrated,
                                  update_errors(expected_integrated,
                                                integrated,
                                                integrated_relative));
    }
    RecordProperty("v1_dual_max_absolute_error", max_absolute);
    RecordProperty("v1_dual_max_relative_error", max_relative);
    RecordProperty("v1_integrated_max_absolute_error", max_integrated);
}

TEST(NativeCiderEvaluator, AppliesRuntimeAndOuterExchangeScalesExactlyOnce)
{
    const std::size_t ngrids = 2;
    const std::vector<double> rho = {0.21, 0.37};
    const std::vector<double> sigma = {0.04, 0.09};
    const std::vector<double> no_tau;
    const auto run = [&](double trained_scale,
                         double xmix,
                         ComponentStorage& baseline,
                         ComponentStorage& feature,
                         ComponentStorage& total,
                         hamilt::cider::EvaluationDiagnostics& diagnostics) {
        const SyntheticModel model(hamilt::cider::SemilocalLevel::Gga, trained_scale);
        const hamilt::cider::DualDensityEvaluator evaluator(model, 1, xmix);
        evaluate_into(evaluator,
                      1,
                      ngrids,
                      rho,
                      sigma,
                      no_tau,
                      rho,
                      sigma,
                      no_tau,
                      baseline,
                      feature,
                      total,
                      &diagnostics);
    };
    ComponentStorage b_ref(2, 2, 2, false), f_ref(2, 2, 2, false), t_ref(2, 2, 2, false);
    hamilt::cider::EvaluationDiagnostics d_ref;
    run(1.0, 1.0, b_ref, f_ref, t_ref, d_ref);
    ComponentStorage b_zero(2, 2, 2, false), f_zero(2, 2, 2, false), t_zero(2, 2, 2, false);
    hamilt::cider::EvaluationDiagnostics d_zero;
    run(1.0, 0.0, b_zero, f_zero, t_zero, d_zero);
    for (double value : f_zero.exc)
    {
        EXPECT_DOUBLE_EQ(0.0, value);
    }
    for (double value : f_zero.vrho)
    {
        EXPECT_DOUBLE_EQ(0.0, value);
    }
    for (double value : f_zero.vsigma)
    {
        EXPECT_DOUBLE_EQ(0.0, value);
    }
    EXPECT_DOUBLE_EQ(0.0, d_zero.runtime_scale());

    const double scales[] = {0.0, 0.5, 1.0};
    const double expected_runtime[] = {0.25, 0.5, 0.25};
    for (std::size_t scale_index = 0; scale_index < 3; ++scale_index)
    {
        ComponentStorage baseline(2, 2, 2, false), feature(2, 2, 2, false), total(2, 2, 2, false);
        hamilt::cider::EvaluationDiagnostics diagnostics;
        run(scales[scale_index], 0.25, baseline, feature, total, diagnostics);
        EXPECT_DOUBLE_EQ(expected_runtime[scale_index], diagnostics.runtime_scale());
        for (std::size_t i = 0; i < feature.exc.size(); ++i)
        {
            EXPECT_NEAR(expected_runtime[scale_index] * f_ref.exc[i], feature.exc[i], 1.0e-14);
            EXPECT_NEAR(0.75 * b_zero.exc[i] + 0.25 * b_ref.exc[i], baseline.exc[i], 1.0e-14);
        }
        for (std::size_t i = 0; i < feature.vrho.size(); ++i)
        {
            EXPECT_NEAR(expected_runtime[scale_index] * f_ref.vrho[i], feature.vrho[i], 1.0e-14);
            EXPECT_NEAR(0.75 * b_zero.vrho[i] + 0.25 * b_ref.vrho[i], baseline.vrho[i], 1.0e-14);
        }
        for (std::size_t i = 0; i < feature.vsigma.size(); ++i)
        {
            EXPECT_NEAR(expected_runtime[scale_index] * f_ref.vsigma[i], feature.vsigma[i], 1.0e-14);
            EXPECT_NEAR(0.75 * b_zero.vsigma[i] + 0.25 * b_ref.vsigma[i], baseline.vsigma[i], 1.0e-14);
        }
    }
    ComponentStorage b_two(2, 2, 2, false), f_two(2, 2, 2, false), t_two(2, 2, 2, false);
    hamilt::cider::EvaluationDiagnostics d_two;
    run(0.5, 1.0, b_two, f_two, t_two, d_two);
    EXPECT_DOUBLE_EQ(2.0, d_two.runtime_scale());
    for (std::size_t i = 0; i < f_two.exc.size(); ++i)
    {
        EXPECT_NEAR(2.0 * f_ref.exc[i], f_two.exc[i], 1.0e-14);
    }
}

TEST(NativeCiderEvaluator, RoutesPseudoCoreOnlyToCallerSelectedDensityRoles)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    const hamilt::cider::ModelLoader loader;
    const std::unique_ptr<const hamilt::cider::Model> model
        = loader.load(std::string(parent) + "/functionals/CIDER23X_SL_GGA.yaml");
    ASSERT_EQ(hamilt::cider::FeatureDensityPolicy::Valence,
              model->metadata().feature_density_policy());
    const hamilt::cider::DualDensityEvaluator evaluator(*model, 1, 0.25);
    const std::vector<double> rho = read_fixture_array(240, 8);
    const std::vector<double> sigma = read_fixture_array(304, 8);
    std::vector<double> rho_core = rho;
    std::vector<double> sigma_core = sigma;
    for (std::size_t i = 0; i < rho.size(); ++i)
    {
        rho_core[i] += 0.015 + 0.001 * static_cast<double>(i);
        sigma_core[i] += 0.004 + 0.0003 * static_cast<double>(i);
    }
    const std::vector<double> no_tau;
    ComponentStorage b0(8, 8, 8, false), f0(8, 8, 8, false), t0(8, 8, 8, false);
    ComponentStorage b1(8, 8, 8, false), f1(8, 8, 8, false), t1(8, 8, 8, false);
    ComponentStorage b2(8, 8, 8, false), f2(8, 8, 8, false), t2(8, 8, 8, false);
    evaluate_into(evaluator, 1, 8, rho, sigma, no_tau, rho, sigma, no_tau, b0, f0, t0);
    evaluate_into(evaluator, 1, 8, rho_core, sigma_core, no_tau, rho, sigma, no_tau, b1, f1, t1);
    evaluate_into(evaluator, 1, 8, rho_core, sigma_core, no_tau,
                  rho_core, sigma_core, no_tau, b2, f2, t2);
    EXPECT_EQ(f0.exc, f1.exc);
    EXPECT_EQ(f0.vrho, f1.vrho);
    EXPECT_EQ(f0.vsigma, f1.vsigma);
    EXPECT_NE(b0.exc, b1.exc);
    EXPECT_NE(f1.exc, f2.exc);
    EXPECT_NE(f1.vrho, f2.vrho);
}

TEST(NativeCiderEvaluator, RegularizesOnlyFeatureRhoSameSpinSigmaAndTau)
{
    const SyntheticModel model(hamilt::cider::SemilocalLevel::Mgga, 1.0);
    const hamilt::cider::DualDensityEvaluator evaluator(model, 2, 0.4);
    const std::vector<double> rho_baseline = {0.25, 0.31};
    const std::vector<double> sigma_baseline = {0.04, 0.02, 0.07};
    const std::vector<double> tau_baseline = {0.12, 0.16};
    const std::vector<double> rho_feature = {-0.2, 0.31};
    const std::vector<double> sigma_feature = {-0.4, -9.0, -0.7};
    const std::vector<double> tau_feature = {-0.3, 0.16};
    ComponentStorage baseline(1, 2, 3, true), feature(1, 2, 3, true), total(1, 2, 3, true);
    hamilt::cider::EvaluationDiagnostics diagnostics;
    evaluate_into(evaluator,
                  2,
                  1,
                  rho_baseline,
                  sigma_baseline,
                  tau_baseline,
                  rho_feature,
                  sigma_feature,
                  tau_feature,
                  baseline,
                  feature,
                  total,
                  &diagnostics);
    EXPECT_EQ(1U, diagnostics.regularized_rho_count());
    EXPECT_EQ(2U, diagnostics.regularized_same_spin_sigma_count());
    EXPECT_EQ(1U, diagnostics.regularized_tau_count());
    EXPECT_DOUBLE_EQ(0.0, feature.vsigma[1]);
    for (const std::vector<double>* values : {&feature.exc, &feature.vrho, &feature.vsigma,
                                              &feature.vtau, &total.exc, &total.vrho,
                                              &total.vsigma, &total.vtau})
    {
        for (double value : *values)
        {
            EXPECT_TRUE(std::isfinite(value));
        }
    }
}

TEST(NativeCiderEvaluator, FeatureDerivativesMatchPhysicalTupleFiniteDifferences)
{
    const SyntheticModel model(hamilt::cider::SemilocalLevel::Mgga, 0.5);
    const hamilt::cider::DualDensityEvaluator evaluator(model, 2, 0.35);
    const std::size_t nspin = 2;
    const std::size_t ngrids = 2;
    const std::vector<double> rho_baseline = {0.23, 0.29, 0.19, 0.25};
    const std::vector<double> sigma_baseline = {0.04, 0.05, 0.01, 0.02, 0.06, 0.07};
    const std::vector<double> tau_baseline = {0.11, 0.14, 0.09, 0.13};
    std::vector<double> rho_feature = {0.21, 0.27, 0.17, 0.24};
    std::vector<double> sigma_feature = {0.035, 0.048, 0.012, 0.018, 0.055, 0.068};
    std::vector<double> tau_feature = {0.10, 0.13, 0.08, 0.12};
    ComponentStorage baseline(ngrids, 4, 6, true), feature(ngrids, 4, 6, true), total(ngrids, 4, 6, true);
    evaluate_into(evaluator, nspin, ngrids, rho_baseline, sigma_baseline, tau_baseline,
                  rho_feature, sigma_feature, tau_feature, baseline, feature, total);
    const double step = 1.0e-6;
    double max_error = 0.0;
    const auto check_buffer = [&](std::vector<double>& values,
                                  const std::vector<double>& derivative,
                                  const char* label) {
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            double objective[2] = {0.0, 0.0};
            for (int sign = 0; sign < 2; ++sign)
            {
                values[index] += sign == 0 ? step : -2.0 * step;
                ComponentStorage displaced_baseline(ngrids, 4, 6, true);
                ComponentStorage displaced_feature(ngrids, 4, 6, true);
                ComponentStorage displaced_total(ngrids, 4, 6, true);
                evaluate_into(evaluator, nspin, ngrids,
                              rho_baseline, sigma_baseline, tau_baseline,
                              rho_feature, sigma_feature, tau_feature,
                              displaced_baseline, displaced_feature, displaced_total);
                objective[sign] = std::accumulate(displaced_feature.exc.begin(),
                                                  displaced_feature.exc.end(), 0.0);
            }
            values[index] += step;
            const double finite_difference = (objective[0] - objective[1]) / (2.0 * step);
            const double error = std::abs(finite_difference - derivative[index]);
            max_error = std::max(max_error, error);
            EXPECT_NEAR(derivative[index], finite_difference, 3.0e-8)
                << label << " index=" << index;
        }
    };
    check_buffer(rho_feature, feature.vrho, "rho");
    check_buffer(sigma_feature, feature.vsigma, "sigma");
    check_buffer(tau_feature, feature.vtau, "tau");
    EXPECT_DOUBLE_EQ(0.0, feature.vsigma[2]);
    EXPECT_DOUBLE_EQ(0.0, feature.vsigma[3]);
    RecordProperty("physical_tuple_finite_difference_max_abs_error", max_error);
}

TEST(NativeCiderEvaluator, RejectsInvalidInputsAndOverlapsWithoutMutation)
{
    const SyntheticModel model(hamilt::cider::SemilocalLevel::Gga, 1.0);
    const SyntheticModel negative_scale_model(hamilt::cider::SemilocalLevel::Gga,
                                              -0.1);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(negative_scale_model, 1, 0.25),
                 std::invalid_argument);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(model, 1, -0.1), std::invalid_argument);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(model, 1, 1.1), std::invalid_argument);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(
                     model, 1, std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    const hamilt::cider::DualDensityEvaluator evaluator(model, 1, 0.25);
    std::vector<double> rho_baseline = {0.23};
    std::vector<double> sigma_baseline = {0.04};
    std::vector<double> rho_feature = {std::numeric_limits<double>::infinity()};
    std::vector<double> sigma_feature = {0.03};
    const std::vector<double> no_tau;
    ComponentStorage baseline(1, 1, 1, false), feature(1, 1, 1, false), total(1, 1, 1, false);
    const ComponentStorage expected_baseline = baseline;
    const ComponentStorage expected_feature = feature;
    const ComponentStorage expected_total = total;
    const hamilt::cider::EvaluationInput input(
        1, 1, density_tuple(rho_baseline, sigma_baseline, no_tau),
        density_tuple(rho_feature, sigma_feature, no_tau));
    hamilt::cider::EvaluationOutput output(baseline.output(), feature.output(), total.output());
    EXPECT_THROW(evaluator.evaluate(input, output), std::invalid_argument);
    EXPECT_EQ(expected_baseline.exc, baseline.exc);
    EXPECT_EQ(expected_baseline.vrho, baseline.vrho);
    EXPECT_EQ(expected_baseline.vsigma, baseline.vsigma);
    EXPECT_EQ(expected_feature.exc, feature.exc);
    EXPECT_EQ(expected_feature.vrho, feature.vrho);
    EXPECT_EQ(expected_feature.vsigma, feature.vsigma);
    EXPECT_EQ(expected_total.exc, total.exc);
    EXPECT_EQ(expected_total.vrho, total.vrho);
    EXPECT_EQ(expected_total.vsigma, total.vsigma);
    EXPECT_DOUBLE_EQ(0.0, output.diagnostics().cider_xmix());

    rho_feature[0] = 0.21;
    const hamilt::cider::EvaluationInput finite_input(
        1, 1, density_tuple(rho_baseline, sigma_baseline, no_tau),
        density_tuple(rho_feature, sigma_feature, no_tau));
    const hamilt::cider::ComponentOutput overlapping_total(
        baseline.exc.data(), 1, total.vrho.data(), 1, total.vsigma.data(), 1);
    hamilt::cider::EvaluationOutput overlap_output(
        baseline.output(), feature.output(), overlapping_total);
    EXPECT_THROW(evaluator.evaluate(finite_input, overlap_output), std::invalid_argument);
    EXPECT_EQ(expected_baseline.exc, baseline.exc);
    EXPECT_EQ(expected_feature.exc, feature.exc);
    EXPECT_EQ(expected_total.vrho, total.vrho);
}

TEST(NativeCiderEvaluator, RejectsAdversarialModelCountsAndNldfProfilesBeforeAllocation)
{
    const std::size_t shape[3] = {1, 1, 1};
    const double cell[9] = {3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0};
    const hamilt::cider::NldfGrid grid(shape, cell);
    const hamilt::cider::NldfAlphaSelection selection(20.0, 1.8, 8);
    const std::size_t huge = std::numeric_limits<std::size_t>::max();

    const AdversarialModel checked_add_overflow(
        hamilt::cider::SemilocalLevel::Gga,
        hamilt::cider::SemilocalMode::Np,
        true,
        2,
        huge,
        1,
        true,
        hamilt::cider::SemilocalLevel::Gga,
        1,
        0,
        0);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(
                     checked_add_overflow, 1, 0.25, grid, selection),
                 std::overflow_error);

    const AdversarialModel huge_semilocal(
        hamilt::cider::SemilocalLevel::Gga,
        hamilt::cider::SemilocalMode::Np,
        false,
        huge,
        0,
        1,
        false,
        hamilt::cider::SemilocalLevel::Gga,
        0,
        0,
        1);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(huge_semilocal, 1, 0.25),
                 std::invalid_argument);

    const AdversarialModel huge_transformed(
        hamilt::cider::SemilocalLevel::Gga,
        hamilt::cider::SemilocalMode::Np,
        false,
        2,
        0,
        huge,
        false,
        hamilt::cider::SemilocalLevel::Gga,
        0,
        2,
        0);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(huge_transformed, 1, 0.25),
                 std::invalid_argument);

    const AdversarialModel count_without_nldf(
        hamilt::cider::SemilocalLevel::Gga,
        hamilt::cider::SemilocalMode::Np,
        false,
        2,
        1,
        1,
        false,
        hamilt::cider::SemilocalLevel::Gga,
        0,
        3,
        1);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(count_without_nldf, 1, 0.25),
                 std::invalid_argument);

    const AdversarialModel flag_without_count(
        hamilt::cider::SemilocalLevel::Gga,
        hamilt::cider::SemilocalMode::Np,
        true,
        2,
        0,
        1,
        true,
        hamilt::cider::SemilocalLevel::Gga,
        1,
        2,
        1);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(
                     flag_without_count, 1, 0.25, grid, selection),
                 std::invalid_argument);

    const AdversarialModel settings_without_flag(
        hamilt::cider::SemilocalLevel::Gga,
        hamilt::cider::SemilocalMode::Np,
        false,
        2,
        0,
        1,
        true,
        hamilt::cider::SemilocalLevel::Gga,
        1,
        2,
        1);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(settings_without_flag, 1, 0.25),
                 std::invalid_argument);

    const AdversarialModel mismatched_setting_count(
        hamilt::cider::SemilocalLevel::Gga,
        hamilt::cider::SemilocalMode::Np,
        true,
        2,
        2,
        1,
        true,
        hamilt::cider::SemilocalLevel::Gga,
        1,
        4,
        1);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(
                     mismatched_setting_count, 1, 0.25, grid, selection),
                 std::invalid_argument);

    const AdversarialModel mismatched_setting_level(
        hamilt::cider::SemilocalLevel::Gga,
        hamilt::cider::SemilocalMode::Np,
        true,
        2,
        1,
        1,
        true,
        hamilt::cider::SemilocalLevel::Mgga,
        1,
        3,
        1);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(
                     mismatched_setting_level, 1, 0.25, grid, selection),
                 std::invalid_argument);

    const AdversarialModel wrong_normalizer_count(
        hamilt::cider::SemilocalLevel::Gga,
        hamilt::cider::SemilocalMode::Np,
        false,
        2,
        0,
        1,
        false,
        hamilt::cider::SemilocalLevel::Gga,
        0,
        1,
        1);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(wrong_normalizer_count, 1, 0.25),
                 std::invalid_argument);

    const AdversarialModel wrong_map_count(
        hamilt::cider::SemilocalLevel::Gga,
        hamilt::cider::SemilocalMode::Np,
        false,
        2,
        0,
        1,
        false,
        hamilt::cider::SemilocalLevel::Gga,
        0,
        2,
        0);
    EXPECT_THROW(hamilt::cider::DualDensityEvaluator(wrong_map_count, 1, 0.25),
                 std::invalid_argument);
}

TEST(NativeCiderEvaluator, ModelDerivedMultiplicationOverflowPrecedesReadsAndWrites)
{
    const SyntheticModel model(hamilt::cider::SemilocalLevel::Gga, 1.0);
    const hamilt::cider::DualDensityEvaluator evaluator(model, 1, 0.25);
    const std::size_t ngrids = std::numeric_limits<std::size_t>::max() / 2 + 1;
    const double nonfinite_input = std::numeric_limits<double>::quiet_NaN();
    double output_storage[9] = {11.0, 12.0, 13.0, 14.0, 15.0,
                                16.0, 17.0, 18.0, 19.0};
    const std::vector<double> expected(output_storage, output_storage + 9);
    const hamilt::cider::DensityTupleInput density(
        &nonfinite_input, ngrids, &nonfinite_input, ngrids);
    const hamilt::cider::EvaluationInput input(1, ngrids, density, density);
    const auto component = [&](std::size_t base) {
        return hamilt::cider::ComponentOutput(output_storage + base,
                                              ngrids,
                                              output_storage + base + 1,
                                              ngrids,
                                              output_storage + base + 2,
                                              ngrids);
    };
    hamilt::cider::EvaluationOutput output(
        component(0), component(3), component(6));

    EXPECT_THROW(evaluator.evaluate(input, output), std::overflow_error);
    EXPECT_EQ(expected, std::vector<double>(output_storage, output_storage + 9));
    EXPECT_DOUBLE_EQ(0.0, output.diagnostics().cider_xmix());
}

TEST(NativeCiderEvaluator, LateNonfiniteFailurePreservesEveryOutput)
{
    const SyntheticModel model(hamilt::cider::SemilocalLevel::Gga,
                               1.0,
                               std::numeric_limits<double>::max());
    const hamilt::cider::DualDensityEvaluator evaluator(model, 1, 1.0);
    const std::vector<double> rho = {1.0};
    const std::vector<double> sigma = {0.2};
    const std::vector<double> no_tau;
    ComponentStorage baseline(1, 1, 1, false), feature(1, 1, 1, false), total(1, 1, 1, false);
    const ComponentStorage expected_baseline = baseline;
    const ComponentStorage expected_feature = feature;
    const ComponentStorage expected_total = total;
    const hamilt::cider::EvaluationInput input(
        1, 1, density_tuple(rho, sigma, no_tau), density_tuple(rho, sigma, no_tau));
    hamilt::cider::EvaluationOutput output(baseline.output(), feature.output(), total.output());
    EXPECT_THROW(evaluator.evaluate(input, output), std::overflow_error);
    EXPECT_EQ(expected_baseline.exc, baseline.exc);
    EXPECT_EQ(expected_baseline.vrho, baseline.vrho);
    EXPECT_EQ(expected_baseline.vsigma, baseline.vsigma);
    EXPECT_EQ(expected_feature.exc, feature.exc);
    EXPECT_EQ(expected_feature.vrho, feature.vrho);
    EXPECT_EQ(expected_feature.vsigma, feature.vsigma);
    EXPECT_EQ(expected_total.exc, total.exc);
    EXPECT_EQ(expected_total.vrho, total.vrho);
    EXPECT_EQ(expected_total.vsigma, total.vsigma);
    EXPECT_DOUBLE_EQ(0.0, output.diagnostics().cider_xmix());
}

} // namespace
