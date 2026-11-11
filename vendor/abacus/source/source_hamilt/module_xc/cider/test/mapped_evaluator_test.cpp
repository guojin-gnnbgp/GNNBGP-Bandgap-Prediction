#include "source_hamilt/module_xc/cider/mapped_evaluator.h"
#include "source_hamilt/module_xc/cider/model_loader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <memory>
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

class TestModel final : public hamilt::cider::Model
{
  public:
    TestModel(const std::vector<hamilt::cider::Normalizer>& normalizers,
              const std::vector<hamilt::cider::FeatureMap>& maps,
              const std::vector<hamilt::cider::SplineTerm>& terms,
              double spline_constant,
              hamilt::cider::AdditiveBaseline additive,
              double output_scale,
              double exchange_scale = 1.0,
              const hamilt::cider::TrainingLabelMetadata& labels
                  = hamilt::cider::TrainingLabelMetadata())
        : metadata_(hamilt::cider::SemilocalLevel::Gga,
                    normalizers.size() > 2,
                    2,
                    normalizers.size() - 2,
                    maps.size(),
                    hamilt::cider::FeatureDensityPolicy::Valence,
                    "test",
                    output_scale,
                    exchange_scale,
                    labels),
          normalizers_(normalizers),
          maps_(maps),
          terms_(terms),
          spline_constant_(spline_constant),
          additive_(additive)
    {
    }

    const hamilt::cider::ModelMetadata& metadata() const override { return metadata_; }
    hamilt::cider::SemilocalMode semilocal_mode() const override
    {
        return hamilt::cider::SemilocalMode::Np;
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
    hamilt::cider::AdditiveBaseline additive_baseline() const override { return additive_; }
    double spline_constant() const override { return spline_constant_; }
    const std::vector<hamilt::cider::SplineTerm>& spline_terms() const override
    {
        return terms_;
    }

  private:
    const hamilt::cider::ModelMetadata metadata_;
    const std::vector<hamilt::cider::Normalizer> normalizers_;
    const std::vector<hamilt::cider::FeatureMap> maps_;
    const std::vector<hamilt::cider::SplineTerm> terms_;
    const double spline_constant_;
    const hamilt::cider::AdditiveBaseline additive_;
};

std::vector<hamilt::cider::Normalizer> identity_normalizers(std::size_t count)
{
    return std::vector<hamilt::cider::Normalizer>(
        count, {hamilt::cider::NormalizerKind::Identity, 1.0});
}

hamilt::cider::FeatureMap umap(std::size_t index = 1)
{
    return {hamilt::cider::FeatureMapKind::U, index, 0.5, 1.0, 0.0, 0, 0.0, 1.0};
}

hamilt::cider::SplineTerm zero_term()
{
    return {1.0,
            {0},
            {{0.0, 1.0, 3}},
            {5},
            std::vector<double>(5, 0.0)};
}

const double test_pi = 3.141592653589793238462643383279502884;
const double test_lda_factor = -0.75 * std::pow(3.0 / test_pi, 1.0 / 3.0);

struct AnalyticBaseline
{
    double energy;
    double density_derivative;
    double s2_derivative;
};

AnalyticBaseline analytic_lda(double rho)
{
    const double rho13 = std::pow(rho, 1.0 / 3.0);
    return {test_lda_factor * rho * rho13,
            (4.0 / 3.0) * test_lda_factor * rho13,
            0.0};
}

AnalyticBaseline analytic_pbe(double rho, double s2)
{
    const double kappa = 0.804;
    const double mu = 0.2195149727645171;
    const double fac = 1.0 / (1.0 + (mu / kappa) * s2);
    const double fx = 1.0 + kappa - kappa * fac;
    const double dfx = mu * fac * fac;
    const double rho13 = std::pow(rho, 1.0 / 3.0);
    return {test_lda_factor * rho * rho13 * fx,
            (4.0 / 3.0) * test_lda_factor * rho13 * fx,
            test_lda_factor * rho * rho13 * dfx};
}

AnalyticBaseline analytic_chachiyo(double rho, double s2)
{
    double fx = 0.0;
    double dfx = 0.0;
    if (s2 < 1.0e-8)
    {
        fx = 1.0 + 8.0 * s2 / 27.0;
        dfx = 8.0 / 27.0;
    }
    else
    {
        const double c = 4.0 * test_pi / 9.0;
        const double root = std::sqrt(s2);
        const double x = c * root;
        const double logarithm = std::log1p(x);
        const double pi2 = test_pi * test_pi;
        fx = (3.0 * x * x + pi2 * logarithm) / ((pi2 + 3.0 * x) * logarithm);
        dfx = (-3.0 * x * x * (pi2 + 3.0 * x)
               + 3.0 * x * (1.0 + x) * (2.0 * pi2 + 3.0 * x) * logarithm
               - 3.0 * pi2 * (1.0 + x) * logarithm * logarithm)
              / ((1.0 + x) * (pi2 + 3.0 * x) * (pi2 + 3.0 * x)
                 * logarithm * logarithm)
              * c / (2.0 * root);
    }
    const double rho13 = std::pow(rho, 1.0 / 3.0);
    return {test_lda_factor * rho * rho13 * fx,
            (4.0 / 3.0) * test_lda_factor * rho13 * fx,
            test_lda_factor * rho * rho13 * dfx};
}

void evaluate(const hamilt::cider::Model& model,
              std::size_t nspin,
              std::size_t ngrids,
              const std::vector<double>& raw,
              const std::vector<double>& rho,
              double cutoff,
              std::vector<double>& energy,
              std::vector<double>& derivative)
{
    const hamilt::cider::MappedEvaluationInput input(
        nspin, ngrids, raw.data(), raw.size(), rho.data(), rho.size(), cutoff);
    hamilt::cider::MappedEvaluationOutput output(
        energy.data(), energy.size(), derivative.data(), derivative.size());
    const hamilt::cider::MappedXC evaluator;
    evaluator.evaluate(model, input, output);
}

TEST(NativeCiderMappedEvaluator, MatchesV1SpinOneGgaRawFeatureOracle)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    const hamilt::cider::ModelLoader loader;
    const std::unique_ptr<const hamilt::cider::Model> model
        = loader.load(std::string(parent) + "/functionals/CIDER23X_SL_GGA.yaml");
    const std::vector<double> raw = read_fixture_array(496, 16);
    const std::vector<double> rho = read_fixture_array(240, 8);
    const std::vector<double> expected_energy = read_fixture_array(1072, 8);
    const std::vector<double> expected_derivative = read_fixture_array(1264, 16);
    std::vector<double> energy(8, -91.0);
    std::vector<double> derivative(16, -93.0);
    const hamilt::cider::MappedEvaluationInput input(
        1, 8, raw.data(), raw.size(), rho.data(), rho.size(), 0.0);
    hamilt::cider::MappedEvaluationOutput output(
        energy.data(), energy.size(), derivative.data(), derivative.size());

    const hamilt::cider::MappedXC evaluator;
    evaluator.evaluate(*model, input, output);

    for (std::size_t i = 0; i < energy.size(); ++i)
    {
        EXPECT_LE(std::abs(energy[i] - expected_energy[i]), 1.0e-12) << i;
    }
    for (std::size_t i = 0; i < derivative.size(); ++i)
    {
        EXPECT_LE(std::abs(derivative[i] - expected_derivative[i]), 1.0e-12) << i;
    }
}

TEST(NativeCiderMappedEvaluator, ExposesConcreteKernelAndMappedXcEvaluators)
{
    EXPECT_FALSE(std::is_abstract<hamilt::cider::MappedDFTKernel>::value);
    EXPECT_FALSE(std::is_abstract<hamilt::cider::MappedXC>::value);
    EXPECT_TRUE((std::is_base_of<hamilt::cider::MappedEvaluator,
                                 hamilt::cider::MappedXC>::value));
}

TEST(NativeCiderMappedEvaluator, MatchesAnalyticLdaAndChachiyoSpinOneBaselines)
{
    const TestModel model(identity_normalizers(2),
                          {umap()},
                          {zero_term()},
                          2.0,
                          hamilt::cider::AdditiveBaseline::GgaXChachiyo,
                          1.0);
    const std::vector<double> rho = {0.37, 0.81};
    const std::vector<double> raw = {0.37, 0.81, 0.0, 0.42};
    std::vector<double> energy(2);
    std::vector<double> derivative(4);
    evaluate(model, 1, 2, raw, rho, 0.0, energy, derivative);
    for (std::size_t grid = 0; grid < 2; ++grid)
    {
        const AnalyticBaseline lda = analytic_lda(raw[grid]);
        const AnalyticBaseline add = analytic_chachiyo(raw[grid], raw[2 + grid]);
        EXPECT_NEAR(2.0 * lda.energy + add.energy, energy[grid], 1.0e-14);
        EXPECT_NEAR(2.0 * lda.density_derivative + add.density_derivative,
                    derivative[grid],
                    1.0e-14);
        EXPECT_NEAR(add.s2_derivative, derivative[2 + grid], 1.0e-14);
    }
}

TEST(NativeCiderMappedEvaluator, MatchesAnalyticLdaAndPbeSeparatedSpinAssembly)
{
    const TestModel model(identity_normalizers(2),
                          {umap()},
                          {zero_term()},
                          -0.4,
                          hamilt::cider::AdditiveBaseline::GgaXPbe,
                          1.0);
    // (nspin=2, nfeature=2, ngrids=2); descriptor rho is 2 * physical rho.
    const std::vector<double> rho = {0.11, 0.17, 0.23, 0.29};
    const std::vector<double> raw = {0.22, 0.34, 0.15, 0.31,
                                     0.46, 0.58, 0.47, 0.63};
    std::vector<double> energy(2);
    std::vector<double> derivative(8);
    evaluate(model, 2, 2, raw, rho, 0.0, energy, derivative);
    for (std::size_t grid = 0; grid < 2; ++grid)
    {
        double expected_energy = 0.0;
        for (std::size_t spin = 0; spin < 2; ++spin)
        {
            const std::size_t density_index = spin * 4 + grid;
            const std::size_t gradient_index = density_index + 2;
            const AnalyticBaseline lda = analytic_lda(raw[density_index]);
            const AnalyticBaseline add
                = analytic_pbe(raw[density_index], raw[gradient_index]);
            expected_energy += (-0.4 * lda.energy + add.energy) / 2.0;
            EXPECT_NEAR((-0.4 * lda.density_derivative + add.density_derivative) / 2.0,
                        derivative[density_index],
                        1.0e-14);
            EXPECT_NEAR(add.s2_derivative / 2.0,
                        derivative[gradient_index],
                        1.0e-14);
        }
        EXPECT_NEAR(expected_energy, energy[grid], 1.0e-14);
    }
}

TEST(NativeCiderMappedEvaluator, AppliesOnlyKernelOutputScaleAndSplineConstant)
{
    const hamilt::cider::TrainingLabelMetadata labels(
        "full_exact_exchange", false, 1.0, {0.25, 0.5});
    const TestModel unscaled(identity_normalizers(2),
                             {umap()},
                             {zero_term()},
                             1.7,
                             hamilt::cider::AdditiveBaseline::GgaXPbe,
                             1.0,
                             0.25,
                             labels);
    const TestModel scaled(identity_normalizers(2),
                           {umap()},
                           {zero_term()},
                           1.7,
                           hamilt::cider::AdditiveBaseline::GgaXPbe,
                           0.4,
                           0.25,
                           labels);
    const std::vector<double> rho = {0.31};
    const std::vector<double> raw = {0.31, 0.27};
    std::vector<double> energy_unscaled(1), derivative_unscaled(2);
    std::vector<double> energy_scaled(1), derivative_scaled(2);
    evaluate(unscaled, 1, 1, raw, rho, 0.0, energy_unscaled, derivative_unscaled);
    evaluate(scaled, 1, 1, raw, rho, 0.0, energy_scaled, derivative_scaled);
    EXPECT_NEAR(0.4 * energy_unscaled[0], energy_scaled[0], 1.0e-15);
    EXPECT_NEAR(0.4 * derivative_unscaled[0], derivative_scaled[0], 1.0e-15);
    EXPECT_NEAR(0.4 * derivative_unscaled[1], derivative_scaled[1], 1.0e-15);
}

TEST(NativeCiderMappedEvaluator, NormalizesDensityBeforeBaselinesAndCutoff)
{
    std::vector<hamilt::cider::Normalizer> normalizers = identity_normalizers(2);
    normalizers[0] = {hamilt::cider::NormalizerKind::Constant, 2.0};
    const double spline_constant = 1.3;
    const TestModel model(normalizers,
                          {umap()},
                          {zero_term()},
                          spline_constant,
                          hamilt::cider::AdditiveBaseline::GgaXPbe,
                          1.0);
    // Raw SEP density is below cutoff, but normalized density is above it.
    const std::vector<double> rho = {0.11};
    const std::vector<double> raw = {0.11, 0.27};
    std::vector<double> energy(1);
    std::vector<double> derivative(2);
    evaluate(model, 1, 1, raw, rho, 0.2, energy, derivative);

    const double normalized_density = 2.0 * raw[0];
    const AnalyticBaseline lda = analytic_lda(normalized_density);
    const AnalyticBaseline add = analytic_pbe(normalized_density, raw[1]);
    EXPECT_NEAR(spline_constant * lda.energy + add.energy, energy[0], 1.0e-14);
    EXPECT_NEAR(2.0 * (spline_constant * lda.density_derivative
                       + add.density_derivative),
                derivative[0],
                1.0e-14);
    EXPECT_NEAR(add.s2_derivative, derivative[1], 1.0e-14);
}

TEST(NativeCiderMappedEvaluator, MasksWhenOnlyNormalizedDensityIsBelowCutoff)
{
    std::vector<hamilt::cider::Normalizer> normalizers = identity_normalizers(2);
    normalizers[0] = {hamilt::cider::NormalizerKind::Constant, 0.5};
    const TestModel model(normalizers,
                          {umap()},
                          {zero_term()},
                          1.3,
                          hamilt::cider::AdditiveBaseline::GgaXPbe,
                          1.0);
    // Raw SEP density is above cutoff, but normalized density is below it.
    const std::vector<double> rho = {0.3};
    const std::vector<double> raw = {0.3, 0.27};
    std::vector<double> energy(1, -1.0);
    std::vector<double> derivative(2, -1.0);
    evaluate(model, 1, 1, raw, rho, 0.2, energy, derivative);
    EXPECT_DOUBLE_EQ(0.0, energy[0]);
    EXPECT_DOUBLE_EQ(0.0, derivative[0]);
    EXPECT_DOUBLE_EQ(0.0, derivative[1]);
}

TEST(NativeCiderMappedEvaluator, UsesStrictPerSpinLowDensityCutoff)
{
    const TestModel model(identity_normalizers(2),
                          {umap()},
                          {zero_term()},
                          1.0,
                          hamilt::cider::AdditiveBaseline::GgaXPbe,
                          1.0);
    const double cutoff = 0.2;
    const std::vector<double> rho = {0.099999999999, 0.1, 0.100000000001,
                                     0.2, 0.2, 0.2};
    const std::vector<double> raw = {0.199999999998, 0.2, 0.200000000002,
                                     0.4, 0.5, 0.6,
                                     0.4, 0.4, 0.4,
                                     0.2, 0.3, 0.4};
    std::vector<double> energy(3);
    std::vector<double> derivative(12);
    evaluate(model, 2, 3, raw, rho, cutoff, energy, derivative);
    EXPECT_DOUBLE_EQ(0.0, derivative[0]);
    EXPECT_DOUBLE_EQ(0.0, derivative[3]);
    EXPECT_NE(0.0, derivative[1]);
    EXPECT_NE(0.0, derivative[2]);
    for (std::size_t grid = 0; grid < 3; ++grid)
    {
        EXPECT_NE(0.0, energy[grid]);
        EXPECT_NE(0.0, derivative[6 + grid]);
        EXPECT_NE(0.0, derivative[9 + grid]);
    }
}

TEST(NativeCiderMappedEvaluator, MatchesAllFiveV1MappedValuesAndRawDerivatives)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    struct Case
    {
        const char* model;
        std::size_t nspin;
        std::size_t nraw;
        std::size_t raw_offset;
        std::size_t rho_offset;
        std::size_t energy_offset;
        std::size_t derivative_offset;
    };
    const Case cases[] = {
        {"functionals/CIDER23X_SL_GGA.yaml", 1, 2, 496, 240, 1072, 1264},
        {"functionals/CIDER23X_SL_MGGA.yaml", 2, 3, 3436, 2604, 5228, 5676},
        {"functionals/CIDER23X_NL_GGA.yaml", 2, 5, 9716, 9140, 12788, 13492},
        {"functionals/CIDER23X_NL_MGGA_DTR.yaml", 1, 6, 18908, 18524, 20764, 21212},
        {"abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/"
         "full/nldf_mgga_cand00_n96_noise0p01.yaml",
         2,
         6,
         26244,
         25412,
         29956,
         30788}};
    const hamilt::cider::ModelLoader loader;
    double max_energy_error = 0.0;
    double max_derivative_error = 0.0;
    for (const Case& item : cases)
    {
        const std::unique_ptr<const hamilt::cider::Model> model
            = loader.load(std::string(parent) + "/" + item.model);
        const std::size_t raw_size = item.nspin * item.nraw * 8;
        const std::vector<double> raw = read_fixture_array(item.raw_offset, raw_size);
        const std::vector<double> rho = read_fixture_array(item.rho_offset, item.nspin * 8);
        const std::vector<double> expected_energy = read_fixture_array(item.energy_offset, 8);
        const std::vector<double> expected_derivative
            = read_fixture_array(item.derivative_offset, raw_size);
        std::vector<double> energy(8);
        std::vector<double> derivative(raw_size);
        evaluate(*model, item.nspin, 8, raw, rho, 0.0, energy, derivative);
        for (std::size_t i = 0; i < energy.size(); ++i)
        {
            const double error = std::abs(energy[i] - expected_energy[i]);
            max_energy_error = std::max(max_energy_error, error);
            EXPECT_LE(error, 1.0e-12) << item.model << " energy " << i;
        }
        for (std::size_t i = 0; i < derivative.size(); ++i)
        {
            const double error = std::abs(derivative[i] - expected_derivative[i]);
            max_derivative_error = std::max(max_derivative_error, error);
            EXPECT_LE(error, 1.0e-12) << item.model << " derivative " << i;
        }
    }
    RecordProperty("v1_mapped_energy_max_abs_error", max_energy_error);
    RecordProperty("v1_mapped_raw_derivative_max_abs_error", max_derivative_error);
}

TEST(NativeCiderMappedEvaluator, RawFeatureDerivativesMatchFiniteDifferences)
{
    std::vector<double> coefficients(25);
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
        coefficients[i] = std::sin(0.31 * static_cast<double>(i));
    }
    std::vector<hamilt::cider::Normalizer> normalizers = identity_normalizers(3);
    normalizers[2] = {hamilt::cider::NormalizerKind::Constant, 1.7};
    const std::vector<hamilt::cider::FeatureMap> maps = {
        umap(),
        {hamilt::cider::FeatureMapKind::V, 2, 0.4, 1.0, 0.5, 0, -0.5, 0.5}};
    const hamilt::cider::SplineTerm term(
        1.3, {0, 1}, {{0.0, 1.0, 3}, {-0.5, 0.5, 3}}, {5, 5}, coefficients);
    const TestModel model(normalizers,
                          maps,
                          {term},
                          -0.2,
                          hamilt::cider::AdditiveBaseline::GgaXPbe,
                          0.7);
    const std::size_t nspin = 2;
    const std::size_t ngrids = 2;
    std::vector<double> rho = {0.13, 0.21, 0.17, 0.28};
    std::vector<double> raw = {0.26, 0.42, 0.19, 0.33, 0.12, 0.24,
                               0.34, 0.56, 0.27, 0.41, 0.18, 0.31};
    std::vector<double> energy(ngrids);
    std::vector<double> derivative(raw.size());
    evaluate(model, nspin, ngrids, raw, rho, 0.0, energy, derivative);
    const double step = 1.0e-6;
    double max_error = 0.0;
    for (std::size_t index = 0; index < raw.size(); ++index)
    {
        double objective[2] = {0.0, 0.0};
        for (int sign = 0; sign < 2; ++sign)
        {
            std::vector<double> displaced_raw = raw;
            std::vector<double> displaced_rho = rho;
            displaced_raw[index] += sign == 0 ? step : -step;
            const std::size_t feature = (index / ngrids) % 3;
            if (feature == 0)
            {
                const std::size_t spin = index / (3 * ngrids);
                const std::size_t grid = index % ngrids;
                displaced_rho[spin * ngrids + grid]
                    += (sign == 0 ? step : -step) / static_cast<double>(nspin);
            }
            std::vector<double> displaced_energy(ngrids);
            std::vector<double> displaced_derivative(raw.size());
            evaluate(model,
                     nspin,
                     ngrids,
                     displaced_raw,
                     displaced_rho,
                     0.0,
                     displaced_energy,
                     displaced_derivative);
            objective[sign] = displaced_energy[0] + displaced_energy[1];
        }
        const double finite_difference = (objective[0] - objective[1]) / (2.0 * step);
        max_error = std::max(max_error, std::abs(derivative[index] - finite_difference));
        EXPECT_NEAR(derivative[index], finite_difference, 3.0e-9)
            << index;
    }
    RecordProperty("raw_feature_finite_difference_max_abs_error", max_error);
}

TEST(NativeCiderMappedEvaluator, RejectsInvalidContractsAndOverlapsWithoutWriting)
{
    const TestModel model(identity_normalizers(2),
                          {umap()},
                          {zero_term()},
                          1.0,
                          hamilt::cider::AdditiveBaseline::GgaXPbe,
                          1.0);
    std::vector<double> rho = {0.3};
    std::vector<double> raw = {0.3, 0.2};
    std::vector<double> energy(1, 17.0);
    std::vector<double> derivative(2, 19.0);
    const std::vector<double> expected_energy = energy;
    const std::vector<double> expected_derivative = derivative;
    const hamilt::cider::MappedXC evaluator;
    const auto expect_invalid = [&](const hamilt::cider::MappedEvaluationInput& input,
                                    hamilt::cider::MappedEvaluationOutput& output) {
        EXPECT_THROW(evaluator.evaluate(model, input, output), std::exception);
        EXPECT_EQ(expected_energy, energy);
        EXPECT_EQ(expected_derivative, derivative);
    };
    hamilt::cider::MappedEvaluationOutput output(
        energy.data(), energy.size(), derivative.data(), derivative.size());
    expect_invalid({3, 1, raw.data(), raw.size(), rho.data(), rho.size(), 0.0}, output);
    expect_invalid({1, 1, raw.data(), 1, rho.data(), rho.size(), 0.0}, output);
    expect_invalid({1, 1, nullptr, raw.size(), rho.data(), rho.size(), 0.0}, output);
    expect_invalid({1, 1, raw.data(), raw.size(), rho.data(), rho.size(), -1.0}, output);
    expect_invalid({1,
                    1,
                    raw.data(),
                    raw.size(),
                    rho.data(),
                    rho.size(),
                    std::numeric_limits<double>::quiet_NaN()},
                   output);
    rho[0] = -0.3;
    expect_invalid({1, 1, raw.data(), raw.size(), rho.data(), rho.size(), 0.0}, output);
    rho[0] = 0.31;
    expect_invalid({1, 1, raw.data(), raw.size(), rho.data(), rho.size(), 0.0}, output);
    rho[0] = 0.3;
    raw[1] = std::numeric_limits<double>::infinity();
    expect_invalid({1, 1, raw.data(), raw.size(), rho.data(), rho.size(), 0.0}, output);
    raw[1] = 0.2;
    const std::vector<double> expected_raw = raw;
    hamilt::cider::MappedEvaluationOutput overlap_output(
        raw.data(), 1, derivative.data(), derivative.size());
    expect_invalid({1, 1, raw.data(), raw.size(), rho.data(), rho.size(), 0.0},
                   overlap_output);
    EXPECT_EQ(expected_raw, raw);
    hamilt::cider::MappedEvaluationOutput output_overlap(
        derivative.data(), 1, derivative.data(), derivative.size());
    expect_invalid({1, 1, raw.data(), raw.size(), rho.data(), rho.size(), 0.0},
                   output_overlap);
}

TEST(NativeCiderMappedEvaluator, LateNumericalFailurePreservesOutputsByteForByte)
{
    const TestModel model(identity_normalizers(2),
                          {umap()},
                          {zero_term()},
                          1.0,
                          hamilt::cider::AdditiveBaseline::GgaXPbe,
                          std::numeric_limits<double>::max());
    const std::vector<double> rho = {1.0};
    const std::vector<double> raw = {1.0, 0.2};
    std::vector<double> energy(1, 17.0);
    std::vector<double> derivative(2, 19.0);
    const std::vector<double> expected_energy = energy;
    const std::vector<double> expected_derivative = derivative;
    const hamilt::cider::MappedEvaluationInput input(
        1, 1, raw.data(), raw.size(), rho.data(), rho.size(), 0.0);
    hamilt::cider::MappedEvaluationOutput output(
        energy.data(), energy.size(), derivative.data(), derivative.size());
    const hamilt::cider::MappedDFTKernel evaluator;
    EXPECT_THROW(evaluator.evaluate(model, input, output), std::overflow_error);
    EXPECT_EQ(expected_energy, energy);
    EXPECT_EQ(expected_derivative, derivative);
}

} // namespace
