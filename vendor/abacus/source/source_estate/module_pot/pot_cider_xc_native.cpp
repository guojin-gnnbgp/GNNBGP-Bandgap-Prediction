#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "pot_cider_xc_native.h"

#include "source_hamilt/module_xc/cider/model_loader.h"
#include "source_hamilt/module_xc/cider/nldf_plan.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace elecstate
{

namespace
{

std::string feature_density_policy_name(
    const hamilt::cider::FeatureDensityPolicy policy)
{
    switch (policy)
    {
        case hamilt::cider::FeatureDensityPolicy::Valence:
            return "valence";
        case hamilt::cider::FeatureDensityPolicy::ValencePseudoCore:
            return "valence_pseudo_core";
    }
    throw std::runtime_error("native CIDER feature-density policy is unknown");
}

} // namespace

PotCiderNativeRuntime::PotCiderNativeRuntime(
    const std::string& model_path,
    const std::size_t grid_shape[3],
    const double cell[9],
    const std::size_t nspin,
    const double cider_xmix)
    : PotCiderNativeRuntime(
          model_path, grid_shape, cell, nspin, cider_xmix, 300.0, 1.8, 0)
{
}

PotCiderNativeRuntime::PotCiderNativeRuntime(
    const std::string& model_path,
    const std::size_t grid_shape[3],
    const double cell[9],
    const std::size_t nspin,
    const double cider_xmix,
    const double nldf_cutoff,
    const double nldf_lambda,
    const std::size_t nldf_alpha_count)
{
    try
    {
        const hamilt::cider::ModelLoader loader;
        std::unique_ptr<const hamilt::cider::Model> model = loader.load(model_path);
        std::unique_ptr<hamilt::cider::DualDensityEvaluator> evaluator;
        if (model->metadata().has_nldf())
        {
            const hamilt::cider::NldfGrid grid(grid_shape, cell);
            const hamilt::cider::NldfAlphaSelection alpha_selection(
                nldf_cutoff, nldf_lambda, nldf_alpha_count);
            evaluator.reset(new hamilt::cider::DualDensityEvaluator(
                *model, nspin, cider_xmix, grid, alpha_selection));
        }
        else
        {
            evaluator.reset(
                new hamilt::cider::DualDensityEvaluator(*model, nspin, cider_xmix));
        }
        model_ = std::move(model);
        evaluator_ = std::move(evaluator);
        metadata_.requires_tau = model_->metadata().requires_tau();
        metadata_.feature_density_policy = feature_density_policy_name(
            model_->metadata().feature_density_policy());
        metadata_.feature_density_policy_source
            = model_->metadata().feature_density_policy_source();
        metadata_.feature_exc_convention = "energy_density";
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error("native CIDER model '" + model_path
                                 + "': " + error.what());
    }
}

PotCiderNativeRuntime::~PotCiderNativeRuntime() = default;

const char* PotCiderNativeRuntime::backend_name() const
{
    return "native";
}

const PotCiderRuntimeMetadata& PotCiderNativeRuntime::metadata() const
{
    return metadata_;
}

void PotCiderNativeRuntime::evaluate(
    const hamilt::cider::EvaluationInput& input,
    const hamilt::cider::ComponentOutput& baseline,
    const hamilt::cider::ComponentOutput& feature) const
{
    const std::size_t nspin = input.nspin();
    const std::size_t ngrids = input.ngrids();
    if (nspin != 1U && nspin != 2U)
    {
        throw std::invalid_argument("native PotCider runtime requires one or two spins");
    }
    if (ngrids > std::numeric_limits<std::size_t>::max() / nspin)
    {
        throw std::overflow_error("native PotCider rho/tau size overflows size_t");
    }
    const std::size_t nsigma = 2 * nspin - 1;
    if (ngrids > std::numeric_limits<std::size_t>::max() / nsigma)
    {
        throw std::overflow_error("native PotCider sigma size overflows size_t");
    }
    const std::size_t nrho = nspin * ngrids;
    const std::size_t sigma_size = nsigma * ngrids;
    std::vector<double> exc_total(ngrids, 0.0);
    std::vector<double> vrho_total(nrho, 0.0);
    std::vector<double> vsigma_total(sigma_size, 0.0);
    std::vector<double> vtau_total(
        model_->metadata().requires_tau() ? nrho : 0, 0.0);
    const hamilt::cider::ComponentOutput total(
        exc_total.data(),
        exc_total.size(),
        vrho_total.data(),
        vrho_total.size(),
        vsigma_total.data(),
        vsigma_total.size(),
        vtau_total.empty() ? nullptr : vtau_total.data(),
        vtau_total.size());
    hamilt::cider::EvaluationOutput output(baseline, feature, total);
    evaluator_->evaluate(input, output);
}

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC
