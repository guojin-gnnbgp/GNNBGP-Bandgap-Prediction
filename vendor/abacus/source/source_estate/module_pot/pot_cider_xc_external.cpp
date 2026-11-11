#if defined(ENABLE_CIDER) && defined(ENABLE_CIDER_EXTERNAL) && defined(USE_LIBXC)

#include "pot_cider_xc_external.h"

#include "cider_bridge.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace elecstate
{

namespace
{

int checked_int(const std::size_t value, const char* label)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(std::string("external CIDER ") + label
                                  + " exceeds int range");
    }
    return static_cast<int>(value);
}

void validate_policy(const std::string& policy)
{
    if (policy != "valence" && policy != "valence_pseudo_core")
    {
        throw std::runtime_error("external CIDER bridge returned unknown feature-density policy");
    }
}

void validate_convention(const std::string& convention)
{
    if (convention != "energy_density" && convention != "per_particle")
    {
        throw std::runtime_error("external CIDER bridge returned unknown energy convention");
    }
}

} // namespace

PotCiderExternalRuntime::PotCiderExternalRuntime(
    const std::string& model_path,
    const std::size_t grid_shape[3],
    const double cell[9],
    const std::size_t nspin,
    const double cider_xmix)
{
    const int grid[3] = {checked_int(grid_shape[0], "grid x"),
                         checked_int(grid_shape[1], "grid y"),
                         checked_int(grid_shape[2], "grid z")};
    context_ = cider_bridge_create(model_path.c_str(),
                                   grid,
                                   cell,
                                   checked_int(nspin, "spin count"),
                                   cider_xmix);
    if (context_ == nullptr)
    {
        throw std::runtime_error("external CIDER model '" + model_path
                                 + "': bridge context creation failed");
    }
    try
    {
        metadata_.requires_tau = cider_bridge_is_mgga(context_) != 0;
        metadata_.feature_density_policy
            = cider_bridge_feature_density_policy_expected(context_);
        metadata_.feature_density_policy_source
            = cider_bridge_feature_density_policy_source(context_);
        metadata_.feature_exc_convention = cider_bridge_exc_convention(context_);
        validate_policy(metadata_.feature_density_policy);
        validate_convention(metadata_.feature_exc_convention);
    }
    catch (...)
    {
        cider_bridge_destroy(&context_);
        throw;
    }
}

PotCiderExternalRuntime::~PotCiderExternalRuntime()
{
    if (context_ != nullptr)
    {
        cider_bridge_destroy(&context_);
    }
}

const char* PotCiderExternalRuntime::backend_name() const
{
    return "external";
}

const PotCiderRuntimeMetadata& PotCiderExternalRuntime::metadata() const
{
    return metadata_;
}

void PotCiderExternalRuntime::evaluate(
    const hamilt::cider::EvaluationInput& input,
    const hamilt::cider::ComponentOutput& baseline,
    const hamilt::cider::ComponentOutput& feature) const
{
    const std::size_t nspin = input.nspin();
    const std::size_t ngrids = input.ngrids();
    if (nspin != 1U && nspin != 2U)
    {
        throw std::invalid_argument("external CIDER runtime requires one or two spins");
    }
    if (ngrids > std::numeric_limits<std::size_t>::max() / nspin)
    {
        throw std::overflow_error("external CIDER rho/tau size overflows size_t");
    }
    const std::size_t nrho = nspin * ngrids;
    const std::size_t nsigma = 2U * nspin - 1U;
    if (ngrids > std::numeric_limits<std::size_t>::max() / nsigma)
    {
        throw std::overflow_error("external CIDER sigma size overflows size_t");
    }
    const std::size_t sigma_size = nsigma * ngrids;
    const bool has_tau = metadata_.requires_tau;

    if (input.baseline().rho_size() != nrho
        || input.feature().rho_size() != nrho
        || input.baseline().sigma_size() != sigma_size
        || input.feature().sigma_size() != sigma_size
        || baseline.exc_size() != ngrids
        || feature.exc_size() != ngrids
        || baseline.vrho_size() != nrho
        || feature.vrho_size() != nrho
        || baseline.vsigma_size() != sigma_size
        || feature.vsigma_size() != sigma_size
        || input.baseline().tau_size() != (has_tau ? nrho : 0U)
        || input.feature().tau_size() != (has_tau ? nrho : 0U)
        || baseline.vtau_size() != (has_tau ? nrho : 0U)
        || feature.vtau_size() != (has_tau ? nrho : 0U))
    {
        throw std::invalid_argument("external CIDER input/output dimensions are inconsistent");
    }

    std::vector<double> baseline_exc(ngrids);
    std::vector<double> baseline_vrho(nrho);
    std::vector<double> baseline_vsigma(sigma_size);
    std::vector<double> baseline_vtau(has_tau ? nrho : 0U);
    std::vector<double> feature_exc(ngrids);
    std::vector<double> feature_vrho(nrho);
    std::vector<double> feature_vsigma(sigma_size);
    std::vector<double> feature_vtau(has_tau ? nrho : 0U);

    const int status = cider_bridge_evaluate_dual(
        context_,
        checked_int(nspin, "spin count"),
        checked_int(ngrids, "grid size"),
        input.baseline().rho(),
        input.baseline().sigma(),
        input.baseline().tau(),
        input.feature().rho(),
        input.feature().sigma(),
        input.feature().tau(),
        baseline_exc.data(),
        baseline_vrho.data(),
        baseline_vsigma.data(),
        has_tau ? baseline_vtau.data() : nullptr,
        feature_exc.data(),
        feature_vrho.data(),
        feature_vsigma.data(),
        has_tau ? feature_vtau.data() : nullptr);
    if (status != 0)
    {
        throw std::runtime_error("external CIDER bridge evaluate_dual failed");
    }

    std::copy(baseline_exc.begin(), baseline_exc.end(), baseline.exc());
    std::copy(baseline_vrho.begin(), baseline_vrho.end(), baseline.vrho());
    std::copy(baseline_vsigma.begin(), baseline_vsigma.end(), baseline.vsigma());
    std::copy(feature_exc.begin(), feature_exc.end(), feature.exc());
    std::copy(feature_vrho.begin(), feature_vrho.end(), feature.vrho());
    std::copy(feature_vsigma.begin(), feature_vsigma.end(), feature.vsigma());
    if (has_tau)
    {
        std::copy(baseline_vtau.begin(), baseline_vtau.end(), baseline.vtau());
        std::copy(feature_vtau.begin(), feature_vtau.end(), feature.vtau());
    }
}

} // namespace elecstate

#endif // ENABLE_CIDER && ENABLE_CIDER_EXTERNAL && USE_LIBXC
