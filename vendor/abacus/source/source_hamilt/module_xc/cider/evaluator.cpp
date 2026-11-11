#include "evaluator.h"

#include "mapped_evaluator.h"
#include "nldf_plan.h"
#include "semilocal_plan.h"

#include <xc.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace hamilt
{
namespace cider
{

DensityTupleInput::DensityTupleInput(const double* rho,
                                     std::size_t rho_size,
                                     const double* sigma,
                                     std::size_t sigma_size,
                                     const double* tau,
                                     std::size_t tau_size)
    : rho_(rho),
      rho_size_(rho_size),
      sigma_(sigma),
      sigma_size_(sigma_size),
      tau_(tau),
      tau_size_(tau_size)
{
}
const double* DensityTupleInput::rho() const { return rho_; }
std::size_t DensityTupleInput::rho_size() const { return rho_size_; }
const double* DensityTupleInput::sigma() const { return sigma_; }
std::size_t DensityTupleInput::sigma_size() const { return sigma_size_; }
const double* DensityTupleInput::tau() const { return tau_; }
std::size_t DensityTupleInput::tau_size() const { return tau_size_; }

EvaluationInput::EvaluationInput(std::size_t nspin,
                                 std::size_t ngrids,
                                 const DensityTupleInput& baseline,
                                 const DensityTupleInput& feature)
    : nspin_(nspin), ngrids_(ngrids), baseline_(baseline), feature_(feature)
{
}
std::size_t EvaluationInput::nspin() const { return nspin_; }
std::size_t EvaluationInput::ngrids() const { return ngrids_; }
const DensityTupleInput& EvaluationInput::baseline() const { return baseline_; }
const DensityTupleInput& EvaluationInput::feature() const { return feature_; }

ComponentOutput::ComponentOutput(double* exc,
                                 std::size_t exc_size,
                                 double* vrho,
                                 std::size_t vrho_size,
                                 double* vsigma,
                                 std::size_t vsigma_size,
                                 double* vtau,
                                 std::size_t vtau_size)
    : exc_(exc),
      exc_size_(exc_size),
      vrho_(vrho),
      vrho_size_(vrho_size),
      vsigma_(vsigma),
      vsigma_size_(vsigma_size),
      vtau_(vtau),
      vtau_size_(vtau_size)
{
}
double* ComponentOutput::exc() const { return exc_; }
std::size_t ComponentOutput::exc_size() const { return exc_size_; }
double* ComponentOutput::vrho() const { return vrho_; }
std::size_t ComponentOutput::vrho_size() const { return vrho_size_; }
double* ComponentOutput::vsigma() const { return vsigma_; }
std::size_t ComponentOutput::vsigma_size() const { return vsigma_size_; }
double* ComponentOutput::vtau() const { return vtau_; }
std::size_t ComponentOutput::vtau_size() const { return vtau_size_; }

EvaluationDiagnostics::EvaluationDiagnostics()
    : feature_density_policy_(FeatureDensityPolicy::Valence),
      feature_density_policy_source_(new std::string()),
      feature_energy_convention_(EnergyConvention::EnergyDensity),
      baseline_exc_convention_(BaselineExcConvention::LibxcExcPerParticle),
      cider_xmix_(0.0),
      trained_exchange_scale_(0.0),
      runtime_scale_(0.0),
      regularized_rho_count_(0),
      regularized_same_spin_sigma_count_(0),
      regularized_tau_count_(0),
      baseline_exc_sum_(0.0),
      feature_exc_sum_(0.0),
      total_exc_sum_(0.0)
{
}
FeatureDensityPolicy EvaluationDiagnostics::feature_density_policy() const
{
    return feature_density_policy_;
}
const std::string& EvaluationDiagnostics::feature_density_policy_source() const
{
    return *feature_density_policy_source_;
}
EnergyConvention EvaluationDiagnostics::feature_energy_convention() const
{
    return feature_energy_convention_;
}
BaselineExcConvention EvaluationDiagnostics::baseline_exc_convention() const
{
    return baseline_exc_convention_;
}
double EvaluationDiagnostics::cider_xmix() const { return cider_xmix_; }
double EvaluationDiagnostics::trained_exchange_scale() const
{
    return trained_exchange_scale_;
}
double EvaluationDiagnostics::runtime_scale() const { return runtime_scale_; }
std::size_t EvaluationDiagnostics::regularized_rho_count() const
{
    return regularized_rho_count_;
}
std::size_t EvaluationDiagnostics::regularized_same_spin_sigma_count() const
{
    return regularized_same_spin_sigma_count_;
}
std::size_t EvaluationDiagnostics::regularized_tau_count() const
{
    return regularized_tau_count_;
}
double EvaluationDiagnostics::baseline_exc_sum() const { return baseline_exc_sum_; }
double EvaluationDiagnostics::feature_exc_sum() const { return feature_exc_sum_; }
double EvaluationDiagnostics::total_exc_sum() const { return total_exc_sum_; }

EvaluationOutput::EvaluationOutput(const ComponentOutput& baseline,
                                   const ComponentOutput& feature,
                                   const ComponentOutput& total)
    : baseline_(baseline), feature_(feature), total_(total)
{
}
const ComponentOutput& EvaluationOutput::baseline() const { return baseline_; }
const ComponentOutput& EvaluationOutput::feature() const { return feature_; }
const ComponentOutput& EvaluationOutput::total() const { return total_; }
const EvaluationDiagnostics& EvaluationOutput::diagnostics() const
{
    return diagnostics_;
}

Evaluator::~Evaluator() = default;

namespace
{

const double feature_rho_floor = 1.0e-12;
const double mapped_density_cutoff = 1.0e-8;
const double libxc_density_threshold = 1.0e-6;

std::size_t checked_product(std::size_t first,
                            std::size_t second,
                            const char* label)
{
    if (first == 0 || second == 0)
    {
        throw std::invalid_argument(std::string("native CIDER evaluator ") + label
                                    + " dimensions must be nonzero");
    }
    if (first > std::numeric_limits<std::size_t>::max() / second)
    {
        throw std::overflow_error(std::string("native CIDER evaluator ") + label
                                  + " dimensions overflow");
    }
    return first * second;
}

std::size_t checked_multiply(std::size_t first,
                             std::size_t second,
                             const char* label)
{
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
    {
        throw std::overflow_error(std::string("native CIDER evaluator ") + label
                                  + " dimensions overflow");
    }
    return first * second;
}

std::size_t checked_add(std::size_t first,
                        std::size_t second,
                        const char* label)
{
    if (first > std::numeric_limits<std::size_t>::max() - second)
    {
        throw std::overflow_error(std::string("native CIDER evaluator ") + label
                                  + " dimensions overflow");
    }
    return first + second;
}

struct BufferRange
{
    std::uintptr_t begin;
    std::uintptr_t end;
};

BufferRange buffer_range(const double* pointer, std::size_t size, const char* label)
{
    if (size == 0)
    {
        return {0, 0};
    }
    if (pointer == nullptr)
    {
        throw std::invalid_argument(std::string("native CIDER evaluator ") + label
                                    + " is null");
    }
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        throw std::overflow_error(std::string("native CIDER evaluator ") + label
                                  + " byte size overflows");
    }
    const std::size_t bytes = size * sizeof(double);
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    {
        throw std::overflow_error(std::string("native CIDER evaluator ") + label
                                  + " address range overflows");
    }
    return {begin, begin + bytes};
}

bool overlap(const BufferRange& first, const BufferRange& second)
{
    return first.begin != first.end && second.begin != second.end
           && first.begin < second.end && second.begin < first.end;
}

void validate_finite(const double* values, std::size_t size, const char* label)
{
    for (std::size_t i = 0; i < size; ++i)
    {
        if (!std::isfinite(values[i]))
        {
            throw std::invalid_argument(std::string("native CIDER evaluator ") + label
                                        + " contains a nonfinite value");
        }
    }
}

struct Shape
{
    std::size_t nrho;
    std::size_t nsigma_channels;
    std::size_t nsigma;
    bool has_tau;
    std::size_t semilocal_feature_count;
    std::size_t nldf_feature_count;
    std::size_t raw_feature_count;
    std::size_t transformed_feature_count;
    std::size_t semilocal_spin_stride;
    std::size_t nldf_spin_stride;
    std::size_t raw_spin_stride;
    std::size_t transformed_spin_stride;
    std::size_t semilocal_size;
    std::size_t nldf_size;
    std::size_t raw_size;
    std::size_t transformed_size;
};

void validate_density(const DensityTupleInput& density,
                      const Shape& shape,
                      const char* role,
                      std::vector<BufferRange>& inputs)
{
    if (density.rho_size() != shape.nrho || density.sigma_size() != shape.nsigma
        || (shape.has_tau && density.tau_size() != shape.nrho)
        || (!shape.has_tau && density.tau_size() != 0))
    {
        throw std::invalid_argument(std::string("native CIDER evaluator ") + role
                                    + " density tuple size is invalid");
    }
    if (density.rho() == nullptr || density.sigma() == nullptr
        || (shape.has_tau && density.tau() == nullptr)
        || (!shape.has_tau && density.tau() != nullptr))
    {
        throw std::invalid_argument(std::string("native CIDER evaluator ") + role
                                    + " density tuple pointer is invalid");
    }
    validate_finite(density.rho(), density.rho_size(), role);
    validate_finite(density.sigma(), density.sigma_size(), role);
    validate_finite(density.tau(), density.tau_size(), role);
    inputs.push_back(buffer_range(density.rho(), density.rho_size(), role));
    inputs.push_back(buffer_range(density.sigma(), density.sigma_size(), role));
    inputs.push_back(buffer_range(density.tau(), density.tau_size(), role));
}

void validate_component(const ComponentOutput& component,
                        const Shape& shape,
                        std::size_t ngrids,
                        const char* label,
                        std::vector<BufferRange>& outputs)
{
    if (component.exc_size() != ngrids
        || component.vrho_size() != shape.nrho
        || component.vsigma_size() != shape.nsigma
        || (shape.has_tau && component.vtau_size() != shape.nrho)
        || (!shape.has_tau && component.vtau_size() != 0))
    {
        throw std::invalid_argument(std::string("native CIDER evaluator ") + label
                                    + " output size is invalid");
    }
    if (component.exc() == nullptr || component.vrho() == nullptr
        || component.vsigma() == nullptr || (shape.has_tau && component.vtau() == nullptr)
        || (!shape.has_tau && component.vtau() != nullptr))
    {
        throw std::invalid_argument(std::string("native CIDER evaluator ") + label
                                    + " output pointer is invalid");
    }
    outputs.push_back(buffer_range(component.exc(), component.exc_size(), label));
    outputs.push_back(buffer_range(component.vrho(), component.vrho_size(), label));
    outputs.push_back(buffer_range(component.vsigma(), component.vsigma_size(), label));
    outputs.push_back(buffer_range(component.vtau(), component.vtau_size(), label));
}

Shape validate_contract(const Model& model,
                        std::size_t expected_nspin,
                        std::size_t semilocal_feature_count,
                        std::size_t nldf_feature_count,
                        std::size_t raw_feature_count,
                        std::size_t transformed_feature_count,
                        std::size_t expected_grid_count,
                        const EvaluationInput& input,
                        const EvaluationOutput& output)
{
    if (input.nspin() != expected_nspin || (input.nspin() != 1 && input.nspin() != 2))
    {
        throw std::invalid_argument("native CIDER evaluator spin count is invalid");
    }
    if (expected_grid_count != 0 && input.ngrids() != expected_grid_count)
    {
        throw std::invalid_argument(
            "native CIDER evaluator NLDF grid does not match evaluation dimensions");
    }
    const std::size_t nrho
        = checked_product(input.nspin(), input.ngrids(), "density");
    const std::size_t nsigma_channels = 2 * input.nspin() - 1;
    const std::size_t nsigma
        = checked_product(nsigma_channels, input.ngrids(), "sigma");
    const std::size_t semilocal_spin_stride
        = checked_product(semilocal_feature_count,
                          input.ngrids(),
                          "semilocal feature stride");
    const std::size_t raw_spin_stride
        = checked_product(raw_feature_count, input.ngrids(), "raw feature stride");
    const std::size_t transformed_spin_stride
        = checked_product(transformed_feature_count,
                          input.ngrids(),
                          "transformed feature stride");
    const std::size_t nldf_spin_stride
        = nldf_feature_count == 0
              ? 0
              : checked_product(nldf_feature_count,
                                input.ngrids(),
                                "NLDF feature stride");
    const std::size_t semilocal_size
        = checked_product(input.nspin(),
                          semilocal_spin_stride,
                          "semilocal feature buffer");
    const std::size_t raw_size
        = checked_product(input.nspin(), raw_spin_stride, "raw feature buffer");
    const std::size_t transformed_size
        = checked_product(input.nspin(),
                          transformed_spin_stride,
                          "transformed feature buffer");
    const std::size_t nldf_size
        = nldf_feature_count == 0
              ? 0
              : checked_product(input.nspin(),
                                nldf_spin_stride,
                                "NLDF feature buffer");
    const Shape shape = {nrho,
                         nsigma_channels,
                         nsigma,
                         model.metadata().requires_tau(),
                         semilocal_feature_count,
                         nldf_feature_count,
                         raw_feature_count,
                         transformed_feature_count,
                         semilocal_spin_stride,
                         nldf_spin_stride,
                         raw_spin_stride,
                         transformed_spin_stride,
                         semilocal_size,
                         nldf_size,
                         raw_size,
                         transformed_size};
    std::vector<BufferRange> inputs;
    validate_density(input.baseline(), shape, "baseline", inputs);
    validate_density(input.feature(), shape, "feature", inputs);
    std::vector<BufferRange> outputs;
    validate_component(output.baseline(), shape, input.ngrids(), "baseline", outputs);
    validate_component(output.feature(), shape, input.ngrids(), "feature", outputs);
    validate_component(output.total(), shape, input.ngrids(), "total", outputs);
    for (std::size_t i = 0; i < outputs.size(); ++i)
    {
        for (std::size_t j = i + 1; j < outputs.size(); ++j)
        {
            if (overlap(outputs[i], outputs[j]))
            {
                throw std::invalid_argument(
                    "native CIDER evaluator output buffers must be disjoint");
            }
        }
        for (std::size_t j = 0; j < inputs.size(); ++j)
        {
            if (overlap(outputs[i], inputs[j]))
            {
                throw std::invalid_argument(
                    "native CIDER evaluator input and output buffers must be disjoint");
            }
        }
    }
    return shape;
}

struct ComponentData
{
    ComponentData(std::size_t ngrids,
                  std::size_t nrho,
                  std::size_t nsigma,
                  bool has_tau)
        : exc(ngrids, 0.0),
          vrho(nrho, 0.0),
          vsigma(nsigma, 0.0),
          vtau(has_tau ? nrho : 0, 0.0)
    {
    }
    std::vector<double> exc;
    std::vector<double> vrho;
    std::vector<double> vsigma;
    std::vector<double> vtau;
};

void validate_component_finite(const ComponentData& component, const char* label)
{
    validate_finite(component.exc.data(), component.exc.size(), label);
    validate_finite(component.vrho.data(), component.vrho.size(), label);
    validate_finite(component.vsigma.data(), component.vsigma.size(), label);
    validate_finite(component.vtau.data(), component.vtau.size(), label);
}

class LibxcFunction
{
  public:
    LibxcFunction(int identifier, int polarization) : initialized_(false)
    {
        if (xc_func_init(&function_, identifier, polarization) != 0)
        {
            throw std::runtime_error("native CIDER evaluator failed to initialize LibXC");
        }
        initialized_ = true;
        xc_func_set_dens_threshold(&function_, libxc_density_threshold);
    }
    ~LibxcFunction()
    {
        if (initialized_)
        {
            xc_func_end(&function_);
        }
    }
    xc_func_type* get() { return &function_; }

  private:
    LibxcFunction(const LibxcFunction&);
    LibxcFunction& operator=(const LibxcFunction&);
    xc_func_type function_;
    bool initialized_;
};

ComponentData evaluate_baseline(const DensityTupleInput& input,
                                std::size_t nspin,
                                std::size_t ngrids,
                                double xmix,
                                bool has_tau)
{
    const std::size_t nsigma_channels = 2 * nspin - 1;
    const std::size_t nrho = checked_product(nspin, ngrids, "LibXC density");
    const std::size_t nsigma
        = checked_product(nsigma_channels, ngrids, "LibXC sigma");
    std::vector<double> rho_interleaved(nrho);
    std::vector<double> sigma_interleaved(nsigma);
    for (std::size_t grid = 0; grid < ngrids; ++grid)
    {
        const std::size_t rho_grid_offset
            = checked_multiply(grid, nspin, "LibXC density grid offset");
        const std::size_t sigma_grid_offset
            = checked_multiply(grid,
                               nsigma_channels,
                               "LibXC sigma grid offset");
        for (std::size_t spin = 0; spin < nspin; ++spin)
        {
            const std::size_t interleaved
                = checked_add(rho_grid_offset, spin, "LibXC density offset");
            const std::size_t spin_offset
                = checked_multiply(spin, ngrids, "density spin offset");
            rho_interleaved[interleaved]
                = input.rho()[checked_add(spin_offset, grid, "density grid offset")];
        }
        for (std::size_t channel = 0; channel < nsigma_channels; ++channel)
        {
            const std::size_t interleaved
                = checked_add(sigma_grid_offset, channel, "LibXC sigma offset");
            const std::size_t channel_offset
                = checked_multiply(channel, ngrids, "sigma channel offset");
            sigma_interleaved[interleaved]
                = input.sigma()[checked_add(channel_offset,
                                            grid,
                                            "sigma grid offset")];
        }
    }
    std::vector<double> exc_x(ngrids), exc_c(ngrids);
    std::vector<double> vrho_x(nrho), vrho_c(nrho);
    std::vector<double> vsigma_x(nsigma), vsigma_c(nsigma);
    const int polarization = nspin == 1 ? XC_UNPOLARIZED : XC_POLARIZED;
    LibxcFunction exchange(XC_GGA_X_PBE, polarization);
    LibxcFunction correlation(XC_GGA_C_PBE, polarization);
    xc_gga_exc_vxc(exchange.get(),
                   ngrids,
                   rho_interleaved.data(),
                   sigma_interleaved.data(),
                   exc_x.data(),
                   vrho_x.data(),
                   vsigma_x.data());
    xc_gga_exc_vxc(correlation.get(),
                   ngrids,
                   rho_interleaved.data(),
                   sigma_interleaved.data(),
                   exc_c.data(),
                   vrho_c.data(),
                   vsigma_c.data());
    ComponentData result(ngrids, nrho, nsigma, has_tau);
    const double exchange_weight = 1.0 - xmix;
    for (std::size_t grid = 0; grid < ngrids; ++grid)
    {
        result.exc[grid] = exchange_weight * exc_x[grid] + exc_c[grid];
        const std::size_t rho_grid_offset
            = checked_multiply(grid, nspin, "LibXC density grid offset");
        const std::size_t sigma_grid_offset
            = checked_multiply(grid,
                               nsigma_channels,
                               "LibXC sigma grid offset");
        for (std::size_t spin = 0; spin < nspin; ++spin)
        {
            const std::size_t interleaved
                = checked_add(rho_grid_offset, spin, "LibXC density offset");
            const std::size_t spin_offset
                = checked_multiply(spin, ngrids, "density spin offset");
            result.vrho[checked_add(spin_offset, grid, "density grid offset")]
                = exchange_weight * vrho_x[interleaved] + vrho_c[interleaved];
        }
        for (std::size_t channel = 0; channel < nsigma_channels; ++channel)
        {
            const std::size_t interleaved
                = checked_add(sigma_grid_offset, channel, "LibXC sigma offset");
            const std::size_t channel_offset
                = checked_multiply(channel, ngrids, "sigma channel offset");
            result.vsigma[checked_add(channel_offset, grid, "sigma grid offset")]
                = exchange_weight * vsigma_x[interleaved] + vsigma_c[interleaved];
        }
    }
    validate_component_finite(result, "baseline output");
    return result;
}

void copy_component(const ComponentData& source, const ComponentOutput& target)
{
    std::copy(source.exc.begin(), source.exc.end(), target.exc());
    std::copy(source.vrho.begin(), source.vrho.end(), target.vrho());
    std::copy(source.vsigma.begin(), source.vsigma.end(), target.vsigma());
    std::copy(source.vtau.begin(), source.vtau.end(), target.vtau());
}

} // namespace

class DualDensityEvaluator::Impl
{
  public:
    Impl(const Model& model,
         std::size_t nspin,
         double cider_xmix,
         const NldfGrid* grid,
         const NldfAlphaSelection* alpha_selection)
        : model_(model),
          nspin_(nspin),
          cider_xmix_(cider_xmix),
          runtime_scale_(model.metadata().runtime_exchange_scale(cider_xmix)),
          semilocal_feature_count_(model.metadata().semilocal_feature_count()),
          nldf_feature_count_(model.metadata().nldf_feature_count()),
          raw_feature_count_(checked_add(semilocal_feature_count_,
                                         nldf_feature_count_,
                                         "raw feature count")),
          transformed_feature_count_(model.metadata().transformed_feature_count()),
          grid_count_(grid == nullptr ? 0 : grid->grid_count()),
          feature_density_policy_source_(
              new std::string(model.metadata().feature_density_policy_source())),
          semilocal_(model.semilocal_mode(), nspin)
    {
        if (nspin != 1 && nspin != 2)
        {
            throw std::invalid_argument("native CIDER evaluator spin count must be one or two");
        }
        if (!std::isfinite(cider_xmix) || cider_xmix < 0.0 || cider_xmix > 1.0)
        {
            throw std::invalid_argument("native CIDER cider_xmix must be in [0,1]");
        }
        const ModelMetadata& metadata = model.metadata();
        if (!std::isfinite(metadata.ml_exchange_scale())
            || metadata.ml_exchange_scale() < 0.0)
        {
            throw std::invalid_argument(
                "native CIDER evaluator trained exchange scale is invalid");
        }
        if (metadata.exchange_baseline() != OuterBaseline::GgaXPbe
            || metadata.correlation_baseline() != OuterBaseline::GgaCPbe)
        {
            throw std::invalid_argument(
                "native CIDER evaluator requires outer PBE exchange and correlation");
        }
        validate_model_contract(false);
        if (!std::isfinite(runtime_scale_))
        {
            throw std::invalid_argument("native CIDER evaluator runtime scale is not finite");
        }
        if (metadata.has_nldf())
        {
            if (grid == nullptr || alpha_selection == nullptr)
            {
                throw std::invalid_argument(
                    "native CIDER evaluator NLDF model requires grid configuration");
            }
            nldf_.reset(new NldfPlan(
                *model.nldf_settings(), nspin, *grid, *alpha_selection));
        }
        else if (grid != nullptr || alpha_selection != nullptr)
        {
            throw std::invalid_argument(
                "native CIDER evaluator non-NLDF model does not accept an NLDF grid");
        }
    }

    void evaluate(const EvaluationInput& input, EvaluationOutput& output) const
    {
        validate_model_contract(true);
        const Shape shape = validate_contract(model_,
                                              nspin_,
                                              semilocal_feature_count_,
                                              nldf_feature_count_,
                                              raw_feature_count_,
                                              transformed_feature_count_,
                                              grid_count_,
                                              input,
                                              output);
        ComponentData baseline = evaluate_baseline(
            input.baseline(), nspin_, input.ngrids(), cider_xmix_, shape.has_tau);

        std::vector<double> rho_feature(
            input.feature().rho(), input.feature().rho() + input.feature().rho_size());
        std::vector<double> sigma_feature(input.feature().sigma(),
                                          input.feature().sigma()
                                              + input.feature().sigma_size());
        std::vector<double> tau_feature;
        if (shape.has_tau)
        {
            tau_feature.assign(input.feature().tau(),
                               input.feature().tau() + input.feature().tau_size());
        }
        std::size_t regularized_rho = 0;
        std::size_t regularized_sigma = 0;
        std::size_t regularized_tau = 0;
        for (std::size_t i = 0; i < rho_feature.size(); ++i)
        {
            if (rho_feature[i] < feature_rho_floor)
            {
                rho_feature[i] = feature_rho_floor;
                ++regularized_rho;
            }
        }
        for (std::size_t spin = 0; spin < nspin_; ++spin)
        {
            const std::size_t channel = 2 * spin;
            const std::size_t channel_offset
                = checked_multiply(channel,
                                   input.ngrids(),
                                   "feature sigma channel offset");
            for (std::size_t grid = 0; grid < input.ngrids(); ++grid)
            {
                double& value
                    = sigma_feature[checked_add(channel_offset,
                                                grid,
                                                "feature sigma grid offset")];
                if (value < 0.0)
                {
                    value = 0.0;
                    ++regularized_sigma;
                }
            }
        }
        for (std::size_t i = 0; i < tau_feature.size(); ++i)
        {
            if (tau_feature[i] < 0.0)
            {
                tau_feature[i] = 0.0;
                ++regularized_tau;
            }
        }

        std::vector<double> semilocal_features(shape.semilocal_size);
        const SemilocalForwardInput semilocal_input(nspin_,
                                                    input.ngrids(),
                                                    rho_feature.data(),
                                                    rho_feature.size(),
                                                    sigma_feature.data(),
                                                    sigma_feature.size(),
                                                    tau_feature.empty()
                                                        ? nullptr
                                                        : tau_feature.data(),
                                                    tau_feature.size());
        SemilocalForwardOutput semilocal_output(semilocal_features.data(),
                                                semilocal_features.size());
        semilocal_.forward(semilocal_input, semilocal_output);
        std::vector<double> raw_features(shape.raw_size, 0.0);
        for (std::size_t spin = 0; spin < nspin_; ++spin)
        {
            const std::size_t source
                = checked_multiply(spin,
                                   shape.semilocal_spin_stride,
                                   "semilocal spin offset");
            const std::size_t target
                = checked_multiply(spin, shape.raw_spin_stride, "raw spin offset");
            std::copy(semilocal_features.begin() + source,
                      semilocal_features.begin()
                          + checked_add(source,
                                        shape.semilocal_spin_stride,
                                        "semilocal copy end"),
                      raw_features.begin() + target);
        }
        if (nldf_)
        {
            std::vector<double> nldf_features(shape.nldf_size);
            const NldfForwardInput nldf_input(nspin_,
                                              input.ngrids(),
                                              rho_feature.data(),
                                              rho_feature.size(),
                                              sigma_feature.data(),
                                              sigma_feature.size(),
                                              tau_feature.empty() ? nullptr
                                                                  : tau_feature.data(),
                                              tau_feature.size());
            NldfForwardOutput nldf_output(nldf_features.data(), nldf_features.size());
            nldf_->forward(nldf_input, nldf_output);
            for (std::size_t spin = 0; spin < nspin_; ++spin)
            {
                const std::size_t source
                    = checked_multiply(spin,
                                       shape.nldf_spin_stride,
                                       "NLDF spin offset");
                const std::size_t raw_spin_offset
                    = checked_multiply(spin,
                                       shape.raw_spin_stride,
                                       "raw spin offset");
                const std::size_t target
                    = checked_add(raw_spin_offset,
                                  shape.semilocal_spin_stride,
                                  "NLDF raw offset");
                std::copy(nldf_features.begin() + source,
                          nldf_features.begin()
                              + checked_add(source,
                                            shape.nldf_spin_stride,
                                            "NLDF copy end"),
                          raw_features.begin() + target);
            }
        }

        ComponentData feature(
            input.ngrids(), shape.nrho, shape.nsigma, shape.has_tau);
        std::vector<double> raw_adjoint(shape.raw_size);
        const MappedEvaluationInput mapped_input(nspin_,
                                                 input.ngrids(),
                                                 raw_features.data(),
                                                 raw_features.size(),
                                                 rho_feature.data(),
                                                 rho_feature.size(),
                                                 mapped_density_cutoff);
        MappedEvaluationOutput mapped_output(feature.exc.data(),
                                             feature.exc.size(),
                                             raw_adjoint.data(),
                                             raw_adjoint.size());
        mapped_.evaluate(model_, mapped_input, mapped_output);
        for (std::size_t i = 0; i < feature.exc.size(); ++i)
        {
            feature.exc[i] *= runtime_scale_;
        }
        for (std::size_t i = 0; i < raw_adjoint.size(); ++i)
        {
            raw_adjoint[i] *= runtime_scale_;
        }

        std::vector<double> semilocal_adjoint(shape.semilocal_size);
        for (std::size_t spin = 0; spin < nspin_; ++spin)
        {
            const std::size_t source
                = checked_multiply(spin, shape.raw_spin_stride, "raw spin offset");
            const std::size_t target
                = checked_multiply(spin,
                                   shape.semilocal_spin_stride,
                                   "semilocal spin offset");
            std::copy(raw_adjoint.begin() + source,
                      raw_adjoint.begin()
                          + checked_add(source,
                                        shape.semilocal_spin_stride,
                                        "semilocal adjoint copy end"),
                      semilocal_adjoint.begin() + target);
        }
        const SemilocalReverseInput semilocal_reverse_input(
            nspin_,
            input.ngrids(),
            rho_feature.data(),
            rho_feature.size(),
            sigma_feature.data(),
            sigma_feature.size(),
            tau_feature.empty() ? nullptr : tau_feature.data(),
            tau_feature.size(),
            semilocal_adjoint.data(),
            semilocal_adjoint.size());
        SemilocalReverseOutput semilocal_reverse_output(feature.vrho.data(),
                                                        feature.vrho.size(),
                                                        feature.vsigma.data(),
                                                        feature.vsigma.size(),
                                                        feature.vtau.empty()
                                                            ? nullptr
                                                            : feature.vtau.data(),
                                                        feature.vtau.size());
        semilocal_.reverse(semilocal_reverse_input, semilocal_reverse_output);
        if (nldf_)
        {
            std::vector<double> nldf_adjoint(shape.nldf_size);
            for (std::size_t spin = 0; spin < nspin_; ++spin)
            {
                const std::size_t raw_spin_offset
                    = checked_multiply(spin,
                                       shape.raw_spin_stride,
                                       "raw spin offset");
                const std::size_t source
                    = checked_add(raw_spin_offset,
                                  shape.semilocal_spin_stride,
                                  "NLDF raw adjoint offset");
                const std::size_t target
                    = checked_multiply(spin,
                                       shape.nldf_spin_stride,
                                       "NLDF spin offset");
                std::copy(raw_adjoint.begin() + source,
                          raw_adjoint.begin()
                              + checked_add(source,
                                            shape.nldf_spin_stride,
                                            "NLDF adjoint copy end"),
                          nldf_adjoint.begin() + target);
            }
            const NldfReverseInput nldf_reverse_input(
                nspin_,
                input.ngrids(),
                rho_feature.data(),
                rho_feature.size(),
                sigma_feature.data(),
                sigma_feature.size(),
                tau_feature.empty() ? nullptr : tau_feature.data(),
                tau_feature.size(),
                nldf_adjoint.data(),
                nldf_adjoint.size());
            NldfReverseOutput nldf_reverse_output(feature.vrho.data(),
                                                  feature.vrho.size(),
                                                  feature.vsigma.data(),
                                                  feature.vsigma.size(),
                                                  feature.vtau.empty()
                                                      ? nullptr
                                                      : feature.vtau.data(),
                                                  feature.vtau.size());
            nldf_->reverse(nldf_reverse_input, nldf_reverse_output);
        }
        validate_component_finite(feature, "feature output");

        ComponentData total(
            input.ngrids(), shape.nrho, shape.nsigma, shape.has_tau);
        for (std::size_t i = 0; i < total.exc.size(); ++i)
        {
            total.exc[i] = baseline.exc[i] + feature.exc[i];
        }
        for (std::size_t i = 0; i < total.vrho.size(); ++i)
        {
            total.vrho[i] = baseline.vrho[i] + feature.vrho[i];
        }
        for (std::size_t i = 0; i < total.vsigma.size(); ++i)
        {
            total.vsigma[i] = baseline.vsigma[i] + feature.vsigma[i];
        }
        for (std::size_t i = 0; i < total.vtau.size(); ++i)
        {
            total.vtau[i] = baseline.vtau[i] + feature.vtau[i];
        }
        validate_component_finite(total, "total output");

        EvaluationDiagnostics diagnostics;
        diagnostics.feature_density_policy_ = model_.metadata().feature_density_policy();
        diagnostics.feature_density_policy_source_ = feature_density_policy_source_;
        diagnostics.feature_energy_convention_ = model_.metadata().energy_convention();
        diagnostics.baseline_exc_convention_
            = BaselineExcConvention::LibxcExcPerParticle;
        diagnostics.cider_xmix_ = cider_xmix_;
        diagnostics.trained_exchange_scale_ = model_.metadata().ml_exchange_scale();
        diagnostics.runtime_scale_ = runtime_scale_;
        diagnostics.regularized_rho_count_ = regularized_rho;
        diagnostics.regularized_same_spin_sigma_count_ = regularized_sigma;
        diagnostics.regularized_tau_count_ = regularized_tau;
        diagnostics.baseline_exc_sum_
            = std::accumulate(baseline.exc.begin(), baseline.exc.end(), 0.0);
        diagnostics.feature_exc_sum_
            = std::accumulate(feature.exc.begin(), feature.exc.end(), 0.0);
        diagnostics.total_exc_sum_
            = std::accumulate(total.exc.begin(), total.exc.end(), 0.0);
        validate_finite(&diagnostics.baseline_exc_sum_, 1, "diagnostic baseline sum");
        validate_finite(&diagnostics.feature_exc_sum_, 1, "diagnostic feature sum");
        validate_finite(&diagnostics.total_exc_sum_, 1, "diagnostic total sum");

        copy_component(baseline, output.baseline());
        copy_component(feature, output.feature());
        copy_component(total, output.total());
        output.diagnostics_ = diagnostics;
    }

  private:
    void validate_model_contract(bool require_backend) const
    {
        const ModelMetadata& metadata = model_.metadata();
        const std::size_t checked_raw
            = checked_add(metadata.semilocal_feature_count(),
                          metadata.nldf_feature_count(),
                          "raw feature count");
        if (metadata.semilocal_feature_count() != semilocal_feature_count_
            || metadata.nldf_feature_count() != nldf_feature_count_
            || checked_raw != raw_feature_count_
            || metadata.raw_feature_count() != raw_feature_count_
            || metadata.transformed_feature_count() != transformed_feature_count_)
        {
            throw std::invalid_argument(
                "native CIDER evaluator model feature counts changed or are inconsistent");
        }
        const SemilocalMode expected_mode
            = metadata.semilocal_level() == SemilocalLevel::Mgga
                  ? SemilocalMode::Npa
                  : SemilocalMode::Np;
        if (model_.semilocal_mode() != expected_mode
            || semilocal_feature_count_ != semilocal_.feature_count())
        {
            throw std::invalid_argument(
                "native CIDER evaluator semilocal profile is inconsistent");
        }
        const bool has_nldf = metadata.has_nldf();
        const NldfSettings* settings = model_.nldf_settings();
        if (has_nldf != (nldf_feature_count_ != 0)
            || has_nldf != (settings != nullptr))
        {
            throw std::invalid_argument(
                "native CIDER evaluator NLDF flag, count, and settings are inconsistent");
        }
        if (settings != nullptr
            && (settings->feature_count() != nldf_feature_count_
                || settings->semilocal_level() != metadata.semilocal_level()))
        {
            throw std::invalid_argument(
                "native CIDER evaluator NLDF settings profile is inconsistent");
        }
        if (model_.normalizers().size() != raw_feature_count_
            || transformed_feature_count_ == 0
            || model_.feature_maps().size() != transformed_feature_count_)
        {
            throw std::invalid_argument(
                "native CIDER evaluator normalizer or feature-map cardinality is inconsistent");
        }
        for (std::size_t i = 0; i < model_.feature_maps().size(); ++i)
        {
            if (model_.feature_maps()[i].raw_feature_index() >= raw_feature_count_)
            {
                throw std::invalid_argument(
                    "native CIDER evaluator feature-map index is out of range");
            }
        }
        if (require_backend && has_nldf != static_cast<bool>(nldf_))
        {
            throw std::invalid_argument(
                "native CIDER evaluator NLDF backend is inconsistent with the model");
        }
    }

    const Model& model_;
    const std::size_t nspin_;
    const double cider_xmix_;
    const double runtime_scale_;
    const std::size_t semilocal_feature_count_;
    const std::size_t nldf_feature_count_;
    const std::size_t raw_feature_count_;
    const std::size_t transformed_feature_count_;
    const std::size_t grid_count_;
    const std::shared_ptr<const std::string> feature_density_policy_source_;
    const SemilocalPlan semilocal_;
    const MappedXC mapped_;
    std::unique_ptr<NldfPlan> nldf_;
};

DualDensityEvaluator::DualDensityEvaluator(const Model& model,
                                           std::size_t nspin,
                                           double cider_xmix)
    : impl_(new Impl(model, nspin, cider_xmix, nullptr, nullptr))
{
}

DualDensityEvaluator::DualDensityEvaluator(const Model& model,
                                           std::size_t nspin,
                                           double cider_xmix,
                                           const NldfGrid& grid,
                                           const NldfAlphaSelection& alpha_selection)
    : impl_(new Impl(model, nspin, cider_xmix, &grid, &alpha_selection))
{
}

DualDensityEvaluator::~DualDensityEvaluator() = default;

void DualDensityEvaluator::evaluate(const EvaluationInput& input,
                                    EvaluationOutput& output) const
{
    impl_->evaluate(input, output);
}

} // namespace cider
} // namespace hamilt
