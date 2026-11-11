#ifndef MODULE_HAMILT_CIDER_EVALUATOR_H
#define MODULE_HAMILT_CIDER_EVALUATOR_H

#include "model.h"

#include <cstddef>
#include <memory>
#include <string>

namespace hamilt
{
namespace cider
{

class NldfAlphaSelection;
class NldfGrid;

// The accepted bridge exposes LibXC's exc (energy per particle) directly,
// whereas the mapped CIDER component follows the model's energy convention.
enum class BaselineExcConvention
{
    LibxcExcPerParticle
};

// One immutable density role. rho/tau are spin-major (nspin,ngrids); sigma is
// channel-major (2*nspin-1,ngrids), [uu,ud,dd] for two spins.
class DensityTupleInput
{
  public:
    DensityTupleInput(const double* rho,
                      std::size_t rho_size,
                      const double* sigma,
                      std::size_t sigma_size,
                      const double* tau = nullptr,
                      std::size_t tau_size = 0);

    const double* rho() const;
    std::size_t rho_size() const;
    const double* sigma() const;
    std::size_t sigma_size() const;
    const double* tau() const;
    std::size_t tau_size() const;

  private:
    const double* rho_;
    std::size_t rho_size_;
    const double* sigma_;
    std::size_t sigma_size_;
    const double* tau_;
    std::size_t tau_size_;
};

class EvaluationInput
{
  public:
    EvaluationInput(std::size_t nspin,
                    std::size_t ngrids,
                    const DensityTupleInput& baseline,
                    const DensityTupleInput& feature);

    std::size_t nspin() const;
    std::size_t ngrids() const;
    const DensityTupleInput& baseline() const;
    const DensityTupleInput& feature() const;

  private:
    std::size_t nspin_;
    std::size_t ngrids_;
    DensityTupleInput baseline_;
    DensityTupleInput feature_;
};

// One writable component packet. exc has ngrids entries; vrho/optional vtau
// are spin-major and vsigma is channel-major. All output and input ranges must
// be mutually disjoint. No output is changed when evaluation throws.
class ComponentOutput
{
  public:
    ComponentOutput(double* exc,
                    std::size_t exc_size,
                    double* vrho,
                    std::size_t vrho_size,
                    double* vsigma,
                    std::size_t vsigma_size,
                    double* vtau = nullptr,
                    std::size_t vtau_size = 0);

    double* exc() const;
    std::size_t exc_size() const;
    double* vrho() const;
    std::size_t vrho_size() const;
    double* vsigma() const;
    std::size_t vsigma_size() const;
    double* vtau() const;
    std::size_t vtau_size() const;

  private:
    double* exc_;
    std::size_t exc_size_;
    double* vrho_;
    std::size_t vrho_size_;
    double* vsigma_;
    std::size_t vsigma_size_;
    double* vtau_;
    std::size_t vtau_size_;
};

class EvaluationDiagnostics
{
  public:
    EvaluationDiagnostics();

    FeatureDensityPolicy feature_density_policy() const;
    const std::string& feature_density_policy_source() const;
    EnergyConvention feature_energy_convention() const;
    BaselineExcConvention baseline_exc_convention() const;
    double cider_xmix() const;
    double trained_exchange_scale() const;
    double runtime_scale() const;
    std::size_t regularized_rho_count() const;
    std::size_t regularized_same_spin_sigma_count() const;
    std::size_t regularized_tau_count() const;
    double baseline_exc_sum() const;
    double feature_exc_sum() const;
    double total_exc_sum() const;

  private:
    friend class DualDensityEvaluator;
    FeatureDensityPolicy feature_density_policy_;
    std::shared_ptr<const std::string> feature_density_policy_source_;
    EnergyConvention feature_energy_convention_;
    BaselineExcConvention baseline_exc_convention_;
    double cider_xmix_;
    double trained_exchange_scale_;
    double runtime_scale_;
    std::size_t regularized_rho_count_;
    std::size_t regularized_same_spin_sigma_count_;
    std::size_t regularized_tau_count_;
    double baseline_exc_sum_;
    double feature_exc_sum_;
    double total_exc_sum_;
};

class EvaluationOutput
{
  public:
    EvaluationOutput(const ComponentOutput& baseline,
                     const ComponentOutput& feature,
                     const ComponentOutput& total);

    const ComponentOutput& baseline() const;
    const ComponentOutput& feature() const;
    const ComponentOutput& total() const;
    const EvaluationDiagnostics& diagnostics() const;

  private:
    friend class DualDensityEvaluator;
    ComponentOutput baseline_;
    ComponentOutput feature_;
    ComponentOutput total_;
    EvaluationDiagnostics diagnostics_;
};

class Evaluator
{
  public:
    virtual ~Evaluator() = 0;
    virtual void evaluate(const EvaluationInput& input, EvaluationOutput& output) const = 0;

  protected:
    Evaluator() = default;
};

// Concrete native dual-density evaluator. The model must outlive this object.
// The grid constructor is required exactly when the model contains NLDF
// features. The object has no mutable per-evaluation state and is noncopyable.
class DualDensityEvaluator final : public Evaluator
{
  public:
    DualDensityEvaluator(const Model& model, std::size_t nspin, double cider_xmix);
    DualDensityEvaluator(const Model& model,
                         std::size_t nspin,
                         double cider_xmix,
                         const NldfGrid& grid,
                         const NldfAlphaSelection& alpha_selection);
    ~DualDensityEvaluator();

    void evaluate(const EvaluationInput& input, EvaluationOutput& output) const override;

  private:
    DualDensityEvaluator(const DualDensityEvaluator&) = delete;
    DualDensityEvaluator& operator=(const DualDensityEvaluator&) = delete;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cider
} // namespace hamilt

#endif // MODULE_HAMILT_CIDER_EVALUATOR_H
