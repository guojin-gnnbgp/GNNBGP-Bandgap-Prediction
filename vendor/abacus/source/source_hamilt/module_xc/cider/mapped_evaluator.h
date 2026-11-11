#ifndef MODULE_HAMILT_CIDER_MAPPED_EVALUATOR_H
#define MODULE_HAMILT_CIDER_MAPPED_EVALUATOR_H

#include <cstddef>

namespace hamilt
{
namespace cider
{

class Model;

// Raw features use contiguous spin-major (nspin, nfeature, ngrids) order.
// rho contains the physical spin densities in (nspin, ngrids) order; raw
// feature zero is the pre-normalization SEP density descriptor nspin * rho.
// Baselines and the strict less-than density cutoff use normalized feature
// zero, matching Python MappedDFTKernel. Reverse normalization is included in
// the returned raw-feature derivatives.
class MappedEvaluationInput
{
  public:
    MappedEvaluationInput(std::size_t nspin,
                          std::size_t ngrids,
                          const double* raw_features,
                          std::size_t raw_feature_size,
                          const double* rho,
                          std::size_t rho_size,
                          double density_cutoff = 0.0);

    std::size_t nspin() const;
    std::size_t ngrids() const;
    const double* raw_features() const;
    std::size_t raw_feature_size() const;
    const double* rho() const;
    std::size_t rho_size() const;
    double density_cutoff() const;

  private:
    std::size_t nspin_;
    std::size_t ngrids_;
    const double* raw_features_;
    std::size_t raw_feature_size_;
    const double* rho_;
    std::size_t rho_size_;
    double density_cutoff_;
};

// energy_density has (ngrids) layout. raw_feature_derivatives has the same
// spin-major layout as the input raw features and stores derivatives of the
// energy density with respect to each raw descriptor. Input and output ranges,
// and the two output ranges, must be disjoint. On any exception both writable
// output ranges remain byte-for-byte unchanged.
class MappedEvaluationOutput
{
  public:
    MappedEvaluationOutput(double* energy_density,
                           std::size_t energy_density_size,
                           double* raw_feature_derivatives,
                           std::size_t raw_feature_derivative_size);

    double* energy_density() const;
    std::size_t energy_density_size() const;
    double* raw_feature_derivatives() const;
    std::size_t raw_feature_derivative_size() const;

  private:
    double* energy_density_;
    std::size_t energy_density_size_;
    double* raw_feature_derivatives_;
    std::size_t raw_feature_derivative_size_;
};

// Evaluates the typed mapped model independently of density-feature plans.
class MappedEvaluator
{
  public:
    virtual ~MappedEvaluator() = 0;
    virtual void evaluate(const Model& model,
                          const MappedEvaluationInput& input,
                          MappedEvaluationOutput& output) const = 0;

  protected:
    MappedEvaluator() = default;
};

// Native counterpart of the accepted Python MappedDFTKernel SEP path.
class MappedDFTKernel
{
  public:
    void evaluate(const Model& model,
                  const MappedEvaluationInput& input,
                  MappedEvaluationOutput& output) const;
};

// Native counterpart of the accepted one-kernel Python MappedXC root.
class MappedXC final : public MappedEvaluator
{
  public:
    void evaluate(const Model& model,
                  const MappedEvaluationInput& input,
                  MappedEvaluationOutput& output) const override;
};

} // namespace cider
} // namespace hamilt

#endif // MODULE_HAMILT_CIDER_MAPPED_EVALUATOR_H
