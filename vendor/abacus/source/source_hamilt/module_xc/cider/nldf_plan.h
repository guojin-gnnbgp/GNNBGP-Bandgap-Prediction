#ifndef MODULE_HAMILT_CIDER_NLDF_PLAN_H
#define MODULE_HAMILT_CIDER_NLDF_PLAN_H

#include "model.h"
#include "nldf_backend.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace hamilt
{
namespace cider
{

class NldfAlphaSelection
{
  public:
    // alpha_count==0 selects it exactly as the accepted Python bridge does.
    NldfAlphaSelection(double cutoff, double lambda, std::size_t alpha_count);
    double cutoff() const;
    double lambda() const;
    std::size_t alpha_count() const;

  private:
    double cutoff_;
    double lambda_;
    std::size_t alpha_count_;
};

struct NldfExponentValue
{
    double value;
    double drho;
    double dsigma;
    double dtau;
};

struct NldfInterpolationArgument
{
    double value;
    double drho;
    double dsigma;
    double dtau;
};

class NldfForwardInput
{
  public:
    // rho/tau are spin-major (nspin,ngrids).  sigma is channel-major
    // (2*nspin-1,ngrids), [uu,ud,dd] for two spins; only uu/dd are read.
    NldfForwardInput(std::size_t nspin,
                     std::size_t ngrids,
                     const double* rho,
                     std::size_t rho_size,
                     const double* sigma,
                     std::size_t sigma_size,
                     const double* tau,
                     std::size_t tau_size);
    std::size_t nspin;
    std::size_t ngrids;
    const double* rho;
    std::size_t rho_size;
    const double* sigma;
    std::size_t sigma_size;
    const double* tau;
    std::size_t tau_size;
};

class NldfForwardOutput
{
  public:
    // features are spin-major (nspin,nfeature,ngrids).  Optional arguments
    // and convolution_function are spin-major (nspin,ngrids).
    NldfForwardOutput(double* features,
                      std::size_t feature_size,
                      double* interpolation_arguments = nullptr,
                      std::size_t argument_size = 0,
                      double* convolution_function = nullptr,
                      std::size_t function_size = 0);
    double* features;
    std::size_t feature_size;
    double* interpolation_arguments;
    std::size_t argument_size;
    double* convolution_function;
    std::size_t function_size;
};

class NldfReverseInput
{
  public:
    NldfReverseInput(std::size_t nspin,
                     std::size_t ngrids,
                     const double* rho,
                     std::size_t rho_size,
                     const double* sigma,
                     std::size_t sigma_size,
                     const double* tau,
                     std::size_t tau_size,
                     const double* feature_adjoint,
                     std::size_t feature_adjoint_size);
    std::size_t nspin;
    std::size_t ngrids;
    const double* rho;
    std::size_t rho_size;
    const double* sigma;
    std::size_t sigma_size;
    const double* tau;
    std::size_t tau_size;
    const double* feature_adjoint;
    std::size_t feature_adjoint_size;
};

class NldfReverseOutput
{
  public:
    // Accumulates into the same layouts as the corresponding forward inputs;
    // the cross-spin sigma channel is never modified.
    NldfReverseOutput(double* vrho,
                      std::size_t vrho_size,
                      double* vsigma,
                      std::size_t vsigma_size,
                      double* vtau,
                      std::size_t vtau_size);
    double* vrho;
    std::size_t vrho_size;
    double* vsigma;
    std::size_t vsigma_size;
    double* vtau;
    std::size_t vtau_size;
};

// Accepted version-J, squared-exponential, density-multiplier-one plan.
class NldfPlan
{
  public:
    NldfPlan(const NldfSettings& settings,
             std::size_t nspin,
             const NldfGrid& grid,
             const NldfAlphaSelection& selection);
    ~NldfPlan();

    std::size_t spin_count() const;
    std::size_t feature_count() const;
    std::size_t alpha_count() const;
    double alpha0() const;
    double lambda() const;
    const std::vector<double>& alphas() const;
    const std::vector<double>& alpha_norms() const;

    // parameter_index==-1 selects theta; otherwise it selects a feature.
    NldfExponentValue exponent(double rho,
                               double same_spin_sigma,
                               double tau,
                               int parameter_index) const;
    NldfInterpolationArgument
    interpolation_argument(const NldfExponentValue& exponent) const;
    void interpolation_coefficients(double argument,
                                    std::vector<double>& coefficients,
                                    std::vector<double>& derivatives) const;
    void forward(const NldfForwardInput& input, NldfForwardOutput& output) const;
    void reverse(const NldfReverseInput& input, NldfReverseOutput& output) const;

  private:
    NldfPlan(const NldfPlan&);
    NldfPlan& operator=(const NldfPlan&);

    NldfSettings settings_;
    std::size_t nspin_;
    NldfGrid grid_;
    double alpha0_;
    double lambda_;
    double rho_cutoff_;
    std::vector<double> alphas_;
    std::vector<double> alpha_norms_;
    std::vector<double> spline_table_;
    std::unique_ptr<NldfBackend> backend_;
};

} // namespace cider
} // namespace hamilt

#endif // MODULE_HAMILT_CIDER_NLDF_PLAN_H
