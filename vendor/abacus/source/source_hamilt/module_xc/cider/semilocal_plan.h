#ifndef MODULE_HAMILT_CIDER_SEMILOCAL_PLAN_H
#define MODULE_HAMILT_CIDER_SEMILOCAL_PLAN_H

#include "model.h"

#include <cstddef>

namespace hamilt
{
namespace cider
{

// rho and optional tau use contiguous (nspin, ngrids) order.  sigma uses the
// native CIDER boundary's channel-major (nsigma, ngrids) layout, after the
// caller converts ABACUS/libxc's grid-interleaved storage: channel 0 for
// nspin=1 and channels [uu, ud, dd] for nspin=2.  Only channels 0 and 2 enter
// spin-polarized semilocal descriptors.  GGA input has tau == nullptr and
// tau_size == 0; MGGA requires a complete tau buffer.
class SemilocalForwardInput
{
  public:
    SemilocalForwardInput(std::size_t nspin,
                          std::size_t ngrids,
                          const double* rho,
                          std::size_t rho_size,
                          const double* sigma,
                          std::size_t sigma_size,
                          const double* tau = nullptr,
                          std::size_t tau_size = 0);

    std::size_t nspin() const;
    std::size_t ngrids() const;
    const double* rho() const;
    std::size_t rho_size() const;
    const double* sigma() const;
    std::size_t sigma_size() const;
    const double* tau() const;
    std::size_t tau_size() const;

  private:
    std::size_t nspin_;
    std::size_t ngrids_;
    const double* rho_;
    std::size_t rho_size_;
    const double* sigma_;
    std::size_t sigma_size_;
    const double* tau_;
    std::size_t tau_size_;
};

// A forward output is exactly the semilocal prefix, in contiguous spin-major
// (nspin, nfeature, ngrids) order.  It never includes an uninitialized NLDF
// tail.  The writable range must be disjoint from every input range.  On any
// exception it remains byte-for-byte unchanged.
class SemilocalForwardOutput
{
  public:
    SemilocalForwardOutput(double* features, std::size_t feature_size);

    double* features() const;
    std::size_t feature_size() const;

  private:
    double* features_;
    std::size_t feature_size_;
};

// Reverse inputs repeat the physical density tuple used by forward and add a
// semilocal-prefix adjoint in contiguous (nspin, nfeature, ngrids) order.
class SemilocalReverseInput
{
  public:
    SemilocalReverseInput(std::size_t nspin,
                          std::size_t ngrids,
                          const double* rho,
                          std::size_t rho_size,
                          const double* sigma,
                          std::size_t sigma_size,
                          const double* tau,
                          std::size_t tau_size,
                          const double* feature_adjoint,
                          std::size_t feature_adjoint_size);

    std::size_t nspin() const;
    std::size_t ngrids() const;
    const double* rho() const;
    std::size_t rho_size() const;
    const double* sigma() const;
    std::size_t sigma_size() const;
    const double* tau() const;
    std::size_t tau_size() const;
    const double* feature_adjoint() const;
    std::size_t feature_adjoint_size() const;

  private:
    std::size_t nspin_;
    std::size_t ngrids_;
    const double* rho_;
    std::size_t rho_size_;
    const double* sigma_;
    std::size_t sigma_size_;
    const double* tau_;
    std::size_t tau_size_;
    const double* feature_adjoint_;
    std::size_t feature_adjoint_size_;
};

// Reverse outputs use (nspin, ngrids) vrho/optional vtau and the same full
// sigma layout as forward.  Contributions are accumulated with +=; the ud
// channel is preserved exactly.  Writable ranges must be mutually disjoint
// and disjoint from every reverse input.  On any exception all outputs remain
// byte-for-byte unchanged.  GGA output omits vtau; MGGA requires it.
class SemilocalReverseOutput
{
  public:
    SemilocalReverseOutput(double* vrho,
                           std::size_t vrho_size,
                           double* vsigma,
                           std::size_t vsigma_size,
                           double* vtau,
                           std::size_t vtau_size);

    double* vrho() const;
    std::size_t vrho_size() const;
    double* vsigma() const;
    std::size_t vsigma_size() const;
    double* vtau() const;
    std::size_t vtau_size() const;

  private:
    double* vrho_;
    std::size_t vrho_size_;
    double* vsigma_;
    std::size_t vsigma_size_;
    double* vtau_;
    std::size_t vtau_size_;
};

// Concrete native counterpart of Python SemilocalPlan2 for the accepted np
// and npa modes.
class SemilocalPlan
{
  public:
    SemilocalPlan(SemilocalMode mode, std::size_t nspin);

    SemilocalMode mode() const;
    std::size_t nspin() const;
    std::size_t feature_count() const;

    void forward(const SemilocalForwardInput& input,
                 SemilocalForwardOutput& output) const;
    void reverse(const SemilocalReverseInput& input,
                 SemilocalReverseOutput& output) const;

  private:
    SemilocalMode mode_;
    std::size_t nspin_;
    std::size_t feature_count_;
};

} // namespace cider
} // namespace hamilt

#endif // MODULE_HAMILT_CIDER_SEMILOCAL_PLAN_H
