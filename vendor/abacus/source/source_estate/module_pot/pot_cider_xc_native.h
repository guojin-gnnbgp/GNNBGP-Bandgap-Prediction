#ifndef POT_CIDER_XC_NATIVE_H
#define POT_CIDER_XC_NATIVE_H

#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "pot_cider_xc_runtime.h"

#include <cstddef>
#include <memory>
#include <string>

namespace elecstate
{

// Native model/evaluator ownership boundary used by PotCiderXC. Input and
// output arrays retain the Task 11 spin-major/channel-major contract.
class PotCiderNativeRuntime final : public PotCiderRuntime
{
  public:
    PotCiderNativeRuntime(const std::string& model_path,
                          const std::size_t grid_shape[3],
                          const double cell[9],
                          std::size_t nspin,
                          double cider_xmix);
    PotCiderNativeRuntime(const std::string& model_path,
                          const std::size_t grid_shape[3],
                          const double cell[9],
                          std::size_t nspin,
                          double cider_xmix,
                          double nldf_cutoff,
                          double nldf_lambda,
                          std::size_t nldf_alpha_count);
    ~PotCiderNativeRuntime();

    const char* backend_name() const override;
    const PotCiderRuntimeMetadata& metadata() const override;
    void evaluate(const hamilt::cider::EvaluationInput& input,
                  const hamilt::cider::ComponentOutput& baseline,
                  const hamilt::cider::ComponentOutput& feature) const;

  private:
    PotCiderNativeRuntime(const PotCiderNativeRuntime&) = delete;
    PotCiderNativeRuntime& operator=(const PotCiderNativeRuntime&) = delete;

    std::unique_ptr<const hamilt::cider::Model> model_;
    std::unique_ptr<hamilt::cider::DualDensityEvaluator> evaluator_;
    PotCiderRuntimeMetadata metadata_;
};

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC
#endif // POT_CIDER_XC_NATIVE_H
