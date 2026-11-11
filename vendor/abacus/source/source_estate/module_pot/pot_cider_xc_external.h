#ifndef POT_CIDER_XC_EXTERNAL_H
#define POT_CIDER_XC_EXTERNAL_H

#if defined(ENABLE_CIDER) && defined(ENABLE_CIDER_EXTERNAL) && defined(USE_LIBXC)

#include "pot_cider_xc_runtime.h"

#include <cstddef>
#include <string>

struct cider_bridge_ctx;

namespace elecstate
{

class PotCiderExternalRuntime final : public PotCiderRuntime
{
  public:
    PotCiderExternalRuntime(const std::string& model_path,
                            const std::size_t grid_shape[3],
                            const double cell[9],
                            std::size_t nspin,
                            double cider_xmix);
    ~PotCiderExternalRuntime() override;

    const char* backend_name() const override;
    const PotCiderRuntimeMetadata& metadata() const override;
    void evaluate(const hamilt::cider::EvaluationInput& input,
                  const hamilt::cider::ComponentOutput& baseline,
                  const hamilt::cider::ComponentOutput& feature) const override;

  private:
    PotCiderExternalRuntime(const PotCiderExternalRuntime&) = delete;
    PotCiderExternalRuntime& operator=(const PotCiderExternalRuntime&) = delete;

    struct cider_bridge_ctx* context_ = nullptr;
    PotCiderRuntimeMetadata metadata_;
};

} // namespace elecstate

#endif // ENABLE_CIDER && ENABLE_CIDER_EXTERNAL && USE_LIBXC
#endif // POT_CIDER_XC_EXTERNAL_H
