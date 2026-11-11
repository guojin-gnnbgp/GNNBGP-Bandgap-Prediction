#ifndef POT_CIDER_XC_RUNTIME_H
#define POT_CIDER_XC_RUNTIME_H

#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "source_hamilt/module_xc/cider/evaluator.h"

#include <string>

namespace elecstate
{

struct PotCiderRuntimeMetadata
{
    bool requires_tau = false;
    std::string feature_density_policy;
    std::string feature_density_policy_source;
    std::string feature_exc_convention;
};

class PotCiderRuntime
{
  public:
    virtual ~PotCiderRuntime();
    virtual const char* backend_name() const = 0;
    virtual const PotCiderRuntimeMetadata& metadata() const = 0;
    virtual void evaluate(
        const hamilt::cider::EvaluationInput& input,
        const hamilt::cider::ComponentOutput& baseline,
        const hamilt::cider::ComponentOutput& feature) const = 0;

  protected:
    PotCiderRuntime() = default;
};

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC
#endif // POT_CIDER_XC_RUNTIME_H
