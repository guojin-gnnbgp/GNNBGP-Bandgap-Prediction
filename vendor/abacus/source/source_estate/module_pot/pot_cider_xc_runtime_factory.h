#ifndef POT_CIDER_XC_RUNTIME_FACTORY_H
#define POT_CIDER_XC_RUNTIME_FACTORY_H

#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "pot_cider_xc_runtime.h"

#include <cstddef>
#include <memory>
#include <string>

namespace elecstate
{

std::unique_ptr<PotCiderRuntime> make_pot_cider_runtime(
    const std::string& backend,
    const std::string& model_path,
    const std::size_t grid_shape[3],
    const double cell[9],
    std::size_t nspin,
    double cider_xmix);

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC
#endif // POT_CIDER_XC_RUNTIME_FACTORY_H
