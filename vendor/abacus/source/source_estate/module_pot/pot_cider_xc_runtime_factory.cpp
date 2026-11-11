#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "pot_cider_xc_runtime_factory.h"

#include "pot_cider_xc_native.h"
#ifdef ENABLE_CIDER_EXTERNAL
#include "pot_cider_xc_external.h"
#endif

#include <stdexcept>

namespace elecstate
{

std::unique_ptr<PotCiderRuntime> make_pot_cider_runtime(
    const std::string& backend,
    const std::string& model_path,
    const std::size_t grid_shape[3],
    const double cell[9],
    const std::size_t nspin,
    const double cider_xmix)
{
    if (backend == "native")
    {
        return std::unique_ptr<PotCiderRuntime>(new PotCiderNativeRuntime(
            model_path, grid_shape, cell, nspin, cider_xmix));
    }
    if (backend == "external")
    {
#ifdef ENABLE_CIDER_EXTERNAL
        return std::unique_ptr<PotCiderRuntime>(new PotCiderExternalRuntime(
            model_path, grid_shape, cell, nspin, cider_xmix));
#else
        throw std::runtime_error(
            "external CIDER backend was not compiled into this executable");
#endif
    }
    throw std::invalid_argument(
        "CIDER runtime factory requires resolved backend native or external");
}

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC
