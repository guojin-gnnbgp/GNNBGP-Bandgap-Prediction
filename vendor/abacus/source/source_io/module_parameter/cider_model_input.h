#ifndef MODULE_IO_PARAMETER_CIDER_MODEL_INPUT_H
#define MODULE_IO_PARAMETER_CIDER_MODEL_INPUT_H

#include <string>

namespace ModuleIO
{

// The small, serializable metadata packet needed by input conversion.  Keep
// the typed native model itself inside cider_native; MPI ranks only need this
// authoritative allocation and density-policy information at this stage.
struct CiderModelInputMetadata
{
    CiderModelInputMetadata();
    CiderModelInputMetadata(bool requires_tau,
                            const std::string& feature_density,
                            const std::string& feature_density_source);

    bool requires_tau;
    std::string feature_density;
    std::string feature_density_source;
};

struct CiderBackendCapabilities
{
    CiderBackendCapabilities(bool native_in, bool external_in)
        : native(native_in), external(external_in)
    {
    }

    bool native;
    bool external;
};

// Inspect a supported native CIDER YAML artifact.  Failures are reported as a
// single deterministic message containing the input model path and the native
// loader's code, YAML node path, and reason.
CiderModelInputMetadata inspect_cider_model_input(const std::string& model_path);

// Canonicalize and validate the user policy against authoritative model
// metadata.  The returned value is one of valence or valence_pseudo_core.
std::string resolve_cider_feature_density(
    const std::string& requested_policy,
    const CiderModelInputMetadata& metadata,
    const std::string& model_path);

// Resolve auto or an explicit backend against the backends compiled into the
// executable. The result is always native or external.
std::string resolve_cider_backend(
    const std::string& requested_backend,
    const CiderBackendCapabilities& capabilities);

} // namespace ModuleIO

#endif // MODULE_IO_PARAMETER_CIDER_MODEL_INPUT_H
