#include "cider_model_input.h"

#include "source_hamilt/module_xc/cider/model.h"
#include "source_hamilt/module_xc/cider/model_loader.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace ModuleIO
{

namespace
{

bool ends_with_case_insensitive(const std::string& value,
                                const std::string& suffix)
{
    if (value.size() < suffix.size())
    {
        return false;
    }
    const std::size_t offset = value.size() - suffix.size();
    for (std::size_t i = 0; i < suffix.size(); ++i)
    {
        if (std::tolower(static_cast<unsigned char>(value[offset + i]))
            != std::tolower(static_cast<unsigned char>(suffix[i])))
        {
            return false;
        }
    }
    return true;
}

std::string density_policy_name(
    const hamilt::cider::FeatureDensityPolicy policy)
{
    switch (policy)
    {
        case hamilt::cider::FeatureDensityPolicy::Valence:
            return "valence";
        case hamilt::cider::FeatureDensityPolicy::ValencePseudoCore:
            return "valence_pseudo_core";
    }
    throw std::runtime_error("unknown native CIDER feature-density policy");
}

std::string canonical_density_policy(std::string policy)
{
    std::transform(policy.begin(),
                   policy.end(),
                   policy.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    if (policy == "total")
    {
        return "valence_pseudo_core";
    }
    return policy;
}

std::string lowercase(std::string value)
{
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    return value;
}

std::invalid_argument inspection_error(const std::string& model_path,
                                       const std::string& code,
                                       const std::string& yaml_path,
                                       const std::string& reason)
{
    std::ostringstream message;
    message << "CIDER model inspection failed: model='" << model_path
            << "' code=" << code
            << " yaml_path=" << yaml_path
            << " reason=" << reason;
    return std::invalid_argument(message.str());
}

} // namespace

CiderModelInputMetadata::CiderModelInputMetadata()
    : requires_tau(false),
      feature_density(),
      feature_density_source()
{
}

CiderModelInputMetadata::CiderModelInputMetadata(
    const bool requires_tau_in,
    const std::string& feature_density_in,
    const std::string& feature_density_source_in)
    : requires_tau(requires_tau_in),
      feature_density(feature_density_in),
      feature_density_source(feature_density_source_in)
{
}

CiderModelInputMetadata inspect_cider_model_input(
    const std::string& model_path)
{
    const bool yaml_suffix = ends_with_case_insensitive(model_path, ".yaml")
                             || ends_with_case_insensitive(model_path, ".yml");
    if (!yaml_suffix)
    {
        const bool joblib_suffix
            = ends_with_case_insensitive(model_path, ".joblib")
              || ends_with_case_insensitive(model_path, ".pkl")
              || ends_with_case_insensitive(model_path, ".pickle");
        throw inspection_error(
            model_path,
            joblib_suffix ? "JOBLIB_UNSUPPORTED" : "ARTIFACT_UNSUPPORTED",
            "<file>",
            joblib_suffix ? "pickle/joblib input is not supported"
                           : "expected a .yaml or .yml artifact");
    }

    try
    {
        const hamilt::cider::ModelMetadata metadata
            = hamilt::cider::ModelLoader().inspect_metadata(model_path);
        return CiderModelInputMetadata(
            metadata.requires_tau(),
            density_policy_name(metadata.feature_density_policy()),
            metadata.feature_density_policy_source());
    }
    catch (const hamilt::cider::LegacyYamlError& error)
    {
        throw inspection_error(model_path,
                               error.code(),
                               error.path(),
                               error.detail());
    }
    catch (const std::exception& error)
    {
        throw inspection_error(model_path,
                               "MODEL_LOAD_FAILED",
                               "<file>",
                               error.what());
    }
}

std::string resolve_cider_feature_density(
    const std::string& requested_policy,
    const CiderModelInputMetadata& metadata,
    const std::string& model_path)
{
    const std::string requested = canonical_density_policy(requested_policy);
    if (requested != "auto"
        && requested != "valence"
        && requested != "valence_pseudo_core")
    {
        throw std::invalid_argument(
            "cider_feature_density must be auto, valence, or "
            "valence_pseudo_core");
    }

    if (requested == "auto")
    {
        return metadata.feature_density;
    }
    if (requested != metadata.feature_density)
    {
        std::ostringstream message;
        message << "cider_feature_density conflicts with loaded CIDER model "
                << "metadata: model='" << model_path
                << "' requested=" << requested
                << " expected=" << metadata.feature_density
                << " source=" << metadata.feature_density_source;
        throw std::invalid_argument(message.str());
    }
    return requested;
}

std::string resolve_cider_backend(
    const std::string& requested_backend,
    const CiderBackendCapabilities& capabilities)
{
    const std::string requested = lowercase(requested_backend);
    if (requested == "auto")
    {
        if (capabilities.external)
        {
            return "external";
        }
        if (capabilities.native)
        {
            return "native";
        }
    }
    else if ((requested == "native" && capabilities.native)
             || (requested == "external" && capabilities.external))
    {
        return requested;
    }

    std::ostringstream message;
    message << "CIDER backend is unavailable: requested=" << requested
            << " native=" << (capabilities.native ? "enabled" : "disabled")
            << " external=" << (capabilities.external ? "enabled" : "disabled");
    throw std::runtime_error(message.str());
}

} // namespace ModuleIO
