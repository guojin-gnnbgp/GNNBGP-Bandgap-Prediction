#include "model_loader.h"

// yaml-cpp is deliberately confined to the loader implementation boundary.
// Task 4 adds safe node validation without exposing parser types publicly.
#include <yaml-cpp/yaml.h>

#include <cerrno>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

#ifndef CIDER_YAML_CPP_NAMESPACE
#define CIDER_YAML_CPP_NAMESPACE YAML
#endif

namespace yaml = CIDER_YAML_CPP_NAMESPACE;

namespace hamilt
{
namespace cider
{

namespace
{

const char* const yaml_tag_prefix = "tag:yaml.org,2002:";

std::string canonical_tag(const yaml::Node& node)
{
    const std::string tag = node.Tag();
    const std::string prefix(yaml_tag_prefix);
    if (tag.compare(0, prefix.size(), prefix) == 0)
    {
        return tag.substr(prefix.size());
    }
    return tag;
}

bool ends_with_case_insensitive(const std::string& value, const std::string& suffix)
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

bool valid_utf8(const std::string& text)
{
    std::size_t i = 0;
    while (i < text.size())
    {
        const unsigned char first = static_cast<unsigned char>(text[i]);
        if (first <= 0x7fU)
        {
            ++i;
            continue;
        }
        std::size_t count = 0;
        std::uint32_t value = 0;
        if (first >= 0xc2U && first <= 0xdfU)
        {
            count = 1;
            value = first & 0x1fU;
        }
        else if (first >= 0xe0U && first <= 0xefU)
        {
            count = 2;
            value = first & 0x0fU;
        }
        else if (first >= 0xf0U && first <= 0xf4U)
        {
            count = 3;
            value = first & 0x07U;
        }
        else
        {
            return false;
        }
        if (i + count >= text.size())
        {
            return false;
        }
        for (std::size_t j = 1; j <= count; ++j)
        {
            const unsigned char next = static_cast<unsigned char>(text[i + j]);
            if ((next & 0xc0U) != 0x80U)
            {
                return false;
            }
            value = (value << 6U) | (next & 0x3fU);
        }
        if ((count == 2 && value < 0x800U) || (count == 3 && value < 0x10000U)
            || value > 0x10ffffU || (value >= 0xd800U && value <= 0xdfffU))
        {
            return false;
        }
        i += count + 1;
    }
    return true;
}

std::string read_input(const std::string& path, const LoaderLimits& limits)
{
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw LegacyYamlError("YAML_SYNTAX", "<file>", "cannot open input file");
    }
    const std::ifstream::pos_type end = input.tellg();
    if (end < 0)
    {
        throw LegacyYamlError("YAML_SYNTAX", "<file>", "cannot determine input size");
    }
    const std::uintmax_t size = static_cast<std::uintmax_t>(end);
    if (size > limits.max_file_bytes || size > std::numeric_limits<std::size_t>::max())
    {
        throw LegacyYamlError("YAML_SYNTAX", "<file>", "input exceeds configured file-byte limit");
    }
    std::string text(static_cast<std::size_t>(size), '\0');
    input.seekg(0);
    if (!text.empty())
    {
        input.read(&text[0], static_cast<std::streamsize>(text.size()));
    }
    if (!input && !text.empty())
    {
        throw LegacyYamlError("YAML_SYNTAX", "<file>", "cannot read complete input file");
    }
    if (!valid_utf8(text))
    {
        throw LegacyYamlError("YAML_SYNTAX", "<file>", "input is not valid UTF-8");
    }
    return text;
}

// yaml-cpp 0.8 rejects the valid PyYAML block-scalar form used for binary
// payloads ending in '='. Canonicalize only exact !!binary | blocks to an
// equivalent quoted scalar before parsing, retaining blank lines so later
// parser diagnostics keep their original line numbers.
bool ends_with_block_scalar_indicator(std::string syntax)
{
    while (!syntax.empty() && std::isspace(static_cast<unsigned char>(syntax.back())))
    {
        syntax.pop_back();
    }
    const std::size_t token_start = syntax.find_last_of(" \t");
    const std::string token
        = token_start == std::string::npos ? syntax : syntax.substr(token_start + 1U);
    if (token.empty() || (token[0] != '|' && token[0] != '>') || token.size() > 3U)
    {
        return false;
    }
    bool saw_chomp = false;
    bool saw_indent = false;
    for (std::size_t i = 1; i < token.size(); ++i)
    {
        if ((token[i] == '+' || token[i] == '-') && !saw_chomp)
        {
            saw_chomp = true;
        }
        else if (token[i] >= '1' && token[i] <= '9' && !saw_indent)
        {
            saw_indent = true;
        }
        else
        {
            return false;
        }
    }
    return true;
}

std::string normalize_binary_blocks(const std::string& text)
{
    std::vector<std::string> lines;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        lines.push_back(line);
    }
    std::ostringstream output;
    bool in_block_scalar = false;
    std::size_t block_scalar_parent_indent = 0;
    for (std::size_t index = 0; index < lines.size(); ++index)
    {
        const std::string& current = lines[index];
        std::size_t current_indent = 0;
        while (current_indent < current.size() && current[current_indent] == ' ')
        {
            ++current_indent;
        }
        const bool blank = current.find_first_not_of(" \t") == std::string::npos;
        if (in_block_scalar)
        {
            if (blank || current_indent > block_scalar_parent_indent)
            {
                output << current << '\n';
                continue;
            }
            in_block_scalar = false;
        }

        bool single_quoted = false;
        bool double_quoted = false;
        std::size_t comment_offset = std::string::npos;
        for (std::size_t i = 0; i < current.size(); ++i)
        {
            const char c = current[i];
            if (c == '\'' && !double_quoted)
            {
                if (single_quoted && i + 1U < current.size() && current[i + 1U] == '\'')
                {
                    ++i;
                }
                else
                {
                    single_quoted = !single_quoted;
                }
            }
            else if (c == '"' && !single_quoted && (i == 0U || current[i - 1U] != '\\'))
            {
                double_quoted = !double_quoted;
            }
            else if (c == '#' && !single_quoted && !double_quoted
                     && (i == 0U || std::isspace(static_cast<unsigned char>(current[i - 1U]))))
            {
                comment_offset = i;
                break;
            }
        }
        const std::string syntax
            = comment_offset == std::string::npos ? current : current.substr(0, comment_offset);
        const std::string marker = "!!binary |";
        const std::size_t marker_offset = syntax.rfind(marker);
        bool exact_marker = marker_offset != std::string::npos;
        if (exact_marker)
        {
            for (std::size_t i = marker_offset + marker.size(); i < syntax.size(); ++i)
            {
                if (!std::isspace(static_cast<unsigned char>(syntax[i])))
                {
                    exact_marker = false;
                    break;
                }
            }
        }
        std::string prefix = marker_offset == std::string::npos ? std::string() : syntax.substr(0, marker_offset);
        while (!prefix.empty() && std::isspace(static_cast<unsigned char>(prefix.back())))
        {
            prefix.pop_back();
        }
        const bool binary_value_position
            = !prefix.empty() && (prefix.back() == ':' || prefix.back() == '-');
        if (!exact_marker || !binary_value_position)
        {
            output << current << '\n';
            if (ends_with_block_scalar_indicator(syntax))
            {
                in_block_scalar = true;
                block_scalar_parent_indent = current_indent;
            }
            continue;
        }
        std::string encoded;
        std::size_t consumed = 0;
        std::size_t next = index + 1U;
        for (; next < lines.size(); ++next)
        {
            const std::string& payload_line = lines[next];
            std::size_t indent = 0;
            while (indent < payload_line.size() && payload_line[indent] == ' ')
            {
                ++indent;
            }
            if (!payload_line.empty() && indent <= current_indent)
            {
                break;
            }
            for (std::size_t i = indent; i < payload_line.size(); ++i)
            {
                if (!std::isspace(static_cast<unsigned char>(payload_line[i])))
                {
                    encoded.push_back(payload_line[i]);
                }
            }
            ++consumed;
        }
        if (encoded.empty())
        {
            throw LegacyYamlError("YAML_SYNTAX", "<file>", "empty !!binary block scalar");
        }
        output << current.substr(0, marker_offset) << "!!binary '" << encoded << "'\n";
        for (std::size_t blank = 0; blank < consumed; ++blank)
        {
            output << '\n';
        }
        index += consumed;
    }
    return output.str();
}

bool is_standard_tag(const std::string& tag)
{
    static const std::set<std::string> tags
        = {"", "?", "!", "map", "seq", "str", "int", "float", "bool", "null", "binary"};
    return tags.count(tag) != 0;
}

std::string python_target(const std::string& tag)
{
    const std::size_t colon = tag.find(':');
    return colon == std::string::npos ? std::string() : tag.substr(colon + 1);
}

bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.compare(0, prefix.size(), prefix) == 0;
}

bool whitelisted_cider_object_tag(const std::string& tag)
{
    static const std::set<std::string> tags = {
        "python/object:ciderpress.dft.xc_evaluator.MappedXC",
        "python/object:ciderpress.dft.xc_evaluator.MappedDFTKernel",
        "python/object:ciderpress.dft.xc_evaluator.SplineSetEvaluator",
        "python/object:ciderpress.dft.transform_data.FeatureList",
        "python/object:ciderpress.dft.transform_data.UMap",
        "python/object:ciderpress.dft.transform_data.VMap",
        "python/object:ciderpress.dft.transform_data.ZMap",
        "python/object:ciderpress.dft.settings.FeatureSettings",
        "python/object:ciderpress.dft.settings.EmptySettings",
        "python/object:ciderpress.dft.settings.SemilocalSettings",
        "python/object:ciderpress.dft.settings.NLDFSettingsVJ",
        "python/object:ciderpress.dft.feat_normalizer.FeatNormalizerList",
        "python/object:ciderpress.dft.feat_normalizer.ConstantNormalizer"};
    return tags.count(tag) != 0;
}

bool indexed_child_path(const std::string& path, const std::string& prefix)
{
    if (!starts_with(path, prefix) || path.size() <= prefix.size() + 1U || path[prefix.size()] != '['
        || path.back() != ']')
    {
        return false;
    }
    for (std::size_t i = prefix.size() + 1U; i + 1U < path.size(); ++i)
    {
        if (!std::isdigit(static_cast<unsigned char>(path[i])))
        {
            return false;
        }
    }
    return true;
}

void throw_tag_error(const std::string& tag, const std::string& path, bool root)
{
    const std::string target = python_target(tag);
    if (starts_with(target, "numpy."))
    {
        throw LegacyYamlError("UNSUPPORTED_NUMPY_TAG", path, "unsupported NumPy tag " + tag);
    }
    if (starts_with(tag, "python/name:ciderpress.dft.baselines."))
    {
        throw LegacyYamlError("UNSUPPORTED_NAMED_FUNCTION", path, "unsupported baseline tag " + tag);
    }
    if (starts_with(tag, "python/object:ciderpress.dft.xc_evaluator.")
        || starts_with(tag, "python/object:ciderpress.dft.xc_evaluator2."))
    {
        throw LegacyYamlError(root ? "UNSUPPORTED_ROOT" : "UNSUPPORTED_EVALUATOR",
                              path,
                              "unsupported evaluator tag " + tag);
    }
    if (starts_with(tag, "python/object:ciderpress.dft.settings.")
        || starts_with(tag, "python/object:ciderpress.dft.transform_data."))
    {
        throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", path, "unsupported descriptor tag " + tag);
    }
    if (starts_with(tag, "python/object:ciderpress.dft.feat_normalizer."))
    {
        throw LegacyYamlError("UNSUPPORTED_NORMALIZER", path, "unsupported normalizer tag " + tag);
    }
    throw LegacyYamlError("UNKNOWN_TAG", path, "unknown tag " + tag);
}

std::string validated_mapping_key(const yaml::Node& node, const std::string& mapping_path)
{
    const std::string key_path = mapping_path + ".<key>";
    if (!node.IsScalar())
    {
        throw LegacyYamlError("YAML_SYNTAX", key_path, "mapping key is not a scalar string");
    }
    const std::string tag = canonical_tag(node);
    if (!is_standard_tag(tag))
    {
        throw_tag_error(tag, key_path, false);
    }
    if (tag != "" && tag != "?" && tag != "!" && tag != "str")
    {
        throw LegacyYamlError("YAML_SYNTAX", key_path, "mapping key must use the YAML string tag");
    }
    return node.Scalar();
}

class Validator
{
  public:
    explicit Validator(const LoaderLimits& limits)
        : limits_(limits), visited_(0), expanded_scalar_bytes_(0)
    {
    }

    void validate(const yaml::Node& root)
    {
        walk(root, "$", true, 0, Role::Root);
    }

  private:
    enum class Role
    {
        Generic,
        Root,
        CoefficientSets,
        CoefficientArray,
        ScaleValue,
        IndSets,
        IndSetEntry,
        DescriptorIndex,
        NormalizerEntries,
        NormalizerEntry,
        NormalizerConstant,
        BoundsTuple,
        SplineGrids,
        SplineGridTerm,
        SplineGridTuple,
        ArrayArgs,
        ArrayName,
        ArrayPlaceholderTuple,
        ArrayPlaceholderBinary,
        ArrayStateTuple,
        ScaleArrayStateTuple,
        ArrayStateVersion,
        ArrayShapeTuple,
        ScaleArrayShapeTuple,
        ArrayDType,
        ArrayOrder,
        ArrayDataBinary,
        ArrayDtypeArgs,
        DtypeArgs,
        DtypeStateTuple,
        ScalarDType,
        ScalarDataBinary
    };

    static bool reconstruct_tag(const std::string& tag)
    {
        return tag == "python/object/apply:numpy.core.multiarray._reconstruct"
               || tag == "python/object/apply:numpy._core.multiarray._reconstruct";
    }

    static bool scalar_tag(const std::string& tag)
    {
        return tag == "python/object/apply:numpy.core.multiarray.scalar"
               || tag == "python/object/apply:numpy._core.multiarray.scalar";
    }

    Role mapping_child_role(const yaml::Node& parent,
                            const std::string& key,
                            Role parent_role) const
    {
        const std::string tag = canonical_tag(parent);
        if (tag == "python/object:ciderpress.dft.xc_evaluator.SplineSetEvaluator")
        {
            if (key == "coeff_sets")
            {
                return Role::CoefficientSets;
            }
            if (key == "scale")
            {
                return Role::ScaleValue;
            }
            if (key == "ind_sets")
            {
                return Role::IndSets;
            }
            if (key == "spline_grids")
            {
                return Role::SplineGrids;
            }
        }
        if (tag == "python/object:ciderpress.dft.feat_normalizer.FeatNormalizerList"
            && key == "_normalizers")
        {
            return Role::NormalizerEntries;
        }
        if (tag == "python/object:ciderpress.dft.feat_normalizer.ConstantNormalizer"
            && key == "const")
        {
            return Role::NormalizerConstant;
        }
        if ((tag == "python/object:ciderpress.dft.transform_data.UMap"
             || tag == "python/object:ciderpress.dft.transform_data.VMap"
             || tag == "python/object:ciderpress.dft.transform_data.ZMap")
            && key == "_bounds")
        {
            return Role::BoundsTuple;
        }
        if (reconstruct_tag(tag))
        {
            if (key == "args")
            {
                return Role::ArrayArgs;
            }
            if (key == "state")
            {
                return parent_role == Role::ScaleValue ? Role::ScaleArrayStateTuple
                                                       : Role::ArrayStateTuple;
            }
        }
        if (tag == "python/object/apply:numpy.dtype")
        {
            if (key == "args")
            {
                return parent_role == Role::ArrayDType ? Role::ArrayDtypeArgs
                                                       : Role::DtypeArgs;
            }
            if (key == "state")
            {
                return Role::DtypeStateTuple;
            }
        }
        return Role::Generic;
    }

    Role sequence_child_role(const yaml::Node& parent, Role parent_role, std::size_t index) const
    {
        if (parent_role == Role::CoefficientSets)
        {
            return Role::CoefficientArray;
        }
        if (parent_role == Role::IndSets)
        {
            return Role::IndSetEntry;
        }
        if (parent_role == Role::IndSetEntry)
        {
            return Role::DescriptorIndex;
        }
        if (parent_role == Role::NormalizerEntries)
        {
            return Role::NormalizerEntry;
        }
        if (parent_role == Role::SplineGrids)
        {
            return Role::SplineGridTerm;
        }
        if (parent_role == Role::SplineGridTerm)
        {
            return Role::SplineGridTuple;
        }
        if (parent_role == Role::ArrayArgs)
        {
            if (index == 0U)
            {
                return Role::ArrayName;
            }
            if (index == 1U)
            {
                return Role::ArrayPlaceholderTuple;
            }
            if (index == 2U)
            {
                return Role::ArrayPlaceholderBinary;
            }
        }
        if (parent_role == Role::ArrayStateTuple || parent_role == Role::ScaleArrayStateTuple)
        {
            if (index == 0U)
            {
                return Role::ArrayStateVersion;
            }
            if (index == 1U)
            {
                return parent_role == Role::ScaleArrayStateTuple ? Role::ScaleArrayShapeTuple
                                                                 : Role::ArrayShapeTuple;
            }
            if (index == 2U)
            {
                return Role::ArrayDType;
            }
            if (index == 3U)
            {
                return Role::ArrayOrder;
            }
            if (index == 4U)
            {
                return Role::ArrayDataBinary;
            }
        }
        if (scalar_tag(canonical_tag(parent)))
        {
            return index == 0U ? Role::ScalarDType : Role::ScalarDataBinary;
        }
        return Role::Generic;
    }

    void validate_mapping_schema(const yaml::Node& node,
                                 const std::string& path,
                                 const std::set<std::string>& allowed,
                                 const std::set<std::string>& required)
    {
        std::set<std::string> present;
        for (yaml::const_iterator it = node.begin(); it != node.end(); ++it)
        {
            const std::string key = validated_mapping_key(it->first, path);
            if (key == "<<")
            {
                throw LegacyYamlError("YAML_SYNTAX", path + ".<<", "YAML merge keys are not allowed");
            }
            if (!present.insert(key).second)
            {
                throw LegacyYamlError("YAML_SYNTAX", path + "." + key, "duplicate mapping key");
            }
            if (allowed.count(key) == 0)
            {
                throw LegacyYamlError("SCHEMA", path + "." + key, "unknown mapping key");
            }
        }
        for (const std::string& key : required)
        {
            if (present.count(key) == 0)
            {
                throw LegacyYamlError("SCHEMA", path + "." + key, "missing required mapping key");
            }
        }
    }

    void validate_tagged_node(const yaml::Node& node,
                              const std::string& tag,
                              const std::string& path,
                              bool root,
                              Role role)
    {
        typedef std::pair<std::set<std::string>, std::set<std::string>> KeySchema;
        static const std::map<std::string, KeySchema> schemas = {
            {"python/object:ciderpress.dft.xc_evaluator.MappedDFTKernel",
             KeySchema({"_add_basefunc", "_mul_basefunc", "feature_list", "fevals", "mode", "output_scale"},
                       {"_add_basefunc", "_mul_basefunc", "feature_list", "fevals", "mode"})},
            {"python/object:ciderpress.dft.xc_evaluator.SplineSetEvaluator",
             KeySchema({"coeff_sets", "const", "ind_sets", "nterms", "scale", "spline_grids"},
                       {"coeff_sets", "const", "ind_sets", "nterms", "scale", "spline_grids"})},
            {"python/object:ciderpress.dft.transform_data.FeatureList",
             KeySchema({"feat_list"}, {"feat_list"})},
            {"python/object:ciderpress.dft.transform_data.UMap",
             KeySchema({"i", "gamma", "_bounds"}, {"i", "gamma", "_bounds"})},
            {"python/object:ciderpress.dft.transform_data.VMap",
             KeySchema({"i", "gamma", "scale", "center", "_bounds"},
                       {"i", "gamma", "scale", "center", "_bounds"})},
            {"python/object:ciderpress.dft.transform_data.ZMap",
             KeySchema({"i", "gamma", "scale", "center", "n", "_bounds"},
                       {"i", "gamma", "scale", "center", "n"})},
            {"python/object:ciderpress.dft.settings.FeatureSettings",
             KeySchema({"sl_settings", "nldf_settings", "nlof_settings", "sdmx_settings", "hyb_settings", "normalizers"},
                       {"sl_settings", "nldf_settings", "nlof_settings", "sdmx_settings", "hyb_settings", "normalizers"})},
            {"python/object:ciderpress.dft.settings.EmptySettings", KeySchema({}, {})},
            {"python/object:ciderpress.dft.settings.SemilocalSettings",
             KeySchema({"level", "mode"}, {"level", "mode"})},
            {"python/object:ciderpress.dft.settings.NLDFSettingsVJ",
             KeySchema({"theta_params", "rho_mult", "feat_specs", "feat_params", "l0_feat_specs", "l1_feat_specs", "l1_feat_dots", "_sl_level"},
                       {"theta_params", "rho_mult", "feat_specs", "feat_params", "l0_feat_specs", "l1_feat_specs", "l1_feat_dots"})},
            {"python/object:ciderpress.dft.feat_normalizer.FeatNormalizerList",
             KeySchema({"_normalizers", "cutoff", "slmode"}, {"_normalizers", "cutoff", "slmode"})},
            {"python/object:ciderpress.dft.feat_normalizer.ConstantNormalizer",
             KeySchema({"const"}, {"const"})},
            {"python/object/apply:numpy.dtype", KeySchema({"args", "state"}, {"args", "state"})},
            {"python/object/apply:numpy.core.multiarray._reconstruct",
             KeySchema({"args", "state"}, {"args", "state"})},
            {"python/object/apply:numpy._core.multiarray._reconstruct",
             KeySchema({"args", "state"}, {"args", "state"})}};

        if (tag == "python/object:ciderpress.dft.xc_evaluator.MappedXC")
        {
            if (!root)
            {
                throw LegacyYamlError("UNSUPPORTED_ROOT", path, "MappedXC is allowed only at the root");
            }
            return;
        }
        if (reconstruct_tag(tag) && role != Role::CoefficientArray && role != Role::ScaleValue)
        {
            throw LegacyYamlError("UNSUPPORTED_NUMPY_TAG", path, "ndarray reconstruction tag is invalid in this role");
        }
        if (tag == "python/object/apply:numpy.dtype" && role != Role::ArrayDType
            && role != Role::ScalarDType)
        {
            throw LegacyYamlError("UNSUPPORTED_NUMPY_TAG", path, "NumPy dtype tag is invalid in this role");
        }
        const std::map<std::string, KeySchema>::const_iterator schema = schemas.find(tag);
        if (schema != schemas.end())
        {
            bool allowed_role = true;
            if (tag == "python/object:ciderpress.dft.xc_evaluator.MappedDFTKernel")
            {
                allowed_role = indexed_child_path(path, "$.kernels");
            }
            else if (tag == "python/object:ciderpress.dft.xc_evaluator.SplineSetEvaluator")
            {
                allowed_role = indexed_child_path(path, "$.kernels[0].fevals");
            }
            else if (tag == "python/object:ciderpress.dft.transform_data.FeatureList")
            {
                allowed_role = path == "$.kernels[0].feature_list";
            }
            else if (tag == "python/object:ciderpress.dft.transform_data.UMap"
                     || tag == "python/object:ciderpress.dft.transform_data.VMap"
                     || tag == "python/object:ciderpress.dft.transform_data.ZMap")
            {
                allowed_role = indexed_child_path(path, "$.kernels[0].feature_list.feat_list");
            }
            else if (tag == "python/object:ciderpress.dft.settings.FeatureSettings")
            {
                allowed_role = path == "$.settings";
            }
            else if (tag == "python/object:ciderpress.dft.settings.EmptySettings")
            {
                allowed_role = path == "$.settings.nldf_settings" || path == "$.settings.nlof_settings"
                               || path == "$.settings.sdmx_settings" || path == "$.settings.hyb_settings";
            }
            else if (tag == "python/object:ciderpress.dft.settings.SemilocalSettings")
            {
                allowed_role = path == "$.settings.sl_settings";
            }
            else if (tag == "python/object:ciderpress.dft.settings.NLDFSettingsVJ")
            {
                allowed_role = path == "$.settings.nldf_settings";
            }
            else if (tag == "python/object:ciderpress.dft.feat_normalizer.FeatNormalizerList")
            {
                allowed_role = path == "$.settings.normalizers";
            }
            else if (tag == "python/object:ciderpress.dft.feat_normalizer.ConstantNormalizer")
            {
                allowed_role = indexed_child_path(path, "$.settings.normalizers._normalizers");
            }
            if (!allowed_role)
            {
                if (starts_with(tag, "python/object:ciderpress.dft.xc_evaluator."))
                {
                    throw LegacyYamlError("UNSUPPORTED_EVALUATOR", path, "evaluator tag is invalid in this role");
                }
                if (starts_with(tag, "python/object:ciderpress.dft.feat_normalizer."))
                {
                    throw LegacyYamlError("UNSUPPORTED_NORMALIZER", path, "normalizer tag is invalid in this role");
                }
                throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", path, "descriptor tag is invalid in this role");
            }
            if (!node.IsMap())
            {
                throw LegacyYamlError(tag.find("numpy.") != std::string::npos ? "MALFORMED_ARRAY" : "SCHEMA",
                                      path,
                                      "tagged object must be a mapping");
            }
            validate_mapping_schema(node, path, schema->second.first, schema->second.second);
            return;
        }
        if (scalar_tag(tag))
        {
            if (role != Role::DescriptorIndex && role != Role::NormalizerConstant)
            {
                throw LegacyYamlError("UNSUPPORTED_NUMPY_TAG", path, "NumPy scalar tag is invalid in this role");
            }
            if (!node.IsSequence())
            {
                throw LegacyYamlError("MALFORMED_ARRAY", path, "NumPy scalar must be a sequence");
            }
            return;
        }
        if (tag == "python/tuple")
        {
            const bool allowed_role = role == Role::BoundsTuple || role == Role::IndSetEntry
                                      || role == Role::SplineGridTerm
                                      || role == Role::SplineGridTuple
                                      || role == Role::ArrayPlaceholderTuple
                                      || role == Role::ArrayStateTuple
                                      || role == Role::ScaleArrayStateTuple
                                      || role == Role::ArrayShapeTuple
                                      || role == Role::ScaleArrayShapeTuple
                                      || role == Role::DtypeStateTuple;
            if (!allowed_role)
            {
                throw LegacyYamlError("UNKNOWN_TAG", path, "python tuple tag is invalid in this role");
            }
            if (!node.IsSequence())
            {
                throw LegacyYamlError("SCHEMA", path, "python tuple must be a sequence");
            }
            return;
        }
        if (tag == "python/name:numpy.ndarray")
        {
            if (role != Role::ArrayName || !node.IsScalar() || !source_is_empty_scalar(node))
            {
                throw LegacyYamlError("UNSUPPORTED_NUMPY_TAG", path, "numpy.ndarray name is invalid in this role");
            }
            return;
        }
        const std::string lda = "python/name:ciderpress.dft.baselines.lda_x";
        const std::string chachiyo = "python/name:ciderpress.dft.baselines.gga_x_chachiyo";
        const std::string pbe = "python/name:ciderpress.dft.baselines.gga_x_pbe";
        if (tag == lda || tag == chachiyo || tag == pbe)
        {
            const bool allowed = (tag == lda && path == "$.kernels[0]._mul_basefunc")
                                 || ((tag == chachiyo || tag == pbe)
                                     && path == "$.kernels[0]._add_basefunc");
            if (!allowed || !node.IsScalar() || !source_is_empty_scalar(node))
            {
                throw LegacyYamlError("UNSUPPORTED_NAMED_FUNCTION", path, "baseline function tag is invalid in this role");
            }
            return;
        }
        throw_tag_error(tag, path, root);
    }

    static bool source_is_empty_scalar(const yaml::Node& node)
    {
        return node.Scalar().empty();
    }

    static bool plain_scalar_equals(const yaml::Node& node, const std::string& value)
    {
        const std::string tag = canonical_tag(node);
        return node.IsScalar() && (tag == "?" || tag == "!" || tag == "str")
               && node.Scalar() == value;
    }

    static bool plain_boolean_equals(const yaml::Node& node, bool value)
    {
        const std::string tag = canonical_tag(node);
        return node.IsScalar() && (tag == "?" || tag == "bool")
               && node.Scalar() == (value ? "true" : "false");
    }

    static bool plain_integer_value(const yaml::Node& node, std::int64_t& value)
    {
        const std::string tag = canonical_tag(node);
        if (!node.IsScalar() || (tag != "?" && tag != "int"))
        {
            return false;
        }
        const std::string text = node.Scalar();
        if (text.empty())
        {
            return false;
        }
        std::size_t start = (text[0] == '+' || text[0] == '-') ? 1U : 0U;
        if (start == text.size())
        {
            return false;
        }
        for (std::size_t i = start; i < text.size(); ++i)
        {
            if (!std::isdigit(static_cast<unsigned char>(text[i])))
            {
                return false;
            }
        }
        errno = 0;
        char* end = nullptr;
        const long long parsed = std::strtoll(text.c_str(), &end, 10);
        if (errno == ERANGE || end == nullptr || *end != '\0')
        {
            return false;
        }
        value = static_cast<std::int64_t>(parsed);
        return true;
    }

    void validate_role_schema(const yaml::Node& node,
                              const std::string& tag,
                              const std::string& path,
                              Role role)
    {
        const auto malformed = [&path](const std::string& detail) {
            throw LegacyYamlError("MALFORMED_ARRAY", path, detail);
        };
        if (role == Role::ArrayArgs && (!node.IsSequence() || node.size() != 3U || !is_standard_tag(tag)))
        {
            malformed("ndarray args must be an ordinary three-element sequence");
        }
        else if (role == Role::ArrayName
                 && (tag != "python/name:numpy.ndarray" || !node.IsScalar()
                     || !node.Scalar().empty()))
        {
            malformed("invalid numpy.ndarray reconstruction argument");
        }
        else if (role == Role::ArrayPlaceholderTuple)
        {
            std::int64_t value = -1;
            if (tag != "python/tuple" || !node.IsSequence() || node.size() != 1U
                || !plain_integer_value(node[0], value) || value != 0)
            {
                malformed("invalid ndarray placeholder shape tuple");
            }
        }
        else if (role == Role::ArrayPlaceholderBinary || role == Role::ArrayDataBinary
                 || role == Role::ScalarDataBinary)
        {
            if (tag != "binary" || !node.IsScalar())
            {
                malformed("expected YAML binary payload");
            }
        }
        else if ((role == Role::ArrayStateTuple || role == Role::ScaleArrayStateTuple)
                 && (tag != "python/tuple" || !node.IsSequence() || node.size() != 5U))
        {
            malformed("ndarray state must be a five-element python tuple");
        }
        else if (role == Role::ArrayStateVersion)
        {
            std::int64_t value = 0;
            if (!plain_integer_value(node, value) || value != 1)
            {
                malformed("unsupported ndarray state version");
            }
        }
        else if (role == Role::ArrayShapeTuple || role == Role::ScaleArrayShapeTuple)
        {
            if (tag != "python/tuple" || !node.IsSequence() || node.size() == 0U
                || node.size() > limits_.max_array_rank)
            {
                malformed("invalid ndarray shape tuple");
            }
            if (role == Role::ScaleArrayShapeTuple && node.size() != 1U)
            {
                malformed("spline scale array must be one-dimensional");
            }
            std::size_t elements = 1U;
            for (std::size_t dimension = 0; dimension < node.size(); ++dimension)
            {
                std::int64_t extent = 0;
                if (!plain_integer_value(node[dimension], extent) || extent <= 0)
                {
                    throw LegacyYamlError("MALFORMED_ARRAY",
                                          path + "[" + std::to_string(dimension) + "]",
                                          "array extent must be a positive integer");
                }
                const std::size_t size = static_cast<std::size_t>(extent);
                if (size > limits_.max_array_elements
                    || elements > limits_.max_array_elements / size)
                {
                    malformed("array element count overflow or limit exceeded");
                }
                elements *= size;
            }
        }
        else if ((role == Role::ArrayDType || role == Role::ScalarDType)
                 && tag != "python/object/apply:numpy.dtype")
        {
            malformed("expected NumPy dtype object");
        }
        else if (role == Role::ArrayOrder && !plain_boolean_equals(node, false))
        {
            malformed("Fortran-order arrays are not supported");
        }
        else if (role == Role::DtypeArgs || role == Role::ArrayDtypeArgs)
        {
            if (!node.IsSequence() || node.size() != 3U || !is_standard_tag(tag))
            {
                malformed("invalid NumPy dtype arguments");
            }
            if (!plain_scalar_equals(node[0], "f8") && !plain_scalar_equals(node[0], "i8"))
            {
                throw LegacyYamlError("MALFORMED_ARRAY", path + "[0]", "unsupported NumPy dtype");
            }
            if (role == Role::ArrayDtypeArgs && !plain_scalar_equals(node[0], "f8"))
            {
                throw LegacyYamlError("MALFORMED_ARRAY", path + "[0]", "ndarrays require float64 dtype");
            }
            if (!plain_boolean_equals(node[1], false) || !plain_boolean_equals(node[2], true))
            {
                malformed("invalid NumPy dtype flags");
            }
        }
        else if (role == Role::DtypeStateTuple)
        {
            if (tag != "python/tuple" || !node.IsSequence() || node.size() != 8U)
            {
                malformed("invalid NumPy dtype state tuple");
            }
            std::int64_t version = 0;
            std::int64_t field_five = 0;
            std::int64_t field_six = 0;
            std::int64_t field_seven = 0;
            if (!plain_integer_value(node[0], version) || version != 3
                || !plain_scalar_equals(node[1], "<") || !node[2].IsNull()
                || !node[3].IsNull() || !node[4].IsNull()
                || !plain_integer_value(node[5], field_five) || field_five != -1
                || !plain_integer_value(node[6], field_six) || field_six != -1
                || !plain_integer_value(node[7], field_seven) || field_seven != 0)
            {
                malformed("unsupported NumPy dtype state");
            }
        }

        if (scalar_tag(tag))
        {
            if (!node.IsSequence() || node.size() != 2U)
            {
                malformed("NumPy scalar must contain dtype and data");
            }
            const yaml::Node dtype = node[0];
            const yaml::Node args = dtype.IsMap() ? dtype["args"] : yaml::Node();
            const std::string expected
                = role == Role::DescriptorIndex ? std::string("i8") : std::string("f8");
            if (!args.IsSequence() || args.size() == 0U || !plain_scalar_equals(args[0], expected))
            {
                malformed("NumPy scalar dtype is not allowed in this role");
            }
        }
    }

    bool already_active(const yaml::Node& node) const
    {
        for (const yaml::Node& active : active_)
        {
            if (node.is(active))
            {
                return true;
            }
        }
        return false;
    }

    void account_expanded_bytes(std::size_t bytes, const std::string& path)
    {
        if (expanded_scalar_bytes_ > limits_.max_yaml_expanded_bytes
            || bytes > limits_.max_yaml_expanded_bytes - expanded_scalar_bytes_)
        {
            throw LegacyYamlError("YAML_SYNTAX", path, "YAML scalar/alias expansion byte limit exceeded");
        }
        expanded_scalar_bytes_ += bytes;
    }

    void validate_standard_node(const yaml::Node& node,
                                const std::string& tag,
                                const std::string& path)
    {
        const bool valid = tag == "" || tag == "?" || tag == "!"
                           || (tag == "map" && node.IsMap()) || (tag == "seq" && node.IsSequence())
                           || ((tag == "str" || tag == "int" || tag == "float" || tag == "bool"
                                || tag == "binary")
                               && node.IsScalar())
                           || (tag == "null" && (node.IsNull() || node.IsScalar()));
        if (!valid)
        {
            throw LegacyYamlError("SCHEMA", path, "ordinary YAML tag has the wrong node type");
        }
    }

    void walk(const yaml::Node& node,
              const std::string& path,
              bool root,
              std::size_t depth,
              Role role)
    {
        if (++visited_ > limits_.max_yaml_nodes)
        {
            throw LegacyYamlError("YAML_SYNTAX", path, "YAML node/alias expansion limit exceeded");
        }
        if (depth > limits_.max_yaml_depth)
        {
            throw LegacyYamlError("YAML_SYNTAX", path, "YAML nesting-depth limit exceeded");
        }

        const std::string tag = canonical_tag(node);
        const std::string mapped_xc = "python/object:ciderpress.dft.xc_evaluator.MappedXC";
        if (root)
        {
            if (tag != mapped_xc)
            {
                if (is_standard_tag(tag))
                {
                    throw LegacyYamlError("UNSUPPORTED_ROOT", path, "root must be MappedXC");
                }
                if (whitelisted_cider_object_tag(tag))
                {
                    throw LegacyYamlError("UNSUPPORTED_ROOT", path, "whitelisted object is invalid at root");
                }
                throw_tag_error(tag, path, true);
            }
            if (!node.IsMap())
            {
                throw LegacyYamlError("SCHEMA", path, "MappedXC root must be a mapping");
            }
            validate_mapping_schema(
                node,
                path,
                {"kernels",
                 "settings",
                 "libxc_baseline",
                 "feature_density_policy",
                 "feature_density_policy_source",
                 "ml_exchange_scale",
                 "ml_exchange_label_convention",
                 "ml_exchange_label_hybrid_scaled",
                 "ml_exchange_label_scale",
                 "ml_exchange_label_source_scales"},
                {"kernels", "settings", "libxc_baseline"});
        }
        if (!is_standard_tag(tag))
        {
            validate_tagged_node(node, tag, path, root, role);
        }
        else
        {
            validate_standard_node(node, tag, path);
        }
        validate_role_schema(node, tag, path, role);

        if (!node.IsSequence() && !node.IsMap())
        {
            if (node.IsScalar())
            {
                account_expanded_bytes(node.Scalar().size(), path);
            }
            return;
        }
        if (already_active(node))
        {
            throw LegacyYamlError("YAML_SYNTAX", path, "recursive YAML alias is not allowed");
        }
        active_.push_back(node);
        if (node.IsSequence())
        {
            for (std::size_t i = 0; i < node.size(); ++i)
            {
                walk(node[i],
                     path + "[" + std::to_string(i) + "]",
                     false,
                     depth + 1,
                     sequence_child_role(node, role, i));
            }
        }
        else
        {
            std::set<std::string> keys;
            for (yaml::const_iterator it = node.begin(); it != node.end(); ++it)
            {
                const std::string key = validated_mapping_key(it->first, path);
                account_expanded_bytes(key.size(), path + "." + key);
                if (key == "<<")
                {
                    throw LegacyYamlError("YAML_SYNTAX", path + ".<<", "YAML merge keys are not allowed");
                }
                if (!keys.insert(key).second)
                {
                    throw LegacyYamlError("YAML_SYNTAX", path + "." + key, "duplicate mapping key");
                }
                walk(it->second,
                     path + "." + key,
                     false,
                     depth + 1,
                     mapping_child_role(node, key, role));
            }
        }
        active_.pop_back();
    }

    const LoaderLimits& limits_;
    std::size_t visited_;
    std::size_t expanded_scalar_bytes_;
    std::vector<yaml::Node> active_;
};

bool is_null_scalar(const std::string& value)
{
    return value.empty() || value == "~" || value == "null" || value == "Null" || value == "NULL";
}

bool parse_integer(const std::string& value, std::int64_t& result)
{
    if (value.empty())
    {
        return false;
    }
    std::size_t start = (value[0] == '+' || value[0] == '-') ? 1U : 0U;
    if (start == value.size())
    {
        return false;
    }
    for (std::size_t i = start; i < value.size(); ++i)
    {
        if (!std::isdigit(static_cast<unsigned char>(value[i])))
        {
            return false;
        }
    }
    errno = 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (errno == ERANGE || end == nullptr || *end != '\0')
    {
        return false;
    }
    result = static_cast<std::int64_t>(parsed);
    return true;
}

bool parse_number(const std::string& value, double& result)
{
    if (value.find('.') == std::string::npos && value.find('e') == std::string::npos
        && value.find('E') == std::string::npos)
    {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    result = std::strtod(value.c_str(), &end);
    return end != value.c_str() && end != nullptr && *end == '\0' && errno != ERANGE
           && std::isfinite(result);
}

std::vector<unsigned char> decode_base64(const yaml::Node& source,
                                         const std::string& path,
                                         std::size_t byte_limit)
{
    if (!source.IsScalar() || canonical_tag(source) != "binary")
    {
        throw LegacyYamlError("MALFORMED_ARRAY", path, "expected YAML binary payload");
    }
    std::string encoded;
    for (char c : source.Scalar())
    {
        if (std::isspace(static_cast<unsigned char>(c)))
        {
            continue;
        }
        encoded.push_back(c);
    }
    if (encoded.empty() || encoded.size() % 4U != 0U)
    {
        throw LegacyYamlError("MALFORMED_ARRAY", path, "invalid base64 length");
    }
    const std::size_t padding = (encoded.back() == '=') + (encoded.size() > 1 && encoded[encoded.size() - 2] == '=');
    const std::size_t decoded_size = (encoded.size() / 4U) * 3U - padding;
    if (decoded_size > byte_limit)
    {
        throw LegacyYamlError("MALFORMED_ARRAY", path, "binary payload exceeds configured byte limit");
    }
    std::vector<unsigned char> decoded;
    decoded.reserve(decoded_size);
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (std::size_t offset = 0; offset < encoded.size(); offset += 4U)
    {
        std::uint32_t bits = 0;
        int quartet_padding = 0;
        for (std::size_t j = 0; j < 4U; ++j)
        {
            const char c = encoded[offset + j];
            if (c == '=')
            {
                if (offset + 4U != encoded.size() || j < 2U)
                {
                    throw LegacyYamlError("MALFORMED_ARRAY", path, "invalid base64 padding");
                }
                ++quartet_padding;
                bits <<= 6U;
            }
            else
            {
                if (quartet_padding != 0)
                {
                    throw LegacyYamlError("MALFORMED_ARRAY", path, "invalid base64 padding");
                }
                const std::size_t value = alphabet.find(c);
                if (value == std::string::npos)
                {
                    throw LegacyYamlError("MALFORMED_ARRAY", path, "invalid base64 character");
                }
                bits = (bits << 6U) | static_cast<std::uint32_t>(value);
            }
        }
        decoded.push_back(static_cast<unsigned char>((bits >> 16U) & 0xffU));
        if (quartet_padding < 2)
        {
            decoded.push_back(static_cast<unsigned char>((bits >> 8U) & 0xffU));
        }
        if (quartet_padding == 0)
        {
            decoded.push_back(static_cast<unsigned char>(bits & 0xffU));
        }
    }
    if (decoded.size() != decoded_size)
    {
        throw LegacyYamlError("MALFORMED_ARRAY", path, "invalid base64 payload size");
    }
    return decoded;
}

bool scalar_is(const yaml::Node& node, const std::string& value)
{
    return node.IsScalar() && node.Scalar() == value;
}

std::int64_t required_integer(const yaml::Node& node,
                              const std::string& path,
                              const std::string& error_code)
{
    const std::string tag = canonical_tag(node);
    if (!node.IsScalar() || (tag != "?" && tag != "int"))
    {
        throw LegacyYamlError(error_code, path, "expected integer scalar");
    }
    std::int64_t value = 0;
    try
    {
        if (!parse_integer(node.Scalar(), value))
        {
            throw LegacyYamlError(error_code, path, "expected integer scalar");
        }
    }
    catch (const LegacyYamlError&)
    {
        throw LegacyYamlError(error_code, path, "integer is outside signed 64-bit range");
    }
    return value;
}

bool required_boolean(const yaml::Node& node, bool expected)
{
    const std::string tag = canonical_tag(node);
    return node.IsScalar() && (tag == "?" || tag == "bool")
           && node.Scalar() == (expected ? "true" : "false");
}

class Converter
{
  public:
    explicit Converter(const LoaderLimits& limits) : limits_(limits), total_array_bytes_(0)
    {
    }

    LegacyNode convert(const yaml::Node& source, const std::string& path)
    {
        const std::string tag = canonical_tag(source);
        if (tag == "python/object/apply:numpy.core.multiarray._reconstruct"
            || tag == "python/object/apply:numpy._core.multiarray._reconstruct")
        {
            return decode_array(source, path);
        }
        if (tag == "python/object/apply:numpy.core.multiarray.scalar"
            || tag == "python/object/apply:numpy._core.multiarray.scalar")
        {
            return decode_scalar(source, path);
        }
        LegacyNode result;
        if (!is_standard_tag(tag))
        {
            result.tag = tag;
        }
        if (source.IsNull())
        {
            result.kind = LegacyNode::Kind::Null;
            return result;
        }
        if (source.IsSequence())
        {
            result.kind = LegacyNode::Kind::Sequence;
            result.sequence_items.reserve(source.size());
            for (std::size_t i = 0; i < source.size(); ++i)
            {
                result.sequence_items.push_back(convert(source[i], path + "[" + std::to_string(i) + "]"));
            }
            return result;
        }
        if (source.IsMap())
        {
            result.kind = LegacyNode::Kind::Mapping;
            for (yaml::const_iterator it = source.begin(); it != source.end(); ++it)
            {
                const std::string key = it->first.Scalar();
                result.mapping_items.emplace(key, convert(it->second, path + "." + key));
            }
            return result;
        }
        if (!source.IsScalar())
        {
            throw LegacyYamlError("SCHEMA", path, "undefined YAML node");
        }
        if (tag == "binary")
        {
            result.kind = LegacyNode::Kind::Binary;
            result.binary_value = decode_base64(source, path, limits_.max_array_bytes);
            account_array_bytes(result.binary_value.size(), path);
            return result;
        }
        const std::string value = source.Scalar();
        if (tag == "null" || (tag == "?" && is_null_scalar(value)))
        {
            result.kind = LegacyNode::Kind::Null;
        }
        else if (tag == "bool" || (tag == "?" && (value == "true" || value == "false")))
        {
            if (value != "true" && value != "false")
            {
                throw LegacyYamlError("SCHEMA", path, "Boolean must be true or false");
            }
            result.kind = LegacyNode::Kind::Boolean;
            result.boolean_value = value == "true";
        }
        else
        {
            std::int64_t integer = 0;
            double number = 0.0;
            if ((tag == "int" || tag == "?") && parse_integer(value, integer))
            {
                result.kind = LegacyNode::Kind::Integer;
                result.integer_value = integer;
            }
            else if ((tag == "float" || tag == "?") && parse_number(value, number))
            {
                result.kind = LegacyNode::Kind::Number;
                result.number_value = number;
            }
            else
            {
                if (tag == "int" || tag == "float")
                {
                    throw LegacyYamlError("SCHEMA", path, "invalid or non-finite numeric scalar");
                }
                result.kind = LegacyNode::Kind::String;
                result.string_value = value;
            }
        }
        return result;
    }

  private:
    LegacyDType decode_dtype(const yaml::Node& node, const std::string& path)
    {
        if (canonical_tag(node) != "python/object/apply:numpy.dtype" || !node.IsMap())
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path, "expected supported NumPy dtype object");
        }
        const yaml::Node args = node["args"];
        const yaml::Node state = node["state"];
        if (!args.IsSequence() || args.size() != 3U || !state.IsSequence() || state.size() != 8U
            || canonical_tag(state) != "python/tuple")
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path, "invalid NumPy dtype reconstruction shape");
        }
        if (!scalar_is(args[0], "f8") && !scalar_is(args[0], "i8"))
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".args[0]", "unsupported NumPy dtype");
        }
        if (!required_boolean(args[1], false) || !required_boolean(args[2], true))
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".args", "invalid NumPy dtype flags");
        }
        if (required_integer(state[0], path + ".state[0]", "MALFORMED_ARRAY") != 3
            || !scalar_is(state[1], "<") || !state[2].IsNull() || !state[3].IsNull()
            || !state[4].IsNull()
            || required_integer(state[5], path + ".state[5]", "MALFORMED_ARRAY") != -1
            || required_integer(state[6], path + ".state[6]", "MALFORMED_ARRAY") != -1
            || required_integer(state[7], path + ".state[7]", "MALFORMED_ARRAY") != 0)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".state", "unsupported NumPy dtype state");
        }
        return args[0].Scalar() == "f8" ? LegacyDType::Float64 : LegacyDType::Int64;
    }

    void account_array_bytes(std::size_t bytes, const std::string& path)
    {
        if (bytes > limits_.max_array_bytes || total_array_bytes_ > limits_.max_total_array_bytes
            || bytes > limits_.max_total_array_bytes - total_array_bytes_)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path, "array exceeds configured byte budget");
        }
        total_array_bytes_ += bytes;
    }

    static std::uint64_t little_endian_word(const std::vector<unsigned char>& data, std::size_t offset)
    {
        std::uint64_t word = 0;
        for (std::size_t byte = 0; byte < 8U; ++byte)
        {
            word |= static_cast<std::uint64_t>(data[offset + byte]) << (8U * byte);
        }
        return word;
    }

    void decode_values(LegacyNumpyValue& value,
                       const std::vector<unsigned char>& data,
                       const std::string& path)
    {
        if (value.dtype == LegacyDType::Float64)
        {
            if (sizeof(double) != 8U || !std::numeric_limits<double>::is_iec559)
            {
                throw LegacyYamlError("MALFORMED_ARRAY", path, "host does not provide IEEE-754 binary64");
            }
            value.float64_values.reserve(data.size() / 8U);
            for (std::size_t offset = 0; offset < data.size(); offset += 8U)
            {
                const std::uint64_t word = little_endian_word(data, offset);
                double decoded = 0.0;
                std::memcpy(&decoded, &word, sizeof(decoded));
                if (!std::isfinite(decoded))
                {
                    throw LegacyYamlError("MALFORMED_ARRAY", path, "non-finite float64 payload value");
                }
                value.float64_values.push_back(decoded);
            }
        }
        else
        {
            value.int64_values.reserve(data.size() / 8U);
            for (std::size_t offset = 0; offset < data.size(); offset += 8U)
            {
                const std::uint64_t word = little_endian_word(data, offset);
                std::int64_t decoded = 0;
                if (word <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                {
                    decoded = static_cast<std::int64_t>(word);
                }
                else if (word == (std::uint64_t(1) << 63U))
                {
                    decoded = std::numeric_limits<std::int64_t>::min();
                }
                else
                {
                    decoded = -static_cast<std::int64_t>((~word) + 1U);
                }
                value.int64_values.push_back(decoded);
            }
        }
    }

    LegacyNode decode_array(const yaml::Node& source, const std::string& path)
    {
        const yaml::Node args = source["args"];
        const yaml::Node state = source["state"];
        if (!args.IsSequence() || args.size() != 3U || !state.IsSequence() || state.size() != 5U
            || canonical_tag(state) != "python/tuple")
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path, "invalid ndarray reconstruction shape");
        }
        if (canonical_tag(args[0]) != "python/name:numpy.ndarray" || !args[0].IsScalar()
            || !args[0].Scalar().empty() || !args[1].IsSequence() || args[1].size() != 1U
            || canonical_tag(args[1]) != "python/tuple"
            || required_integer(args[1][0], path + ".args[1][0]", "MALFORMED_ARRAY") != 0)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".args", "invalid ndarray placeholder arguments");
        }
        const std::vector<unsigned char> placeholder
            = decode_base64(args[2], path + ".args[2]", 1U);
        if (placeholder.size() != 1U || placeholder[0] != 0x62U)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".args[2]", "invalid ndarray placeholder bytes");
        }
        if (required_integer(state[0], path + ".state[0]", "MALFORMED_ARRAY") != 1)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".state[0]", "unsupported ndarray state version");
        }
        const yaml::Node shape = state[1];
        if (!shape.IsSequence() || canonical_tag(shape) != "python/tuple" || shape.size() == 0U
            || shape.size() > limits_.max_array_rank)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".state[1]", "invalid ndarray rank");
        }
        LegacyNode result;
        result.kind = LegacyNode::Kind::Numpy;
        result.tag = "numpy.ndarray";
        result.numpy_value.scalar = false;
        std::size_t elements = 1U;
        for (std::size_t dimension = 0; dimension < shape.size(); ++dimension)
        {
            const std::int64_t extent
                = required_integer(shape[dimension],
                                   path + ".state[1][" + std::to_string(dimension) + "]",
                                   "MALFORMED_ARRAY");
            if (extent <= 0)
            {
                throw LegacyYamlError("MALFORMED_ARRAY",
                                      path + ".state[1][" + std::to_string(dimension) + "]",
                                      "array extent must be positive");
            }
            const std::size_t size = static_cast<std::size_t>(extent);
            if (size > limits_.max_array_elements || elements > limits_.max_array_elements / size)
            {
                throw LegacyYamlError("MALFORMED_ARRAY", path + ".state[1]", "array element count overflow or limit exceeded");
            }
            elements *= size;
            result.numpy_value.shape.push_back(size);
        }
        result.numpy_value.dtype = decode_dtype(state[2], path + ".state[2]");
        if (result.numpy_value.dtype != LegacyDType::Float64)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".state[2]", "ndarrays require float64 dtype");
        }
        if (!required_boolean(state[3], false))
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".state[3]", "Fortran-order arrays are not supported");
        }
        if (elements > std::numeric_limits<std::size_t>::max() / 8U)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".state[1]", "array byte count overflow");
        }
        const std::size_t bytes = elements * 8U;
        account_array_bytes(bytes, path);
        const std::vector<unsigned char> data
            = decode_base64(state[4], path + ".state[4]", limits_.max_array_bytes);
        if (data.size() != bytes)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + ".state[4]", "decoded byte count does not match shape");
        }
        decode_values(result.numpy_value, data, path + ".state[4]");
        return result;
    }

    LegacyNode decode_scalar(const yaml::Node& source, const std::string& path)
    {
        if (!source.IsSequence() || source.size() != 2U)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path, "NumPy scalar must contain dtype and data");
        }
        LegacyNode result;
        result.kind = LegacyNode::Kind::Numpy;
        result.tag = "numpy.scalar";
        result.numpy_value.scalar = true;
        result.numpy_value.dtype = decode_dtype(source[0], path + "[0]");
        account_array_bytes(8U, path);
        const std::vector<unsigned char> data
            = decode_base64(source[1], path + "[1]", 8U);
        if (data.size() != 8U)
        {
            throw LegacyYamlError("MALFORMED_ARRAY", path + "[1]", "NumPy scalar payload must contain eight bytes");
        }
        decode_values(result.numpy_value, data, path + "[1]");
        return result;
    }

    const LoaderLimits& limits_;
    std::size_t total_array_bytes_;
};

} // namespace

LoaderLimits::LoaderLimits()
    : max_file_bytes(64U * 1024U * 1024U),
      max_yaml_nodes(1000000U),
      max_yaml_expanded_bytes(128U * 1024U * 1024U),
      max_yaml_depth(128U),
      max_array_rank(3U),
      max_array_elements(10000000U),
      max_array_bytes(64U * 1024U * 1024U),
      max_total_array_bytes(128U * 1024U * 1024U)
{
}

LegacyNumpyValue::LegacyNumpyValue() : dtype(LegacyDType::Float64), scalar(false)
{
}

LegacyNode::LegacyNode()
    : kind(Kind::Null), boolean_value(false), integer_value(0), number_value(0.0)
{
}

LegacyYamlError::LegacyYamlError(const std::string& code,
                                 const std::string& path,
                                 const std::string& detail)
    : std::runtime_error("CIDER legacy YAML error [" + code + "] at " + path + ": " + detail),
      code_(code),
      path_(path),
      detail_(detail)
{
}

const std::string& LegacyYamlError::code() const
{
    return code_;
}

const std::string& LegacyYamlError::path() const
{
    return path_;
}

const std::string& LegacyYamlError::detail() const
{
    return detail_;
}

LegacyDocument ModelLoader::read_legacy_document(const std::string& path, const LoaderLimits& limits)
{
    const std::string suffixes[] = {".joblib", ".pkl", ".pickle"};
    for (const std::string& suffix : suffixes)
    {
        if (ends_with_case_insensitive(path, suffix))
        {
            throw LegacyYamlError("JOBLIB_UNSUPPORTED", "<file>", "pickle/joblib input is not supported");
        }
    }
    const std::string text = read_input(path, limits);
    std::vector<yaml::Node> documents;
    try
    {
        documents = yaml::LoadAll(normalize_binary_blocks(text));
    }
    catch (const yaml::Exception& error)
    {
        throw LegacyYamlError("YAML_SYNTAX", "<file>", error.what());
    }
    if (documents.size() != 1U || !documents[0].IsDefined())
    {
        throw LegacyYamlError("YAML_SYNTAX", "<file>", "document must contain exactly one root");
    }
    Validator(limits).validate(documents[0]);
    LegacyDocument document;
    document.root = Converter(limits).convert(documents[0], "$");
    return document;
}

namespace
{

const LegacyNode& member(const LegacyNode& node,
                         const std::string& key,
                         const std::string& path)
{
    const std::map<std::string, LegacyNode>::const_iterator item = node.mapping_items.find(key);
    if (node.kind != LegacyNode::Kind::Mapping || item == node.mapping_items.end())
    {
        throw LegacyYamlError("SCHEMA", path + "." + key, "required field is missing");
    }
    return item->second;
}

const LegacyNode* optional_member(const LegacyNode& node, const std::string& key)
{
    const std::map<std::string, LegacyNode>::const_iterator item = node.mapping_items.find(key);
    return item == node.mapping_items.end() ? nullptr : &item->second;
}

void require_tag(const LegacyNode& node,
                 const std::string& expected,
                 const std::string& path,
                 const std::string& code = "SCHEMA")
{
    if (node.tag != expected)
    {
        throw LegacyYamlError(code, path, "unexpected typed object");
    }
}

const std::vector<LegacyNode>& sequence(const LegacyNode& node,
                                        const std::string& path,
                                        const std::string& code = "SCHEMA")
{
    if (node.kind != LegacyNode::Kind::Sequence)
    {
        throw LegacyYamlError(code, path, "expected sequence");
    }
    return node.sequence_items;
}

std::string string_value(const LegacyNode& node,
                         const std::string& path,
                         const std::string& code = "SCHEMA")
{
    if (node.kind != LegacyNode::Kind::String)
    {
        throw LegacyYamlError(code, path, "expected string");
    }
    return node.string_value;
}

double number_value(const LegacyNode& node,
                    const std::string& path,
                    bool allow_numpy_scalar = false)
{
    double value = 0.0;
    if (node.kind == LegacyNode::Kind::Integer)
    {
        value = static_cast<double>(node.integer_value);
    }
    else if (node.kind == LegacyNode::Kind::Number)
    {
        value = node.number_value;
    }
    else if (allow_numpy_scalar && node.kind == LegacyNode::Kind::Numpy
             && node.numpy_value.scalar && node.numpy_value.dtype == LegacyDType::Float64
             && node.numpy_value.float64_values.size() == 1U)
    {
        value = node.numpy_value.float64_values[0];
    }
    else
    {
        throw LegacyYamlError("SCHEMA", path, "expected finite number");
    }
    if (!std::isfinite(value))
    {
        throw LegacyYamlError("SCHEMA", path, "number must be finite");
    }
    return value;
}

std::int64_t integer_value(const LegacyNode& node,
                           const std::string& path,
                           bool allow_numpy_scalar = false)
{
    if (node.kind == LegacyNode::Kind::Integer)
    {
        return node.integer_value;
    }
    if (allow_numpy_scalar && node.kind == LegacyNode::Kind::Numpy && node.numpy_value.scalar
        && node.numpy_value.dtype == LegacyDType::Int64
        && node.numpy_value.int64_values.size() == 1U)
    {
        return node.numpy_value.int64_values[0];
    }
    throw LegacyYamlError("SCHEMA", path, "expected integer");
}

bool boolean_value(const LegacyNode& node, const std::string& path)
{
    if (node.kind != LegacyNode::Kind::Boolean)
    {
        throw LegacyYamlError("SCHEMA", path, "expected Boolean");
    }
    return node.boolean_value;
}

ParameterTriple parameter_triple(const LegacyNode& node, const std::string& path)
{
    const std::vector<LegacyNode>& values = sequence(node, path, "UNSUPPORTED_DESCRIPTOR");
    if (values.size() != 3U)
    {
        throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", path, "parameter tuple must have three values");
    }
    const double a0 = number_value(values[0], path + "[0]");
    const double gradient = number_value(values[1], path + "[1]");
    const double tau = number_value(values[2], path + "[2]");
    if (!(a0 > 0.0) || gradient < 0.0 || tau < 0.0)
    {
        throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", path, "unsupported exponent parameters");
    }
    return ParameterTriple(a0, gradient, tau);
}

std::pair<double, double> bounds(const LegacyNode& node, const std::string& path)
{
    const std::vector<LegacyNode>& values = sequence(node, path);
    if (values.size() != 2U)
    {
        throw LegacyYamlError("SCHEMA", path, "bounds must contain two values");
    }
    const double lower = number_value(values[0], path + "[0]");
    const double upper = number_value(values[1], path + "[1]");
    if (!(lower < upper))
    {
        throw LegacyYamlError("SCHEMA", path, "bounds must be strictly increasing");
    }
    return std::make_pair(lower, upper);
}

bool basename_contains_cider23x(const std::string& path)
{
    const std::size_t separator = path.find_last_of("/\\");
    std::string basename = separator == std::string::npos ? path : path.substr(separator + 1U);
    std::transform(basename.begin(), basename.end(), basename.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return basename.find("cider23x") != std::string::npos;
}

class NativeModel final : public Model
{
  public:
    NativeModel(const ModelMetadata& metadata,
                SemilocalMode semilocal_mode,
                std::unique_ptr<const NldfSettings> nldf_settings,
                double normalizer_cutoff,
                const std::vector<Normalizer>& normalizers,
                const std::vector<FeatureMap>& feature_maps,
                MultiplicativeBaseline multiplicative_baseline,
                AdditiveBaseline additive_baseline,
                double spline_constant,
                const std::vector<SplineTerm>& spline_terms)
        : metadata_(metadata),
          semilocal_mode_(semilocal_mode),
          nldf_settings_(std::move(nldf_settings)),
          normalizer_cutoff_(normalizer_cutoff),
          normalizers_(normalizers),
          feature_maps_(feature_maps),
          multiplicative_baseline_(multiplicative_baseline),
          additive_baseline_(additive_baseline),
          spline_constant_(spline_constant),
          spline_terms_(spline_terms)
    {
    }

    const ModelMetadata& metadata() const override { return metadata_; }
    SemilocalMode semilocal_mode() const override { return semilocal_mode_; }
    const NldfSettings* nldf_settings() const override { return nldf_settings_.get(); }
    double normalizer_cutoff() const override { return normalizer_cutoff_; }
    const std::vector<Normalizer>& normalizers() const override { return normalizers_; }
    const std::vector<FeatureMap>& feature_maps() const override { return feature_maps_; }
    MultiplicativeBaseline multiplicative_baseline() const override
    {
        return multiplicative_baseline_;
    }
    AdditiveBaseline additive_baseline() const override { return additive_baseline_; }
    double spline_constant() const override { return spline_constant_; }
    const std::vector<SplineTerm>& spline_terms() const override { return spline_terms_; }

  private:
    const ModelMetadata metadata_;
    const SemilocalMode semilocal_mode_;
    const std::unique_ptr<const NldfSettings> nldf_settings_;
    const double normalizer_cutoff_;
    const std::vector<Normalizer> normalizers_;
    const std::vector<FeatureMap> feature_maps_;
    const MultiplicativeBaseline multiplicative_baseline_;
    const AdditiveBaseline additive_baseline_;
    const double spline_constant_;
    const std::vector<SplineTerm> spline_terms_;
};

class TypedModelBuilder
{
  public:
    TypedModelBuilder(const LegacyDocument& document, const std::string& path)
        : root_(document.root), input_path_(path)
    {
    }

    std::unique_ptr<const Model> build() const
    {
        const LegacyNode& kernels_node = member(root_, "kernels", "$");
        const std::vector<LegacyNode>& kernels = sequence(kernels_node, "$.kernels");
        if (kernels.size() != 1U)
        {
            throw LegacyYamlError("UNSUPPORTED_EVALUATOR", "$.kernels", "exactly one mapped kernel is required");
        }
        const LegacyNode& kernel = kernels[0];
        require_tag(kernel,
                    "python/object:ciderpress.dft.xc_evaluator.MappedDFTKernel",
                    "$.kernels[0]",
                    "UNSUPPORTED_EVALUATOR");
        if (string_value(member(kernel, "mode", "$.kernels[0]"), "$.kernels[0].mode") != "SEP")
        {
            throw LegacyYamlError("SCHEMA", "$.kernels[0].mode", "only SEP mode is supported");
        }
        const std::vector<LegacyNode>& evaluators
            = sequence(member(kernel, "fevals", "$.kernels[0]"), "$.kernels[0].fevals");
        if (evaluators.size() != 1U)
        {
            throw LegacyYamlError("UNSUPPORTED_EVALUATOR", "$.kernels[0].fevals", "exactly one evaluator is required");
        }
        const LegacyNode& evaluator = evaluators[0];
        require_tag(evaluator,
                    "python/object:ciderpress.dft.xc_evaluator.SplineSetEvaluator",
                    "$.kernels[0].fevals[0]",
                    "UNSUPPORTED_EVALUATOR");

        const LegacyNode& baseline = member(root_, "libxc_baseline", "$");
        if (baseline.kind != LegacyNode::Kind::Null)
        {
            throw LegacyYamlError("SCHEMA", "$.libxc_baseline", "libxc_baseline must be null");
        }
        const MultiplicativeBaseline multiplicative = parse_multiplicative_baseline(kernel);
        const AdditiveBaseline additive = parse_additive_baseline(kernel);
        const double output_scale = parse_output_scale(kernel);

        const LegacyNode& settings = member(root_, "settings", "$");
        require_tag(settings,
                    "python/object:ciderpress.dft.settings.FeatureSettings",
                    "$.settings",
                    "UNSUPPORTED_DESCRIPTOR");
        require_empty_settings(settings, "nlof_settings");
        require_empty_settings(settings, "sdmx_settings");
        require_empty_settings(settings, "hyb_settings");

        SemilocalLevel semilocal_level;
        SemilocalMode semilocal_mode;
        std::size_t semilocal_count = 0U;
        parse_semilocal(settings, semilocal_level, semilocal_mode, semilocal_count);

        std::unique_ptr<const NldfSettings> nldf_settings
            = parse_nldf(settings, semilocal_level);
        const std::size_t nldf_count
            = nldf_settings.get() == nullptr ? 0U : nldf_settings->feature_count();
        const std::size_t raw_count = semilocal_count + nldf_count;

        double normalizer_cutoff = 0.0;
        const std::vector<Normalizer> normalizers
            = parse_normalizers(settings, semilocal_mode, raw_count, normalizer_cutoff);
        const std::vector<FeatureMap> feature_maps = parse_feature_maps(kernel, raw_count);
        const double spline_constant
            = number_value(member(evaluator, "const", "$.kernels[0].fevals[0]"),
                           "$.kernels[0].fevals[0].const");
        const std::vector<SplineTerm> spline_terms
            = parse_spline_terms(evaluator, feature_maps);

        const std::pair<FeatureDensityPolicy, std::string> density
            = parse_density_policy(nldf_count != 0U);
        const double exchange_scale = parse_exchange_scale();
        const TrainingLabelMetadata labels = parse_training_labels();
        const ModelMetadata metadata(semilocal_level,
                                     nldf_count != 0U,
                                     semilocal_count,
                                     nldf_count,
                                     feature_maps.size(),
                                     density.first,
                                     density.second,
                                     output_scale,
                                     exchange_scale,
                                     labels);
        return std::unique_ptr<const Model>(new NativeModel(metadata,
                                                            semilocal_mode,
                                                            std::move(nldf_settings),
                                                            normalizer_cutoff,
                                                            normalizers,
                                                            feature_maps,
                                                            multiplicative,
                                                            additive,
                                                            spline_constant,
                                                            spline_terms));
    }

  private:
    MultiplicativeBaseline parse_multiplicative_baseline(const LegacyNode& kernel) const
    {
        const LegacyNode& node = member(kernel, "_mul_basefunc", "$.kernels[0]");
        if (node.tag != "python/name:ciderpress.dft.baselines.lda_x")
        {
            throw LegacyYamlError("UNSUPPORTED_NAMED_FUNCTION",
                                  "$.kernels[0]._mul_basefunc",
                                  "unsupported multiplicative baseline");
        }
        return MultiplicativeBaseline::LdaX;
    }

    AdditiveBaseline parse_additive_baseline(const LegacyNode& kernel) const
    {
        const LegacyNode& node = member(kernel, "_add_basefunc", "$.kernels[0]");
        if (node.tag == "python/name:ciderpress.dft.baselines.gga_x_chachiyo")
        {
            return AdditiveBaseline::GgaXChachiyo;
        }
        if (node.tag == "python/name:ciderpress.dft.baselines.gga_x_pbe")
        {
            return AdditiveBaseline::GgaXPbe;
        }
        throw LegacyYamlError("UNSUPPORTED_NAMED_FUNCTION",
                              "$.kernels[0]._add_basefunc",
                              "unsupported additive baseline");
    }

    double parse_output_scale(const LegacyNode& kernel) const
    {
        const LegacyNode* node = optional_member(kernel, "output_scale");
        return node == nullptr ? 1.0 : number_value(*node, "$.kernels[0].output_scale");
    }

    void require_empty_settings(const LegacyNode& settings, const std::string& key) const
    {
        const std::string path = "$.settings." + key;
        const LegacyNode& node = member(settings, key, "$.settings");
        require_tag(node,
                    "python/object:ciderpress.dft.settings.EmptySettings",
                    path,
                    "UNSUPPORTED_DESCRIPTOR");
        if (node.kind != LegacyNode::Kind::Mapping || !node.mapping_items.empty())
        {
            throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", path, "settings must be empty");
        }
    }

    void parse_semilocal(const LegacyNode& settings,
                         SemilocalLevel& level,
                         SemilocalMode& mode,
                         std::size_t& count) const
    {
        const LegacyNode& semilocal = member(settings, "sl_settings", "$.settings");
        require_tag(semilocal,
                    "python/object:ciderpress.dft.settings.SemilocalSettings",
                    "$.settings.sl_settings",
                    "UNSUPPORTED_DESCRIPTOR");
        const std::string level_text
            = string_value(member(semilocal, "level", "$.settings.sl_settings"),
                           "$.settings.sl_settings.level",
                           "UNSUPPORTED_DESCRIPTOR");
        const std::string mode_text
            = string_value(member(semilocal, "mode", "$.settings.sl_settings"),
                           "$.settings.sl_settings.mode",
                           "UNSUPPORTED_DESCRIPTOR");
        if (level_text == "GGA" && mode_text == "np")
        {
            level = SemilocalLevel::Gga;
            mode = SemilocalMode::Np;
            count = 2U;
            return;
        }
        if (level_text == "MGGA" && mode_text == "npa")
        {
            level = SemilocalLevel::Mgga;
            mode = SemilocalMode::Npa;
            count = 3U;
            return;
        }
        throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR",
                              "$.settings.sl_settings",
                              "unsupported semilocal level/mode pair");
    }

    std::unique_ptr<const NldfSettings> parse_nldf(const LegacyNode& settings,
                                                   SemilocalLevel semilocal_level) const
    {
        const std::string path = "$.settings.nldf_settings";
        const LegacyNode& node = member(settings, "nldf_settings", "$.settings");
        if (node.tag == "python/object:ciderpress.dft.settings.EmptySettings")
        {
            if (node.kind != LegacyNode::Kind::Mapping || !node.mapping_items.empty())
            {
                throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", path, "settings must be empty");
            }
            return std::unique_ptr<const NldfSettings>();
        }
        require_tag(node,
                    "python/object:ciderpress.dft.settings.NLDFSettingsVJ",
                    path,
                    "UNSUPPORTED_DESCRIPTOR");
        SemilocalLevel effective_level = SemilocalLevel::Mgga;
        const LegacyNode* level_node = optional_member(node, "_sl_level");
        if (level_node != nullptr)
        {
            const std::string text
                = string_value(*level_node, path + "._sl_level", "UNSUPPORTED_DESCRIPTOR");
            if (text == "GGA")
            {
                effective_level = SemilocalLevel::Gga;
            }
            else if (text != "MGGA")
            {
                throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR",
                                      path + "._sl_level",
                                      "unsupported NLDF semilocal level");
            }
        }
        if (effective_level != semilocal_level)
        {
            throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR",
                                  path + "._sl_level",
                                  "NLDF and semilocal levels must match");
        }
        if (string_value(member(node, "rho_mult", path), path + ".rho_mult", "UNSUPPORTED_DESCRIPTOR")
            != "one")
        {
            throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", path + ".rho_mult", "rho_mult must be one");
        }
        for (const std::string& key : {std::string("l0_feat_specs"),
                                       std::string("l1_feat_specs"),
                                       std::string("l1_feat_dots")})
        {
            if (!sequence(member(node, key, path), path + "." + key, "UNSUPPORTED_DESCRIPTOR").empty())
            {
                throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", path + "." + key, "legacy feature family must be empty");
            }
        }
        const ParameterTriple theta_parameters
            = parameter_triple(member(node, "theta_params", path), path + ".theta_params");
        const std::vector<LegacyNode>& specs
            = sequence(member(node, "feat_specs", path), path + ".feat_specs", "UNSUPPORTED_DESCRIPTOR");
        if (specs.empty())
        {
            throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", path + ".feat_specs", "at least one NLDF feature is required");
        }
        for (std::size_t i = 0; i < specs.size(); ++i)
        {
            if (string_value(specs[i], path + ".feat_specs[" + std::to_string(i) + "]", "UNSUPPORTED_DESCRIPTOR")
                != "se")
            {
                throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR",
                                      path + ".feat_specs[" + std::to_string(i) + "]",
                                      "only se features are supported");
            }
        }
        const std::vector<LegacyNode>& parameters
            = sequence(member(node, "feat_params", path), path + ".feat_params", "UNSUPPORTED_DESCRIPTOR");
        if (parameters.size() != specs.size())
        {
            throw LegacyYamlError("FEATURE_COUNT", path + ".feat_params", "feature specs and parameters differ in length");
        }
        std::vector<ParameterTriple> feature_parameters;
        feature_parameters.reserve(parameters.size());
        for (std::size_t i = 0; i < parameters.size(); ++i)
        {
            feature_parameters.push_back(parameter_triple(
                parameters[i], path + ".feat_params[" + std::to_string(i) + "]"));
        }
        return std::unique_ptr<const NldfSettings>(
            new NldfSettings(effective_level, theta_parameters, feature_parameters));
    }

    std::vector<Normalizer> parse_normalizers(const LegacyNode& settings,
                                              SemilocalMode mode,
                                              std::size_t raw_count,
                                              double& cutoff) const
    {
        const std::string path = "$.settings.normalizers";
        const LegacyNode& node = member(settings, "normalizers", "$.settings");
        require_tag(node,
                    "python/object:ciderpress.dft.feat_normalizer.FeatNormalizerList",
                    path,
                    "UNSUPPORTED_NORMALIZER");
        cutoff = number_value(member(node, "cutoff", path), path + ".cutoff");
        if (!(cutoff > 0.0))
        {
            throw LegacyYamlError("SCHEMA", path + ".cutoff", "normalizer cutoff must be positive");
        }
        const std::string expected_mode = mode == SemilocalMode::Np ? "np" : "npa";
        if (string_value(member(node, "slmode", path), path + ".slmode") != expected_mode)
        {
            throw LegacyYamlError("FEATURE_COUNT", path + ".slmode", "normalizer mode differs from semilocal mode");
        }
        const std::vector<LegacyNode>& entries
            = sequence(member(node, "_normalizers", path), path + "._normalizers");
        if (entries.size() != raw_count)
        {
            throw LegacyYamlError("FEATURE_COUNT", path + "._normalizers", "normalizer count differs from raw feature count");
        }
        std::vector<Normalizer> result;
        result.reserve(entries.size());
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            const std::string entry_path = path + "._normalizers[" + std::to_string(i) + "]";
            if (entries[i].kind == LegacyNode::Kind::Null)
            {
                result.push_back(Normalizer(NormalizerKind::Identity, 1.0));
            }
            else
            {
                require_tag(entries[i],
                            "python/object:ciderpress.dft.feat_normalizer.ConstantNormalizer",
                            entry_path,
                            "UNSUPPORTED_NORMALIZER");
                result.push_back(Normalizer(
                    NormalizerKind::Constant,
                    number_value(member(entries[i], "const", entry_path), entry_path + ".const", true)));
            }
        }
        return result;
    }

    std::vector<FeatureMap> parse_feature_maps(const LegacyNode& kernel,
                                               std::size_t raw_count) const
    {
        const std::string list_path = "$.kernels[0].feature_list";
        const LegacyNode& list = member(kernel, "feature_list", "$.kernels[0]");
        require_tag(list,
                    "python/object:ciderpress.dft.transform_data.FeatureList",
                    list_path,
                    "UNSUPPORTED_DESCRIPTOR");
        const std::vector<LegacyNode>& entries
            = sequence(member(list, "feat_list", list_path), list_path + ".feat_list", "UNSUPPORTED_DESCRIPTOR");
        if (entries.empty())
        {
            throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", list_path + ".feat_list", "feature map list must be nonempty");
        }
        std::vector<FeatureMap> result;
        result.reserve(entries.size());
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            const LegacyNode& entry = entries[i];
            const std::string path = list_path + ".feat_list[" + std::to_string(i) + "]";
            const std::int64_t raw_index = integer_value(member(entry, "i", path), path + ".i");
            if (raw_index < 0 || static_cast<std::uint64_t>(raw_index) >= raw_count)
            {
                throw LegacyYamlError("FEATURE_COUNT", path + ".i", "raw feature index is out of bounds");
            }
            const double gamma = number_value(member(entry, "gamma", path), path + ".gamma");
            if (!(gamma > 0.0))
            {
                throw LegacyYamlError("SCHEMA", path + ".gamma", "gamma must be positive");
            }
            FeatureMapKind kind;
            double scale = 1.0;
            double center = 0.0;
            int exponent = 0;
            std::pair<double, double> map_bounds;
            if (entry.tag == "python/object:ciderpress.dft.transform_data.UMap")
            {
                kind = FeatureMapKind::U;
                map_bounds = bounds(member(entry, "_bounds", path), path + "._bounds");
            }
            else if (entry.tag == "python/object:ciderpress.dft.transform_data.VMap"
                     || entry.tag == "python/object:ciderpress.dft.transform_data.ZMap")
            {
                kind = entry.tag.find(".VMap") != std::string::npos ? FeatureMapKind::V
                                                                    : FeatureMapKind::Z;
                scale = number_value(member(entry, "scale", path), path + ".scale");
                center = number_value(member(entry, "center", path), path + ".center");
                if (!(scale > 0.0))
                {
                    throw LegacyYamlError("SCHEMA", path + ".scale", "scale must be positive");
                }
                if (kind == FeatureMapKind::Z)
                {
                    const std::int64_t n = integer_value(member(entry, "n", path), path + ".n");
                    if (n != 1)
                    {
                        throw LegacyYamlError("SCHEMA", path + ".n", "ZMap exponent must equal one");
                    }
                    exponent = 1;
                }
                const LegacyNode* explicit_bounds = optional_member(entry, "_bounds");
                map_bounds = explicit_bounds == nullptr
                                 ? std::make_pair(-center, scale - center)
                                 : bounds(*explicit_bounds, path + "._bounds");
            }
            else
            {
                throw LegacyYamlError("UNSUPPORTED_DESCRIPTOR", path, "unsupported feature map");
            }
            result.push_back(FeatureMap(kind,
                                        static_cast<std::size_t>(raw_index),
                                        gamma,
                                        scale,
                                        center,
                                        exponent,
                                        map_bounds.first,
                                        map_bounds.second));
        }
        return result;
    }

    std::vector<double> parse_scales(const LegacyNode& node, const std::string& path) const
    {
        if (node.kind == LegacyNode::Kind::Numpy)
        {
            if (node.numpy_value.scalar || node.numpy_value.dtype != LegacyDType::Float64
                || node.numpy_value.shape.size() != 1U)
            {
                throw LegacyYamlError("MALFORMED_ARRAY", path, "scale must be a one-dimensional float64 array");
            }
            return node.numpy_value.float64_values;
        }
        const std::vector<LegacyNode>& entries = sequence(node, path);
        std::vector<double> result;
        result.reserve(entries.size());
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            result.push_back(number_value(entries[i], path + "[" + std::to_string(i) + "]"));
        }
        return result;
    }

    std::vector<SplineTerm> parse_spline_terms(const LegacyNode& evaluator,
                                               const std::vector<FeatureMap>& maps) const
    {
        const std::string path = "$.kernels[0].fevals[0]";
        const std::int64_t signed_nterms
            = integer_value(member(evaluator, "nterms", path), path + ".nterms");
        if (signed_nterms <= 0)
        {
            throw LegacyYamlError("SCHEMA", path + ".nterms", "nterms must be positive");
        }
        const std::size_t nterms = static_cast<std::size_t>(signed_nterms);
        const std::vector<double> scales = parse_scales(member(evaluator, "scale", path), path + ".scale");
        const std::vector<LegacyNode>& index_sets
            = sequence(member(evaluator, "ind_sets", path), path + ".ind_sets");
        const std::vector<LegacyNode>& grids
            = sequence(member(evaluator, "spline_grids", path), path + ".spline_grids");
        const std::vector<LegacyNode>& coefficients
            = sequence(member(evaluator, "coeff_sets", path), path + ".coeff_sets");
        if (scales.size() != nterms)
        {
            throw LegacyYamlError("FEATURE_COUNT", path + ".scale", "scale length differs from nterms");
        }
        if (index_sets.size() != nterms)
        {
            throw LegacyYamlError("FEATURE_COUNT", path + ".ind_sets", "index-set count differs from nterms");
        }
        if (grids.size() != nterms)
        {
            throw LegacyYamlError("FEATURE_COUNT", path + ".spline_grids", "grid count differs from nterms");
        }
        if (coefficients.size() != nterms)
        {
            throw LegacyYamlError("FEATURE_COUNT", path + ".coeff_sets", "coefficient count differs from nterms");
        }
        std::vector<SplineTerm> result;
        result.reserve(nterms);
        for (std::size_t term = 0; term < nterms; ++term)
        {
            const std::string term_suffix = "[" + std::to_string(term) + "]";
            const std::vector<LegacyNode>& index_nodes
                = sequence(index_sets[term], path + ".ind_sets" + term_suffix);
            if (index_nodes.empty() || index_nodes.size() > 3U)
            {
                throw LegacyYamlError("FEATURE_COUNT",
                                      path + ".ind_sets" + term_suffix,
                                      "spline dimension must be between one and three");
            }
            std::vector<std::size_t> indices;
            for (std::size_t d = 0; d < index_nodes.size(); ++d)
            {
                const std::string index_path
                    = path + ".ind_sets" + term_suffix + "[" + std::to_string(d) + "]";
                const std::int64_t index = integer_value(index_nodes[d], index_path, true);
                if (index < 0 || static_cast<std::uint64_t>(index) >= maps.size())
                {
                    throw LegacyYamlError("FEATURE_COUNT", index_path, "transformed feature index is out of bounds");
                }
                if (std::find(indices.begin(), indices.end(), static_cast<std::size_t>(index))
                    != indices.end())
                {
                    throw LegacyYamlError("FEATURE_COUNT", index_path, "descriptor indices must be distinct");
                }
                indices.push_back(static_cast<std::size_t>(index));
            }
            const std::vector<LegacyNode>& grid_nodes
                = sequence(grids[term], path + ".spline_grids" + term_suffix);
            if (grid_nodes.size() != indices.size())
            {
                throw LegacyYamlError("FEATURE_COUNT",
                                      path + ".spline_grids" + term_suffix,
                                      "grid dimension differs from descriptor dimension");
            }
            std::vector<SplineGrid> term_grids;
            for (std::size_t d = 0; d < grid_nodes.size(); ++d)
            {
                const std::string grid_path = path + ".spline_grids" + term_suffix + "["
                                              + std::to_string(d) + "]";
                const std::vector<LegacyNode>& values = sequence(grid_nodes[d], grid_path);
                if (values.size() != 3U)
                {
                    throw LegacyYamlError("SCHEMA", grid_path, "grid must contain lower, upper, and order");
                }
                const double lower = number_value(values[0], grid_path + "[0]");
                const double upper = number_value(values[1], grid_path + "[1]");
                const std::int64_t order = integer_value(values[2], grid_path + "[2]");
                if (!(lower < upper) || order <= 0)
                {
                    throw LegacyYamlError("SCHEMA", grid_path, "grid range/order is invalid");
                }
                const FeatureMap& map = maps[indices[d]];
                if (lower != map.lower_bound() || upper != map.upper_bound())
                {
                    throw LegacyYamlError("FEATURE_COUNT", grid_path, "grid bounds differ from feature-map bounds");
                }
                term_grids.push_back(SplineGrid(lower, upper, static_cast<std::size_t>(order)));
            }
            const LegacyNode& array = coefficients[term];
            const std::string coefficient_path = path + ".coeff_sets" + term_suffix;
            if (array.kind != LegacyNode::Kind::Numpy || array.numpy_value.scalar
                || array.numpy_value.dtype != LegacyDType::Float64)
            {
                throw LegacyYamlError("MALFORMED_ARRAY", coefficient_path, "coefficient set must be a float64 array");
            }
            if (array.numpy_value.shape.size() != indices.size())
            {
                throw LegacyYamlError("MALFORMED_ARRAY", coefficient_path, "coefficient rank differs from spline dimension");
            }
            for (std::size_t d = 0; d < term_grids.size(); ++d)
            {
                if (term_grids[d].order() > std::numeric_limits<std::size_t>::max() - 2U
                    || array.numpy_value.shape[d] != term_grids[d].order() + 2U)
                {
                    throw LegacyYamlError("MALFORMED_ARRAY",
                                          coefficient_path,
                                          "coefficient extent differs from grid order plus two");
                }
            }
            result.push_back(SplineTerm(scales[term],
                                        indices,
                                        term_grids,
                                        array.numpy_value.shape,
                                        array.numpy_value.float64_values));
        }
        return result;
    }

    std::pair<FeatureDensityPolicy, std::string> parse_density_policy(bool has_nldf) const
    {
        const LegacyNode* policy_node = optional_member(root_, "feature_density_policy");
        const LegacyNode* source_node = optional_member(root_, "feature_density_policy_source");
        bool explicit_policy = false;
        FeatureDensityPolicy policy = FeatureDensityPolicy::Valence;
        if (policy_node != nullptr && policy_node->kind != LegacyNode::Kind::Null)
        {
            if (policy_node->kind != LegacyNode::Kind::String)
            {
                throw LegacyYamlError("DENSITY_POLICY_INVALID",
                                      "$.feature_density_policy",
                                      "policy must be a string or missing sentinel");
            }
            const std::string& text = policy_node->string_value;
            if (!text.empty() && text != "auto")
            {
                explicit_policy = true;
                if (text == "valence")
                {
                    policy = FeatureDensityPolicy::Valence;
                }
                else if (text == "valence_pseudo_core" || text == "total")
                {
                    policy = FeatureDensityPolicy::ValencePseudoCore;
                }
                else
                {
                    throw LegacyYamlError("DENSITY_POLICY_INVALID",
                                          "$.feature_density_policy",
                                          "unsupported density policy");
                }
            }
        }
        if (explicit_policy)
        {
            std::string source = "model";
            if (source_node != nullptr)
            {
                if (source_node->kind != LegacyNode::Kind::String)
                {
                    throw LegacyYamlError("DENSITY_POLICY_INVALID",
                                          "$.feature_density_policy_source",
                                          "policy source must be a string");
                }
                if (!source_node->string_value.empty())
                {
                    source = source_node->string_value;
                }
            }
            return std::make_pair(policy, source);
        }
        if (source_node != nullptr)
        {
            throw LegacyYamlError("DENSITY_POLICY_INVALID",
                                  "$.feature_density_policy_source",
                                  "policy source requires an explicit policy");
        }
        if (!has_nldf)
        {
            return std::make_pair(FeatureDensityPolicy::Valence, "default_valence");
        }
        if (basename_contains_cider23x(input_path_))
        {
            return std::make_pair(FeatureDensityPolicy::ValencePseudoCore,
                                  "inferred_legacy_cider23x");
        }
        throw LegacyYamlError("DENSITY_POLICY_REQUIRED",
                              "$.feature_density_policy",
                              "NLDF model requires an explicit density policy");
    }

    double parse_exchange_scale() const
    {
        const LegacyNode* node = optional_member(root_, "ml_exchange_scale");
        if (node == nullptr)
        {
            return 1.0;
        }
        if (node->kind == LegacyNode::Kind::Null)
        {
            return 0.0;
        }
        const double value = number_value(*node, "$.ml_exchange_scale");
        if (value < 0.0)
        {
            throw LegacyYamlError("SCHEMA", "$.ml_exchange_scale", "exchange scale must be nonnegative");
        }
        return value;
    }

    TrainingLabelMetadata parse_training_labels() const
    {
        const std::string keys[] = {"ml_exchange_label_convention",
                                    "ml_exchange_label_hybrid_scaled",
                                    "ml_exchange_label_scale",
                                    "ml_exchange_label_source_scales"};
        std::size_t present = 0U;
        for (const std::string& key : keys)
        {
            present += optional_member(root_, key) == nullptr ? 0U : 1U;
        }
        if (present == 0U)
        {
            return TrainingLabelMetadata();
        }
        if (present != 4U)
        {
            for (const std::string& key : keys)
            {
                if (optional_member(root_, key) == nullptr)
                {
                    throw LegacyYamlError("SCHEMA", "$." + key, "training-label metadata is an atomic group");
                }
            }
        }
        const std::string convention
            = string_value(*optional_member(root_, keys[0]), "$." + keys[0]);
        if (convention != "full_exact_exchange")
        {
            throw LegacyYamlError("SCHEMA", "$." + keys[0], "unsupported label convention");
        }
        const bool hybrid_scaled
            = boolean_value(*optional_member(root_, keys[1]), "$." + keys[1]);
        if (hybrid_scaled)
        {
            throw LegacyYamlError("SCHEMA", "$." + keys[1], "hybrid-scaled labels are not supported");
        }
        const double scale = number_value(*optional_member(root_, keys[2]), "$." + keys[2]);
        if (scale != 1.0)
        {
            throw LegacyYamlError("SCHEMA", "$." + keys[2], "label scale must equal one");
        }
        const std::vector<LegacyNode>& nodes
            = sequence(*optional_member(root_, keys[3]), "$." + keys[3]);
        if (nodes.empty())
        {
            throw LegacyYamlError("SCHEMA", "$." + keys[3], "source scales must be nonempty");
        }
        std::vector<double> source_scales;
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            const double value
                = number_value(nodes[i], "$." + keys[3] + "[" + std::to_string(i) + "]");
            if (!(value > 0.0))
            {
                throw LegacyYamlError("SCHEMA",
                                      "$." + keys[3] + "[" + std::to_string(i) + "]",
                                      "source scale must be positive");
            }
            source_scales.push_back(value);
        }
        return TrainingLabelMetadata(convention, hybrid_scaled, scale, source_scales);
    }

    const LegacyNode& root_;
    const std::string& input_path_;
};

} // namespace

std::unique_ptr<const Model> ModelLoader::load(const std::string& path,
                                               const LoaderLimits& limits) const
{
    const LegacyDocument document = read_legacy_document(path, limits);
    return TypedModelBuilder(document, path).build();
}

ModelMetadata ModelLoader::inspect_metadata(const std::string& path,
                                            const LoaderLimits& limits) const
{
    return load(path, limits)->metadata();
}

} // namespace cider
} // namespace hamilt
