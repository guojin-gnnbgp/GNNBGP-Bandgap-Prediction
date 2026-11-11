#ifndef MODULE_HAMILT_CIDER_MODEL_LOADER_H
#define MODULE_HAMILT_CIDER_MODEL_LOADER_H

#include "model.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hamilt
{
namespace cider
{

struct LoaderLimits
{
    LoaderLimits();

    std::size_t max_file_bytes;
    std::size_t max_yaml_nodes;
    std::size_t max_yaml_expanded_bytes;
    std::size_t max_yaml_depth;
    std::size_t max_array_rank;
    std::size_t max_array_elements;
    std::size_t max_array_bytes;
    std::size_t max_total_array_bytes;
};

enum class LegacyDType
{
    Float64,
    Int64
};

struct LegacyNumpyValue
{
    LegacyNumpyValue();

    LegacyDType dtype;
    bool scalar;
    std::vector<std::size_t> shape;
    std::vector<double> float64_values;
    std::vector<std::int64_t> int64_values;
};

struct LegacyNode
{
    enum class Kind
    {
        Null,
        Boolean,
        Integer,
        Number,
        String,
        Binary,
        Sequence,
        Mapping,
        Numpy
    };

    LegacyNode();

    Kind kind;
    std::string tag;
    bool boolean_value;
    std::int64_t integer_value;
    double number_value;
    std::string string_value;
    std::vector<unsigned char> binary_value;
    std::vector<LegacyNode> sequence_items;
    std::map<std::string, LegacyNode> mapping_items;
    LegacyNumpyValue numpy_value;
};

struct LegacyDocument
{
    LegacyNode root;
};

class LegacyYamlError : public std::runtime_error
{
  public:
    LegacyYamlError(const std::string& code, const std::string& path, const std::string& detail);

    const std::string& code() const;
    const std::string& path() const;
    const std::string& detail() const;

  private:
    std::string code_;
    std::string path_;
    std::string detail_;
};

// Converts a supported artifact into a validated native model.
class ModelLoader
{
  public:
    ModelLoader() = default;
    ~ModelLoader() = default;

    std::unique_ptr<const Model> load(const std::string& path,
                                      const LoaderLimits& limits = LoaderLimits()) const;
    ModelMetadata inspect_metadata(const std::string& path,
                                   const LoaderLimits& limits = LoaderLimits()) const;

    // Reads the safe, validated legacy YAML representation consumed by the
    // typed model construction implemented separately.
    static LegacyDocument read_legacy_document(const std::string& path,
                                               const LoaderLimits& limits = LoaderLimits());

};

} // namespace cider
} // namespace hamilt

#endif // MODULE_HAMILT_CIDER_MODEL_LOADER_H
