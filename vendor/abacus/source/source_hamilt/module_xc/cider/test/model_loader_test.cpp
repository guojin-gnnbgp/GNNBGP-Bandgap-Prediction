#include "source_hamilt/module_xc/cider/model_loader.h"
#include "oracle_sha256.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

class TempYaml
{
  public:
    explicit TempYaml(const std::string& contents)
    {
        static unsigned int serial = 0;
        path_ = "/tmp/cider-native-loader-" + std::to_string(++serial) + ".yaml";
        std::ofstream output(path_.c_str(), std::ios::binary);
        output << contents;
    }

    ~TempYaml()
    {
        std::remove(path_.c_str());
    }

    const std::string& path() const
    {
        return path_;
    }

  private:
    std::string path_;
};

std::string array_model(const std::string& numpy_module,
                        const std::string& dtype,
                        const std::string& byte_order,
                        const std::string& fortran_order,
                        const std::string& shape,
                        const std::string& data)
{
    return "!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
           "kernels:\n"
           "- !!python/object:ciderpress.dft.xc_evaluator.MappedDFTKernel\n"
           "  _add_basefunc: !!python/name:ciderpress.dft.baselines.gga_x_chachiyo ''\n"
           "  _mul_basefunc: !!python/name:ciderpress.dft.baselines.lda_x ''\n"
           "  feature_list: !!python/object:ciderpress.dft.transform_data.FeatureList\n"
           "    feat_list: []\n"
           "  fevals:\n"
           "  - !!python/object:ciderpress.dft.xc_evaluator.SplineSetEvaluator\n"
           "    coeff_sets:\n"
           "    - !!python/object/apply:numpy."
           + numpy_module
           + ".multiarray._reconstruct\n"
             "      args:\n"
             "      - !!python/name:numpy.ndarray ''\n"
             "      - !!python/tuple [0]\n"
             "      - !!binary 'Yg=='\n"
             "      state: !!python/tuple\n"
             "      - 1\n"
             "      - !!python/tuple "
           + shape
           + "\n"
             "      - !!python/object/apply:numpy.dtype\n"
             "        args: ["
           + dtype
           + ", false, true]\n"
             "        state: !!python/tuple [3, '"
           + byte_order
           + "', null, null, null, -1, -1, 0]\n"
             "      - "
           + fortran_order
           + "\n"
             "      - !!binary '"
           + data
           + "'\n"
             "    const: 0.0\n"
             "    ind_sets: []\n"
             "    nterms: 1\n"
             "    scale: [1.0]\n"
             "    spline_grids: []\n"
             "  mode: SEP\n"
             "settings: !!python/object:ciderpress.dft.settings.FeatureSettings\n"
             "  sl_settings: !!python/object:ciderpress.dft.settings.SemilocalSettings\n"
             "    level: GGA\n"
             "    mode: np\n"
             "  nldf_settings: !!python/object:ciderpress.dft.settings.EmptySettings {}\n"
             "  nlof_settings: !!python/object:ciderpress.dft.settings.EmptySettings {}\n"
             "  sdmx_settings: !!python/object:ciderpress.dft.settings.EmptySettings {}\n"
             "  hyb_settings: !!python/object:ciderpress.dft.settings.EmptySettings {}\n"
             "  normalizers: !!python/object:ciderpress.dft.feat_normalizer.FeatNormalizerList\n"
             "    _normalizers: []\n"
             "    cutoff: 1.0e-10\n"
             "    slmode: np\n"
             "libxc_baseline: null\n";
}

void replace_once(std::string& text, const std::string& from, const std::string& to)
{
    const std::size_t offset = text.find(from);
    ASSERT_NE(std::string::npos, offset);
    text.replace(offset, from.size(), to);
}

void replace_between(std::string& text,
                     const std::string& begin,
                     const std::string& end,
                     const std::string& replacement)
{
    const std::size_t begin_offset = text.find(begin);
    ASSERT_NE(std::string::npos, begin_offset);
    const std::size_t end_offset = text.find(end, begin_offset + begin.size());
    ASSERT_NE(std::string::npos, end_offset);
    text.replace(begin_offset, end_offset - begin_offset, replacement);
}

std::string numpy_scalar(const std::string& module,
                         const std::string& dtype,
                         const std::string& data,
                         const std::string& indent)
{
    return indent + "!!python/object/apply:numpy." + module + ".multiarray.scalar\n" + indent
           + "- !!python/object/apply:numpy.dtype\n" + indent + "  args: [" + dtype
           + ", false, true]\n" + indent
           + "  state: !!python/tuple [3, '<', null, null, null, -1, -1, 0]\n" + indent
           + "- !!binary '" + data + "'\n";
}

std::string typed_model()
{
    std::string yaml
        = array_model("core", "f8", "<", "false", "[3]", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
    replace_once(yaml,
                 "    feat_list: []\n",
                 "    feat_list:\n"
                 "    - !!python/object:ciderpress.dft.transform_data.UMap\n"
                 "      i: 1\n"
                 "      gamma: 1.0\n"
                 "      _bounds: [0.0, 1.0]\n");
    replace_once(yaml, "    ind_sets: []\n", "    ind_sets:\n    - [0]\n");
    replace_once(yaml, "    spline_grids: []\n", "    spline_grids:\n    - [[0.0, 1.0, 1]]\n");
    replace_once(yaml, "    _normalizers: []\n", "    _normalizers: [null, null]\n");
    return yaml;
}

void expect_error(const std::string& yaml,
                  const std::string& code,
                  const std::string& path,
                  const hamilt::cider::LoaderLimits& limits = hamilt::cider::LoaderLimits())
{
    const TempYaml input(yaml);
    try
    {
        hamilt::cider::ModelLoader::read_legacy_document(input.path(), limits);
        FAIL() << "expected " << code << " at " << path;
    }
    catch (const hamilt::cider::LegacyYamlError& error)
    {
        EXPECT_EQ(code, error.code());
        EXPECT_EQ(path, error.path());
        EXPECT_NE(std::string::npos, std::string(error.what()).find("CIDER legacy YAML error [" + code + "] at " + path + ":"));
    }
}

std::string read_file(const std::string& path)
{
    std::ifstream input(path.c_str(), std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::string fixture_path()
{
    const std::string source(__FILE__);
    return source.substr(0, source.find_last_of('/')) + "/fixtures/v1/arrays.bin";
}

template <typename Value>
std::vector<Value> read_fixture_array(std::size_t offset, std::size_t count)
{
    std::ifstream input(fixture_path().c_str(), std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("failed to open native CIDER v1 fixture");
    }
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<Value> values(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(Value)));
    if (!input)
    {
        throw std::runtime_error("failed to read native CIDER v1 fixture array");
    }
    return values;
}

void expect_typed_error(const std::string& yaml,
                        const std::string& code,
                        const std::string& path)
{
    const TempYaml input(yaml);
    try
    {
        hamilt::cider::ModelLoader().load(input.path());
        FAIL() << "expected " << code << " at " << path;
    }
    catch (const hamilt::cider::LegacyYamlError& error)
    {
        EXPECT_EQ(code, error.code());
        EXPECT_EQ(path, error.path());
    }
}

TEST(NativeCiderModelLoader, RejectsPickleSuffixBeforeOpeningInput)
{
    for (const std::string& suffix : {std::string(".PiCkLe"), std::string(".PKL"), std::string(".JobLib")})
    {
        try
        {
            hamilt::cider::ModelLoader::read_legacy_document("does-not-exist" + suffix);
            FAIL() << "expected legacy loader error";
        }
        catch (const std::runtime_error& error)
        {
            EXPECT_EQ(
                "CIDER legacy YAML error [JOBLIB_UNSUPPORTED] at <file>: pickle/joblib input is not supported",
                std::string(error.what()));
        }
    }
}

TEST(NativeCiderModelLoader, ReadsWhitelistedRootIntoIntermediateNodes)
{
    const TempYaml input(
        "!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
        "kernels: []\n"
        "settings: null\n"
        "libxc_baseline: null\n");

    const hamilt::cider::LegacyDocument document
        = hamilt::cider::ModelLoader::read_legacy_document(input.path());

    EXPECT_EQ(hamilt::cider::LegacyNode::Kind::Mapping, document.root.kind);
    EXPECT_EQ("python/object:ciderpress.dft.xc_evaluator.MappedXC", document.root.tag);
    ASSERT_EQ(3U, document.root.mapping_items.size());
    EXPECT_EQ(hamilt::cider::LegacyNode::Kind::Sequence,
              document.root.mapping_items.at("kernels").kind);
    EXPECT_EQ(hamilt::cider::LegacyNode::Kind::Null,
              document.root.mapping_items.at("settings").kind);
}

TEST(NativeCiderModelLoader, DecodesNumpyCoreAliasesToTheSameFloat64Array)
{
    for (const std::string& module : {std::string("core"), std::string("_core")})
    {
        const TempYaml input(array_model(
            module, "f8", "<", "false", "[2]", "AAAAAAAA8D8AAAAAAAAEwA=="));
        const hamilt::cider::LegacyDocument document
            = hamilt::cider::ModelLoader::read_legacy_document(input.path());
        const hamilt::cider::LegacyNode& array
            = document.root.mapping_items.at("kernels")
                  .sequence_items.at(0)
                  .mapping_items.at("fevals")
                  .sequence_items.at(0)
                  .mapping_items.at("coeff_sets")
                  .sequence_items.at(0);

        ASSERT_EQ(hamilt::cider::LegacyNode::Kind::Numpy, array.kind);
        EXPECT_FALSE(array.numpy_value.scalar);
        EXPECT_EQ(hamilt::cider::LegacyDType::Float64, array.numpy_value.dtype);
        ASSERT_EQ(1U, array.numpy_value.shape.size());
        EXPECT_EQ(2U, array.numpy_value.shape[0]);
        ASSERT_EQ(2U, array.numpy_value.float64_values.size());
        EXPECT_DOUBLE_EQ(1.0, array.numpy_value.float64_values[0]);
        EXPECT_DOUBLE_EQ(-2.5, array.numpy_value.float64_values[1]);
    }
}

TEST(NativeCiderModelLoader, RejectsFloat64ScalarInInt64DescriptorIndexRole)
{
    std::string yaml = array_model("core", "f8", "<", "false", "[2]", "AAAAAAAA8D8AAAAAAAAEwA==");
    replace_once(yaml,
                 "    ind_sets: []\n",
                 "    ind_sets:\n"
                 "    - !!python/tuple\n"
                 "      - "
                     + numpy_scalar("core", "f8", "AAAAAAAA+D8=", "        "));
    const TempYaml input(yaml);

    try
    {
        hamilt::cider::ModelLoader::read_legacy_document(input.path());
        FAIL() << "expected scalar role error";
    }
    catch (const hamilt::cider::LegacyYamlError& error)
    {
        EXPECT_EQ("MALFORMED_ARRAY", error.code());
        EXPECT_EQ("$.kernels[0].fevals[0].ind_sets[0][0]", error.path());
    }
}

TEST(NativeCiderModelLoader, RejectsNumpyArrayOutsideSplineArrayRoles)
{
    std::string yaml = array_model("core", "f8", "<", "false", "[2]", "AAAAAAAA8D8AAAAAAAAEwA==");
    const std::string array_start = "!!python/object/apply:numpy.core.multiarray._reconstruct\n";
    const std::size_t start = yaml.find(array_start);
    ASSERT_NE(std::string::npos, start);
    const std::size_t end = yaml.find("    const:", start);
    ASSERT_NE(std::string::npos, end);
    const std::string array = yaml.substr(start, end - start);
    const std::size_t settings = yaml.find("settings:");
    const std::size_t baseline = yaml.find("libxc_baseline:", settings);
    ASSERT_NE(std::string::npos, settings);
    ASSERT_NE(std::string::npos, baseline);
    yaml.replace(settings, baseline - settings, "settings: " + array);
    const TempYaml input(yaml);

    try
    {
        hamilt::cider::ModelLoader::read_legacy_document(input.path());
        FAIL() << "expected array role error";
    }
    catch (const hamilt::cider::LegacyYamlError& error)
    {
        EXPECT_EQ("UNSUPPORTED_NUMPY_TAG", error.code());
        EXPECT_EQ("$.settings", error.path());
    }
}

TEST(NativeCiderModelLoader, RoutesTagKeyAndYamlFailuresDeterministically)
{
    const std::string minimal
        = "!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
          "kernels: []\nsettings: null\nlibxc_baseline: null\n";

    expect_error("!!python/object:evil.Payload {}\n", "UNKNOWN_TAG", "$");
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC2 {}\n",
                 "UNSUPPORTED_ROOT",
                 "$");
    expect_error("!!python/object:ciderpress.dft.settings.FeatureSettings {}\n",
                 "UNSUPPORTED_ROOT",
                 "$");
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels:\n- !!python/object:ciderpress.dft.xc_evaluator.RBFEvaluator {}\n"
                 "settings: null\nlibxc_baseline: null\n",
                 "UNSUPPORTED_EVALUATOR",
                 "$.kernels[0]");
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\nsettings: !!python/object:ciderpress.dft.settings.NLOFSettings {}\n"
                 "libxc_baseline: null\n",
                 "UNSUPPORTED_DESCRIPTOR",
                 "$.settings");
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\nsettings: !!python/object:ciderpress.dft.feat_normalizer.GeneralNormalizer {}\n"
                 "libxc_baseline: null\n",
                 "UNSUPPORTED_NORMALIZER",
                 "$.settings");
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\nsettings: !!python/object/apply:numpy.core.multiarray.frombuffer []\n"
                 "libxc_baseline: null\n",
                 "UNSUPPORTED_NUMPY_TAG",
                 "$.settings");
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\nsettings: !!python/name:ciderpress.dft.baselines.lda_x ''\n"
                 "libxc_baseline: null\n",
                 "UNSUPPORTED_NAMED_FUNCTION",
                 "$.settings");
    expect_error(minimal + "extra: 1\n", "SCHEMA", "$.extra");
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\nkernels: []\nsettings: null\nlibxc_baseline: null\n",
                 "YAML_SYNTAX",
                 "$.kernels");
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\nsettings: null\nlibxc_baseline: null\n<<: {}\n",
                 "YAML_SYNTAX",
                 "$.<<");
}

TEST(NativeCiderModelLoader, RejectsMalformedNumpyArraysWithPayloadPaths)
{
    const std::string valid
        = array_model("core", "f8", "<", "false", "[2]", "AAAAAAAA8D8AAAAAAAAEwA==");
    struct Case
    {
        std::string from;
        std::string to;
        std::string path;
    };
    const Case cases[] = {
        {"[3, '<', null", "[3, '>', null", ".state[2].state"},
        {"      - false\n      - !!binary", "      - true\n      - !!binary", ".state[3]"},
        {"AAAAAAAA8D8AAAAAAAAEwA==", "AAAAAAAA8D8=", ".state[4]"},
        {"!!python/tuple [2]", "!!python/tuple []", ".state[1]"},
        {"!!python/tuple [2]", "!!python/tuple [1, 1, 1, 1]", ".state[1]"},
        {"!!python/tuple [2]", "!!python/tuple [0]", ".state[1][0]"},
        {"      state: !!python/tuple\n      - 1\n",
         "      state: !!python/tuple\n      - 2\n",
         ".state[0]"},
        {"args: [f8, false, true]", "args: [i8, false, true]", ".state[2].args[0]"},
        {"args: [f8, false, true]", "args: [O8, false, true]", ".state[2].args[0]"},
        {"AAAAAAAA8D8AAAAAAAAEwA==", "!!!!", ".state[4]"},
        {"!!binary 'Yg=='", "!!binary 'Yw=='", ".args[2]"}};
    const std::string prefix = "$.kernels[0].fevals[0].coeff_sets[0]";
    for (const Case& item : cases)
    {
        SCOPED_TRACE(item.from + " -> " + item.to);
        std::string yaml = valid;
        replace_once(yaml, item.from, item.to);
        expect_error(yaml, "MALFORMED_ARRAY", prefix + item.path);
    }

    std::string nonfinite = array_model("core", "f8", "<", "false", "[1]", "AAAAAAAA8H8=");
    expect_error(nonfinite, "MALFORMED_ARRAY", prefix + ".state[4]");
}

TEST(NativeCiderModelLoader, EnforcesConfiguredFileNodeAndArrayLimits)
{
    const std::string yaml
        = array_model("core", "f8", "<", "false", "[2]", "AAAAAAAA8D8AAAAAAAAEwA==");
    hamilt::cider::LoaderLimits limits;
    limits.max_file_bytes = 8U;
    expect_error(yaml, "YAML_SYNTAX", "<file>", limits);

    limits = hamilt::cider::LoaderLimits();
    limits.max_yaml_nodes = 3U;
    expect_error(yaml, "YAML_SYNTAX", "$.kernels[0]._add_basefunc", limits);

    limits = hamilt::cider::LoaderLimits();
    limits.max_array_elements = 1U;
    expect_error(yaml,
                 "MALFORMED_ARRAY",
                 "$.kernels[0].fevals[0].coeff_sets[0].state[1]",
                 limits);

    limits = hamilt::cider::LoaderLimits();
    limits.max_array_bytes = 8U;
    expect_error(yaml,
                 "MALFORMED_ARRAY",
                 "$.kernels[0].fevals[0].coeff_sets[0]",
                 limits);

    limits = hamilt::cider::LoaderLimits();
    limits.max_total_array_bytes = 8U;
    expect_error(yaml,
                 "MALFORMED_ARRAY",
                 "$.kernels[0].fevals[0].coeff_sets[0]",
                 limits);

    std::string overflow = yaml;
    replace_once(overflow, "!!python/tuple [2]", "!!python/tuple [9223372036854775807, 3]");
    limits = hamilt::cider::LoaderLimits();
    limits.max_array_elements = std::numeric_limits<std::size_t>::max();
    expect_error(overflow,
                 "MALFORMED_ARRAY",
                 "$.kernels[0].fevals[0].coeff_sets[0].state[1]",
                 limits);

    limits = hamilt::cider::LoaderLimits();
    limits.max_yaml_expanded_bytes = 8U;
    expect_error(yaml, "YAML_SYNTAX", "$.kernels[0]._add_basefunc", limits);
}

TEST(NativeCiderModelLoader, DecodesWhitelistedFloat64AndInt64Scalars)
{
    std::string yaml
        = array_model("_core", "f8", "<", "false", "[2]", "AAAAAAAA8D8AAAAAAAAEwA==");
    replace_once(yaml,
                 "    ind_sets: []\n",
                 "    ind_sets:\n"
                 "    - !!python/tuple\n"
                 "      - !!python/object/apply:numpy._core.multiarray.scalar\n"
                 "        - !!python/object/apply:numpy.dtype\n"
                 "          args: [i8, false, true]\n"
                 "          state: !!python/tuple [3, '<', null, null, null, -1, -1, 0]\n"
                 "        - !!binary 'KgAAAAAAAAA='\n");
    replace_once(yaml,
                 "    _normalizers: []\n",
                 "    _normalizers:\n"
                 "    - !!python/object:ciderpress.dft.feat_normalizer.ConstantNormalizer\n"
                 "      const: !!python/object/apply:numpy._core.multiarray.scalar\n"
                 "      - !!python/object/apply:numpy.dtype\n"
                 "        args: [f8, false, true]\n"
                 "        state: !!python/tuple [3, '<', null, null, null, -1, -1, 0]\n"
                 "      - !!binary 'AAAAAAAA+D8='\n");
    const TempYaml input(yaml);
    const hamilt::cider::LegacyDocument document
        = hamilt::cider::ModelLoader::read_legacy_document(input.path());
    const hamilt::cider::LegacyNode& index
        = document.root.mapping_items.at("kernels")
              .sequence_items.at(0)
              .mapping_items.at("fevals")
              .sequence_items.at(0)
              .mapping_items.at("ind_sets")
              .sequence_items.at(0)
              .sequence_items.at(0);
    ASSERT_EQ(hamilt::cider::LegacyDType::Int64, index.numpy_value.dtype);
    ASSERT_EQ(1U, index.numpy_value.int64_values.size());
    EXPECT_EQ(42, index.numpy_value.int64_values[0]);

    const hamilt::cider::LegacyNode& constant
        = document.root.mapping_items.at("settings")
              .mapping_items.at("normalizers")
              .mapping_items.at("_normalizers")
              .sequence_items.at(0)
              .mapping_items.at("const");
    ASSERT_EQ(hamilt::cider::LegacyDType::Float64, constant.numpy_value.dtype);
    ASSERT_EQ(1U, constant.numpy_value.float64_values.size());
    EXPECT_DOUBLE_EQ(1.5, constant.numpy_value.float64_values[0]);
}

TEST(NativeCiderModelLoader, RejectsRecursiveAliasesAndMultipleDocuments)
{
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: &cycle\n- *cycle\nsettings: null\nlibxc_baseline: null\n",
                 "YAML_SYNTAX",
                 "$.kernels[0]");
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\nsettings: null\nlibxc_baseline: null\n"
                 "---\nnull\n",
                 "YAML_SYNTAX",
                 "<file>");
}

TEST(NativeCiderModelLoader, BinaryBlockWorkaroundIsNarrowAndRejectsMalformedBlocks)
{
    const TempYaml ordinary(
        "!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
        "# !!binary |\n"
        "kernels: []\nsettings: null\nlibxc_baseline: null\n"
        "feature_density_policy: 'literal !!binary |'\n");
    const hamilt::cider::LegacyDocument document
        = hamilt::cider::ModelLoader::read_legacy_document(ordinary.path());
    EXPECT_EQ("literal !!binary |",
              document.root.mapping_items.at("feature_density_policy").string_value);

    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\nsettings: !!binary |\nYg==\nlibxc_baseline: null\n",
                 "YAML_SYNTAX",
                 "<file>");
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\nsettings: !!binary |\n  !!!!\nlibxc_baseline: null\n",
                 "MALFORMED_ARRAY",
                 "$.settings");
}

TEST(NativeCiderModelLoader, BinaryMarkerTextInSequenceCommentsCannotConsumeNodes)
{
    for (const std::string& comment : {std::string("# note: !!binary |"),
                                       std::string("# note - !!binary |")})
    {
        SCOPED_TRACE(comment);
        std::string yaml
            = array_model("core", "f8", "<", "false", "[2]", "AAAAAAAA8D8AAAAAAAAEwA==");
        replace_once(yaml,
                     "- !!python/object:ciderpress.dft.xc_evaluator.MappedDFTKernel\n",
                     "- " + comment
                         + "\n  !!python/object:ciderpress.dft.xc_evaluator.MappedDFTKernel\n");
        const TempYaml input(yaml);
        const hamilt::cider::LegacyDocument document
            = hamilt::cider::ModelLoader::read_legacy_document(input.path());
        ASSERT_EQ(1U, document.root.mapping_items.at("kernels").sequence_items.size());
        EXPECT_EQ("python/object:ciderpress.dft.xc_evaluator.MappedDFTKernel",
                  document.root.mapping_items.at("kernels").sequence_items[0].tag);
    }

    const TempYaml block_text(
        "!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
        "kernels: []\nsettings: null\nlibxc_baseline: null\n"
        "feature_density_policy: |-\n"
        "  - !!binary |\n"
        "    Yg==\n");
    const hamilt::cider::LegacyDocument block_document
        = hamilt::cider::ModelLoader::read_legacy_document(block_text.path());
    const std::string& block_value
        = block_document.root.mapping_items.at("feature_density_policy").string_value;
    EXPECT_NE(std::string::npos, block_value.find("- !!binary |"));
    EXPECT_NE(std::string::npos, block_value.find("Yg=="));
}

TEST(NativeCiderModelLoader, RejectsPythonTaggedMappingKeysBeforeUsingTheirText)
{
    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "? !!python/name:numpy.ndarray 'kernels'\n"
                 ": []\n"
                 "settings: null\nlibxc_baseline: null\n",
                 "UNSUPPORTED_NUMPY_TAG",
                 "$.<key>");

    std::string nested
        = array_model("core", "f8", "<", "false", "[2]", "AAAAAAAA8D8AAAAAAAAEwA==");
    replace_once(nested,
                 "  mode: SEP\n",
                 "  ? !!python/object:evil.Key 'mode'\n"
                 "  : SEP\n");
    expect_error(nested, "UNKNOWN_TAG", "$.kernels[0].<key>");
}

TEST(NativeCiderModelLoader, RejectsNumpyTagsAtAdversarialNearMatchRoles)
{
    const std::string valid
        = array_model("core", "f8", "<", "false", "[2]", "AAAAAAAA8D8AAAAAAAAEwA==");
    const std::string tag = "!!python/object/apply:numpy.core.multiarray._reconstruct\n";
    const std::size_t array_start = valid.find(tag);
    const std::size_t array_end = valid.find("    const:", array_start);
    ASSERT_NE(std::string::npos, array_start);
    ASSERT_NE(std::string::npos, array_end);
    const std::string array_node = valid.substr(array_start, array_end - array_start);

    const std::string nested_array
        = "!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
          "kernels: []\n"
          "settings:\n"
          "  coeff_sets:\n"
          "  - nested: "
          + array_node
          + "libxc_baseline: null\n";
    expect_error(nested_array,
                 "UNSUPPORTED_NUMPY_TAG",
                 "$.settings.coeff_sets[0].nested");

    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\n"
                 "settings:\n"
                 "  args:\n"
                 "  - nested: !!python/name:numpy.ndarray ''\n"
                 "libxc_baseline: null\n",
                 "UNSUPPORTED_NUMPY_TAG",
                 "$.settings.args[0].nested");

    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\nsettings: !!python/tuple []\nlibxc_baseline: null\n",
                 "UNKNOWN_TAG",
                 "$.settings");

    expect_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                 "kernels: []\n"
                 "settings:\n"
                 "  ind_sets:\n"
                 "  - nested: !!python/object/apply:numpy.core.multiarray.scalar\n"
                 "    - !!python/object/apply:numpy.dtype\n"
                 "      args: [i8, false, true]\n"
                 "      state: !!python/tuple [3, '<', null, null, null, -1, -1, 0]\n"
                 "    - !!binary 'KgAAAAAAAAA='\n"
                 "libxc_baseline: null\n",
                 "UNSUPPORTED_NUMPY_TAG",
                 "$.settings.ind_sets[0].nested");
}

TEST(NativeCiderModelLoader, ValidatesWrongNumpyRoleBeforeDecodingEarlierPayload)
{
    std::string yaml = array_model("core", "f8", "<", "false", "[2]", "!!!!");
    const std::string dtype
        = "!!python/object/apply:numpy.dtype\n"
          "  args: [f8, false, true]\n"
          "  state: !!python/tuple [3, '<', null, null, null, -1, -1, 0]\n";
    const std::size_t settings = yaml.find("settings:");
    const std::size_t baseline = yaml.find("libxc_baseline:", settings);
    ASSERT_NE(std::string::npos, settings);
    ASSERT_NE(std::string::npos, baseline);
    yaml.replace(settings, baseline - settings, "settings: " + dtype);

    expect_error(yaml, "UNSUPPORTED_NUMPY_TAG", "$.settings");

    std::string malformed_dtype
        = array_model("core", "f8", ">", "false", "[1]", "AAAAAAAA8D8=");
    const std::string array_tag
        = "!!python/object/apply:numpy.core.multiarray._reconstruct\n";
    const std::size_t scale_start = malformed_dtype.find(array_tag);
    const std::size_t scale_end = malformed_dtype.find("    const:", scale_start);
    ASSERT_NE(std::string::npos, scale_start);
    ASSERT_NE(std::string::npos, scale_end);
    const std::string scale_array
        = malformed_dtype.substr(scale_start, scale_end - scale_start);

    std::string schema_ordering
        = array_model("core", "f8", "<", "false", "[2]", "!!!!");
    replace_once(schema_ordering, "    scale: [1.0]\n", "    scale: " + scale_array);
    expect_error(schema_ordering,
                 "MALFORMED_ARRAY",
                 "$.kernels[0].fevals[0].scale.state[2].state");
}

TEST(NativeCiderModelLoader, TraversesNormativeCorpusWhenParentIsProvided)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus smoke check";
    }
    const std::string files[] = {
        "functionals/CIDER23X_SL_GGA.yaml",
        "functionals/CIDER23X_NL_GGA.yaml",
        "functionals/CIDER23X_SL_MGGA.yaml",
        "functionals/CIDER23X_NL_MGGA_DTR.yaml",
        "abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/full/"
        "nldf_mgga_cand00_n96_noise0p01.yaml"};
    for (const std::string& relative : files)
    {
        SCOPED_TRACE(relative);
        const hamilt::cider::LegacyDocument document
            = hamilt::cider::ModelLoader::read_legacy_document(std::string(parent) + "/" + relative);
        EXPECT_EQ(hamilt::cider::LegacyNode::Kind::Mapping, document.root.kind);
        EXPECT_EQ("python/object:ciderpress.dft.xc_evaluator.MappedXC", document.root.tag);
    }
}

TEST(NativeCiderModelLoader, LoadsTypedModelWithAuthoritativeMetadata)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }

    hamilt::cider::ModelLoader loader;
    const std::unique_ptr<const hamilt::cider::Model> model
        = loader.load(std::string(parent) + "/functionals/CIDER23X_SL_GGA.yaml");
    const hamilt::cider::ModelMetadata& metadata = model->metadata();

    EXPECT_EQ(hamilt::cider::SemilocalLevel::Gga, metadata.semilocal_level());
    EXPECT_FALSE(metadata.requires_tau());
    EXPECT_FALSE(metadata.has_nldf());
    EXPECT_EQ(2U, metadata.raw_feature_count());
    EXPECT_EQ(1U, metadata.transformed_feature_count());
    EXPECT_EQ(hamilt::cider::FeatureDensityPolicy::Valence,
              metadata.feature_density_policy());
    EXPECT_EQ("default_valence", metadata.feature_density_policy_source());
    EXPECT_EQ(hamilt::cider::EnergyConvention::EnergyDensity,
              metadata.energy_convention());
    EXPECT_DOUBLE_EQ(1.0, metadata.kernel_output_scale());
    EXPECT_DOUBLE_EQ(1.0, metadata.ml_exchange_scale());
    EXPECT_DOUBLE_EQ(0.25, metadata.runtime_exchange_scale(0.25));

    ASSERT_EQ(2U, model->normalizers().size());
    EXPECT_EQ(hamilt::cider::NormalizerKind::Identity,
              model->normalizers()[0].kind());
    ASSERT_EQ(1U, model->feature_maps().size());
    EXPECT_EQ(hamilt::cider::FeatureMapKind::U, model->feature_maps()[0].kind());
    ASSERT_EQ(1U, model->spline_terms().size());
    EXPECT_EQ(1U, model->spline_terms()[0].dimension());
}

TEST(NativeCiderModelLoader, OracleSha256HelperMatchesKnownVector)
{
    const std::vector<std::uint8_t> abc = {'a', 'b', 'c'};
    EXPECT_EQ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
              cider_test::detail::sha256(abc));
}

TEST(NativeCiderModelLoader, CorpusMetadataMatchesV1Oracle)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    struct Case
    {
        const char* path;
        bool mgga;
        bool nldf;
        std::size_t raw_count;
        std::size_t transformed_count;
        hamilt::cider::FeatureDensityPolicy density_policy;
        const char* density_source;
        bool labels_specified;
    };
    const Case cases[] = {
        {"functionals/CIDER23X_SL_GGA.yaml", false, false, 2U, 1U,
         hamilt::cider::FeatureDensityPolicy::Valence, "default_valence", false},
        {"functionals/CIDER23X_SL_MGGA.yaml", true, false, 3U, 2U,
         hamilt::cider::FeatureDensityPolicy::Valence, "default_valence", false},
        {"functionals/CIDER23X_NL_GGA.yaml", false, true, 5U, 4U,
         hamilt::cider::FeatureDensityPolicy::ValencePseudoCore,
         "inferred_legacy_cider23x", false},
        {"functionals/CIDER23X_NL_MGGA_DTR.yaml", true, true, 6U, 5U,
         hamilt::cider::FeatureDensityPolicy::ValencePseudoCore,
         "inferred_legacy_cider23x", false},
        {"abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/full/"
         "nldf_mgga_cand00_n96_noise0p01.yaml", true, true, 6U, 5U,
         hamilt::cider::FeatureDensityPolicy::Valence, "abacus_training", true}};

    const hamilt::cider::ModelLoader loader;
    for (const Case& item : cases)
    {
        SCOPED_TRACE(item.path);
        const hamilt::cider::ModelMetadata metadata
            = loader.inspect_metadata(std::string(parent) + "/" + item.path);
        EXPECT_EQ(item.mgga, metadata.is_mgga());
        EXPECT_EQ(item.mgga, metadata.requires_tau());
        EXPECT_EQ(item.nldf, metadata.has_nldf());
        EXPECT_EQ(item.raw_count, metadata.raw_feature_count());
        EXPECT_EQ(item.raw_count, metadata.normalized_feature_count());
        EXPECT_EQ(item.transformed_count, metadata.transformed_feature_count());
        EXPECT_EQ(item.density_policy, metadata.feature_density_policy());
        EXPECT_EQ(item.density_source, metadata.feature_density_policy_source());
        EXPECT_EQ(item.labels_specified, metadata.training_label_metadata().specified());
        EXPECT_DOUBLE_EQ(1.0, metadata.kernel_output_scale());
        EXPECT_DOUBLE_EQ(1.0, metadata.ml_exchange_scale());
        EXPECT_EQ(hamilt::cider::EnergyConvention::EnergyDensity,
                  metadata.energy_convention());
    }
}

TEST(NativeCiderModelLoader, CorpusSplineStructureHashesAndSamplesMatchV1Oracle)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    struct Case
    {
        const char* path;
        std::size_t term_count;
        std::size_t maximum_dimension;
        std::size_t scale_offset;
        std::size_t dimension_offset;
        std::size_t index_offset;
        std::size_t grid_offset;
        std::size_t sample_offset;
        std::vector<std::string> coefficient_sha256;
    };
    const Case cases[] = {
        {"functionals/CIDER23X_SL_GGA.yaml", 1U, 1U, 0U, 8U, 12U, 16U, 40U,
         {"201e3d7fde044e8bbbfba692352868f370171b80456764079ec65244a311687a"}},
        {"functionals/CIDER23X_SL_MGGA.yaml", 1U, 2U, 2016U, 2024U, 2028U,
         2036U, 2084U,
         {"4cc5b4fd269dcc4366bfe1e4f797158d0422d8aae8302508c45a6e3bc900145c"}},
        {"functionals/CIDER23X_NL_GGA.yaml", 7U, 3U, 7644U, 7700U, 7728U,
         7812U, 8316U,
         {"38bc3a7464ac5409ed8c1054b72f49d92a9744f1a41a583a5e31e1dcd925ed3d",
          "d0e9c27b3756f6f0672819c4a3b690c99b8eec2f0547026c347f39b5091634e5",
          "d58ba4743f6e234935b7db6eb8e9a2586ae51e6501e1ad17eb6700b473bcb798",
          "98f44836fdaedf66eefee3b2cb1f0e027482f9348c611d856613e9e8207e1623",
          "751ae682782477839deceec02647040a56c14b40f1aa14e21bc8739fd8804706",
          "07d8f7cee565fc7473984a8ed897a10b3e4761bc39bd146b472913d697d145c0",
          "c9978a824696534ff7e1076c1f5069e2aca47514fb2e2f1c378b107baee97d34"}},
        {"functionals/CIDER23X_NL_MGGA_DTR.yaml", 11U, 3U, 16484U, 16572U,
         16616U, 16748U, 17540U,
         {"23d95a67fcb377844adc0cbb4e12da14c7a456bcb6ebd8d435aba018dab2507a",
          "65b3bba09867f5cb35ada71602443715ffabbd715be70b68b7dc1827b47d7f18",
          "28c0a2f22e212a64f38f4a7815f990088a4b8adf75ee8cdcf98922e135645585",
          "1f42ee79f473e5f3cca1f2a9816085c049874b69662ffb80adcc07a254906967",
          "cb9cc7292c9b9b00ec502fb798d2deb4a7c2661e3debc539093c23d1503cfd5e",
          "a08accc080b714e7365d91fc375cd0e81a8c142917bf82a4265b6b40f9a693cf",
          "795ad8fb16aa52c331c92730155a10c7f9f8c6d1a8a445584148ff3f6968740e",
          "b77cd433093049f368b17236adae24957688820f1c7bdd0be37cc55920ca7f32",
          "f2db13dedca035e9813826f3f6977e4f84909b907dbbdb1e46e1fa135f422c08",
          "b1e679ce63c9ccd611f22129f7c406e783d1b31c5e51ab924f761aa55458a77c",
          "a234af57d9d742726e2f18dfdbb672918230fe9c7ac35de798fb131387986b7f"}},
        {"abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/"
         "full/nldf_mgga_cand00_n96_noise0p01.yaml",
         11U, 3U, 23116U, 23204U, 23248U, 23380U, 24172U,
         {"d08dd0ed390634883affa1ae570a4885a29a7f790faf97eb2ddbb3c5873ea3f1",
          "19e4537c097c886caeebb67b0b9aa9f1d4afd288359b3ed46deea9a9b0245372",
          "40c61d725ed4acc3f7ead53653aef8f47e2778e6d0bd9e40053bb504750d3117",
          "89436c0d286d03fd0e7a1246614b7371f37d46cc595436506e5b3664bdb53c82",
          "3c4482e148d21295b2eba0e91b295a8e6b4c1ebc05c985f5aa01903c8ced21f2",
          "7d12f87dde47ae012a9f9d99ea052573051fee56caa7cce3c6e5f19b28d6f943",
          "67acf5fe8c2a77caf58b5ceb3678e52d089af544754c8d7a407f9c8e91f370b8",
          "8f39be095cb689ce114684a526faa094dad5925778af638c527d5f0b8e04d791",
          "30e5af6ae8446edc8f4bb0037148e793d33bbd6b395ecb11eec53fddc78d79e9",
          "ae295bd2cdd5de1d01e6e8be9bd332f8cc5461c3402df86bd52f86c1c6401ad5",
          "8171799f7b2e7353c91110e745374ccb06eabc1fa4671824239a5e881502afd9"}}};

    const hamilt::cider::ModelLoader loader;
    for (const Case& item : cases)
    {
        SCOPED_TRACE(item.path);
        const std::unique_ptr<const hamilt::cider::Model> model
            = loader.load(std::string(parent) + "/" + item.path);
        ASSERT_EQ(item.term_count, model->spline_terms().size());
        ASSERT_EQ(item.term_count, item.coefficient_sha256.size());
        const std::vector<double> expected_scales
            = read_fixture_array<double>(item.scale_offset, item.term_count);
        const std::vector<std::int32_t> expected_dimensions
            = read_fixture_array<std::int32_t>(item.dimension_offset, item.term_count);
        const std::vector<std::int32_t> expected_indices
            = read_fixture_array<std::int32_t>(
                item.index_offset, item.term_count * item.maximum_dimension);
        const std::vector<double> expected_grids = read_fixture_array<double>(
            item.grid_offset, item.term_count * item.maximum_dimension * 3U);
        const std::vector<double> expected_samples
            = read_fixture_array<double>(item.sample_offset, item.term_count * 9U);

        for (std::size_t term_index = 0; term_index < item.term_count; ++term_index)
        {
            SCOPED_TRACE(term_index);
            const hamilt::cider::SplineTerm& term = model->spline_terms()[term_index];
            ASSERT_GT(expected_dimensions[term_index], 0);
            const std::size_t dimension
                = static_cast<std::size_t>(expected_dimensions[term_index]);
            ASSERT_LE(dimension, item.maximum_dimension);
            EXPECT_DOUBLE_EQ(expected_scales[term_index], term.scale());
            ASSERT_EQ(dimension, term.dimension());
            ASSERT_EQ(dimension, term.descriptor_indices().size());
            ASSERT_EQ(dimension, term.grids().size());
            ASSERT_EQ(dimension, term.coefficient_extents().size());

            std::size_t coefficient_count = 1U;
            for (std::size_t axis = 0; axis < dimension; ++axis)
            {
                const std::size_t flat = term_index * item.maximum_dimension + axis;
                ASSERT_GE(expected_indices[flat], 0);
                EXPECT_EQ(static_cast<std::size_t>(expected_indices[flat]),
                          term.descriptor_indices()[axis]);
                const std::size_t grid_flat = flat * 3U;
                EXPECT_DOUBLE_EQ(expected_grids[grid_flat], term.grids()[axis].lower());
                EXPECT_DOUBLE_EQ(expected_grids[grid_flat + 1U], term.grids()[axis].upper());
                ASSERT_GE(expected_grids[grid_flat + 2U], 0.0);
                const std::size_t order
                    = static_cast<std::size_t>(expected_grids[grid_flat + 2U]);
                EXPECT_EQ(order, term.grids()[axis].order());
                EXPECT_EQ(order + 2U, term.coefficient_extents()[axis]);
                coefficient_count *= term.coefficient_extents()[axis];
            }
            ASSERT_EQ(coefficient_count, term.coefficients().size());
            EXPECT_EQ(item.coefficient_sha256[term_index],
                      cider_test::sha256_little_endian_doubles(term.coefficients()));
            for (std::size_t sample = 0; sample < 9U; ++sample)
            {
                const std::size_t coefficient_index
                    = sample * (coefficient_count - 1U) / 8U;
                EXPECT_DOUBLE_EQ(expected_samples[term_index * 9U + sample],
                                 term.coefficients()[coefficient_index])
                    << "coefficient sample " << sample;
            }
        }
    }
}

TEST(NativeCiderModelLoader, MinimalTypedModelUsesDocumentedDefaults)
{
    const TempYaml input(typed_model());
    const std::unique_ptr<const hamilt::cider::Model> model
        = hamilt::cider::ModelLoader().load(input.path());
    EXPECT_DOUBLE_EQ(1.0, model->metadata().kernel_output_scale());
    EXPECT_DOUBLE_EQ(1.0, model->metadata().ml_exchange_scale());
    EXPECT_FALSE(model->metadata().training_label_metadata().specified());
    EXPECT_EQ(hamilt::cider::FeatureDensityPolicy::Valence,
              model->metadata().feature_density_policy());
    EXPECT_EQ("default_valence", model->metadata().feature_density_policy_source());
}

TEST(NativeCiderModelLoader, RetainsTypedCorpusGraphWithoutYamlTypes)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    const hamilt::cider::ModelLoader loader;
    const std::unique_ptr<const hamilt::cider::Model> mgga
        = loader.load(std::string(parent) + "/functionals/CIDER23X_SL_MGGA.yaml");
    ASSERT_EQ(2U, mgga->feature_maps().size());
    EXPECT_EQ(hamilt::cider::FeatureMapKind::Z, mgga->feature_maps()[1].kind());
    EXPECT_DOUBLE_EQ(-1.0, mgga->feature_maps()[1].lower_bound());
    EXPECT_DOUBLE_EQ(1.0, mgga->feature_maps()[1].upper_bound());

    const std::unique_ptr<const hamilt::cider::Model> nldf
        = loader.load(std::string(parent) + "/functionals/CIDER23X_NL_GGA.yaml");
    ASSERT_NE(nullptr, nldf->nldf_settings());
    EXPECT_EQ(3U, nldf->nldf_settings()->feature_count());
    ASSERT_EQ(5U, nldf->normalizers().size());
    EXPECT_EQ(hamilt::cider::NormalizerKind::Constant, nldf->normalizers()[2].kind());
    ASSERT_EQ(4U, nldf->feature_maps().size());
    EXPECT_EQ(hamilt::cider::FeatureMapKind::V, nldf->feature_maps()[1].kind());
    ASSERT_EQ(7U, nldf->spline_terms().size());
    EXPECT_EQ(3U, nldf->spline_terms()[4].dimension());
    EXPECT_EQ(27U, nldf->spline_terms()[4].coefficient_extents()[0]);
}

TEST(NativeCiderModelLoader, RejectsRenamedLegacyNldfWithoutExplicitDensityPolicy)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    expect_typed_error(read_file(std::string(parent) + "/functionals/CIDER23X_NL_GGA.yaml"),
                       "DENSITY_POLICY_REQUIRED",
                       "$.feature_density_policy");
}

TEST(NativeCiderModelLoader, ResolvesOptionalRuntimeAndDensityMetadata)
{
    std::string yaml = typed_model();
    replace_once(yaml,
                 "kernels:\n",
                 "feature_density_policy: total\n"
                 "feature_density_policy_source: ''\n"
                 "ml_exchange_scale: null\n"
                 "kernels:\n");
    replace_once(yaml, "  mode: SEP\n", "  mode: SEP\n  output_scale: 0.5\n");
    const TempYaml input(yaml);
    const hamilt::cider::ModelMetadata metadata
        = hamilt::cider::ModelLoader().inspect_metadata(input.path());
    EXPECT_EQ(hamilt::cider::FeatureDensityPolicy::ValencePseudoCore,
              metadata.feature_density_policy());
    EXPECT_EQ("model", metadata.feature_density_policy_source());
    EXPECT_DOUBLE_EQ(0.5, metadata.kernel_output_scale());
    EXPECT_DOUBLE_EQ(0.0, metadata.ml_exchange_scale());
    EXPECT_DOUBLE_EQ(0.25, metadata.runtime_exchange_scale(0.25));
}

TEST(NativeCiderModelLoader, RejectsSyntheticFeatureCountAndSplineInvariants)
{
    struct Case
    {
        const char* from;
        const char* to;
        const char* code;
        const char* path;
    };
    const Case cases[] = {
        {"_normalizers: [null, null]", "_normalizers: [null]", "FEATURE_COUNT",
         "$.settings.normalizers._normalizers"},
        {"      i: 1", "      i: 2", "FEATURE_COUNT",
         "$.kernels[0].feature_list.feat_list[0].i"},
        {"    nterms: 1", "    nterms: 2", "FEATURE_COUNT",
         "$.kernels[0].fevals[0].scale"},
        {"    - [0]", "    - []", "FEATURE_COUNT",
         "$.kernels[0].fevals[0].ind_sets[0]"},
        {"    - [0]", "    - [0, 0, 0, 0]", "FEATURE_COUNT",
         "$.kernels[0].fevals[0].ind_sets[0]"},
        {"    - [0]", "    - [0, 0]", "FEATURE_COUNT",
         "$.kernels[0].fevals[0].ind_sets[0][1]"},
        {"    - [[0.0, 1.0, 1]]", "    - []", "FEATURE_COUNT",
         "$.kernels[0].fevals[0].spline_grids[0]"},
        {"[[0.0, 1.0, 1]]", "[[0.0, 0.5, 1]]", "FEATURE_COUNT",
         "$.kernels[0].fevals[0].spline_grids[0][0]"},
        {"[[0.0, 1.0, 1]]", "[[0.0, 1.0, 2]]", "MALFORMED_ARRAY",
         "$.kernels[0].fevals[0].coeff_sets[0]"}};
    for (const Case& item : cases)
    {
        SCOPED_TRACE(std::string(item.from) + " -> " + item.to);
        std::string yaml = typed_model();
        replace_once(yaml, item.from, item.to);
        expect_typed_error(yaml, item.code, item.path);
    }
}

TEST(NativeCiderModelLoader, RejectsSyntheticSettingsAndMetadataInvariants)
{
    struct Case
    {
        const char* insertion;
        const char* code;
        const char* path;
    };
    const Case cases[] = {
        {"feature_density_policy_source: orphan\n", "DENSITY_POLICY_INVALID",
         "$.feature_density_policy_source"},
        {"feature_density_policy: unsupported\n", "DENSITY_POLICY_INVALID",
         "$.feature_density_policy"},
        {"ml_exchange_scale: -0.1\n", "SCHEMA", "$.ml_exchange_scale"},
        {"ml_exchange_label_convention: full_exact_exchange\n", "SCHEMA",
         "$.ml_exchange_label_hybrid_scaled"}};
    for (const Case& item : cases)
    {
        SCOPED_TRACE(item.insertion);
        std::string yaml = typed_model();
        replace_once(yaml, "kernels:\n", std::string(item.insertion) + "kernels:\n");
        expect_typed_error(yaml, item.code, item.path);
    }

    std::string mode = typed_model();
    replace_once(mode, "    mode: np\n", "    mode: npa\n");
    expect_typed_error(mode, "UNSUPPORTED_DESCRIPTOR", "$.settings.sl_settings");

    std::string normalizer_mode = typed_model();
    replace_once(normalizer_mode, "    slmode: np\n", "    slmode: npa\n");
    expect_typed_error(normalizer_mode, "FEATURE_COUNT", "$.settings.normalizers.slmode");
}

TEST(NativeCiderModelLoader, RejectsKernelEvaluatorModeAndCoefficientRankInvariants)
{
    expect_typed_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                       "kernels: []\nsettings: null\nlibxc_baseline: null\n",
                       "UNSUPPORTED_EVALUATOR",
                       "$.kernels");
    expect_typed_error("!!python/object:ciderpress.dft.xc_evaluator.MappedXC\n"
                       "kernels: [{}, {}]\nsettings: null\nlibxc_baseline: null\n",
                       "UNSUPPORTED_EVALUATOR",
                       "$.kernels");

    std::string mode = typed_model();
    replace_once(mode, "  mode: SEP\n", "  mode: POL\n");
    expect_typed_error(mode, "SCHEMA", "$.kernels[0].mode");

    std::string evaluator = typed_model();
    replace_between(evaluator, "  fevals:\n", "  mode: SEP\n", "  fevals: []\n");
    expect_typed_error(evaluator, "UNSUPPORTED_EVALUATOR", "$.kernels[0].fevals");

    std::string rank = typed_model();
    replace_once(rank,
                 "      _bounds: [0.0, 1.0]\n",
                 "      _bounds: [0.0, 1.0]\n"
                 "    - !!python/object:ciderpress.dft.transform_data.UMap\n"
                 "      i: 0\n"
                 "      gamma: 1.0\n"
                 "      _bounds: [0.0, 1.0]\n");
    replace_once(rank, "    - [0]\n", "    - [0, 1]\n");
    replace_once(rank,
                 "    - [[0.0, 1.0, 1]]\n",
                 "    - [[0.0, 1.0, 1], [0.0, 1.0, 1]]\n");
    expect_typed_error(rank, "MALFORMED_ARRAY", "$.kernels[0].fevals[0].coeff_sets[0]");
}

TEST(NativeCiderModelLoader, RejectsUnsupportedVersionJSubsets)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    const std::string valid
        = read_file(std::string(parent) + "/functionals/CIDER23X_NL_GGA.yaml");
    struct Case
    {
        const char* from;
        const char* to;
        const char* code;
        const char* path;
    };
    const Case cases[] = {
        {"    _sl_level: GGA", "    _sl_level: MGGA", "UNSUPPORTED_DESCRIPTOR",
         "$.settings.nldf_settings._sl_level"},
        {"    rho_mult: one", "    rho_mult: expnt", "UNSUPPORTED_DESCRIPTOR",
         "$.settings.nldf_settings.rho_mult"},
        {"    l0_feat_specs: []", "    l0_feat_specs: [se]", "UNSUPPORTED_DESCRIPTOR",
         "$.settings.nldf_settings.l0_feat_specs"},
        {"    theta_params:\n    - 0.25", "    theta_params:\n    - 0.0",
         "UNSUPPORTED_DESCRIPTOR", "$.settings.nldf_settings.theta_params"},
        {"    feat_specs:\n    - se\n    - se\n    - se",
         "    feat_specs:\n    - other\n    - se\n    - se",
         "UNSUPPORTED_DESCRIPTOR", "$.settings.nldf_settings.feat_specs[0]"},
        {"    feat_specs:\n    - se\n    - se\n    - se",
         "    feat_specs:\n    - se\n    - se",
         "FEATURE_COUNT", "$.settings.nldf_settings.feat_params"}};
    for (const Case& item : cases)
    {
        SCOPED_TRACE(std::string(item.from) + " -> " + item.to);
        std::string yaml = valid;
        replace_once(yaml, item.from, item.to);
        expect_typed_error(yaml, item.code, item.path);
    }
}

TEST(NativeCiderModelLoader, RetainsCompleteTrainingLabelMetadata)
{
    const char* parent = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (parent == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the read-only corpus check";
    }
    const hamilt::cider::ModelMetadata metadata = hamilt::cider::ModelLoader().inspect_metadata(
        std::string(parent)
        + "/abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/"
          "full/nldf_mgga_cand00_n96_noise0p01.yaml");
    const hamilt::cider::TrainingLabelMetadata& labels = metadata.training_label_metadata();
    EXPECT_TRUE(labels.specified());
    EXPECT_EQ("full_exact_exchange", labels.convention());
    EXPECT_FALSE(labels.hybrid_scaled());
    EXPECT_DOUBLE_EQ(1.0, labels.scale());
    ASSERT_EQ(1U, labels.source_scales().size());
    EXPECT_DOUBLE_EQ(1.0, labels.source_scales()[0]);
}

} // namespace
