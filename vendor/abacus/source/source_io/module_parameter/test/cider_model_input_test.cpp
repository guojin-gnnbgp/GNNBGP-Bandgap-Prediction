#include "source_io/module_parameter/cider_model_input.h"
#include "source_io/module_parameter/input_parameter.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

class TemporaryModel
{
  public:
    TemporaryModel(const std::string& source, const std::string& name)
    {
        path_ = std::string("/tmp/") + name;
        std::ifstream input(source.c_str(), std::ios::binary);
        if (!input)
        {
            throw std::runtime_error("cannot open test model " + source);
        }
        std::ofstream output(path_.c_str(), std::ios::binary | std::ios::trunc);
        output << input.rdbuf();
        if (!output)
        {
            throw std::runtime_error("cannot copy test model to " + path_);
        }
    }

    ~TemporaryModel() { std::remove(path_.c_str()); }

    const std::string& path() const { return path_; }

  private:
    std::string path_;
};

std::string corpus_model(const std::string& relative)
{
    const char* root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (root == nullptr || *root == '\0')
    {
        throw std::runtime_error("CIDER_NATIVE_CORPUS_ROOT is not set");
    }
    return std::string(root) + "/" + relative;
}

} // namespace

TEST(CiderModelInput, UsesModelMetadataInsteadOfFilenameForTau)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr || *corpus_root == '\0')
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for the metadata replay";
    }
    const TemporaryModel renamed_mgga(
        corpus_model("functionals/CIDER23X_SL_MGGA.yaml"),
        "abacus_cider_renamed_model.yaml");
    const ModuleIO::CiderModelInputMetadata mgga
        = ModuleIO::inspect_cider_model_input(renamed_mgga.path());
    EXPECT_TRUE(mgga.requires_tau);

    const TemporaryModel misleading_gga(
        corpus_model("functionals/CIDER23X_SL_GGA.yaml"),
        "abacus_cider_looks_MGGA.yaml");
    const ModuleIO::CiderModelInputMetadata gga
        = ModuleIO::inspect_cider_model_input(misleading_gga.path());
    EXPECT_FALSE(gga.requires_tau);
}

TEST(CiderModelInput, ReportsJoblibAndNonYamlArtifactsActionably)
{
    for (const std::string& path : {"missing.joblib", "missing.pkl", "missing.pickle"})
    {
        try
        {
            ModuleIO::inspect_cider_model_input(path);
            FAIL() << "expected inspection failure for " << path;
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message(error.what());
            EXPECT_NE(std::string::npos, message.find("model='" + path + "'"));
            EXPECT_NE(std::string::npos, message.find("code=JOBLIB_UNSUPPORTED"));
            EXPECT_NE(std::string::npos, message.find("yaml_path=<file>"));
            EXPECT_NE(std::string::npos, message.find("reason="));
        }
    }

    try
    {
        ModuleIO::inspect_cider_model_input("missing.txt");
        FAIL() << "expected non-YAML inspection failure";
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message(error.what());
        EXPECT_NE(std::string::npos, message.find("model='missing.txt'"));
        EXPECT_NE(std::string::npos, message.find("code=ARTIFACT_UNSUPPORTED"));
        EXPECT_NE(std::string::npos, message.find("yaml_path=<file>"));
        EXPECT_NE(std::string::npos, message.find("reason=expected a .yaml or .yml artifact"));
    }
}

TEST(CiderModelInput, PreservesNativeYamlFailureDetails)
{
    const std::string path = "/tmp/abacus_cider_malformed.yaml";
    {
        std::ofstream output(path.c_str(), std::ios::trunc);
        output << "root: [unterminated\n";
    }
    try
    {
        ModuleIO::inspect_cider_model_input(path);
        FAIL() << "expected malformed YAML failure";
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message(error.what());
        EXPECT_NE(std::string::npos, message.find("model='" + path + "'"));
        EXPECT_NE(std::string::npos, message.find("code=YAML_SYNTAX"));
        EXPECT_NE(std::string::npos, message.find("yaml_path=<file>"));
        EXPECT_NE(std::string::npos, message.find("reason="));
    }
    std::remove(path.c_str());
}

TEST(CiderModelInput, ReportsUnsupportedYamlFamilyWithNativeNodePath)
{
    const std::string path = "/tmp/abacus_cider_unsupported.yaml";
    {
        std::ofstream output(path.c_str(), std::ios::trunc);
        output << "!!python/object:ciderpress.dft.xc_evaluator.MappedXC2 {}\n";
    }
    try
    {
        ModuleIO::inspect_cider_model_input(path);
        FAIL() << "expected unsupported YAML family failure";
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message(error.what());
        EXPECT_NE(std::string::npos, message.find("model='" + path + "'"));
        EXPECT_NE(std::string::npos, message.find("code=UNSUPPORTED_ROOT"));
        EXPECT_NE(std::string::npos, message.find("yaml_path=$"));
        EXPECT_NE(std::string::npos, message.find("reason=unsupported evaluator tag"));
    }
    std::remove(path.c_str());
}

TEST(CiderModelInput, ResolvesAutoAndRejectsExplicitDensityConflict)
{
    const ModuleIO::CiderModelInputMetadata metadata(
        false,
        "valence_pseudo_core",
        "model");
    EXPECT_EQ("valence_pseudo_core",
              ModuleIO::resolve_cider_feature_density("auto", metadata, "model.yaml"));
    EXPECT_EQ("valence_pseudo_core",
              ModuleIO::resolve_cider_feature_density("total", metadata, "model.yaml"));
    EXPECT_EQ("valence_pseudo_core",
              ModuleIO::resolve_cider_feature_density("VALENCE_PSEUDO_CORE", metadata, "model.yaml"));

    try
    {
        ModuleIO::resolve_cider_feature_density("valence", metadata, "model.yaml");
        FAIL() << "expected density-policy conflict";
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message(error.what());
        EXPECT_NE(std::string::npos, message.find("model='model.yaml'"));
        EXPECT_NE(std::string::npos, message.find("requested=valence"));
        EXPECT_NE(std::string::npos, message.find("expected=valence_pseudo_core"));
        EXPECT_NE(std::string::npos, message.find("source=model"));
    }


    EXPECT_THROW(
        ModuleIO::resolve_cider_feature_density("core", metadata, "model.yaml"),
        std::invalid_argument);
}

TEST(CiderModelInput, ResolvesBackendAgainstCompiledCapabilities)
{
    const ModuleIO::CiderBackendCapabilities native_only(true, false);
    EXPECT_EQ("native", ModuleIO::resolve_cider_backend("auto", native_only));
    EXPECT_EQ("native", ModuleIO::resolve_cider_backend("NATIVE", native_only));
    EXPECT_THROW(ModuleIO::resolve_cider_backend("external", native_only),
                 std::runtime_error);

    const ModuleIO::CiderBackendCapabilities dual(true, true);
    EXPECT_EQ("external", ModuleIO::resolve_cider_backend("auto", dual));
    EXPECT_EQ("external", ModuleIO::resolve_cider_backend("EXTERNAL", dual));
    EXPECT_EQ("native", ModuleIO::resolve_cider_backend("native", dual));
    EXPECT_THROW(ModuleIO::resolve_cider_backend("fallback", dual),
                 std::runtime_error);
}

TEST(CiderModelInput, DefaultsBackendSelectionToAuto)
{
    Input_para input;
    EXPECT_EQ("auto", input.cider_backend);
}
