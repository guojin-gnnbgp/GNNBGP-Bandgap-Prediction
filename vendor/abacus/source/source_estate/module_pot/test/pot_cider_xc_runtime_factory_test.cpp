#include "source_estate/module_pot/pot_cider_xc_runtime_factory.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

TEST(PotCiderRuntimeFactory, SelectsOnlyResolvedCompiledBackends)
{
    const char* corpus_root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    if (corpus_root == nullptr)
    {
        GTEST_SKIP() << "set CIDER_NATIVE_CORPUS_ROOT for runtime construction";
    }
    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    const auto make_runtime = [&](const std::string& backend) {
        return elecstate::make_pot_cider_runtime(
            backend,
            std::string(corpus_root) + "/functionals/CIDER23X_SL_GGA.yaml",
            shape,
            cell,
            1,
            0.25);
    };

    EXPECT_STREQ("native", make_runtime("native")->backend_name());
#ifdef ENABLE_CIDER_EXTERNAL
    EXPECT_STREQ("external", make_runtime("external")->backend_name());
#else
    EXPECT_THROW(make_runtime("external"), std::runtime_error);
#endif
    EXPECT_THROW(make_runtime("auto"), std::invalid_argument);
    EXPECT_THROW(make_runtime("fallback"), std::invalid_argument);
}
