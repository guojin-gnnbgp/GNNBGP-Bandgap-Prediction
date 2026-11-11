#include "source_hamilt/module_xc/cider/evaluator.h"
#include "source_hamilt/module_xc/cider/mapped_evaluator.h"
#include "source_hamilt/module_xc/cider/model.h"
#include "source_hamilt/module_xc/cider/model_loader.h"
#include "source_hamilt/module_xc/cider/nldf_backend.h"
#include "source_hamilt/module_xc/cider/nldf_plan.h"
#include "source_hamilt/module_xc/cider/semilocal_plan.h"

#include <gtest/gtest.h>

#include <type_traits>

namespace
{

TEST(NativeCiderBoundary, ResponsibilitiesAreIndependentInterfaces)
{
    EXPECT_TRUE(std::is_abstract<hamilt::cider::Model>::value);
    EXPECT_FALSE(std::is_abstract<hamilt::cider::ModelLoader>::value);
    EXPECT_TRUE(std::is_abstract<hamilt::cider::MappedEvaluator>::value);
    EXPECT_FALSE(std::is_abstract<hamilt::cider::SemilocalPlan>::value);
    EXPECT_FALSE(std::is_abstract<hamilt::cider::NldfPlan>::value);
    EXPECT_FALSE(std::is_abstract<hamilt::cider::NldfBackend>::value);
    EXPECT_TRUE(std::is_abstract<hamilt::cider::Evaluator>::value);
}

} // namespace
