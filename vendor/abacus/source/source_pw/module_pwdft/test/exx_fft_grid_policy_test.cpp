#include "source_basis/module_pw/pw_basis.h"
#include "source_pw/module_pwdft/exx_fft_grid_policy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace
{
hamilt::ExxSymmetryGridOperation zno_operation()
{
    hamilt::ExxSymmetryGridOperation operation;
    operation.symop = 6;
    operation.rotation = ModuleBase::Matrix3(0, -1, 0,
                                             1, 1, 0,
                                             0, 0, 1);
    operation.translation = ModuleBase::Vector3<double>(1.0 / 3.0, -1.0 / 3.0, -0.5);
    return operation;
}
} // namespace

TEST(ExxFftGridPolicy, ReportsRotationRatioFailure)
{
    hamilt::ExxSymmetryGridOperation operation;
    operation.symop = 4;
    operation.rotation = ModuleBase::Matrix3(0, 1, 0,
                                             1, 0, 0,
                                             0, 0, 1);

    const hamilt::ExxFftGridCompatibility compatibility
        = hamilt::check_exx_fft_grid_compatibility(hamilt::ExxFftGrid(24, 30, 36), {operation});

    ASSERT_FALSE(compatibility.compatible);
    ASSERT_FALSE(compatibility.failures.empty());
    EXPECT_EQ(compatibility.failures[0].condition, hamilt::ExxFftGridCondition::rotation_ratio);
    EXPECT_EQ(compatibility.failures[0].symop, 4);
    EXPECT_EQ(compatibility.failures[0].target_axis, 0);
    EXPECT_EQ(compatibility.failures[0].source_axis, 1);
    EXPECT_DOUBLE_EQ(compatibility.failures[0].value, 24.0 / 30.0);
}

TEST(ExxFftGridPolicy, ReportsZnOTranslationFailures)
{
    const hamilt::ExxFftGridCompatibility compatibility
        = hamilt::check_exx_fft_grid_compatibility(hamilt::ExxFftGrid(32, 32, 50), {zno_operation()});

    ASSERT_FALSE(compatibility.compatible);
    ASSERT_EQ(compatibility.failures.size(), 2u);
    EXPECT_EQ(compatibility.failures[0].condition, hamilt::ExxFftGridCondition::translation);
    EXPECT_EQ(compatibility.failures[0].symop, 6);
    EXPECT_EQ(compatibility.failures[0].target_axis, 0);
    EXPECT_NEAR(compatibility.failures[0].value, 32.0 / 3.0, 1.0e-12);
    EXPECT_EQ(compatibility.failures[1].target_axis, 1);
    EXPECT_NEAR(compatibility.failures[1].value, -32.0 / 3.0, 1.0e-12);
}

TEST(ExxFftGridPolicy, AcceptsSymmetryPrecisionNoiseOnCommensurateTranslation)
{
    hamilt::ExxSymmetryGridOperation operation;
    operation.symop = 24;
    operation.rotation = ModuleBase::Matrix3(-1, 0, 0,
                                             0, -1, 0,
                                             0, 0, -1);
    operation.translation = ModuleBase::Vector3<double>(0.25000029858165368,
                                                        -0.24999990047217582,
                                                        -0.24999970141834638);

    const hamilt::ExxFftGridCompatibility compatibility
        = hamilt::check_exx_fft_grid_compatibility(hamilt::ExxFftGrid(72, 72, 72), {operation});

    EXPECT_TRUE(compatibility.compatible);
    EXPECT_TRUE(compatibility.failures.empty());
}

TEST(ExxFftGridPolicy, SelectsSmallestCompatibleZnOGrid)
{
    hamilt::ExxFftGridPolicyInput input;
    input.initial_grid = hamilt::ExxFftGrid(32, 32, 50);
    input.operations = {zno_operation()};
    input.max_grid_points = 100000;
    input.max_candidate_trials = hamilt::EXX_FFT_GRID_MAX_CANDIDATE_TRIALS;

    const hamilt::ExxFftGridPolicyResult result = hamilt::choose_exx_fft_grid(input);

    ASSERT_TRUE(result.found);
    EXPECT_TRUE(result.adjusted);
    EXPECT_EQ(result.grid, hamilt::ExxFftGrid(36, 36, 50));
    EXPECT_EQ(result.grid_points, 64800u);
    EXPECT_EQ(result.candidate_trials, 9u);
    EXPECT_TRUE(hamilt::check_exx_fft_grid_compatibility(result.grid, input.operations).compatible);
}

TEST(ExxFftGridPolicy, StopsAtCandidateTrialLimitWithActionableMessage)
{
    hamilt::ExxFftGridPolicyInput input;
    input.initial_grid = hamilt::ExxFftGrid(32, 32, 50);
    input.operations = {zno_operation()};
    input.max_grid_points = 100000;
    input.max_candidate_trials = 8;

    const hamilt::ExxFftGridPolicyResult result = hamilt::choose_exx_fft_grid(input);

    EXPECT_FALSE(result.found);
    EXPECT_TRUE(result.requires_full_k_fallback);
    EXPECT_EQ(result.stop_reason, hamilt::ExxFftGridStopReason::trial_limit);
    EXPECT_EQ(result.candidate_trials, 8u);
    EXPECT_NE(result.last_attempted_grid, input.initial_grid);
    EXPECT_FALSE(result.last_compatibility.compatible);

    const std::string message = hamilt::describe_exx_fft_grid_policy_failure(input, result);
    EXPECT_NE(message.find("stopped after 8 of 8 enlarged-grid trials"), std::string::npos);
    EXPECT_NE(message.find("initial FFT grid 32x32x50"), std::string::npos);
    EXPECT_NE(message.find("last attempted FFT grid 40x32x50"), std::string::npos);
    EXPECT_NE(message.find("different ecutexx"), std::string::npos);
}

TEST(ExxFftGridPolicy, ProductionCandidateTrialLimitIsSixteen)
{
    EXPECT_EQ(hamilt::EXX_FFT_GRID_MAX_CANDIDATE_TRIALS, 16u);
}

TEST(ExxFftGridPolicy, ClampsCallerTrialLimitToProductionMaximum)
{
    hamilt::ExxSymmetryGridOperation operation = zno_operation();
    operation.translation = ModuleBase::Vector3<double>(1.0 / 7.0, 0.0, 0.0);

    hamilt::ExxFftGridPolicyInput input;
    input.initial_grid = hamilt::ExxFftGrid(32, 32, 50);
    input.operations = {operation};
    input.max_grid_points = 1000000;
    input.max_candidate_trials = 100;

    const hamilt::ExxFftGridPolicyResult result = hamilt::choose_exx_fft_grid(input);

    EXPECT_FALSE(result.found);
    EXPECT_EQ(result.stop_reason, hamilt::ExxFftGridStopReason::trial_limit);
    EXPECT_EQ(result.candidate_trials, 16u);
    const std::string message = hamilt::describe_exx_fft_grid_policy_failure(input, result);
    EXPECT_NE(message.find("stopped after 16 of 16 enlarged-grid trials"), std::string::npos);
}

TEST(ExxFftGridPolicy, LeavesCompatibleControlGridUnchanged)
{
    hamilt::ExxFftGridPolicyInput input;
    input.initial_grid = hamilt::ExxFftGrid(24, 24, 36);
    input.operations = {zno_operation()};
    input.max_grid_points = 100000;
    input.max_candidate_trials = hamilt::EXX_FFT_GRID_MAX_CANDIDATE_TRIALS;

    const hamilt::ExxFftGridPolicyResult result = hamilt::choose_exx_fft_grid(input);

    ASSERT_TRUE(result.found);
    EXPECT_FALSE(result.adjusted);
    EXPECT_EQ(result.grid, input.initial_grid);
    EXPECT_EQ(result.candidate_trials, 0u);
}

TEST(ExxFftGridPolicy, MemoryCeilingDoesNotRejectUnchangedCompatibleGrid)
{
    hamilt::ExxFftGridPolicyInput input;
    input.initial_grid = hamilt::ExxFftGrid(24, 24, 36);
    input.operations = {zno_operation()};
    input.max_grid_points = 1;
    input.max_candidate_trials = hamilt::EXX_FFT_GRID_MAX_CANDIDATE_TRIALS;

    const hamilt::ExxFftGridPolicyResult result = hamilt::choose_exx_fft_grid(input);

    ASSERT_TRUE(result.found);
    EXPECT_FALSE(result.adjusted);
    EXPECT_FALSE(result.requires_full_k_fallback);
    EXPECT_EQ(result.grid, input.initial_grid);
}

TEST(ExxFftGridPolicy, RequestsFullKFallbackAtMemoryLimit)
{
    hamilt::ExxFftGridPolicyInput input;
    input.initial_grid = hamilt::ExxFftGrid(32, 32, 50);
    input.operations = {zno_operation()};
    input.max_grid_points = 32u * 32u * 50u;
    input.max_candidate_trials = hamilt::EXX_FFT_GRID_MAX_CANDIDATE_TRIALS;

    const hamilt::ExxFftGridPolicyResult result = hamilt::choose_exx_fft_grid(input);

    EXPECT_FALSE(result.found);
    EXPECT_TRUE(result.requires_full_k_fallback);
    EXPECT_EQ(result.stop_reason, hamilt::ExxFftGridStopReason::memory_limit);
    EXPECT_EQ(result.candidate_trials, 0u);
    EXPECT_EQ(result.max_grid_points, 51200u);
}

TEST(ExxFftGridPolicy, ConvertsMemoryBudgetToOverflowSafeSearchLimit)
{
    const hamilt::ExxFftGridSearchLimit limit = hamilt::make_exx_fft_grid_search_limit(1.0, 16);

    ASSERT_TRUE(limit.valid);
    EXPECT_EQ(limit.budget_bytes, 1024u * 1024u);
    EXPECT_EQ(limit.max_grid_points, 65536u);

    const hamilt::ExxFftGridSearchLimit overflow
        = hamilt::make_exx_fft_grid_search_limit(std::numeric_limits<double>::infinity(), 16);
    EXPECT_FALSE(overflow.valid);
}

TEST(ExxFftGridPolicy, EnlargedEmbeddingPreservesPhysicalGVectorSphere)
{
#ifdef __MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    const bool owns_mpi = mpi_initialized == 0;
    if (owns_mpi)
    {
        MPI_Init(nullptr, nullptr);
    }
#endif
    {
        ModulePW::PW_Basis original("cpu", "double");
        ModulePW::PW_Basis adjusted("cpu", "double");
#ifdef __MPI
        original.initmpi(1, 0, MPI_COMM_WORLD);
        adjusted.initmpi(1, 0, MPI_COMM_WORLD);
#endif
        const ModuleBase::Matrix3 lattice(1, 0, 0,
                                          0, 1, 0,
                                          0, 0, 1);
        original.initgrids(1.0, lattice, 32, 32, 50);
        adjusted.initgrids(1.0, lattice, 36, 36, 50);
        original.initparameters(false, 240.0, 1, true);
        adjusted.initparameters(false, 240.0, 1, true);
        original.setuptransform();
        adjusted.setuptransform();
        original.collect_local_pw();
        adjusted.collect_local_pw();

        EXPECT_DOUBLE_EQ(original.ggecut, adjusted.ggecut);
        ASSERT_EQ(original.npw, adjusted.npw);

        std::vector<std::array<int, 3>> original_g;
        std::vector<std::array<int, 3>> adjusted_g;
        for (int ig = 0; ig < original.npw; ++ig)
        {
            EXPECT_LE(original.gg[ig], original.ggecut + 1.0e-12);
            original_g.push_back({{static_cast<int>(original.gdirect[ig].x),
                                   static_cast<int>(original.gdirect[ig].y),
                                   static_cast<int>(original.gdirect[ig].z)}});
        }
        for (int ig = 0; ig < adjusted.npw; ++ig)
        {
            EXPECT_LE(adjusted.gg[ig], adjusted.ggecut + 1.0e-12);
            adjusted_g.push_back({{static_cast<int>(adjusted.gdirect[ig].x),
                                   static_cast<int>(adjusted.gdirect[ig].y),
                                   static_cast<int>(adjusted.gdirect[ig].z)}});
        }
        std::sort(original_g.begin(), original_g.end());
        std::sort(adjusted_g.begin(), adjusted_g.end());
        EXPECT_EQ(original_g, adjusted_g);

        ModulePW::PW_Basis zno_adjusted("cpu", "double");
#ifdef __MPI
        zno_adjusted.initmpi(1, 0, MPI_COMM_WORLD);
#endif
        const ModuleBase::Matrix3 zno_lattice(3.25, 0, 0,
                                              -1.625, 2.814582562299, 0,
                                              0, 0, 5.207);
        zno_adjusted.initgrids(1.889726125457828, zno_lattice, 36, 36, 50);
        zno_adjusted.initparameters(false, 240.0, 1, true);
        zno_adjusted.setuptransform();
        zno_adjusted.collect_local_pw();
        EXPECT_EQ(zno_adjusted.npw, 20185);
    }
#ifdef __MPI
    if (owns_mpi)
    {
        MPI_Finalize();
    }
#endif
}

TEST(ExxReducedMeshPolicy, DetectsNativeAndDeferredExactExchangeConsumers)
{
    EXPECT_TRUE(hamilt::is_pw_exact_exchange_functional("HSE"));
    EXPECT_TRUE(hamilt::is_pw_exact_exchange_functional("pbe0"));
    EXPECT_FALSE(hamilt::is_pw_exact_exchange_functional("PBE"));
    EXPECT_FALSE(hamilt::needs_pw_exx_grid(false, false, false));
    EXPECT_TRUE(hamilt::needs_pw_exx_grid(true, false, false));
    EXPECT_TRUE(hamilt::needs_pw_exx_grid(false, true, false));
    EXPECT_TRUE(hamilt::needs_pw_exx_grid(false, false, true));
}

TEST(ExxReducedMeshPolicy, AcceptsSemilocalAndCompatibleExactExchangeMeshes)
{
    hamilt::ExxReducedMeshAcceptanceInput input;
    input.grid_result.found = false;
    EXPECT_TRUE(hamilt::assess_exx_reduced_mesh(input).accepted);

    input.exx_required = true;
    input.grid_result.found = true;
    input.grid_result.adjusted = false;
    EXPECT_TRUE(hamilt::assess_exx_reduced_mesh(input).accepted);
}

TEST(ExxReducedMeshPolicy, RejectsAdjustedEnergyDensityGrid)
{
    hamilt::ExxReducedMeshAcceptanceInput input;
    input.exx_required = true;
    input.requires_charge_grid_match = true;
    input.charge_grid_matches = false;
    input.grid_result.found = true;
    input.grid_result.adjusted = true;

    const hamilt::ExxReducedMeshAcceptance result = hamilt::assess_exx_reduced_mesh(input);
    EXPECT_FALSE(result.accepted);
    EXPECT_EQ(result.reason, hamilt::ExxReducedMeshFallbackReason::charge_grid_mismatch);
}

TEST(ExxReducedMeshPolicy, EnergyDensityUsesActualChargeGridDimensions)
{
    EXPECT_TRUE(hamilt::exx_fft_grid_matches(hamilt::ExxFftGrid(36, 36, 50), 36, 36, 50));
    EXPECT_FALSE(hamilt::exx_fft_grid_matches(hamilt::ExxFftGrid(36, 36, 50), 32, 32, 50));

    hamilt::ExxReducedMeshAcceptanceInput input;
    input.exx_required = true;
    input.requires_charge_grid_match = true;
    input.charge_grid_matches = true;
    input.grid_result.found = true;
    input.grid_result.adjusted = true;
    EXPECT_TRUE(hamilt::assess_exx_reduced_mesh(input).accepted);

    input.charge_grid_matches = false;
    EXPECT_EQ(hamilt::assess_exx_reduced_mesh(input).reason,
              hamilt::ExxReducedMeshFallbackReason::charge_grid_mismatch);
}

TEST(ExxReducedMeshPolicy, RejectsSearchAndMemoryFailures)
{
    hamilt::ExxReducedMeshAcceptanceInput input;
    input.exx_required = true;
    input.grid_result.found = false;
    EXPECT_EQ(hamilt::assess_exx_reduced_mesh(input).reason,
              hamilt::ExxReducedMeshFallbackReason::grid_search_failed);

    input.grid_result.found = true;
    input.grid_result.adjusted = true;
    input.operator_memory_fits = false;
    EXPECT_EQ(hamilt::assess_exx_reduced_mesh(input).reason,
              hamilt::ExxReducedMeshFallbackReason::operator_memory);

    input.operator_memory_fits = true;
    input.stress_requested = true;
    input.stress_memory_fits = false;
    EXPECT_EQ(hamilt::assess_exx_reduced_mesh(input).reason,
              hamilt::ExxReducedMeshFallbackReason::stress_memory);
}

TEST(ExxReducedMeshPolicy, MemoryFallbackDiagnosticIncludesActionableContext)
{
    hamilt::ExxFftGridPolicyInput input;
    input.initial_grid = hamilt::ExxFftGrid(32, 32, 50);
    hamilt::ExxFftGridPolicyResult result;
    result.found = true;
    result.adjusted = true;
    result.grid = hamilt::ExxFftGrid(36, 36, 50);

    const std::string message = hamilt::describe_exx_reduced_mesh_memory_fallback(
        input, result, 73400320u, 62914560u, true, 240.0);

    EXPECT_NE(message.find("initial grid 32x32x50"), std::string::npos);
    EXPECT_NE(message.find("compatible grid 36x36x50"), std::string::npos);
    EXPECT_NE(message.find("required 73400320 bytes"), std::string::npos);
    EXPECT_NE(message.find("budget 62914560 bytes"), std::string::npos);
    EXPECT_NE(message.find("stress"), std::string::npos);
    EXPECT_NE(message.find("ecutexx = 240"), std::string::npos);
}
