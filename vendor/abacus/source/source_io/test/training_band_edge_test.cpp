#include "source_io/module_training/training_band_edge.h"

#include "source_base/matrix.h"
#include "source_cell/klist.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

K_Vectors make_kvectors(const std::vector<int>& spins)
{
    K_Vectors kv;
    kv.set_nks(static_cast<int>(spins.size()));
    kv.set_nspin(std::find(spins.begin(), spins.end(), 1) == spins.end() ? 1 : 2);
    kv.isk = spins;
    for (int ik = 0; ik < static_cast<int>(spins.size()); ++ik)
    {
        kv.kvec_d.emplace_back(0.1 * ik, 0.2 * ik, 0.3 * ik);
        kv.kvec_c.emplace_back(0.4 * ik, 0.5 * ik, 0.6 * ik);
    }
    return kv;
}

ModuleBase::matrix make_matrix(const std::vector<std::vector<double>>& rows)
{
    ModuleBase::matrix result(static_cast<int>(rows.size()), static_cast<int>(rows.front().size()));
    for (int row = 0; row < result.nr; ++row)
    {
        for (int column = 0; column < result.nc; ++column)
        {
            result(row, column) = rows[row][column];
        }
    }
    return result;
}

TEST(TrainingBandEdge, SelectsDirectEdgesAndConvertsRyToHa)
{
    const K_Vectors kv = make_kvectors({0});
    const auto ekb = make_matrix({{-0.8, -0.2, 0.4, 0.9}});
    const auto wg = make_matrix({{2.0, 2.0, 0.0, 0.0}});

    const auto edges = ModuleIO::select_training_band_edges(kv, ekb, wg, 1.0e-6);

    ASSERT_EQ(edges.size(), 2);
    EXPECT_EQ(edges[0].role, ModuleIO::TrainingBandEdgeRole::Vbm);
    EXPECT_EQ(edges[0].k, 0);
    EXPECT_EQ(edges[0].band, 1);
    EXPECT_DOUBLE_EQ(edges[0].occupation, 2.0);
    EXPECT_NEAR(edges[0].eigenvalue_Ha, -0.1, 1.0e-14);
    EXPECT_EQ(edges[1].role, ModuleIO::TrainingBandEdgeRole::Cbm);
    EXPECT_EQ(edges[1].band, 2);
}

TEST(TrainingBandEdge, SelectsIndirectDegenerateEdgesInDeterministicOrder)
{
    const K_Vectors kv = make_kvectors({1, 0, 1, 0});
    const auto ekb = make_matrix({{-1.0, -0.2, 0.7},
                                  {-1.0, -0.5, 0.4},
                                  {-1.0, -0.2000019, 0.4000019},
                                  {-1.0, -0.8, 0.9}});
    const auto wg = make_matrix({{1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}});

    const auto edges = ModuleIO::select_training_band_edges(kv, ekb, wg, 1.0e-6);

    ASSERT_EQ(edges.size(), 4);
    EXPECT_EQ(edges[0].role, ModuleIO::TrainingBandEdgeRole::Vbm);
    EXPECT_EQ(edges[0].spin, 1);
    EXPECT_EQ(edges[0].k, 0);
    EXPECT_EQ(edges[1].spin, 1);
    EXPECT_EQ(edges[1].k, 2);
    EXPECT_EQ(edges[2].role, ModuleIO::TrainingBandEdgeRole::Cbm);
    EXPECT_EQ(edges[2].spin, 0);
    EXPECT_EQ(edges[2].k, 1);
    EXPECT_EQ(edges[3].spin, 1);
    EXPECT_EQ(edges[3].k, 2);
    EXPECT_EQ(edges[1].kpt_direct[0], 0.2);
    EXPECT_EQ(edges[1].kpt_cart[2], 1.2);
}

TEST(TrainingBandEdge, IncludesToleranceBoundaryAndExcludesOutside)
{
    const K_Vectors kv = make_kvectors({0, 0, 0});
    const auto ekb = make_matrix({{-0.5, 0.5}, {-0.500002, 0.500002}, {-0.5000021, 0.5000021}});
    const auto wg = make_matrix({{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}});

    const auto edges = ModuleIO::select_training_band_edges(kv, ekb, wg, 1.0e-6);

    ASSERT_EQ(edges.size(), 4);
    EXPECT_EQ(edges[0].k, 0);
    EXPECT_EQ(edges[1].k, 1);
    EXPECT_EQ(edges[2].k, 0);
    EXPECT_EQ(edges[3].k, 1);
}

TEST(TrainingBandEdge, TreatsFractionalOccupationAboveThresholdAsOccupied)
{
    const K_Vectors kv = make_kvectors({0});
    const auto ekb = make_matrix({{-0.4, -0.1, 0.2}});
    const auto wg = make_matrix({{1.0, 2.0e-8, 1.0e-8}});

    const auto edges = ModuleIO::select_training_band_edges(kv, ekb, wg, 0.0);

    ASSERT_EQ(edges.size(), 2);
    EXPECT_EQ(edges[0].band, 1);
    EXPECT_EQ(edges[1].band, 2);
}

TEST(TrainingBandEdge, RejectsKpointMetadataShorterThanBandMatrices)
{
    const auto ekb = make_matrix({{-0.4, 0.2}, {-0.3, 0.3}});
    const auto wg = make_matrix({{1.0, 0.0}, {1.0, 0.0}});

    auto missing_spin = make_kvectors({0, 1});
    missing_spin.isk.pop_back();
    EXPECT_THROW(ModuleIO::select_training_band_edges(missing_spin, ekb, wg, 1.0e-6), std::runtime_error);

    auto missing_direct_k = make_kvectors({0, 1});
    missing_direct_k.kvec_d.pop_back();
    EXPECT_THROW(ModuleIO::select_training_band_edges(missing_direct_k, ekb, wg, 1.0e-6), std::runtime_error);

    auto missing_cartesian_k = make_kvectors({0, 1});
    missing_cartesian_k.kvec_c.pop_back();
    EXPECT_THROW(ModuleIO::select_training_band_edges(missing_cartesian_k, ekb, wg, 1.0e-6), std::runtime_error);
}

TEST(TrainingBandEdge, RejectsNonFiniteKpointCoordinates)
{
    const auto ekb = make_matrix({{-0.4, 0.2}});
    const auto wg = make_matrix({{1.0, 0.0}});

    auto non_finite_direct_k = make_kvectors({0});
    non_finite_direct_k.kvec_d[0].y = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(ModuleIO::select_training_band_edges(non_finite_direct_k, ekb, wg, 1.0e-6), std::runtime_error);

    auto non_finite_cartesian_k = make_kvectors({0});
    non_finite_cartesian_k.kvec_c[0].z = std::numeric_limits<double>::infinity();
    EXPECT_THROW(ModuleIO::select_training_band_edges(non_finite_cartesian_k, ekb, wg, 1.0e-6),
                 std::runtime_error);
}

TEST(TrainingBandEdge, OrdersSameRoleSpinAndKDegenerateBandsByBandIndex)
{
    const K_Vectors kv = make_kvectors({0});
    const auto ekb = make_matrix({{-0.8, -0.2, -0.2, 0.4, 0.4, 0.9}});
    const auto wg = make_matrix({{2.0, 2.0, 2.0, 0.0, 0.0, 0.0}});

    const auto edges = ModuleIO::select_training_band_edges(kv, ekb, wg, 0.0);

    ASSERT_EQ(edges.size(), 4);
    EXPECT_EQ(edges[0].role, ModuleIO::TrainingBandEdgeRole::Vbm);
    EXPECT_EQ(edges[0].spin, 0);
    EXPECT_EQ(edges[0].k, 0);
    EXPECT_EQ(edges[0].band, 1);
    EXPECT_EQ(edges[1].band, 2);
    EXPECT_EQ(edges[2].role, ModuleIO::TrainingBandEdgeRole::Cbm);
    EXPECT_EQ(edges[2].spin, 0);
    EXPECT_EQ(edges[2].k, 0);
    EXPECT_EQ(edges[2].band, 3);
    EXPECT_EQ(edges[3].band, 4);
}

TEST(TrainingBandEdge, RejectsMissingRolesAndNonFiniteInput)
{
    const K_Vectors kv = make_kvectors({0});
    const auto all_occupied_ekb = make_matrix({{-0.4, -0.1}});
    const auto all_occupied_wg = make_matrix({{1.0, 1.0}});
    EXPECT_THROW(ModuleIO::select_training_band_edges(kv, all_occupied_ekb, all_occupied_wg, 1.0e-6),
                 std::runtime_error);

    const auto all_empty_wg = make_matrix({{0.0, 0.0}});
    EXPECT_THROW(ModuleIO::select_training_band_edges(kv, all_occupied_ekb, all_empty_wg, 1.0e-6),
                 std::runtime_error);

    auto non_finite_ekb = make_matrix({{-0.4, 0.2}});
    const auto mixed_wg = make_matrix({{1.0, 0.0}});
    non_finite_ekb(0, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(ModuleIO::select_training_band_edges(kv, non_finite_ekb, mixed_wg, 1.0e-6),
                 std::runtime_error);
}

} // namespace
