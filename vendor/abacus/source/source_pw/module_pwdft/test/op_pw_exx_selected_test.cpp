#include "source_pw/module_pwdft/op_pw_exx.h"

#include <gtest/gtest.h>

#include <complex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

using Complex = std::complex<double>;
using Cpu = base_device::DEVICE_CPU;
using Operator = hamilt::OperatorEXXPW<Complex, Cpu>;
using Psi = psi::Psi<Complex, Cpu>;

TEST(OperatorExxPwSelectedTest, ExposesSelectedDiagonalInterface)
{
    using Return = decltype(std::declval<Operator&>().cal_exx_diagonal_selected(
        static_cast<Psi*>(nullptr),
        std::declval<const std::vector<std::pair<int, int>>&>()));
    EXPECT_TRUE((std::is_same<Return, std::vector<double>>::value));
}

TEST(OperatorExxPwSelectedTest, GroupsTargetsByKAndKeepsInputPositions)
{
    const auto grouped = hamilt::exx_selected_detail::validate_and_group_targets(
        2,
        4,
        {{1, 3}, {0, 1}, {1, 0}, {0, 1}});

    ASSERT_EQ(grouped.size(), 2);
    ASSERT_EQ(grouped.at(0).size(), 2);
    EXPECT_EQ(grouped.at(0)[0], std::make_pair(1, std::size_t{1}));
    EXPECT_EQ(grouped.at(0)[1], std::make_pair(1, std::size_t{3}));
    ASSERT_EQ(grouped.at(1).size(), 2);
    EXPECT_EQ(grouped.at(1)[0], std::make_pair(3, std::size_t{0}));
    EXPECT_EQ(grouped.at(1)[1], std::make_pair(0, std::size_t{2}));
}

TEST(OperatorExxPwSelectedTest, RejectsOutOfRangeTargets)
{
    EXPECT_THROW(hamilt::exx_selected_detail::validate_and_group_targets(2, 4, {{-1, 0}}),
                 std::out_of_range);
    EXPECT_THROW(hamilt::exx_selected_detail::validate_and_group_targets(2, 4, {{2, 0}}),
                 std::out_of_range);
    EXPECT_THROW(hamilt::exx_selected_detail::validate_and_group_targets(2, 4, {{0, -1}}),
                 std::out_of_range);
    EXPECT_THROW(hamilt::exx_selected_detail::validate_and_group_targets(2, 4, {{0, 4}}),
                 std::out_of_range);
}

TEST(OperatorExxPwSelectedTest, RestoresCurrentKAfterExceptionalExit)
{
    const std::vector<int> ngk{3, 2};
    Psi psi(2, 4, 3, ngk, true);
    psi.fix_k(1);

    EXPECT_THROW(
        {
            hamilt::exx_selected_detail::CurrentKGuard<Psi> guard(&psi);
            psi.fix_k(0);
            throw std::runtime_error("selected diagonal failure");
        },
        std::runtime_error);
    EXPECT_EQ(psi.get_current_k(), 1);
    EXPECT_EQ(psi.get_current_nbas(), 2);
}

TEST(OperatorExxPwSelectedTest, CpuSourceTileUsesConfiguredBandTileForOneTarget)
{
    constexpr int configured_band_tile_size = 4;
    constexpr int occupied_source_bands = 6;
    constexpr int selected_target_count = 1;

    EXPECT_EQ(hamilt::exx_selected_detail::resolve_cpu_source_tile_size(configured_band_tile_size,
                                                                        occupied_source_bands),
              4);
    EXPECT_NE(hamilt::exx_selected_detail::resolve_cpu_source_tile_size(configured_band_tile_size,
                                                                        occupied_source_bands),
              selected_target_count);
}

TEST(OperatorExxPwSelectedTest, DisabledExxDoesNotRequireCoulombParameters)
{
    const std::map<Conv_Coulomb_Pot_K::Coulomb_Type,
                   std::vector<std::map<std::string, std::string>>>
        parameters;

    EXPECT_TRUE(hamilt::exx_options_detail::coulomb_params_if_enabled(
                    false,
                    parameters,
                    Conv_Coulomb_Pot_K::Coulomb_Type::Fock)
                    .empty());
}

TEST(OperatorExxPwSelectedTest, EnabledExxAllowsUnusedCoulombPotentialFamily)
{
    const hamilt::exx_options_detail::CoulombParameterMap parameters = {
        {Conv_Coulomb_Pot_K::Coulomb_Type::Erfc, {{{"alpha", "1"}, {"omega", "0.11"}}}}};

    EXPECT_TRUE(hamilt::exx_options_detail::coulomb_params_if_enabled(
                    true,
                    parameters,
                    Conv_Coulomb_Pot_K::Coulomb_Type::Fock)
                    .empty());
    EXPECT_EQ(hamilt::exx_options_detail::coulomb_params_if_enabled(
                  true,
                  parameters,
                  Conv_Coulomb_Pot_K::Coulomb_Type::Erfc)
                  .size(),
              1);
}

TEST(OperatorExxPwSelectedTest, EnabledExxRequiresAtLeastOneCoulombPotentialFamily)
{
    const hamilt::exx_options_detail::CoulombParameterMap parameters;

    EXPECT_THROW(hamilt::exx_options_detail::coulomb_params_if_enabled(
                     true,
                     parameters,
                     Conv_Coulomb_Pot_K::Coulomb_Type::Fock),
                 std::invalid_argument);
}

} // namespace
