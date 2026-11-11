#include "source_io/module_training/training_dump.h"

#include <gtest/gtest.h>

TEST(TrainingExxEnergyDensityState, payload_is_one_shot_and_clear_prevents_stale_reuse)
{
    ModuleIO::clear_last_training_exx_energy_density();

    ModuleIO::TrainingExxEnergyDensity consumed;
    EXPECT_FALSE(ModuleIO::take_last_training_exx_energy_density(consumed));

    ModuleIO::TrainingExxEnergyDensity current;
    current.present = true;
    current.values_local_Ha_bohr3 = {-0.25, -0.75};
    current.scalar_energy_Ha = -1.0;
    current.k_policy = "pair-map";
    current.nx = 2;
    current.ny = 1;
    current.nz = 1;
    ModuleIO::set_last_training_exx_energy_density(current);

    ASSERT_TRUE(ModuleIO::take_last_training_exx_energy_density(consumed));
    EXPECT_TRUE(consumed.present);
    EXPECT_EQ(consumed.values_local_Ha_bohr3, current.values_local_Ha_bohr3);
    EXPECT_DOUBLE_EQ(consumed.scalar_energy_Ha, current.scalar_energy_Ha);
    EXPECT_EQ(consumed.k_policy, current.k_policy);

    ModuleIO::TrainingExxEnergyDensity stale;
    EXPECT_FALSE(ModuleIO::take_last_training_exx_energy_density(stale));

    ModuleIO::set_last_training_exx_energy_density(current);
    ModuleIO::clear_last_training_exx_energy_density();
    EXPECT_FALSE(ModuleIO::take_last_training_exx_energy_density(stale));
}
