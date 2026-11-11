#include "training_dump.h"

#include <utility>

namespace ModuleIO
{

namespace
{
bool last_training_exx_energy_density_valid = false;
TrainingExxEnergyDensity last_training_exx_energy_density;
}

void set_last_training_exx_energy_density(const TrainingExxEnergyDensity& energy_density)
{
    last_training_exx_energy_density = energy_density;
    last_training_exx_energy_density_valid = energy_density.present;
}

void clear_last_training_exx_energy_density()
{
    last_training_exx_energy_density = TrainingExxEnergyDensity();
    last_training_exx_energy_density_valid = false;
}

bool take_last_training_exx_energy_density(TrainingExxEnergyDensity& energy_density)
{
    if (!last_training_exx_energy_density_valid)
    {
        return false;
    }
    energy_density = std::move(last_training_exx_energy_density);
    last_training_exx_energy_density = TrainingExxEnergyDensity();
    last_training_exx_energy_density_valid = false;
    return true;
}

} // namespace ModuleIO
