#ifndef TRAINING_DUMP_H
#define TRAINING_DUMP_H

#include "training_band_edge.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"
#include "source_estate/elecstate.h"

#include <string>
#include <vector>

class Charge;

namespace ModuleIO
{

struct TrainingBandEdgeLabel
{
    bool present = false;
    std::vector<TrainingBandEdgeState> states;
    std::vector<double> orbital_rho_tsg;
    std::vector<double> orbital_sigma_txg;
    std::vector<double> orbital_tau_tsg;
};

struct TrainingSDMXWavefunction
{
    bool present = false;
    int nspin = 0;
    int nkpt = 0;
    int nband = 0;
    int npwk_max = 0;
    std::vector<double> coeffs_real_skbg;
    std::vector<double> coeffs_imag_skbg;
    std::vector<double> npw_k;
    std::vector<double> g_index_kg;
};

struct TrainingExxEnergyDensity
{
    bool present = false;
    std::vector<double> values_local_Ha_bohr3;
    double scalar_energy_Ha = 0.0;
    double max_imaginary_abs_Ha_bohr3 = 0.0;
    std::string k_policy;
    int nx = 0;
    int ny = 0;
    int nz = 0;
};

void write_training_dump(const UnitCell& ucell,
                         const elecstate::ElecState& elec,
                         const ModulePW::PW_Basis& rho_basis,
                         const Charge& chr,
                         const bool scf_converged,
                         const int istep,
                         const std::string& output_root,
                         const TrainingBandEdgeLabel* band_edge_label = nullptr,
                         const TrainingSDMXWavefunction* sdmx_wavefunction = nullptr);

void set_last_training_band_edge_label(const TrainingBandEdgeLabel& label);
void clear_last_training_band_edge_label();
void set_last_training_sdmx_wavefunction(const TrainingSDMXWavefunction& wavefunction);
void clear_last_training_sdmx_wavefunction();
void set_last_training_exx_energy_density(const TrainingExxEnergyDensity& energy_density);
void clear_last_training_exx_energy_density();
bool take_last_training_exx_energy_density(TrainingExxEnergyDensity& energy_density);

} // namespace ModuleIO

#endif
