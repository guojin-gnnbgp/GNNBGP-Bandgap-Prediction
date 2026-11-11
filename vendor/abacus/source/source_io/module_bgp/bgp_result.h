#ifndef MODULE_BGP_RESULT_H
#define MODULE_BGP_RESULT_H

#include <string>
#include <vector>

namespace ModuleIO
{

struct BgpElectronicState
{
    bool converged = false;
    int nspin = 1;
    double fermi_energy_ry = 0.0;
    double printed_gap_ry = 0.0;
    std::vector<double> k_weights;
    std::vector<std::vector<double>> eigenvalues_ry;
    std::vector<std::vector<double>> occupations;
};

void write_bgp_electronic_state(const std::string& path, const BgpElectronicState& state, int rank);

} // namespace ModuleIO

#endif
