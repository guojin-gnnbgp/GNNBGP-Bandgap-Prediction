#ifndef TRAINING_BAND_EDGE_H
#define TRAINING_BAND_EDGE_H

#include <array>
#include <vector>

class K_Vectors;

namespace ModuleBase
{
class matrix;
}

namespace ModuleIO
{

enum class TrainingBandEdgeRole
{
    Vbm = 0,
    Cbm = 1
};

struct TrainingBandEdgeState
{
    TrainingBandEdgeRole role;
    int spin;
    int k;
    int band;
    double occupation;
    double eigenvalue_Ha;
    double exx_shift_Ha = 0.0;
    std::array<double, 3> kpt_direct{};
    std::array<double, 3> kpt_cart{};
};

std::vector<TrainingBandEdgeState> select_training_band_edges(const K_Vectors& kv,
                                                               const ModuleBase::matrix& ekb,
                                                               const ModuleBase::matrix& wg,
                                                               double tolerance_Ha,
                                                               double occupation_threshold = 1.0e-8);

} // namespace ModuleIO

#endif
