#include "training_band_edge.h"

#include "source_base/matrix.h"
#include "source_cell/klist.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace ModuleIO
{

std::vector<TrainingBandEdgeState> select_training_band_edges(const K_Vectors& kv,
                                                               const ModuleBase::matrix& ekb,
                                                               const ModuleBase::matrix& wg,
                                                               const double tolerance_Ha,
                                                               const double occupation_threshold)
{
    const int nks = kv.get_nks();
    if (nks <= 0 || ekb.nc <= 0 || ekb.nr != nks || wg.nr != nks || wg.nc != ekb.nc)
    {
        throw std::runtime_error("inconsistent band-edge selector dimensions");
    }
    if (kv.isk.size() < static_cast<std::size_t>(nks) || kv.kvec_d.size() < static_cast<std::size_t>(nks)
        || kv.kvec_c.size() < static_cast<std::size_t>(nks))
    {
        throw std::runtime_error("incomplete band-edge selector k-point metadata");
    }
    if (!std::isfinite(tolerance_Ha) || tolerance_Ha < 0.0 || !std::isfinite(occupation_threshold))
    {
        throw std::runtime_error("invalid band-edge selector threshold");
    }

    for (int ik = 0; ik < nks; ++ik)
    {
        const auto& direct_k = kv.kvec_d[ik];
        const auto& cartesian_k = kv.kvec_c[ik];
        if (!std::isfinite(direct_k.x) || !std::isfinite(direct_k.y) || !std::isfinite(direct_k.z)
            || !std::isfinite(cartesian_k.x) || !std::isfinite(cartesian_k.y) || !std::isfinite(cartesian_k.z))
        {
            throw std::runtime_error("non-finite band-edge selector k-point metadata");
        }
    }

    double vbm_Ry = -std::numeric_limits<double>::infinity();
    double cbm_Ry = std::numeric_limits<double>::infinity();
    for (int ik = 0; ik < nks; ++ik)
    {
        for (int ib = 0; ib < ekb.nc; ++ib)
        {
            const double eigenvalue_Ry = ekb(ik, ib);
            const double occupation = wg(ik, ib);
            if (!std::isfinite(eigenvalue_Ry) || !std::isfinite(occupation))
            {
                throw std::runtime_error("non-finite band-edge selector data");
            }
            if (occupation > occupation_threshold)
            {
                vbm_Ry = std::max(vbm_Ry, eigenvalue_Ry);
            }
            else
            {
                cbm_Ry = std::min(cbm_Ry, eigenvalue_Ry);
            }
        }
    }
    if (!std::isfinite(vbm_Ry) || !std::isfinite(cbm_Ry))
    {
        throw std::runtime_error("band-edge selector requires occupied and empty states");
    }

    const double tolerance_Ry = 2.0 * tolerance_Ha;
    std::vector<TrainingBandEdgeState> states;
    for (int ik = 0; ik < nks; ++ik)
    {
        for (int ib = 0; ib < ekb.nc; ++ib)
        {
            const double eigenvalue_Ry = ekb(ik, ib);
            const double occupation = wg(ik, ib);
            const bool occupied = occupation > occupation_threshold;
            const double distance_Ry = occupied ? vbm_Ry - eigenvalue_Ry : eigenvalue_Ry - cbm_Ry;
            if (distance_Ry > tolerance_Ry)
            {
                continue;
            }

            TrainingBandEdgeState state;
            state.role = occupied ? TrainingBandEdgeRole::Vbm : TrainingBandEdgeRole::Cbm;
            state.spin = kv.isk[ik];
            state.k = ik;
            state.band = ib;
            state.occupation = occupation;
            state.eigenvalue_Ha = 0.5 * eigenvalue_Ry;
            state.kpt_direct = {kv.kvec_d[ik].x, kv.kvec_d[ik].y, kv.kvec_d[ik].z};
            state.kpt_cart = {kv.kvec_c[ik].x, kv.kvec_c[ik].y, kv.kvec_c[ik].z};
            states.push_back(state);
        }
    }

    std::stable_sort(states.begin(),
                     states.end(),
                     [](const TrainingBandEdgeState& lhs, const TrainingBandEdgeState& rhs) {
                         return std::tie(lhs.role, lhs.spin, lhs.k, lhs.band)
                                < std::tie(rhs.role, rhs.spin, rhs.k, rhs.band);
                     });
    return states;
}

} // namespace ModuleIO
