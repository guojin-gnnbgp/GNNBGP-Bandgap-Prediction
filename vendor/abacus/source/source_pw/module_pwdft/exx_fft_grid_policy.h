#ifndef EXX_FFT_GRID_POLICY_H
#define EXX_FFT_GRID_POLICY_H

#include "source_base/matrix3.h"
#include "source_base/vector3.h"

#include <cstddef>
#include <string>
#include <vector>

namespace hamilt
{

constexpr std::size_t EXX_FFT_GRID_MAX_CANDIDATE_TRIALS = 16;

struct ExxFftGrid
{
    ExxFftGrid() {}
    ExxFftGrid(int nx_in, int ny_in, int nz_in) : nx(nx_in), ny(ny_in), nz(nz_in) {}

    bool operator==(const ExxFftGrid& rhs) const
    {
        return nx == rhs.nx && ny == rhs.ny && nz == rhs.nz;
    }

    bool operator!=(const ExxFftGrid& rhs) const
    {
        return !(*this == rhs);
    }

    int nx = 0;
    int ny = 0;
    int nz = 0;
};

struct ExxSymmetryGridOperation
{
    int symop = -1;
    ModuleBase::Matrix3 rotation;
    ModuleBase::Vector3<double> translation;
};

enum class ExxFftGridCondition
{
    invalid_dimensions,
    rotation_ratio,
    translation
};

struct ExxFftGridFailure
{
    ExxFftGridCondition condition = ExxFftGridCondition::invalid_dimensions;
    int operation_index = -1;
    int symop = -1;
    int target_axis = -1;
    int source_axis = -1;
    double coefficient = 0.0;
    double value = 0.0;
};

struct ExxFftGridCompatibility
{
    bool compatible = false;
    std::vector<ExxFftGridFailure> failures;
};

enum class ExxFftGridStopReason
{
    none,
    invalid_initial_grid,
    memory_limit,
    trial_limit,
    candidates_exhausted
};

struct ExxFftGridPolicyInput
{
    ExxFftGrid initial_grid;
    std::vector<ExxSymmetryGridOperation> operations;
    std::size_t max_grid_points = 0;
    std::size_t max_candidate_trials = 0;
};

struct ExxFftGridPolicyResult
{
    bool found = false;
    bool adjusted = false;
    bool requires_full_k_fallback = false;
    ExxFftGrid grid;
    std::size_t grid_points = 0;
    std::size_t max_grid_points = 0;
    std::size_t candidate_trials = 0;
    ExxFftGridStopReason stop_reason = ExxFftGridStopReason::none;
    ExxFftGridCompatibility initial_compatibility;
    ExxFftGrid last_attempted_grid;
    ExxFftGridCompatibility last_compatibility;
};

struct ExxFftGridSearchLimit
{
    bool valid = false;
    std::size_t budget_bytes = 0;
    std::size_t max_grid_points = 0;
};

enum class ExxReducedMeshFallbackReason
{
    none,
    grid_search_failed,
    charge_grid_mismatch,
    operator_memory,
    stress_memory
};

struct ExxReducedMeshAcceptanceInput
{
    bool exx_required = false;
    bool requires_charge_grid_match = false;
    bool charge_grid_matches = true;
    bool operator_memory_fits = true;
    bool stress_requested = false;
    bool stress_memory_fits = true;
    ExxFftGridPolicyResult grid_result;
};

struct ExxReducedMeshAcceptance
{
    bool accepted = true;
    ExxReducedMeshFallbackReason reason = ExxReducedMeshFallbackReason::none;
};

bool checked_exx_fft_grid_points(const ExxFftGrid& grid, std::size_t& points);

bool is_exx_fft_friendly_dimension(int dimension);

ExxFftGridCompatibility check_exx_fft_grid_compatibility(
    const ExxFftGrid& grid,
    const std::vector<ExxSymmetryGridOperation>& operations);

ExxFftGridPolicyResult choose_exx_fft_grid(const ExxFftGridPolicyInput& input);

ExxFftGridSearchLimit make_exx_fft_grid_search_limit(double memory_budget_mb,
                                                       std::size_t complex_value_bytes);

bool is_pw_exact_exchange_functional(std::string functional);

bool exx_fft_grid_matches(const ExxFftGrid& grid, int nx, int ny, int nz);

bool needs_pw_exx_grid(bool native_exact_exchange,
                       bool scalar_label_requested,
                       bool band_label_requested);

ExxReducedMeshAcceptance assess_exx_reduced_mesh(const ExxReducedMeshAcceptanceInput& input);

std::string describe_exx_fft_grid_incompatibility(
    const ExxFftGrid& grid,
    const std::vector<ExxSymmetryGridOperation>& operations,
    const ExxFftGridCompatibility& compatibility);

std::string describe_exx_fft_grid_policy_failure(const ExxFftGridPolicyInput& input,
                                                 const ExxFftGridPolicyResult& result);

std::string describe_exx_reduced_mesh_memory_fallback(const ExxFftGridPolicyInput& grid_input,
                                                       const ExxFftGridPolicyResult& grid_result,
                                                       std::size_t required_bytes,
                                                       std::size_t budget_bytes,
                                                       bool stress_workload,
                                                       double ecut_exx);

} // namespace hamilt

#endif // EXX_FFT_GRID_POLICY_H
