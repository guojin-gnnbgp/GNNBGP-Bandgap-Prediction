#include "exx_fft_grid_policy.h"

#include "source_pw/module_pwdft/exx_size_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>

namespace hamilt
{
namespace
{
const double INTEGRAL_TOLERANCE = 1.0e-8;
const double SYMMETRY_COORDINATE_TOLERANCE = 1.0e-6;
const double DEFAULT_MEMORY_BUDGET_MIB = 1024.0;
const std::size_t BYTES_PER_MIB = 1024u * 1024u;

bool is_integral(double value)
{
    return std::isfinite(value) && std::abs(value - std::round(value)) < INTEGRAL_TOLERANCE;
}

bool is_commensurate_translation(double translation, int dimension)
{
    const double value = translation * dimension;
    const double tolerance = std::max(INTEGRAL_TOLERANCE,
                                      SYMMETRY_COORDINATE_TOLERANCE * std::abs(dimension));
    return std::isfinite(value) && std::abs(value - std::round(value)) < tolerance;
}

int grid_dimension(const ExxFftGrid& grid, int axis)
{
    if (axis == 0)
    {
        return grid.nx;
    }
    if (axis == 1)
    {
        return grid.ny;
    }
    return grid.nz;
}

double rotation_coefficient(const ModuleBase::Matrix3& rotation, int source_axis, int target_axis)
{
    const double coefficients[3][3] = {
        {rotation.e11, rotation.e12, rotation.e13},
        {rotation.e21, rotation.e22, rotation.e23},
        {rotation.e31, rotation.e32, rotation.e33}};
    return coefficients[source_axis][target_axis];
}

double translation_component(const ModuleBase::Vector3<double>& translation, int axis)
{
    if (axis == 0)
    {
        return translation.x;
    }
    if (axis == 1)
    {
        return translation.y;
    }
    return translation.z;
}

void append_failure(std::vector<ExxFftGridFailure>& failures,
                    ExxFftGridCondition condition,
                    int operation_index,
                    int symop,
                    int target_axis,
                    int source_axis,
                    double coefficient,
                    double value)
{
    ExxFftGridFailure failure;
    failure.condition = condition;
    failure.operation_index = operation_index;
    failure.symop = symop;
    failure.target_axis = target_axis;
    failure.source_axis = source_axis;
    failure.coefficient = coefficient;
    failure.value = value;
    failures.push_back(failure);
}

std::vector<int> fft_friendly_dimensions(int minimum, int maximum)
{
    std::vector<int> dimensions;
    if (minimum <= 0 || maximum < minimum)
    {
        return dimensions;
    }

    for (long long factor2 = 1; factor2 <= maximum; factor2 *= 2)
    {
        for (long long factor3 = factor2; factor3 <= maximum; factor3 *= 3)
        {
            for (long long value = factor3; value <= maximum; value *= 5)
            {
                if (value >= minimum)
                {
                    dimensions.push_back(static_cast<int>(value));
                }
                if (value > maximum / 5)
                {
                    break;
                }
            }
            if (factor3 > maximum / 3)
            {
                break;
            }
        }
        if (factor2 > maximum / 2)
        {
            break;
        }
    }

    std::sort(dimensions.begin(), dimensions.end());
    dimensions.erase(std::unique(dimensions.begin(), dimensions.end()), dimensions.end());
    return dimensions;
}

bool grid_precedes(const ExxFftGrid& lhs,
                   std::size_t lhs_points,
                   const ExxFftGrid& rhs,
                   std::size_t rhs_points)
{
    if (lhs_points != rhs_points)
    {
        return lhs_points < rhs_points;
    }
    if (lhs.nx != rhs.nx)
    {
        return lhs.nx < rhs.nx;
    }
    if (lhs.ny != rhs.ny)
    {
        return lhs.ny < rhs.ny;
    }
    return lhs.nz < rhs.nz;
}

struct GridCandidate
{
    std::size_t x_index = 0;
    std::size_t y_index = 0;
    std::size_t z_index = 0;
    ExxFftGrid grid;
    std::size_t points = 0;
};

struct GridCandidateLater
{
    bool operator()(const GridCandidate& lhs, const GridCandidate& rhs) const
    {
        return grid_precedes(rhs.grid, rhs.points, lhs.grid, lhs.points);
    }
};

const char* stop_reason_description(ExxFftGridStopReason reason)
{
    if (reason == ExxFftGridStopReason::invalid_initial_grid)
    {
        return "the initial FFT dimensions are invalid or overflow the grid size";
    }
    if (reason == ExxFftGridStopReason::memory_limit)
    {
        return "the next enlarged FFT grid exceeds the defensive memory search limit";
    }
    if (reason == ExxFftGridStopReason::trial_limit)
    {
        return "the candidate-trial limit was reached";
    }
    if (reason == ExxFftGridStopReason::candidates_exhausted)
    {
        return "no further FFT-friendly candidate exists within the search limits";
    }
    return "the search ended without a compatible grid";
}

std::size_t effective_candidate_trial_limit(const ExxFftGridPolicyInput& input)
{
    return std::min(input.max_candidate_trials, EXX_FFT_GRID_MAX_CANDIDATE_TRIALS);
}

const char* axis_name(int axis)
{
    const char* names[] = {"x", "y", "z"};
    return axis >= 0 && axis < 3 ? names[axis] : "?";
}
} // namespace

bool checked_exx_fft_grid_points(const ExxFftGrid& grid, std::size_t& points)
{
    points = 0;
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0)
    {
        return false;
    }
    return checked_exx_size_product({static_cast<std::size_t>(grid.nx),
                                     static_cast<std::size_t>(grid.ny),
                                     static_cast<std::size_t>(grid.nz)},
                                    points);
}

bool is_exx_fft_friendly_dimension(int dimension)
{
    if (dimension <= 0)
    {
        return false;
    }
    for (const int factor: {2, 3, 5})
    {
        while (dimension % factor == 0)
        {
            dimension /= factor;
        }
    }
    return dimension == 1;
}

ExxFftGridCompatibility check_exx_fft_grid_compatibility(
    const ExxFftGrid& grid,
    const std::vector<ExxSymmetryGridOperation>& operations)
{
    ExxFftGridCompatibility result;
    std::size_t ignored_points = 0;
    if (!checked_exx_fft_grid_points(grid, ignored_points))
    {
        append_failure(result.failures,
                       ExxFftGridCondition::invalid_dimensions,
                       -1,
                       -1,
                       -1,
                       -1,
                       0.0,
                       0.0);
        return result;
    }

    for (std::size_t operation_index = 0; operation_index < operations.size(); ++operation_index)
    {
        const ExxSymmetryGridOperation& operation = operations[operation_index];
        for (int target_axis = 0; target_axis < 3; ++target_axis)
        {
            for (int source_axis = 0; source_axis < 3; ++source_axis)
            {
                const double coefficient = rotation_coefficient(operation.rotation, source_axis, target_axis);
                if (std::abs(coefficient) < INTEGRAL_TOLERANCE)
                {
                    continue;
                }
                const double value = coefficient * grid_dimension(grid, target_axis)
                                     / grid_dimension(grid, source_axis);
                if (!is_integral(value))
                {
                    append_failure(result.failures,
                                   ExxFftGridCondition::rotation_ratio,
                                   static_cast<int>(operation_index),
                                   operation.symop,
                                   target_axis,
                                   source_axis,
                                   coefficient,
                                   value);
                }
            }
        }
        for (int target_axis = 0; target_axis < 3; ++target_axis)
        {
            const double coefficient = translation_component(operation.translation, target_axis);
            const double value = coefficient * grid_dimension(grid, target_axis);
            if (!is_commensurate_translation(coefficient, grid_dimension(grid, target_axis)))
            {
                append_failure(result.failures,
                               ExxFftGridCondition::translation,
                               static_cast<int>(operation_index),
                               operation.symop,
                               target_axis,
                               -1,
                               coefficient,
                               value);
            }
        }
    }
    result.compatible = result.failures.empty();
    return result;
}

ExxFftGridPolicyResult choose_exx_fft_grid(const ExxFftGridPolicyInput& input)
{
    ExxFftGridPolicyResult result;
    result.grid = input.initial_grid;
    result.max_grid_points = input.max_grid_points;
    result.initial_compatibility = check_exx_fft_grid_compatibility(input.initial_grid, input.operations);
    result.last_attempted_grid = input.initial_grid;
    result.last_compatibility = result.initial_compatibility;
    const std::size_t candidate_trial_limit = effective_candidate_trial_limit(input);

    std::size_t initial_points = 0;
    if (!checked_exx_fft_grid_points(input.initial_grid, initial_points))
    {
        result.requires_full_k_fallback = true;
        result.stop_reason = ExxFftGridStopReason::invalid_initial_grid;
        return result;
    }
    if (result.initial_compatibility.compatible)
    {
        result.found = true;
        result.grid_points = initial_points;
        return result;
    }
    if (input.max_grid_points < initial_points)
    {
        result.requires_full_k_fallback = true;
        result.stop_reason = ExxFftGridStopReason::memory_limit;
        return result;
    }
    if (candidate_trial_limit == 0)
    {
        result.requires_full_k_fallback = true;
        result.stop_reason = ExxFftGridStopReason::trial_limit;
        return result;
    }

    const std::size_t max_x_size = input.max_grid_points
                                   / static_cast<std::size_t>(input.initial_grid.ny)
                                   / static_cast<std::size_t>(input.initial_grid.nz);
    const std::size_t max_y_size = input.max_grid_points
                                   / static_cast<std::size_t>(input.initial_grid.nx)
                                   / static_cast<std::size_t>(input.initial_grid.nz);
    const std::size_t max_z_size = input.max_grid_points
                                   / static_cast<std::size_t>(input.initial_grid.nx)
                                   / static_cast<std::size_t>(input.initial_grid.ny);
    const std::size_t int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    const bool dimensions_bounded_by_memory
        = max_x_size < int_max || max_y_size < int_max || max_z_size < int_max;
    const std::vector<int> x_candidates
        = fft_friendly_dimensions(input.initial_grid.nx, static_cast<int>(std::min(max_x_size, int_max)));
    const std::vector<int> y_candidates
        = fft_friendly_dimensions(input.initial_grid.ny, static_cast<int>(std::min(max_y_size, int_max)));
    const std::vector<int> z_candidates
        = fft_friendly_dimensions(input.initial_grid.nz, static_cast<int>(std::min(max_z_size, int_max)));

    if (x_candidates.empty() || y_candidates.empty() || z_candidates.empty())
    {
        result.requires_full_k_fallback = true;
        result.stop_reason = ExxFftGridStopReason::memory_limit;
        return result;
    }

    // Traverse the Cartesian product lazily so both compatibility checks and
    // successor generation stop at the hard trial and memory limits.
    std::priority_queue<GridCandidate, std::vector<GridCandidate>, GridCandidateLater> candidates;
    std::set<std::tuple<std::size_t, std::size_t, std::size_t>> queued_indices;
    const auto enqueue = [&](std::size_t x_index, std::size_t y_index, std::size_t z_index)
    {
        if (x_index >= x_candidates.size() || y_index >= y_candidates.size() || z_index >= z_candidates.size())
        {
            return;
        }
        const std::tuple<std::size_t, std::size_t, std::size_t> indices(x_index, y_index, z_index);
        if (!queued_indices.insert(indices).second)
        {
            return;
        }
        GridCandidate candidate;
        candidate.x_index = x_index;
        candidate.y_index = y_index;
        candidate.z_index = z_index;
        candidate.grid = ExxFftGrid(x_candidates[x_index], y_candidates[y_index], z_candidates[z_index]);
        if (checked_exx_fft_grid_points(candidate.grid, candidate.points))
        {
            candidates.push(candidate);
        }
    };

    enqueue(0, 0, 0);
    while (!candidates.empty())
    {
        const GridCandidate candidate = candidates.top();
        candidates.pop();
        if (candidate.points > input.max_grid_points)
        {
            result.requires_full_k_fallback = true;
            result.stop_reason = ExxFftGridStopReason::memory_limit;
            return result;
        }

        if (candidate.grid == input.initial_grid)
        {
            enqueue(candidate.x_index + 1, candidate.y_index, candidate.z_index);
            enqueue(candidate.x_index, candidate.y_index + 1, candidate.z_index);
            enqueue(candidate.x_index, candidate.y_index, candidate.z_index + 1);
            continue;
        }

        ++result.candidate_trials;
        result.last_attempted_grid = candidate.grid;
        result.last_compatibility = check_exx_fft_grid_compatibility(candidate.grid, input.operations);
        if (result.last_compatibility.compatible)
        {
            result.found = true;
            result.adjusted = true;
            result.grid = candidate.grid;
            result.grid_points = candidate.points;
            return result;
        }
        if (result.candidate_trials == candidate_trial_limit)
        {
            result.requires_full_k_fallback = true;
            result.stop_reason = ExxFftGridStopReason::trial_limit;
            return result;
        }

        enqueue(candidate.x_index + 1, candidate.y_index, candidate.z_index);
        enqueue(candidate.x_index, candidate.y_index + 1, candidate.z_index);
        enqueue(candidate.x_index, candidate.y_index, candidate.z_index + 1);
    }

    result.requires_full_k_fallback = true;
    result.stop_reason = dimensions_bounded_by_memory ? ExxFftGridStopReason::memory_limit
                                                      : ExxFftGridStopReason::candidates_exhausted;
    return result;
}

ExxFftGridSearchLimit make_exx_fft_grid_search_limit(double memory_budget_mb,
                                                       std::size_t complex_value_bytes)
{
    ExxFftGridSearchLimit result;
    const double budget_mib = memory_budget_mb == 0.0 ? DEFAULT_MEMORY_BUDGET_MIB : memory_budget_mb;
    if (!std::isfinite(budget_mib) || budget_mib <= 0.0 || complex_value_bytes == 0)
    {
        return result;
    }
    const long double bytes = static_cast<long double>(budget_mib)
                              * static_cast<long double>(BYTES_PER_MIB);
    if (bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
    {
        return result;
    }
    result.budget_bytes = static_cast<std::size_t>(bytes);
    result.max_grid_points = result.budget_bytes / complex_value_bytes;
    result.valid = result.max_grid_points > 0;
    return result;
}

bool needs_pw_exx_grid(bool native_exact_exchange,
                       bool scalar_label_requested,
                       bool band_label_requested)
{
    return native_exact_exchange || scalar_label_requested || band_label_requested;
}

bool is_pw_exact_exchange_functional(std::string functional)
{
    std::transform(functional.begin(),
                   functional.end(),
                   functional.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    const std::vector<std::string> exact_exchange_functionals = {
        "hf", "pbe0", "b3lyp", "hse", "scan0", "muller", "power", "cwp22", "wp22",
        "lc_pbe", "lc_wpbe", "lrc_wpbe", "lrc_wpbeh", "cam_pbeh"};
    return std::find(exact_exchange_functionals.begin(), exact_exchange_functionals.end(), functional)
           != exact_exchange_functionals.end();
}

bool exx_fft_grid_matches(const ExxFftGrid& grid, int nx, int ny, int nz)
{
    return grid.nx == nx && grid.ny == ny && grid.nz == nz;
}

ExxReducedMeshAcceptance assess_exx_reduced_mesh(const ExxReducedMeshAcceptanceInput& input)
{
    ExxReducedMeshAcceptance result;
    if (!input.exx_required)
    {
        return result;
    }
    if (!input.grid_result.found)
    {
        result.accepted = false;
        result.reason = ExxReducedMeshFallbackReason::grid_search_failed;
        return result;
    }
    if (input.requires_charge_grid_match && !input.charge_grid_matches)
    {
        result.accepted = false;
        result.reason = ExxReducedMeshFallbackReason::charge_grid_mismatch;
        return result;
    }
    if (!input.operator_memory_fits)
    {
        result.accepted = false;
        result.reason = ExxReducedMeshFallbackReason::operator_memory;
        return result;
    }
    if (input.stress_requested && !input.stress_memory_fits)
    {
        result.accepted = false;
        result.reason = ExxReducedMeshFallbackReason::stress_memory;
    }
    return result;
}

std::string describe_exx_fft_grid_incompatibility(
    const ExxFftGrid& grid,
    const std::vector<ExxSymmetryGridOperation>& operations,
    const ExxFftGridCompatibility& compatibility)
{
    std::ostringstream description;
    description << "PW EXX real-space symmetry operation is incompatible with FFT grid "
                << grid.nx << "x" << grid.ny << "x" << grid.nz;
    for (const ExxFftGridFailure& failure: compatibility.failures)
    {
        description << "\n  operation map index " << failure.operation_index << ", symop " << failure.symop;
        if (failure.operation_index >= 0
            && failure.operation_index < static_cast<int>(operations.size()))
        {
            const ExxSymmetryGridOperation& operation = operations[failure.operation_index];
            description << ", R=[[" << operation.rotation.e11 << "," << operation.rotation.e12 << ","
                        << operation.rotation.e13 << "],[" << operation.rotation.e21 << ","
                        << operation.rotation.e22 << "," << operation.rotation.e23 << "],["
                        << operation.rotation.e31 << "," << operation.rotation.e32 << ","
                        << operation.rotation.e33 << "]]"
                        << ", tau=(" << operation.translation.x << "," << operation.translation.y << ","
                        << operation.translation.z << ")";
        }
        if (failure.condition == ExxFftGridCondition::rotation_ratio)
        {
            description << ": R(" << axis_name(failure.source_axis) << "->" << axis_name(failure.target_axis)
                        << ")*N" << axis_name(failure.target_axis) << "/N" << axis_name(failure.source_axis)
                        << " = " << failure.value << " is not integral";
        }
        else if (failure.condition == ExxFftGridCondition::translation)
        {
            description << ": tau_" << axis_name(failure.target_axis) << "*N"
                        << axis_name(failure.target_axis) << " = " << failure.value << " is not integral";
        }
        else
        {
            description << ": FFT dimensions must be positive and non-overflowing";
        }
    }
    return description.str();
}

std::string describe_exx_fft_grid_policy_failure(const ExxFftGridPolicyInput& input,
                                                 const ExxFftGridPolicyResult& result)
{
    std::ostringstream description;
    const std::size_t candidate_trial_limit = effective_candidate_trial_limit(input);
    description << "PW EXX symmetry-compatible FFT-grid search stopped after "
                << result.candidate_trials << " of " << candidate_trial_limit
                << " enlarged-grid trials because " << stop_reason_description(result.stop_reason) << ".\n"
                << "  initial FFT grid " << input.initial_grid.nx << "x" << input.initial_grid.ny << "x"
                << input.initial_grid.nz << "; last attempted FFT grid " << result.last_attempted_grid.nx << "x"
                << result.last_attempted_grid.ny << "x" << result.last_attempted_grid.nz
                << "; memory search limit " << result.max_grid_points << " FFT points";
    if (!result.last_compatibility.compatible)
    {
        description << "\n"
                    << describe_exx_fft_grid_incompatibility(result.last_attempted_grid,
                                                             input.operations,
                                                             result.last_compatibility);
    }
    description << "\n  Try a different ecutexx value to obtain a different initial FFT grid. "
                   "Otherwise, use the automatic full-k fallback or disable k-point symmetry for PW EXX.";
    return description.str();
}

std::string describe_exx_reduced_mesh_memory_fallback(const ExxFftGridPolicyInput& grid_input,
                                                       const ExxFftGridPolicyResult& grid_result,
                                                       std::size_t required_bytes,
                                                       std::size_t budget_bytes,
                                                       bool stress_workload,
                                                       double ecut_exx)
{
    std::ostringstream description;
    description << "  The symmetry-compatible EXX " << (stress_workload ? "stress " : "operator ")
                << "grid exceeds the defensive tile-memory budget: initial grid "
                << grid_input.initial_grid.nx << "x" << grid_input.initial_grid.ny << "x"
                << grid_input.initial_grid.nz << ", compatible grid " << grid_result.grid.nx << "x"
                << grid_result.grid.ny << "x" << grid_result.grid.nz << ", required "
                << required_bytes << " bytes, budget " << budget_bytes << " bytes.\n"
                << "  Try a different ecutexx value (current ecutexx = " << ecut_exx
                << " Ry) or allow the automatic full-k fallback.";
    return description.str();
}

} // namespace hamilt
