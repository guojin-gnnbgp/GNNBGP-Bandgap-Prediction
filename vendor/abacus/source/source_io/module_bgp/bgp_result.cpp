#include "bgp_result.h"

#include "source_base/constants.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace
{

void require_finite(double value, const char* field)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error(std::string("non-finite BGP result field: ") + field);
    }
}

void validate_state(const ModuleIO::BgpElectronicState& state)
{
    if (state.nspin != 1)
    {
        throw std::runtime_error("BGP electronic-state output requires nspin = 1");
    }
    if (state.k_weights.empty())
    {
        throw std::runtime_error("BGP electronic-state output requires at least one k point");
    }
    if (state.eigenvalues_ry.size() != state.k_weights.size()
        || state.occupations.size() != state.k_weights.size())
    {
        throw std::runtime_error("BGP electronic-state k-point dimensions are inconsistent");
    }

    require_finite(state.fermi_energy_ry, "fermi_energy_ry");
    require_finite(state.printed_gap_ry, "printed_gap_ry");
    std::size_t bands = 0;
    for (std::size_t ik = 0; ik < state.k_weights.size(); ++ik)
    {
        require_finite(state.k_weights[ik], "k_weights");
        if (ik == 0)
        {
            bands = state.eigenvalues_ry[ik].size();
            if (bands == 0)
            {
                throw std::runtime_error("BGP electronic-state output requires at least one band");
            }
        }
        if (state.eigenvalues_ry[ik].size() != bands || state.occupations[ik].size() != bands)
        {
            throw std::runtime_error("BGP electronic-state band dimensions are inconsistent");
        }
        for (std::size_t ib = 0; ib < bands; ++ib)
        {
            require_finite(state.eigenvalues_ry[ik][ib], "eigenvalues_ry");
            require_finite(state.occupations[ik][ib], "occupations");
        }
    }
}

void write_number(std::ostream& output, double value)
{
    output << std::setprecision(15) << value;
}

void write_vector(std::ostream& output, const std::vector<double>& values, double scale)
{
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0)
        {
            output << ", ";
        }
        write_number(output, values[index] * scale);
    }
    output << ']';
}

void write_matrix(std::ostream& output, const std::vector<std::vector<double>>& rows, double scale)
{
    output << '[';
    for (std::size_t row = 0; row < rows.size(); ++row)
    {
        if (row != 0)
        {
            output << ", ";
        }
        write_vector(output, rows[row], scale);
    }
    output << ']';
}

} // namespace

namespace ModuleIO
{

void write_bgp_electronic_state(const std::string& path, const BgpElectronicState& state, int rank)
{
    if (rank != 0)
    {
        return;
    }
    validate_state(state);
    if (path.empty())
    {
        throw std::runtime_error("BGP electronic-state output path is empty");
    }

    const std::string temporary_path = path + ".tmp";
    try
    {
        std::ofstream output(temporary_path.c_str(), std::ios::out | std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error("could not open temporary BGP electronic-state output");
        }

        output << "{\n"
               << "  \"schema_version\": \"1.0\",\n"
               << "  \"converged\": " << (state.converged ? "true" : "false") << ",\n"
               << "  \"nspin\": 1,\n"
               << "  \"occupation_weighting\": \"abacus_k_weighted\",\n"
               << "  \"spin_capacity\": 2.0,\n"
               << "  \"energy_unit\": \"eV\",\n"
               << "  \"fermi_energy_ev\": ";
        write_number(output, state.fermi_energy_ry * ModuleBase::Ry_to_eV);
        output << ",\n  \"printed_gap_ev\": ";
        write_number(output, state.printed_gap_ry * ModuleBase::Ry_to_eV);
        output << ",\n  \"k_weights\": ";
        write_vector(output, state.k_weights, 1.0);
        output << ",\n  \"eigenvalues_ev\": ";
        write_matrix(output, state.eigenvalues_ry, ModuleBase::Ry_to_eV);
        output << ",\n  \"occupations\": ";
        write_matrix(output, state.occupations, 1.0);
        output << "\n}\n";
        output.close();
        if (!output)
        {
            throw std::runtime_error("could not flush temporary BGP electronic-state output");
        }
        if (std::rename(temporary_path.c_str(), path.c_str()) != 0)
        {
            throw std::runtime_error("could not atomically replace BGP electronic-state output");
        }
    }
    catch (...)
    {
        std::remove(temporary_path.c_str());
        throw;
    }
}

} // namespace ModuleIO
