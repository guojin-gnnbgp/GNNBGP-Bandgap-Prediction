#include "esolver_ks_pw.h"

#include "source_estate/cal_ux.h"
#include "source_estate/elecstate_pw.h"
#include "source_estate/module_charge/symmetry_rho.h"

#include "source_hsolver/diago_iter_assist.h"
#include "source_hsolver/hsolver_pw.h"
#include "source_hsolver/diago_params.h"

#include "source_hsolver/kernels/hegvd_op.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/module_bgp/bgp_result.h"
#include "source_io/module_training/training_dump.h"
#include "source_lcao/module_deltaspin/spin_constrain.h"
#include "source_pw/module_pwdft/onsite_proj.h"
#include "source_lcao/module_dftu/dftu.h"
#include "source_pw/module_pwdft/vsep_pw.h"
#include "source_pw/module_pwdft/hamilt_pw.h"

#include "source_pw/module_pwdft/forces.h"
#include "source_pw/module_pwdft/stress_pw.h"
#include "source_hamilt/module_xc/xc_functional.h" // use XC_Functional

#ifdef __DSP
#include "source_base/kernels/dsp/dsp_connector.h"
#endif

#include "source_pw/module_pwdft/setup_pot.h" // mohan add 20250929
#include "source_estate/setup_estate_pw.h" // mohan add 20251005
#include "source_io/module_ctrl/ctrl_output_pw.h" // mohan add 20250927
#include "source_estate/module_charge/chgmixing.h" // use charge mixing, mohan add 20251006 
#include "source_estate/update_pot.h" // mohan add 20251016
#include "source_pw/module_pwdft/update_cell_pw.h" // mohan add 20250309
#include "source_pw/module_pwdft/dftu_pw.h" // mohan add 20250309
#include "source_pw/module_pwdft/deltaspin_pw.h" // mohan add 20250309

#include "source_hamilt/module_xc/exx_info.h" // use GlobalC::exx_info
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/parallel_device.h"
#include "source_io/module_energy/write_bands.h"
#include "source_pw/module_pwdft/op_pw_exx.h"
#include "source_pw/module_pwdft/exx_fft_grid_policy.h"
#include "source_pw/module_pwdft/exx_tile_policy.h"
#include "source_pw/module_pwdft/op_pw_veff.h"
#include "source_pw/module_pwdft/setup_pwwfc.h"

#if defined(ENABLE_CIDER) && defined(USE_LIBXC)
#include "source_estate/module_pot/pot_cider_xc.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
hamilt::ExxOperatorOptions make_exx_operator_options(const Input_para& inp, const Exx_Info_Global& exx_info)
{
    hamilt::ExxOperatorOptions options;
    options.enabled = exx_info.cal_exx;
    options.batch_fft_size = inp.exx_batch_fft_size;
    options.band_tile_size = inp.exx_band_tile_size;
    options.q_tile_size = inp.exx_q_tile_size;
    options.configured_nbands = inp.nbands;
    options.nspin = inp.nspin;
    options.ecutexx = inp.ecutexx;
    options.ecutrho = inp.ecutrho;
    options.gamma_extrapolation = inp.exx_gamma_extrapolation;
    options.exxace = inp.exxace;
    options.separate_loop = exx_info.separate_loop;
    options.hybrid_alpha = exx_info.hybrid_alpha;
    options.auto_tiling = inp.exx_auto_tiling;
    options.tile_memory_budget_mb = inp.exx_tile_memory_budget_mb;
    options.fock_params = hamilt::exx_options_detail::coulomb_params_if_enabled(
        options.enabled,
        exx_info.coulomb_param,
        Conv_Coulomb_Pot_K::Coulomb_Type::Fock);
    options.erfc_params = hamilt::exx_options_detail::coulomb_params_if_enabled(
        options.enabled,
        exx_info.coulomb_param,
        Conv_Coulomb_Pot_K::Coulomb_Type::Erfc);
    return options;
}
} // namespace

namespace ModuleESolver
{

namespace
{

std::vector<hamilt::ExxSymmetryGridOperation> reduced_mesh_grid_operations(const K_Vectors& kv)
{
    std::map<int, hamilt::ExxSymmetryGridOperation> operations_by_symop;
    for (const K_Vectors::ExxFullPoint& point: kv.exx_full_k_map)
    {
        if (!point.active)
        {
            continue;
        }
        hamilt::ExxSymmetryGridOperation operation;
        operation.symop = point.symop;
        operation.rotation = point.gmatrix;
        operation.translation = point.gtrans;
        operations_by_symop[operation.symop] = operation;
    }
    std::vector<hamilt::ExxSymmetryGridOperation> operations;
    operations.reserve(operations_by_symop.size());
    for (const auto& entry: operations_by_symop)
    {
        operations.push_back(entry.second);
    }
    return operations;
}

int active_reduced_mesh_q_count(const K_Vectors& kv)
{
    return std::max(1,
                    static_cast<int>(std::count_if(kv.exx_full_q_map.begin(),
                                                   kv.exx_full_q_map.end(),
                                                   [](const K_Vectors::ExxFullPoint& point) { return point.active; })));
}

std::size_t reduced_mesh_max_star_size(const K_Vectors& kv)
{
    std::map<std::pair<int, int>, std::size_t> star_sizes;
    std::size_t maximum = 1;
    for (const K_Vectors::ExxFullPoint& point: kv.exx_full_k_map)
    {
        if (point.active)
        {
            maximum = std::max(maximum, ++star_sizes[std::make_pair(point.rep_pool, point.rep_local_index)]);
        }
    }
    return maximum;
}

template <typename T, typename Device>
using RealType = typename GetTypeReal<T>::type;

template <typename T, typename Device>
struct TrainingExxLabel
{
    double energy_ry = 0.0;
    std::vector<ModuleIO::TrainingBandEdgeState> selected_edges;
    std::vector<double> energy_density_ry_bohr3;
    double max_imaginary_abs_ry_bohr3 = 0.0;
    std::string k_policy;
    int nx = 0;
    int ny = 0;
    int nz = 0;
};

ModuleIO::TrainingSDMXWavefunction make_training_sdmx_wavefunction(
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU>* psi_in,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K* pw_wfc)
{
    ModuleIO::TrainingSDMXWavefunction out;
    if (psi_in == nullptr || pw_wfc == nullptr)
    {
        return out;
    }
    out.present = true;
    out.nspin = PARAM.inp.nspin;
    out.nkpt = psi_in->get_nk();
    out.nband = psi_in->get_nbands();
    out.npwk_max = pw_wfc->npwk_max;
    const std::size_t coeff_size = static_cast<std::size_t>(out.nspin)
                                   * static_cast<std::size_t>(out.nkpt)
                                   * static_cast<std::size_t>(out.nband)
                                   * static_cast<std::size_t>(out.npwk_max);
    const std::size_t kg_size = static_cast<std::size_t>(out.nkpt)
                                * static_cast<std::size_t>(out.npwk_max);
    out.coeffs_real_skbg.assign(coeff_size, 0.0);
    out.coeffs_imag_skbg.assign(coeff_size, 0.0);
    out.npw_k.assign(out.nkpt, 0.0);
    out.g_index_kg.assign(kg_size, 0.0);
    for (int ik = 0; ik < out.nkpt; ++ik)
    {
        const int ispin = PARAM.inp.nspin == 2 ? kv.isk[ik] : 0;
        const int npw = pw_wfc->npwk[ik];
        out.npw_k[ik] = static_cast<double>(npw);
        for (int ig = 0; ig < out.npwk_max; ++ig)
        {
            const std::size_t kg = static_cast<std::size_t>(ik) * out.npwk_max + ig;
            out.g_index_kg[kg] = static_cast<double>(ig < npw ? pw_wfc->getigl2ig(ik, ig) : 0);
        }
        for (int ib = 0; ib < out.nband; ++ib)
        {
            for (int ig = 0; ig < npw; ++ig)
            {
                const std::size_t offset =
                    (((static_cast<std::size_t>(ispin) * out.nkpt + ik) * out.nband + ib)
                     * out.npwk_max)
                    + ig;
                const auto coeff = psi_in->operator()(ik, ib, ig);
                out.coeffs_real_skbg[offset] = static_cast<double>(coeff.real());
                out.coeffs_imag_skbg[offset] = static_cast<double>(coeff.imag());
            }
        }
    }
    return out;
}

void setup_training_exx_label()
{
    GlobalC::exx_info.info_global.coulomb_param.clear();
    GlobalC::exx_info.info_global.cal_exx = true;
    GlobalC::exx_info.info_global.hybrid_alpha = 1.0;
    GlobalC::exx_info.info_global.hse_omega = 0.11;
    GlobalC::exx_info.info_global.ccp_type = Conv_Coulomb_Pot_K::Ccp_Type::Hf;
    GlobalC::exx_info.info_global.separate_loop = false;
    GlobalC::exx_info.info_global.hybrid_step = 1;
    GlobalC::exx_info.info_global.coulomb_param[Conv_Coulomb_Pot_K::Coulomb_Type::Fock] = {
        {{"alpha", "1"}}};
}

ModuleIO::TrainingBandEdgeLabel make_training_band_edge_label(
    std::vector<ModuleIO::TrainingBandEdgeState> selected_edges)
{
    ModuleIO::TrainingBandEdgeLabel label;
    if (selected_edges.empty())
    {
        return label;
    }
    for (const auto& edge: selected_edges)
    {
        if (!std::isfinite(edge.exx_shift_Ha))
        {
            return label;
        }
    }
    label.present = true;
    label.states = std::move(selected_edges);
    return label;
}

template <typename T, typename Device>
TrainingExxLabel<T, Device> evaluate_training_exx_label(const UnitCell& ucell,
                                                         const int* isk,
                                                         const ModulePW::PW_Basis_K* wfcpw,
                                                         const ModulePW::PW_Basis* rhopw,
                                                         K_Vectors* kv,
                                                         const ModuleBase::matrix* wg,
                                                         psi::Psi<T, Device>* psi,
                                                         const bool include_scalar,
                                                         const std::vector<ModuleIO::TrainingBandEdgeState>* selected_edges)
{
    const bool saved_cal_exx = GlobalC::exx_info.info_global.cal_exx;
    const auto saved_coulomb_param = GlobalC::exx_info.info_global.coulomb_param;
    const auto saved_ccp_type = GlobalC::exx_info.info_global.ccp_type;
    const double saved_hybrid_alpha = GlobalC::exx_info.info_global.hybrid_alpha;
    const double saved_hse_omega = GlobalC::exx_info.info_global.hse_omega;
    const bool saved_separate_loop = GlobalC::exx_info.info_global.separate_loop;
    const size_t saved_hybrid_step = GlobalC::exx_info.info_global.hybrid_step;

    using OperatorEXX = hamilt::OperatorEXXPW<T, Device>;
    const auto saved_fock_div = OperatorEXX::fock_div;
    const auto saved_erfc_div = OperatorEXX::erfc_div;
    auto restore_exx_state = [&]() {
        GlobalC::exx_info.info_global.cal_exx = saved_cal_exx;
        GlobalC::exx_info.info_global.coulomb_param = saved_coulomb_param;
        GlobalC::exx_info.info_global.ccp_type = saved_ccp_type;
        GlobalC::exx_info.info_global.hybrid_alpha = saved_hybrid_alpha;
        GlobalC::exx_info.info_global.hse_omega = saved_hse_omega;
        GlobalC::exx_info.info_global.separate_loop = saved_separate_loop;
        GlobalC::exx_info.info_global.hybrid_step = saved_hybrid_step;
        OperatorEXX::fock_div = saved_fock_div;
        OperatorEXX::erfc_div = saved_erfc_div;
    };

    try
    {
        setup_training_exx_label();
        OperatorEXX::fock_div.clear();
        OperatorEXX::erfc_div.clear();

        if (psi == nullptr || psi->get_pointer() == nullptr)
        {
            ModuleBase::WARNING_QUIT("evaluate_training_exx_label",
                                     "missing PW wavefunction data for one-shot EXX label evaluation");
        }

        hamilt::ExxOperatorOptions options;
        options.batch_fft_size = PARAM.inp.exx_batch_fft_size;
        options.band_tile_size = PARAM.inp.exx_band_tile_size;
        options.q_tile_size = PARAM.inp.exx_q_tile_size;
        options.nspin = PARAM.inp.nspin;
        options.ecutexx = PARAM.inp.ecutexx;
        options.ecutrho = PARAM.inp.ecutrho;
        options.gamma_extrapolation = false;
        options.exxace = false;
        options.separate_loop = false;
        options.hybrid_alpha = 1.0;
        options.fock_params = {{{"alpha", "1"}}};
        options.auto_tiling = PARAM.inp.exx_auto_tiling;
        options.tile_memory_budget_mb = PARAM.inp.exx_tile_memory_budget_mb;

        if (GlobalV::MY_RANK == 0)
        {
            GlobalV::ofs_running << " training_exx_label: requested one-shot full EXX tiles: batch FFT size = "
                                 << options.batch_fft_size
                                 << ", band tile size = " << options.band_tile_size
                                 << ", q tile size = " << options.q_tile_size
                                 << ", auto tiling = " << (options.auto_tiling ? "on" : "off")
                                 << ", memory budget = " << options.tile_memory_budget_mb << " MB" << std::endl;
        }

        hamilt::ExxExecutionContext execution_context;
        execution_context.my_rank = GlobalV::MY_RANK;
        execution_context.my_pool = kv->para_k.my_pool;
        execution_context.kpar = kv->para_k.kpar;
        execution_context.running_log = &GlobalV::ofs_running;
        OperatorEXX op_exx(isk, wfcpw, rhopw, kv, &ucell, options, execution_context);
        op_exx.set_psi(*psi);
        op_exx.set_wg(wg);
        TrainingExxLabel<T, Device> label;
        if (include_scalar && PARAM.inp.out_exx_energy_density)
        {
            auto result = op_exx.cal_exx_energy_with_density(psi);
            label.energy_ry = result.energy_ry;
            label.energy_density_ry_bohr3 = std::move(result.energy_density_ry_bohr3);
            label.max_imaginary_abs_ry_bohr3 = result.max_imaginary_abs_ry_bohr3;
            label.k_policy = std::move(result.k_policy);
            label.nx = result.nx;
            label.ny = result.ny;
            label.nz = result.nz;
            if (label.energy_density_ry_bohr3.size() != static_cast<std::size_t>(rhopw->nrxx))
            {
                ModuleBase::WARNING_QUIT("evaluate_training_exx_label",
                                         "one-shot EXX energy density has an inconsistent local grid size");
            }
        }
        else if (include_scalar)
        {
            label.energy_ry = op_exx.cal_exx_energy(psi);
        }
        if (include_scalar && !std::isfinite(label.energy_ry))
        {
            ModuleBase::WARNING_QUIT("evaluate_training_exx_label",
                                     "one-shot EXX label evaluation produced a non-finite energy");
        }
        if (selected_edges != nullptr)
        {
            op_exx.first_iter = false;
            std::vector<std::pair<int, int>> targets;
            targets.reserve(selected_edges->size());
            for (const auto& edge: *selected_edges)
            {
                targets.emplace_back(edge.k, edge.band);
            }
            const auto diagonal_ry = op_exx.cal_exx_diagonal_selected(psi, targets);
            label.selected_edges = *selected_edges;
            for (std::size_t index = 0; index < label.selected_edges.size(); ++index)
            {
                label.selected_edges[index].exx_shift_Ha = 0.5 * diagonal_ry[index];
            }
        }
        restore_exx_state();
        return label;
    }
    catch (...)
    {
        restore_exx_state();
        throw;
    }
}

template <typename T, typename Device>
std::vector<double> cal_local_potential_band_diagonal(const ModuleBase::matrix& potential,
                                                      const int* isk,
                                                      const ModulePW::PW_Basis_K* pw_wfc,
                                                      psi::Psi<T, Device>* psi_in)
{
    using Real = typename GetTypeReal<T>::type;
    using resize_op = base_device::memory::resize_memory_op<T, Device>;
    using delete_op = base_device::memory::delete_memory_op<T, Device>;
    using set_op = base_device::memory::set_memory_op<T, Device>;
    using dot_op = ModuleBase::dot_real_op<T, Device>;

    const int nk = psi_in->get_nk();
    const int nbands = psi_in->get_nbands();
    const int nbasis = psi_in->get_nbasis();
    std::vector<Real> potential_real(static_cast<std::size_t>(potential.nr) * potential.nc);
    for (std::size_t i = 0; i < potential_real.size(); ++i)
    {
        potential_real[i] = static_cast<Real>(potential.c[i]);
    }

    hamilt::Veff<hamilt::OperatorPW<T, Device>> veff(isk,
                                                     potential_real.data(),
                                                     potential.nr,
                                                     potential.nc,
                                                     pw_wfc);
    std::vector<double> diagonal(static_cast<std::size_t>(nk) * nbands, 0.0);
    T* hpsi = nullptr;
    resize_op()(hpsi, static_cast<std::size_t>(nbands) * nbasis);

    for (int ik = 0; ik < nk; ++ik)
    {
        psi_in->fix_k(ik);
        veff.init(ik);
        set_op()(hpsi, 0, static_cast<std::size_t>(nbands) * nbasis);
        veff.act(nbands, nbasis, psi_in->get_npol(), psi_in->get_pointer(), hpsi, psi_in->get_current_nbas(), true);
        for (int ib = 0; ib < nbands; ++ib)
        {
            const T* psi_band = psi_in->get_pointer() + static_cast<std::size_t>(ib) * nbasis;
            const T* hpsi_band = hpsi + static_cast<std::size_t>(ib) * nbasis;
            diagonal[static_cast<std::size_t>(ik) * nbands + ib] =
                dot_op()(psi_in->get_current_nbas(), psi_band, hpsi_band, false);
        }
    }

#ifdef __MPI
    Parallel_Common::reduce_data(diagonal.data(), static_cast<int>(diagonal.size()), POOL_WORLD);
#endif
    delete_op()(hpsi);
    return diagonal;
}

template <typename T, typename Device>
void maybe_write_exchange_shift_dump(const char* prefix,
                                     const int istep,
                                     const int iter,
                                     const K_Vectors& kv,
                                     const ModuleBase::matrix& ekb,
                                     const ModuleBase::matrix& wg,
                                     const double fermi_ry,
                                     const double cider_xmix,
                                     const double exx_hybrid_alpha,
                                     const std::vector<double>* cider_feature_diag,
                                     const std::vector<double>* exx_hybrid_diag)
{
    if (prefix == nullptr || std::string(prefix).empty())
    {
        return;
    }
    if (GlobalV::RANK_IN_POOL != 0)
    {
        return;
    }

    char filename[4096];
    std::snprintf(filename, sizeof(filename), "%s_step%04d_iter%04d.jsonl", prefix, istep, iter);
    std::ofstream out(filename);
    if (!out.good())
    {
        GlobalV::ofs_warning << "ExchangeShift: failed to open dump " << filename << std::endl;
        return;
    }
    out << std::setprecision(17);
    out << "{\"kind\":\"metadata\""
        << ",\"schema\":\"ABACUS_EXCHANGE_SHIFT_JSONL_V1\""
        << ",\"step\":" << istep
        << ",\"iter\":" << iter
        << ",\"nks\":" << kv.get_nks()
        << ",\"nbands\":" << ekb.nc
        << ",\"fermi_ry\":" << fermi_ry
        << ",\"cider_xmix\":" << cider_xmix
        << ",\"exx_hybrid_alpha\":" << exx_hybrid_alpha
        << ",\"pbe_x_available\":false"
        << ",\"cider_feature_available\":" << (cider_feature_diag == nullptr ? "false" : "true")
        << ",\"exx_hybrid_available\":" << (exx_hybrid_diag == nullptr ? "false" : "true")
        << "}\n";
    for (int ik = 0; ik < kv.get_nks(); ++ik)
    {
        const int ikglobal = kv.ik2iktot[ik];
        for (int ib = 0; ib < ekb.nc; ++ib)
        {
            const std::size_t idx = static_cast<std::size_t>(ik) * ekb.nc + ib;
            const bool have_cider = cider_feature_diag != nullptr && std::isfinite((*cider_feature_diag)[idx]);
            const bool have_exx = exx_hybrid_diag != nullptr && std::isfinite((*exx_hybrid_diag)[idx]);
            out << "{\"kind\":\"band\""
                << ",\"ik\":" << ik
                << ",\"ik_global\":" << ikglobal
                << ",\"ib\":" << ib
                << ",\"eigenvalue_ry\":" << ekb(ik, ib)
                << ",\"occupation\":" << wg(ik, ib)
                << ",\"pbe_x_ry\":0.0"
                << ",\"cider_feature_ry\":";
            if (have_cider)
            {
                out << (*cider_feature_diag)[idx];
            }
            else
            {
                out << "null";
            }
            out << ",\"exx_ry\":";
            if (have_exx)
            {
                out << (*exx_hybrid_diag)[idx];
            }
            else
            {
                out << "null";
            }
            out << ",\"exx_hybrid_ry\":";
            if (have_exx)
            {
                out << (*exx_hybrid_diag)[idx];
            }
            else
            {
                out << "null";
            }
            out << "}\n";
        }
    }
    GlobalV::ofs_running << "ExchangeShift: wrote dump " << filename << std::endl;
}

} // namespace

template <typename T, typename Device>
ESolver_KS_PW<T, Device>::ESolver_KS_PW()
{
    this->classname = "ESolver_KS_PW";
    this->basisname = "PW";
}

template <typename T, typename Device>
ESolver_KS_PW<T, Device>::~ESolver_KS_PW()
{
    //****************************************************
    // do not add any codes in this deconstructor funcion
    //****************************************************
    // delete Hamilt
    if (this->p_hamilt != nullptr)
    {
        delete this->p_hamilt;
        this->p_hamilt = nullptr;
    }

    // delete exx_helper
    if (this->exx_helper != nullptr)
    {
        delete this->exx_helper;
        this->exx_helper = nullptr;
    }

    // mohan add 2025-10-12
    this->stp.clean();
}

template <typename T, typename Device>
bool ESolver_KS_PW<T, Device>::accept_reduced_k_mesh(const UnitCell& ucell,
                                                      const Input_para& inp,
                                                      const K_Vectors& proposed_kv,
                                                      std::ofstream& log) const
{
    const bool native_exact_exchange
        = GlobalC::exx_info.info_global.cal_exx
          || hamilt::is_pw_exact_exchange_functional(inp.dft_functional)
          || (ucell.ntype > 0 && hamilt::is_pw_exact_exchange_functional(ucell.atoms[0].ncpp.xc_func));
    const bool exx_required = hamilt::needs_pw_exx_grid(native_exact_exchange,
                                                        inp.out_exx_label,
                                                        inp.out_bandgap_label);
    if (!exx_required)
    {
        return true;
    }

    double ecut_exx = inp.ecutexx;
    if (ecut_exx == 0.0)
    {
        ecut_exx = inp.ecutrho;
    }
    ModulePW::PW_Basis cutoff_basis("cpu", "double");
#ifdef __MPI
    cutoff_basis.initmpi(this->pw_rho->poolnproc, this->pw_rho->poolrank, this->pw_rho->pool_world);
#endif
    cutoff_basis.initgrids(this->pw_rho->lat0, this->pw_rho->latvec, ecut_exx);

    hamilt::ExxFftGridPolicyInput grid_input;
    grid_input.initial_grid = hamilt::ExxFftGrid(cutoff_basis.nx, cutoff_basis.ny, cutoff_basis.nz);
    if (inp.out_exx_energy_density
        && !hamilt::exx_fft_grid_matches(grid_input.initial_grid,
                                         this->pw_rhod->nx,
                                         this->pw_rhod->ny,
                                         this->pw_rhod->nz))
    {
        ModuleBase::WARNING_QUIT(
            "ESolver_KS_PW::accept_reduced_k_mesh",
            "out_exx_energy_density requires the cutoff-derived EXX FFT grid to match the charge-density grid; "
            "the EXX grid is "
                + std::to_string(grid_input.initial_grid.nx) + "x"
                + std::to_string(grid_input.initial_grid.ny) + "x"
                + std::to_string(grid_input.initial_grid.nz) + " while the charge grid is "
                + std::to_string(this->pw_rhod->nx) + "x" + std::to_string(this->pw_rhod->ny) + "x"
                + std::to_string(this->pw_rhod->nz)
                + ". Use ecutexx = 0 or an ecutexx/explicit charge grid combination that produces identical dimensions.");
    }
    grid_input.operations = reduced_mesh_grid_operations(proposed_kv);
    const hamilt::ExxFftGridSearchLimit search_limit
        = hamilt::make_exx_fft_grid_search_limit(inp.exx_tile_memory_budget_mb, sizeof(T));
    std::size_t tile_budget_bytes = 0;
    const bool tile_budget_valid
        = hamilt::exx_tile_effective_budget_bytes(inp.exx_tile_memory_budget_mb,
                                                  inp.exx_auto_tiling ? 0.75 : 1.0,
                                                  tile_budget_bytes);
    grid_input.max_grid_points = search_limit.max_grid_points;
    grid_input.max_candidate_trials = hamilt::EXX_FFT_GRID_MAX_CANDIDATE_TRIALS;

    hamilt::ExxReducedMeshAcceptanceInput acceptance_input;
    acceptance_input.exx_required = true;
    acceptance_input.requires_charge_grid_match = inp.out_exx_energy_density;
    acceptance_input.stress_requested = inp.cal_stress;
    if (search_limit.valid && tile_budget_valid)
    {
        acceptance_input.grid_result = hamilt::choose_exx_fft_grid(grid_input);
        acceptance_input.charge_grid_matches
            = acceptance_input.grid_result.found
              && hamilt::exx_fft_grid_matches(acceptance_input.grid_result.grid,
                                              this->pw_rhod->nx,
                                              this->pw_rhod->ny,
                                              this->pw_rhod->nz);
    }
    log << " PW EXX reduced-mesh precheck: initial FFT grid = " << grid_input.initial_grid.nx << "x"
        << grid_input.initial_grid.ny << "x" << grid_input.initial_grid.nz
        << ", active symmetry operations = " << grid_input.operations.size();
    if (search_limit.valid && acceptance_input.grid_result.found)
    {
        log << ", selected FFT grid = " << acceptance_input.grid_result.grid.nx << "x"
            << acceptance_input.grid_result.grid.ny << "x" << acceptance_input.grid_result.grid.nz
            << ", adjusted = " << (acceptance_input.grid_result.adjusted ? "yes" : "no");
    }
    log << std::endl;

    hamilt::ExxTilePolicyResult operator_tiles;
    hamilt::ExxTilePolicyResult stress_tiles;
    bool operator_estimate_overflow = false;
    bool stress_estimate_overflow = false;
    if (search_limit.valid && acceptance_input.grid_result.found && acceptance_input.grid_result.adjusted
        && !acceptance_input.requires_charge_grid_match)
    {
        hamilt::ExxTilePolicyInput tile_input;
        tile_input.auto_tiling = inp.exx_auto_tiling;
        tile_input.device = inp.device;
        tile_input.precision = std::is_same<RealType<T, Device>, float>::value ? "single" : "double";
        tile_input.memory_budget_mb = inp.exx_tile_memory_budget_mb;
        tile_input.nbands = std::max(1, inp.nbands);
        tile_input.active_q_count = active_reduced_mesh_q_count(proposed_kv);
        tile_input.max_star_size = reduced_mesh_max_star_size(proposed_kv);
        tile_input.nrxx = acceptance_input.grid_result.grid_points;
        tile_input.npw = acceptance_input.grid_result.grid_points;
        tile_input.npwk_max = acceptance_input.grid_result.grid_points;
        tile_input.target_npwk_max = acceptance_input.grid_result.grid_points;
        tile_input.fft_nx = acceptance_input.grid_result.grid.nx;
        tile_input.fft_ny = acceptance_input.grid_result.grid.ny;
        tile_input.fft_nz = acceptance_input.grid_result.grid.nz;
        tile_input.requested_batch_fft_size = inp.exx_batch_fft_size;
        tile_input.requested_band_tile_size = inp.exx_band_tile_size;
        tile_input.requested_q_tile_size = inp.exx_q_tile_size;
        if (!hamilt::estimate_exx_fixed_scratch_bytes(tile_input.nrxx,
                                                      tile_input.npw,
                                                      tile_input.npwk_max,
                                                      tile_input.target_npwk_max,
                                                      sizeof(T),
                                                      sizeof(RealType<T, Device>),
                                                      tile_input.fixed_scratch_bytes))
        {
            acceptance_input.operator_memory_fits = false;
            operator_estimate_overflow = true;
            operator_tiles.budget_bytes = tile_budget_bytes;
        }
        else
        {
            std::size_t required_bytes = 0;
            acceptance_input.operator_memory_fits
                = hamilt::estimate_exx_reduced_mesh_guard_bytes(tile_input, false, required_bytes)
                  && required_bytes <= tile_budget_bytes;
            operator_tiles.minimum_required_bytes = required_bytes;
            operator_tiles.requested_bytes = required_bytes;
            operator_tiles.budget_bytes = tile_budget_bytes;
            operator_tiles.fits = acceptance_input.operator_memory_fits;
        }

        if (inp.cal_stress)
        {
            if (!hamilt::estimate_exx_stress_fixed_bytes(tile_input.nrxx,
                                                         tile_input.npw,
                                                         tile_input.npwk_max,
                                                         static_cast<std::size_t>(proposed_kv.get_nks()),
                                                         sizeof(std::complex<double>),
                                                         sizeof(double),
                                                         tile_input.fixed_scratch_bytes))
            {
                acceptance_input.stress_memory_fits = false;
                stress_estimate_overflow = true;
                stress_tiles.budget_bytes = tile_budget_bytes;
            }
            else
            {
                tile_input.precision = "double";
                std::size_t required_bytes = 0;
                acceptance_input.stress_memory_fits
                    = hamilt::estimate_exx_reduced_mesh_guard_bytes(tile_input, true, required_bytes)
                      && required_bytes <= tile_budget_bytes;
                stress_tiles.minimum_required_bytes = required_bytes;
                stress_tiles.requested_bytes = required_bytes;
                stress_tiles.budget_bytes = tile_budget_bytes;
                stress_tiles.fits = acceptance_input.stress_memory_fits;
            }
        }
    }

    const hamilt::ExxReducedMeshAcceptance acceptance = hamilt::assess_exx_reduced_mesh(acceptance_input);
    if (acceptance.accepted)
    {
        return true;
    }

    log << " WARNING: PW EXX disabled k-point symmetry reduction before wavefunction allocation." << std::endl;
    if (!search_limit.valid || !tile_budget_valid)
    {
        log << "  The EXX FFT-grid memory search limit is invalid." << std::endl;
    }
    else if (acceptance.reason == hamilt::ExxReducedMeshFallbackReason::grid_search_failed)
    {
        log << hamilt::describe_exx_fft_grid_policy_failure(grid_input, acceptance_input.grid_result) << std::endl;
    }
    else if (acceptance.reason == hamilt::ExxReducedMeshFallbackReason::charge_grid_mismatch)
    {
        log << "  out_exx_energy_density requires the EXX FFT grid to match the charge-density grid; "
               "the reduced mesh requires an enlarged symmetry-compatible embedding."
            << std::endl;
    }
    else if (acceptance.reason == hamilt::ExxReducedMeshFallbackReason::operator_memory)
    {
        if (operator_estimate_overflow)
        {
            log << "  The PW EXX operator memory estimate overflowed size_t for the compatible grid; "
                   "the configured budget is "
                << operator_tiles.budget_bytes << " bytes." << std::endl;
        }
        else
        {
            const std::size_t required_bytes = operator_tiles.requested_bytes > 0
                                                   ? operator_tiles.requested_bytes
                                                   : operator_tiles.minimum_required_bytes;
            log << hamilt::describe_exx_reduced_mesh_memory_fallback(grid_input,
                                                                     acceptance_input.grid_result,
                                                                     required_bytes,
                                                                     operator_tiles.budget_bytes,
                                                                     false,
                                                                     ecut_exx)
                << std::endl;
        }
    }
    else if (acceptance.reason == hamilt::ExxReducedMeshFallbackReason::stress_memory)
    {
        if (stress_estimate_overflow)
        {
            log << "  The PW EXX stress memory estimate overflowed size_t for the compatible grid; "
                   "the configured budget is "
                << stress_tiles.budget_bytes << " bytes." << std::endl;
        }
        else
        {
            const std::size_t required_bytes = stress_tiles.requested_bytes > 0
                                                   ? stress_tiles.requested_bytes
                                                   : stress_tiles.minimum_required_bytes;
            log << hamilt::describe_exx_reduced_mesh_memory_fallback(grid_input,
                                                                     acceptance_input.grid_result,
                                                                     required_bytes,
                                                                     stress_tiles.budget_bytes,
                                                                     true,
                                                                     ecut_exx)
                << std::endl;
        }
    }
    log << "  Rebuilding the full-k mesh for PW EXX." << std::endl;
    return false;
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::allocate_hamilt(const UnitCell& ucell)
{
	this->p_hamilt = new hamilt::HamiltPW<T, Device>(
			this->pelec->pot, 
			this->pw_wfc, 
			&this->kv, 
			&this->ppcell, 
			&this->dftu,
			&ucell,
            *this->exx_options_,
            *this->exx_context_);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::validate_mixed_band_targets(
    const Input_para& inp,
    const hamilt::ExxOperatorOptions& exx_options) const
{
    if (inp.mem_saver == 1 && inp.calculation == "scf" && !this->kv.has_band_kpoints())
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mem_saver=1 for scf PW calculations requires K_POINTS_BAND target k-points");
    }
    if (!this->kv.has_band_kpoints())
    {
        return;
    }

    if (inp.calculation != "scf")
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mixed hybrid band targets require calculation scf");
    }
    if (inp.basis_type != "pw")
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mixed hybrid band targets require basis_type pw");
    }
    if (!inp.out_band[0])
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "K_POINTS_BAND requires out_band 1");
    }
    if (!exx_options.enabled || !inp.exxace || !exx_options.separate_loop)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mixed hybrid band targets require PW hybrid EXX with exx_ace 1 and exx_separate_loop 1");
    }
    if (inp.kpar > 1)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mixed hybrid band targets do not support KPAR>1 in v1");
    }
    if (inp.mem_saver == 1 && inp.ks_solver == "cg")
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mem_saver=1 for mixed hybrid band targets does not support ks_solver cg; "
                                 "use dav, dav_subspace, or bpcg");
    }
    if (inp.mem_saver == 1 && inp.out_wfc_pw != 0)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mixed hybrid band targets write eigenvalues only and do not support out_wfc_pw "
                                 "with mem_saver=1");
    }
}



template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::before_all_runners(UnitCell& ucell, const Input_para& inp)
{
    ESolver_KS::before_all_runners(ucell, inp);
    this->input_parameters_.reset(new const Input_para(inp));
    this->system_parameters_.reset(new const System_para(PARAM.globalv));
    this->my_bndgroup_ = GlobalV::MY_BNDGROUP;
    this->nproc_in_pool_ = this->kv.para_k.get_nproc_in_pool(inp.bndpar);
    this->running_log_ = &GlobalV::ofs_running;
    this->exx_options_.reset(
        new const hamilt::ExxOperatorOptions(make_exx_operator_options(inp, GlobalC::exx_info.info_global)));
    hamilt::ExxExecutionContext exx_context;
    exx_context.my_rank = this->system_parameters_->myrank;
    exx_context.my_pool = this->kv.para_k.my_pool;
    exx_context.kpar = inp.kpar;
    exx_context.running_log = this->running_log_;
    this->exx_context_.reset(new const hamilt::ExxExecutionContext(exx_context));
    this->validate_mixed_band_targets(inp, *this->exx_options_);

    //! setup and allocation for pelec, potentials, etc. 
    elecstate::setup_estate_pw(ucell, this->kv, this->sf, this->pelec, this->chr,
      this->locpp, this->ppcell, this->vsep_cell, this->pw_wfc, this->pw_rho,
      this->pw_rhod, this->pw_big, this->solvent, inp);

    this->stp.before_runner(ucell,
                            this->kv,
                            this->sf,
                            *this->pw_wfc,
                            this->ppcell,
                            *this->input_parameters_,
                            *this->system_parameters_,
                            this->my_bndgroup_,
                            *this->running_log_,
                            false);

    ModuleBase::GlobalFunc::DONE(*this->running_log_, "INIT BASIS");

    //! Create exx_helper based on device and precision
    const bool is_gpu = (inp.device == "gpu");
    const bool is_single = (inp.precision == "single");

#if ((defined __CUDA) || (defined __ROCM))
    if (is_gpu)
    {
        if (is_single)
        {
            this->exx_helper = new Exx_Helper<std::complex<float>, base_device::DEVICE_GPU>();
        }
        else
        {
            this->exx_helper = new Exx_Helper<std::complex<double>, base_device::DEVICE_GPU>();
        }
    }
    else
#endif
    {
        if (is_single)
        {
            this->exx_helper = new Exx_Helper<std::complex<float>, base_device::DEVICE_CPU>();
        }
        else
        {
            this->exx_helper = new Exx_Helper<std::complex<double>, base_device::DEVICE_CPU>();
        }
    }

    //! Initialize exx pw
    this->exx_helper->init(ucell, inp, this->pelec->wg);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::before_scf(UnitCell& ucell, const int istep)
{
    ModuleBase::TITLE("ESolver_KS_PW", "before_scf");
    ModuleBase::timer::start("ESolver_KS_PW", "before_scf");

    ESolver_KS::before_scf(ucell, istep);

    //! Init variables (once the cell has changed)
    pw::update_cell_pw(ucell, this->ppcell, this->kv, this->pw_wfc, PARAM.inp);

    if (ucell.cell_parameter_updated)
    {
        this->stp.p_psi_init->prepare_init(PARAM.inp.pw_seed);
    }

    //! Init Hamiltonian (cell changed)
    //! Operators in HamiltPW should be reallocated once cell changed
    //! delete Hamilt if not first scf
    if (this->p_hamilt != nullptr)
    {
        delete this->p_hamilt;
        this->p_hamilt = nullptr;
    }

    //! Allocate HamiltPW
    this->allocate_hamilt(ucell);

    //! Setup potentials (local, non-local, sc, +U, DFT-1/2)
    // note: init DFT+U is done here for pw basis for every scf iteration, however,
    // init DFT+U is done in "before_all_runners" in LCAO basis. This should be refactored, mohan note 2025-11-06
    pw::setup_pot(istep, ucell, this->kv, this->sf, this->pelec, this->Pgrid,
              this->chr, this->locpp, this->ppcell, this->dftu, this->vsep_cell,
              this->stp.template get_psi_t<T, Device>(),
	      this->p_hamilt,
	      this->pw_wfc, this->pw_rhod, PARAM.globalv.global_out_dir, PARAM.inp);

    // setup psi (electronic wave functions)
    this->stp.init(this->p_hamilt);

    //! Setup EXX helper for Hamiltonian and psi
    exx_helper->before_scf(this->p_hamilt, this->stp.template get_psi_t<T, Device>(), PARAM.inp);

    ModuleBase::timer::end("ESolver_KS_PW", "before_scf");
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::iter_init(UnitCell& ucell, const int istep, const int iter)
{
    ESolver_KS::iter_init(ucell, istep, iter);

    module_charge::chgmixing_ks_pw(iter, this->p_chgmix, this->dftu, PARAM.inp);

    // mohan move harris functional here, 2012-06-05
    // use 'rho(in)' and 'v_h and v_xc'(in)
    this->pelec->f_en.deband_harris = this->pelec->cal_delta_eband(ucell);

    // update local occupations for DFT+U
    // should before lambda loop in DeltaSpin
    pw::iter_init_dftu_pw(iter, istep, this->dftu, this->stp.template get_psi_t<T, Device>(), this->pelec->wg, ucell, PARAM.inp);
}

// Temporary, it should be replaced by hsolver later.
template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::hamilt2rho_single(UnitCell& ucell, const int istep, const int iter, const double ethr)
{
    ModuleBase::timer::start("ESolver_KS_PW", "hamilt2rho_single");

    // reset energy
    this->pelec->f_en.eband = 0.0;
    this->pelec->f_en.demet = 0.0;

    // setup diagonalization parameters
    hsolver::setup_diago_params_pw<T, Device>(istep, iter, ethr, PARAM.inp);

    bool skip_charge = PARAM.inp.calculation == "nscf" ? true : false;

    // run the inner lambda loop to contrain atomic moments with the DeltaSpin method
    bool skip_solve = pw::run_deltaspin_lambda_loop(iter - 1, this->drho, PARAM.inp);

    if (!skip_solve)
    {
        hsolver::HSolverPW<T, Device> hsolver_pw_obj(this->pw_wfc,
                                                     PARAM.inp.calculation,
                                                     PARAM.inp.basis_type,
                                                     PARAM.inp.ks_solver,
                                                     PARAM.globalv.use_uspp,
                                                     PARAM.inp.nspin,
                                                     hsolver::DiagoIterAssist<T, Device>::SCF_ITER,
                                                     hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX,
                                                     hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR,
                                                     hsolver::DiagoIterAssist<T, Device>::need_subspace,
                                                     PARAM.inp.use_k_continuity);

        hsolver_pw_obj.solve(static_cast<hamilt::Hamilt<T, Device>*>(this->p_hamilt), *this->stp.template get_psi_t<T, Device>(), this->pelec, this->pelec->ekb.c,
          GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL, skip_charge, ucell.tpiba, ucell.nat);
    }

    const char* exchange_shift_prefix = std::getenv("ABACUS_EXCHANGE_SHIFT_DUMP_PREFIX");
    if (exchange_shift_prefix != nullptr && !std::string(exchange_shift_prefix).empty() && !skip_solve)
    {
        std::vector<double> cider_feature_diag;
        std::vector<double>* cider_feature_diag_ptr = nullptr;
#if defined(ENABLE_CIDER) && defined(USE_LIBXC)
        if (!PARAM.inp.cider_model.empty())
        {
            const ModuleBase::matrix* v_feature = elecstate::PotCiderXC::debug_last_feature_potential();
            if (v_feature != nullptr)
            {
                cider_feature_diag = cal_local_potential_band_diagonal(
                    *v_feature,
                    this->kv.isk.data(),
                    this->pw_wfc,
                    this->stp.template get_psi_t<T, Device>());
                cider_feature_diag_ptr = &cider_feature_diag;
            }
        }
#endif

        std::vector<double> exx_hybrid_diag;
        std::vector<double>* exx_hybrid_diag_ptr = nullptr;
        if (GlobalC::exx_info.info_global.cal_exx && this->exx_helper != nullptr
            && this->exx_helper->cal_exx_hybrid_band_diagonal(this->stp.template get_psi_t<T, Device>(), exx_hybrid_diag))
        {
            exx_hybrid_diag_ptr = &exx_hybrid_diag;
        }

        maybe_write_exchange_shift_dump<T, Device>(
            exchange_shift_prefix,
            istep,
            iter,
            this->kv,
            this->pelec->ekb,
            this->pelec->wg,
            this->pelec->eferm.ef,
            PARAM.inp.cider_xmix,
            GlobalC::exx_info.info_global.hybrid_alpha,
            cider_feature_diag_ptr,
            exx_hybrid_diag_ptr);
    }

    // symmetrize the charge density
    Symmetry_rho::symmetrize_rho(PARAM.inp.nspin, this->chr, this->pw_rhod, ucell.symm);

    ModuleBase::timer::end("ESolver_KS_PW", "hamilt2rho_single");
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::iter_finish(UnitCell& ucell, const int istep, int& iter, bool& conv_esolver)
{
    // Related to EXX
    if (GlobalC::exx_info.info_global.cal_exx && !exx_helper->get_op_first_iter())
    {
        this->pelec->set_exx(exx_helper->cal_exx_energy(this->stp.template get_psi_t<T, Device>()));
    }

    // deband is calculated from "output" charge density
    this->pelec->f_en.deband = this->pelec->cal_delta_eband(ucell);

    // Call iter_finish() of ESolver_KS
    ESolver_KS::iter_finish(ucell, istep, iter, conv_esolver);

    // D in USPP needs vloc, thus needs update when veff updated
    // calculate the effective coefficient matrix for non-local
    // pp projectors, liuyu 2023-10-24
    if (PARAM.globalv.use_uspp)
    {
        ModuleBase::matrix veff = this->pelec->pot->get_eff_v();
        this->ppcell.cal_effective_D(veff, this->pw_rhod, ucell);
    }

    // Handle EXX-related operations after SCF iteration
    exx_helper->iter_finish(this->pelec, &this->chr, this->stp.template get_psi_t<T, Device>(), ucell, PARAM.inp, conv_esolver, iter);

    // check if oscillate for delta_spin method
    pw::check_deltaspin_oscillation(iter, this->drho, this->p_chgmix, PARAM.inp);

    // the output quantities
    ModuleIO::ctrl_iter_pw(istep, iter, conv_esolver, this->stp.psi_cpu, 
              this->kv, this->pw_wfc, PARAM.inp);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::after_scf(UnitCell& ucell, const int istep, const bool conv_esolver)
{
    ModuleBase::TITLE("ESolver_KS_PW", "after_scf");
    ModuleBase::timer::start("ESolver_KS_PW", "after_scf");

    const bool write_training_data = PARAM.inp.out_training_data;
    const bool write_scalar_exx = PARAM.inp.out_exx_label;
    const bool write_band_edges = PARAM.inp.out_bandgap_label;
    const double saved_exx_energy = this->pelec->f_en.exx;
    ModuleIO::clear_last_training_band_edge_label();
    ModuleIO::clear_last_training_exx_energy_density();
    if (write_scalar_exx || write_band_edges)
    {
        std::vector<ModuleIO::TrainingBandEdgeState> selected_edges;
        if (write_band_edges)
        {
            selected_edges = ModuleIO::select_training_band_edges(this->kv,
                                                                   this->pelec->ekb,
                                                                   this->pelec->wg,
                                                                   PARAM.inp.band_edge_degeneracy_tolerance);
        }
        ModuleBase::TITLE("ESolver_KS_PW", "training_exx_label");
        ModuleBase::timer::start("ESolver_KS_PW", "training_exx_label");
        auto exx_label = evaluate_training_exx_label<T, Device>(ucell,
                                                                 this->kv.isk.data(),
                                                                 this->pw_wfc,
                                                                 this->pw_rhod,
                                                                 &this->kv,
                                                                 &this->pelec->wg,
                                                                 this->stp.template get_psi_t<T, Device>(),
                                                                 write_scalar_exx,
                                                                 write_band_edges ? &selected_edges : nullptr);
        if (write_scalar_exx)
        {
            this->pelec->f_en.exx = exx_label.energy_ry;
            if (PARAM.inp.out_exx_energy_density)
            {
                ModuleIO::TrainingExxEnergyDensity energy_density;
                energy_density.present = true;
                energy_density.values_local_Ha_bohr3 = std::move(exx_label.energy_density_ry_bohr3);
                for (double& value: energy_density.values_local_Ha_bohr3)
                {
                    value *= 0.5;
                }
                energy_density.scalar_energy_Ha = 0.5 * exx_label.energy_ry;
                energy_density.max_imaginary_abs_Ha_bohr3
                    = 0.5 * exx_label.max_imaginary_abs_ry_bohr3;
                energy_density.k_policy = std::move(exx_label.k_policy);
                energy_density.nx = exx_label.nx;
                energy_density.ny = exx_label.ny;
                energy_density.nz = exx_label.nz;
                ModuleIO::set_last_training_exx_energy_density(energy_density);
            }
        }
        if (write_band_edges)
        {
            auto band_edge_label = make_training_band_edge_label(std::move(exx_label.selected_edges));
            if (!band_edge_label.present)
            {
                ModuleBase::WARNING_QUIT("ESolver_KS_PW",
                                         "out_bandgap_label could not identify finite VBM/CBM EXX band-edge labels");
            }
            auto* elec_pw = static_cast<elecstate::ElecStatePW<T, Device>*>(this->pelec);
            auto& psi = *this->stp.template get_psi_t<T, Device>();
            elec_pw->cal_band_edge_orbital_responses(psi,
                                                     band_edge_label.states,
                                                     band_edge_label.orbital_rho_tsg,
                                                     band_edge_label.orbital_tau_tsg);
            ModuleIO::set_last_training_band_edge_label(band_edge_label);
            GlobalV::ofs_running << "training_exx_label: selected band-edge EXX diagonals complete"
                                 << std::endl;
        }
        if (write_scalar_exx)
        {
            const std::streamsize saved_precision = GlobalV::ofs_running.precision();
            GlobalV::ofs_running << "training_exx_label: full EXX energy = " << std::setprecision(16)
                                 << 0.5 * this->pelec->f_en.exx << " Ha" << std::endl;
            GlobalV::ofs_running.precision(saved_precision);
        }
        ModuleBase::timer::end("ESolver_KS_PW", "training_exx_label");
    }

    // Calculate kinetic energy density tau immediately before output paths
    // that consume it. Keep this independent of the XC meta-GGA flag so PBE
    // training dumps do not enter the meta-GGA XC path.
    if ((PARAM.inp.out_elf[0] > 0 || PARAM.inp.out_training_data || PARAM.inp.out_cider_label)
        && this->chr.kin_r != nullptr)
    {
        auto* elec_pw = static_cast<elecstate::ElecStatePW<T, Device>*>(this->pelec);
        auto& psi = *this->stp.template get_psi_t<T, Device>();
        elec_pw->cal_tau(psi);
    }

#if defined(ENABLE_CIDER) && defined(USE_LIBXC)
    if (PARAM.inp.out_cider_label)
    {
        ModuleBase::TITLE("ESolver_KS_PW", "training_cider_label");
        ModuleBase::timer::start("ESolver_KS_PW", "training_cider_label");
        double cider_etxc = 0.0;
        double cider_vtxc = 0.0;
        ModuleBase::matrix cider_vofk;
        cider_vofk.create(PARAM.inp.nspin, this->pw_rhod->nrxx);
        elecstate::PotCiderXC cider_eval(this->pw_rhod, &ucell, &cider_etxc, &cider_vtxc, &cider_vofk);
        ModuleBase::matrix dummy_v_eff(PARAM.inp.nspin, this->pw_rhod->nrxx);
        cider_eval.cal_v_eff(&this->chr, &ucell, dummy_v_eff);
        const auto& cider_energy = elecstate::PotCiderXC::last_energy_components();
        if (!cider_energy.valid)
        {
            ModuleBase::WARNING_QUIT("ESolver_KS_PW", "post-SCF CIDER label evaluation did not produce energy components");
        }
        GlobalV::ofs_running
            << "training_cider_label: feature scaled exchange energy = "
            << cider_energy.feature_scaled_Ha << " Ha" << std::endl;
        GlobalV::ofs_running
            << "training_cider_label: feature unmixed exchange energy = "
            << cider_energy.feature_unmixed_Ha << " Ha" << std::endl;
        GlobalV::ofs_running
            << "training_cider_label: baseline exchange-correlation energy = "
            << cider_energy.baseline_Ha << " Ha" << std::endl;
        GlobalV::ofs_running
            << "training_cider_label: total evaluated energy = "
            << cider_energy.total_Ha
            << " Ha feature_density=" << cider_energy.feature_density_policy
            << " exc_convention=" << cider_energy.feature_exc_convention
            << " xmix=" << PARAM.inp.cider_xmix << std::endl;
        ModuleBase::timer::end("ESolver_KS_PW", "training_cider_label");
    }
#else
    if (PARAM.inp.out_cider_label)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW", "out_cider_label requires ENABLE_CIDER and USE_LIBXC");
    }
#endif

    if (PARAM.inp.out_sdmx_descriptor_data)
    {
        this->stp.copy_d2h();
        ModuleIO::set_last_training_sdmx_wavefunction(
            make_training_sdmx_wavefunction(this->stp.psi_cpu, this->kv, this->pw_wfc));
    }
    else
    {
        ModuleIO::clear_last_training_sdmx_wavefunction();
    }

    // Call 'after_scf' of ESolver_KS
    ESolver_KS::after_scf(ucell, istep, conv_esolver);

    if (PARAM.inp.out_bgp_result)
    {
        ModuleIO::BgpElectronicState state;
        state.converged = conv_esolver;
        state.nspin = PARAM.inp.nspin;
        state.fermi_energy_ry = this->pelec->eferm.ef;
        state.printed_gap_ry = this->pelec->bandgap;
        state.k_weights = this->kv.wk;
        state.eigenvalues_ry.resize(this->pelec->ekb.nr);
        state.occupations.resize(this->pelec->wg.nr);
        for (int ik = 0; ik < this->pelec->ekb.nr; ++ik)
        {
            state.eigenvalues_ry[ik].resize(this->pelec->ekb.nc);
            state.occupations[ik].resize(this->pelec->wg.nc);
            for (int ib = 0; ib < this->pelec->ekb.nc; ++ib)
            {
                state.eigenvalues_ry[ik][ib] = this->pelec->ekb(ik, ib);
                state.occupations[ik][ib] = this->pelec->wg(ik, ib);
            }
        }
        ModuleIO::write_bgp_electronic_state(PARAM.globalv.global_out_dir + "bgp_electronic_state.json",
                                              state,
                                              GlobalV::MY_RANK);
    }
    if (write_scalar_exx && !write_training_data)
    {
        this->pelec->f_en.exx = saved_exx_energy;
    }

    // Output quantities
    ModuleIO::ctrl_scf_pw<T, Device>(istep, ucell, this->pelec, this->chr, this->kv, this->pw_wfc,
              this->pw_rho, this->pw_rhod, this->pw_big, this->stp,
              this->Pgrid, PARAM.inp);
    ModuleIO::clear_last_training_exx_energy_density();

    if (write_scalar_exx && write_training_data)
    {
        this->pelec->f_en.exx = saved_exx_energy;
    }

    if (conv_esolver)
    {
        this->solve_mixed_band_targets(ucell, *this->input_parameters_, *this->system_parameters_);
    }

    ModuleBase::timer::end("ESolver_KS_PW", "after_scf");
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::solve_mixed_band_targets(UnitCell& ucell,
                                                        const Input_para& inp,
                                                        const System_para& sys)
{
    if (!this->kv.has_band_kpoints())
    {
        return;
    }

    ModuleBase::TITLE("ESolver_KS_PW", "solve_mixed_band_targets");
    ModuleBase::timer::start("ESolver_KS_PW", "mixed_band_targets");

    auto* source_exx = static_cast<hamilt::OperatorEXXPW<T, Device>*>(this->exx_helper->get_op_exx());
    if (source_exx == nullptr)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::solve_mixed_band_targets",
                                 "SCF EXX source operator is not initialized");
    }

    if (inp.mem_saver == 1)
    {
        this->solve_mixed_band_targets_mem_saver(ucell, source_exx, inp, sys);
    }
    else
    {
        this->solve_mixed_band_targets_full(ucell, source_exx, inp, sys);
    }

    ModuleBase::timer::end("ESolver_KS_PW", "mixed_band_targets");
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::solve_mixed_band_targets_full(UnitCell& ucell,
                                                             hamilt::OperatorEXXPW<T, Device>* source_exx,
                                                             const Input_para& inp,
                                                             const System_para& sys)
{
    K_Vectors band_kv = this->kv.make_band_target_kvectors(inp.nspin);
    ModulePW::PW_Basis_K* band_pw_wfc = nullptr;
    pw::setup_pwwfc(inp, ucell, *this->pw_rho, band_kv, band_pw_wfc);

    pseudopot_cell_vnl band_ppcell;
    band_ppcell.init(ucell, &this->sf, band_pw_wfc);
    band_ppcell.init_vnl(ucell, this->pw_rhod);
    ModuleBase::matrix veff = this->pelec->pot->get_eff_v();
    band_ppcell.cal_effective_D(veff, this->pw_rhod, ucell);

    Setup_Psi_pw band_stp;
    band_stp.before_runner(ucell,
                           band_kv,
                           this->sf,
                           *band_pw_wfc,
                           band_ppcell,
                           inp,
                           sys,
                           this->my_bndgroup_,
                           *this->running_log_,
                           false);

    auto* band_hamilt = new hamilt::HamiltPW<T, Device>(this->pelec->pot,
                                                         band_pw_wfc,
                                                         &band_kv,
                                                         &band_ppcell,
                                                         &this->dftu,
                                                         &ucell,
                                                         source_exx,
                                                         *this->exx_options_,
                                                         *this->exx_context_);
    band_stp.init(band_hamilt);

    elecstate::ElecStatePW<T, Device> band_elec(band_pw_wfc,
                                                &this->chr,
                                                &band_kv,
                                                &ucell,
                                                &band_ppcell,
                                                this->pw_rho,
                                                this->pw_big);
    band_elec.pot = this->pelec->pot;
    band_elec.skip_weights = true;
    band_elec.wg.zero_out();

    const double target_ethr = std::max(inp.scf_thr, 1.0e-8);
    hsolver::setup_diago_params_pw<T, Device>(0, 1, target_ethr, inp);
    hsolver::HSolverPW<T, Device> hsolver_pw_obj(band_pw_wfc,
                                                 "nscf",
                                                 inp.basis_type,
                                                 inp.ks_solver,
                                                 sys.use_uspp,
                                                 inp.nspin,
                                                 hsolver::DiagoIterAssist<T, Device>::SCF_ITER,
                                                 hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX,
                                                 hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR,
                                                 hsolver::DiagoIterAssist<T, Device>::need_subspace,
                                                 false);

    hsolver_pw_obj.solve(static_cast<hamilt::Hamilt<T, Device>*>(band_hamilt),
                         *band_stp.template get_psi_t<T, Device>(),
                         &band_elec,
                         band_elec.ekb.c,
                         band_kv.para_k.rank_in_pool,
                         this->nproc_in_pool_,
                         true,
                         ucell.tpiba,
                         ucell.nat);

    ModuleIO::write_bands(inp, band_elec.ekb, band_kv);
    band_elec.pot = nullptr;
    delete band_hamilt;
    band_stp.clean();
    pw::teardown_pwwfc(band_pw_wfc);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::solve_mixed_band_targets_mem_saver(UnitCell& ucell,
                                                                  hamilt::OperatorEXXPW<T, Device>* source_exx,
                                                                  const Input_para& inp,
                                                                  const System_para& sys)
{
    K_Vectors band_kv = this->kv.make_band_target_kvectors(inp.nspin);
    ModulePW::PW_Basis_K* band_pw_wfc = nullptr;
    pw::setup_pwwfc(inp, ucell, *this->pw_rho, band_kv, band_pw_wfc);

    pseudopot_cell_vnl band_ppcell;
    band_ppcell.init(ucell, &this->sf, band_pw_wfc);
    band_ppcell.init_vnl(ucell, this->pw_rhod);

    ModuleBase::matrix band_ekb(band_kv.get_nks(), sys.nbands_l);
    const double target_ethr = std::max(inp.scf_thr, 1.0e-8);
    ModuleBase::matrix veff = this->pelec->pot->get_eff_v();
    band_ppcell.cal_effective_D(veff, this->pw_rhod, ucell);

    Setup_Psi_pw band_stp;
    band_stp.before_runner(ucell,
                           band_kv,
                           this->sf,
                           *band_pw_wfc,
                           band_ppcell,
                           inp,
                           sys,
                           this->my_bndgroup_,
                           *this->running_log_,
                           true);

    auto* band_hamilt = new hamilt::HamiltPW<T, Device>(this->pelec->pot,
                                                         band_pw_wfc,
                                                         &band_kv,
                                                         &band_ppcell,
                                                         &this->dftu,
                                                         &ucell,
                                                         source_exx,
                                                         *this->exx_options_,
                                                         *this->exx_context_);
    elecstate::ElecStatePW<T, Device> band_elec(band_pw_wfc,
                                                &this->chr,
                                                &band_kv,
                                                &ucell,
                                                &band_ppcell,
                                                this->pw_rho,
                                                this->pw_big);
    band_elec.pot = this->pelec->pot;
    band_elec.skip_weights = true;
    band_elec.wg.zero_out();

    hsolver::setup_diago_params_pw<T, Device>(0, 1, target_ethr, inp);
    hsolver::HSolverPW<T, Device> hsolver_pw_obj(band_pw_wfc,
                                                 "nscf",
                                                 inp.basis_type,
                                                 inp.ks_solver,
                                                 sys.use_uspp,
                                                 inp.nspin,
                                                 hsolver::DiagoIterAssist<T, Device>::SCF_ITER,
                                                 hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX,
                                                 hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR,
                                                 hsolver::DiagoIterAssist<T, Device>::need_subspace,
                                                 false);

    for (int ik = 0; ik < band_pw_wfc->nks; ++ik)
    {
        band_stp.init_ik(band_hamilt, ik);
        hsolver_pw_obj.solve_ik(static_cast<hamilt::Hamilt<T, Device>*>(band_hamilt),
                                *band_stp.template get_psi_t<T, Device>(),
                                &band_elec,
                                band_ekb.c + ik * sys.nbands_l,
                                ik,
                                band_kv.para_k.rank_in_pool,
                                this->nproc_in_pool_,
                                true,
                                inp.diago_smooth_ethr);
    }

    band_elec.pot = nullptr;
    delete band_hamilt;
    band_stp.clean();
    pw::teardown_pwwfc(band_pw_wfc);

    ModuleIO::write_bands(inp, band_ekb, band_kv);
}

template <typename T, typename Device>
double ESolver_KS_PW<T, Device>::cal_energy()
{
    return this->pelec->f_en.etot;
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::cal_force(UnitCell& ucell, ModuleBase::matrix& force)
{
    Forces<double, Device> ff(ucell.nat);

    // mohan add 2025-10-12
    this->stp.update_psi_d();

    // Calculate forces
    ff.cal_force(ucell, force, *this->pelec, this->pw_rhod, &ucell.symm,
                 &this->sf, this->solvent, &this->dftu, &this->locpp, &this->ppcell, 
                 &this->kv, this->pw_wfc, this->stp.template get_psi_d<T, Device>());
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::cal_stress(UnitCell& ucell, ModuleBase::matrix& stress)
{
    Stress_PW<double, Device> ss(this->pelec);

    // mohan add 2025-10-12
    this->stp.update_psi_d();

    ss.cal_stress(stress, ucell, this->dftu, this->locpp, this->ppcell, this->pw_rhod,
                  &ucell.symm, &this->sf, &this->kv, this->pw_wfc, this->stp.template get_psi_d<T, Device>(),
                  *this->exx_options_, *this->exx_context_);

    // external stress
    double unit_transform = 0.0;
    unit_transform = ModuleBase::RYDBERG_SI / pow(ModuleBase::BOHR_RADIUS_SI, 3) * 1.0e-8;
    double external_stress[3] = {PARAM.inp.press1, PARAM.inp.press2, PARAM.inp.press3};
    for (int i = 0; i < 3; i++)
    {
        stress(i, i) -= external_stress[i] / unit_transform;
    }
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::after_all_runners(UnitCell& ucell)
{
    ESolver_KS::after_all_runners(ucell);

    ModuleIO::ctrl_runner_pw<T, Device>(ucell, this->pelec, this->pw_wfc,
            this->pw_rho, this->pw_rhod, this->chr, this->kv, this->stp,
            this->sf, this->ppcell, this->solvent, this->Pgrid, PARAM.inp);

    elecstate::teardown_estate_pw(this->pelec, this->vsep_cell);

}

template class ESolver_KS_PW<std::complex<float>, base_device::DEVICE_CPU>;
template class ESolver_KS_PW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ESolver_KS_PW<std::complex<float>, base_device::DEVICE_GPU>;
template class ESolver_KS_PW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace ModuleESolver
