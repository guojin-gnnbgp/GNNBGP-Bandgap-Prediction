#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "pot_cider_xc.h"
#include "pot_cider_xc_runtime_factory.h"

#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_global.h"
#include "source_base/parallel_reduce.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_hamilt/module_xc/libxc_abacus.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/module_parameter/cider_model_input.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#ifdef __MPI
#include <mpi.h>
#endif

namespace
{

constexpr double CIDER_GGA_RHO_THRESHOLD = 1.0e-6;
constexpr double CIDER_GGA_GRHO_THRESHOLD = 1.0e-10;
constexpr double CIDER_MGGA_RHO_THRESHOLD = 1.0e-8;
constexpr double CIDER_MGGA_GRHO_THRESHOLD = 1.0e-12;
constexpr double CIDER_MGGA_TAU_THRESHOLD = 1.0e-8;

ModuleBase::matrix g_last_cider_feature_v;
bool g_last_cider_feature_v_valid = false;
elecstate::PotCiderXC::EnergyComponents g_last_energy_components;

struct CiderPotentialDiagnostics
{
    double max_abs_v = 0.0;
    double rms_v = 0.0;
    double max_abs_vtau = 0.0;
    double rms_vtau = 0.0;
    double max_abs_v_density = 0.0;
    double max_abs_v_gradient = 0.0;
    double max_abs_vtau_density = 0.0;
    double max_abs_vtau_tau = 0.0;
    double min_rho_at_max_v = 0.0;
    double min_grho_at_max_v = 0.0;
    double min_tau_at_max_v = 0.0;
    double min_rho_at_max_vtau = 0.0;
    double min_grho_at_max_vtau = 0.0;
    double min_tau_at_max_vtau = 0.0;
    std::size_t masked_count = 0;
    std::size_t active_count = 0;
    int max_v_spin = 0;
    int max_vtau_spin = 0;
    std::size_t max_v_ir = 0;
    std::size_t max_vtau_ir = 0;
};

struct CiderPotentialClipStats
{
    std::size_t clipped_count = 0;
    std::size_t total_count = 0;
    double max_abs_before = 0.0;
    double max_abs_after = 0.0;
};

std::string canonical_cider_feature_density(std::string policy)
{
    std::transform(policy.begin(), policy.end(), policy.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (policy == "total") {
        return "valence_pseudo_core";
    }
    return policy;
}

bool cider_features_use_pseudo_core(const std::string& policy)
{
    return policy == "valence_pseudo_core";
}

double cider_feature_policy_code(const std::string& policy)
{
    return cider_features_use_pseudo_core(policy) ? 1.0 : 0.0;
}

double cider_exc_convention_code(const std::string& convention)
{
    return convention == "energy_density" ? 1.0 : 0.0;
}

double positive_env_double(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return 0.0;
    }
    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || !std::isfinite(value) || value <= 0.0) {
        GlobalV::ofs_warning
            << "PotCiderXC: ignoring invalid positive numeric environment variable "
            << name << "=" << raw << std::endl;
        return 0.0;
    }
    return value;
}

double env_double_or_default(const char* name, const double default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || !std::isfinite(value)) {
        GlobalV::ofs_warning
            << "PotCiderXC: ignoring invalid numeric environment variable "
            << name << "=" << raw << std::endl;
        return default_value;
    }
    return value;
}

bool env_flag_enabled(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value != "0" && value != "false" && value != "off" && value != "no";
}

double clip_to_cap(const double value, const double cap)
{
    if (cap <= 0.0) {
        return value;
    }
    return std::max(-cap, std::min(cap, value));
}

CiderPotentialClipStats clip_feature_matrix_in_place(
    ModuleBase::matrix& values,
    const double cap)
{
    CiderPotentialClipStats stats;
    stats.total_count = static_cast<std::size_t>(values.nr) * static_cast<std::size_t>(values.nc);
    if (cap <= 0.0) {
        return stats;
    }
    for (int is = 0; is < values.nr; ++is) {
        for (int ir = 0; ir < values.nc; ++ir) {
            const double before = values(is, ir);
            const double after = clip_to_cap(before, cap);
            stats.max_abs_before = std::max(stats.max_abs_before, std::abs(before));
            stats.max_abs_after = std::max(stats.max_abs_after, std::abs(after));
            if (after != before) {
                ++stats.clipped_count;
                values(is, ir) = after;
            }
        }
    }
    return stats;
}

std::vector<double> scaled_vector_copy(
    const std::vector<double>& values,
    const double scale)
{
    if (scale == 1.0) {
        return values;
    }
    std::vector<double> scaled(values);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < scaled.size(); ++i) {
        scaled[i] *= scale;
    }
    return scaled;
}

CiderPotentialClipStats clip_vector_in_place(
    std::vector<double>& values,
    const double cap)
{
    CiderPotentialClipStats stats;
    stats.total_count = values.size();
    if (cap <= 0.0) {
        return stats;
    }
    for (double& value : values) {
        const double before = value;
        const double after = clip_to_cap(before, cap);
        stats.max_abs_before = std::max(stats.max_abs_before, std::abs(before));
        stats.max_abs_after = std::max(stats.max_abs_after, std::abs(after));
        if (after != before) {
            ++stats.clipped_count;
            value = after;
        }
    }
    return stats;
}

void log_cider_potential_clipping(
    const int eval_count,
    const double feature_v_cap,
    const CiderPotentialClipStats& v_stats,
    const double feature_vtau_cap,
    const CiderPotentialClipStats& vtau_stats)
{
    if (feature_v_cap <= 0.0 && feature_vtau_cap <= 0.0) {
        return;
    }
    GlobalV::ofs_running
        << "PotCiderXC: potential clipping"
        << " call=" << eval_count
        << " feature_v_cap=" << feature_v_cap
        << " feature_v_clipped=" << v_stats.clipped_count
        << " feature_v_total=" << v_stats.total_count
        << " feature_v_max_abs_before=" << v_stats.max_abs_before
        << " feature_v_max_abs_after=" << v_stats.max_abs_after
        << " feature_vtau_cap=" << feature_vtau_cap
        << " feature_vtau_clipped=" << vtau_stats.clipped_count
        << " feature_vtau_total=" << vtau_stats.total_count
        << " feature_vtau_max_abs_before=" << vtau_stats.max_abs_before
        << " feature_vtau_max_abs_after=" << vtau_stats.max_abs_after
        << std::endl;
}

void log_same_density_xc_comparison(
    const Charge* const chg,
    const UnitCell* const ucell,
    const double cider_etxc,
    const double cider_vtxc,
    const ModuleBase::matrix& cider_v,
    const ModuleBase::matrix* cider_vofk = nullptr)
{
    const auto native_xc = XC_Functional_Libxc::v_xc_libxc(
        XC_Functional::get_func_id(),
        chg->nrxx,
        ucell->omega,
        ucell->tpiba,
        chg,
        nullptr);

    const double native_etxc = std::get<0>(native_xc);
    const double native_vtxc = std::get<1>(native_xc);
    const ModuleBase::matrix& native_v = std::get<2>(native_xc);

    double max_abs_v_diff = 0.0;
    double rms_v_diff = 0.0;
    double max_abs_v_native = 0.0;
    const double denom = static_cast<double>(cider_v.nr * cider_v.nc);

    for (int is = 0; is < cider_v.nr; ++is) {
        for (int ir = 0; ir < cider_v.nc; ++ir) {
            const double diff = cider_v(is, ir) - native_v(is, ir);
            max_abs_v_diff = std::max(max_abs_v_diff, std::abs(diff));
            rms_v_diff += diff * diff;
            max_abs_v_native = std::max(max_abs_v_native, std::abs(native_v(is, ir)));
        }
    }
    rms_v_diff = std::sqrt(rms_v_diff / std::max(denom, 1.0));

    GlobalV::ofs_running
        << "PotCiderXC: same-density native-libxc comparison"
        << " etxc_cider=" << cider_etxc
        << " etxc_native=" << native_etxc
        << " delta_etxc=" << (cider_etxc - native_etxc)
        << " vtxc_cider=" << cider_vtxc
        << " vtxc_native=" << native_vtxc
        << " delta_vtxc=" << (cider_vtxc - native_vtxc)
        << " max_abs_v_diff=" << max_abs_v_diff
        << " rms_v_diff=" << rms_v_diff
        << " max_abs_v_native=" << max_abs_v_native
        << std::endl;

    GlobalV::ofs_running
        << "PotCiderXC: same-density sample"
        << " cider_v00=" << cider_v(0, 0)
        << " native_v00=" << native_v(0, 0)
        << " delta_v00=" << (cider_v(0, 0) - native_v(0, 0))
        << std::endl;

    if (cider_vofk != nullptr)
    {
        double max_abs_vofk = 0.0;
        double rms_vofk = 0.0;
        const double vofk_denom = static_cast<double>(cider_vofk->nr * cider_vofk->nc);
        for (int is = 0; is < cider_vofk->nr; ++is)
        {
            for (int ir = 0; ir < cider_vofk->nc; ++ir)
            {
                const double value = (*cider_vofk)(is, ir);
                max_abs_vofk = std::max(max_abs_vofk, std::abs(value));
                rms_vofk += value * value;
            }
        }
        rms_vofk = std::sqrt(rms_vofk / std::max(vofk_denom, 1.0));

        GlobalV::ofs_running
            << "PotCiderXC: same-density vofk comparison"
            << " max_abs_vofk_diff=" << max_abs_vofk
            << " rms_vofk_diff=" << rms_vofk
            << " cider_vofk00=" << ((*cider_vofk).nr > 0 && (*cider_vofk).nc > 0 ? (*cider_vofk)(0, 0) : 0.0)
            << std::endl;
    }
}

std::vector<double> build_cider_sgn(
    const int nspin,
    const std::size_t nrxx,
    const std::vector<double>& rho_interleaved,
    const std::vector<double>& sigma_interleaved,
    const std::vector<double>& tau_interleaved,
    const bool is_mgga)
{
    std::vector<double> sgn(nrxx * nspin, 1.0);
    const double rho_threshold =
        is_mgga ? CIDER_MGGA_RHO_THRESHOLD : CIDER_GGA_RHO_THRESHOLD;
    const double grho_threshold =
        is_mgga ? CIDER_MGGA_GRHO_THRESHOLD : CIDER_GGA_GRHO_THRESHOLD;

    if (nspin == 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            const bool low_density = rho_interleaved[ir] < rho_threshold;
            const bool low_gradient =
                is_mgga
                && !sigma_interleaved.empty()
                && std::sqrt(std::abs(sigma_interleaved[ir])) < grho_threshold;
            const bool low_tau =
                is_mgga
                && !tau_interleaved.empty()
                && std::abs(tau_interleaved[ir]) < CIDER_MGGA_TAU_THRESHOLD;
            if (low_density || low_gradient || low_tau) {
                sgn[ir] = 0.0;
            }
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            const std::size_t up = ir * 2;
            const std::size_t dw = up + 1;
            const bool low_density_up =
                rho_interleaved[up] < rho_threshold;
            const bool low_density_dw =
                rho_interleaved[dw] < rho_threshold;
            const bool low_gradient_up =
                is_mgga
                && !sigma_interleaved.empty()
                && std::sqrt(std::abs(sigma_interleaved[ir * 3])) < grho_threshold;
            const bool low_gradient_dw =
                is_mgga
                && !sigma_interleaved.empty()
                && std::sqrt(std::abs(sigma_interleaved[ir * 3 + 2])) < grho_threshold;
            const bool low_tau_up =
                is_mgga
                && !tau_interleaved.empty()
                && std::abs(tau_interleaved[up]) < CIDER_MGGA_TAU_THRESHOLD;
            const bool low_tau_dw =
                is_mgga
                && !tau_interleaved.empty()
                && std::abs(tau_interleaved[dw]) < CIDER_MGGA_TAU_THRESHOLD;
            if (low_density_up || low_gradient_up || low_tau_up) {
                sgn[up] = 0.0;
            }
            if (low_density_dw || low_gradient_dw || low_tau_dw) {
                sgn[dw] = 0.0;
            }
        }
    }

    return sgn;
}

double feature_spin_gradient_magnitude(
    const int nspin,
    const std::vector<double>& sigma_interleaved,
    const std::size_t nrxx,
    const int is,
    const std::size_t ir)
{
    if (sigma_interleaved.empty()) {
        return 0.0;
    }
    if (nspin == 1) {
        return std::sqrt(std::abs(sigma_interleaved[ir]));
    }
    const std::size_t offset = ir * 3 + (is == 0 ? 0 : 2);
    return offset < sigma_interleaved.size()
        ? std::sqrt(std::abs(sigma_interleaved[offset]))
        : 0.0;
}

CiderPotentialDiagnostics collect_cider_potential_diagnostics(
    const int nspin,
    const std::size_t nrxx,
    const std::vector<double>& sgn_interleaved,
    const std::vector<double>& rho_interleaved,
    const std::vector<double>& sigma_interleaved,
    const std::vector<double>& tau_interleaved,
    const ModuleBase::matrix& v_feature,
    const std::vector<double>& vtau_feature_interleaved)
{
    CiderPotentialDiagnostics stats;
    double sum_v2 = 0.0;
    double sum_vtau2 = 0.0;
    double max_abs_v = -1.0;
    double max_abs_vtau = -1.0;
    const std::size_t total = nrxx * static_cast<std::size_t>(nspin);

    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            const std::size_t index = ir * nspin + is;
            const double rho = rho_interleaved.empty() ? 0.0 : rho_interleaved[index];
            const double grho = feature_spin_gradient_magnitude(
                nspin, sigma_interleaved, nrxx, is, ir);
            const double tau =
                tau_interleaved.empty() ? 0.0 : tau_interleaved[index];
            const double sgn =
                sgn_interleaved.empty() ? 1.0 : sgn_interleaved[index];
            const bool masked = std::abs(sgn) < 0.5;
            if (masked) {
                ++stats.masked_count;
            } else {
                ++stats.active_count;
            }

            const double v = v_feature(is, ir);
            const double abs_v = std::abs(v);
            sum_v2 += v * v;
            if (rho < 1.0e-6) {
                stats.max_abs_v_density = std::max(stats.max_abs_v_density, abs_v);
            }
            if (grho < 1.0e-8) {
                stats.max_abs_v_gradient = std::max(stats.max_abs_v_gradient, abs_v);
            }
            if (abs_v > max_abs_v) {
                max_abs_v = abs_v;
                stats.max_abs_v = abs_v;
                stats.max_v_spin = is;
                stats.max_v_ir = ir;
                stats.min_rho_at_max_v = rho;
                stats.min_grho_at_max_v = grho;
                stats.min_tau_at_max_v = tau;
            }

            if (!vtau_feature_interleaved.empty()) {
                const double vtau = vtau_feature_interleaved[index] * sgn;
                const double abs_vtau = std::abs(vtau);
                sum_vtau2 += vtau * vtau;
                if (rho < 1.0e-6) {
                    stats.max_abs_vtau_density =
                        std::max(stats.max_abs_vtau_density, abs_vtau);
                }
                if (std::abs(tau) < 1.0e-6) {
                    stats.max_abs_vtau_tau =
                        std::max(stats.max_abs_vtau_tau, abs_vtau);
                }
                if (abs_vtau > max_abs_vtau) {
                    max_abs_vtau = abs_vtau;
                    stats.max_abs_vtau = abs_vtau;
                    stats.max_vtau_spin = is;
                    stats.max_vtau_ir = ir;
                    stats.min_rho_at_max_vtau = rho;
                    stats.min_grho_at_max_vtau = grho;
                    stats.min_tau_at_max_vtau = tau;
                }
            }
        }
    }

    const double denom = static_cast<double>(std::max<std::size_t>(total, 1));
    stats.rms_v = std::sqrt(sum_v2 / denom);
    stats.rms_vtau = std::sqrt(sum_vtau2 / denom);
    return stats;
}

void log_cider_potential_diagnostics(
    const int eval_count,
    const double grid_weight,
    const CiderPotentialDiagnostics& stats)
{
    const double total_count =
        static_cast<double>(stats.active_count + stats.masked_count);
    const double masked_fraction =
        total_count > 0.0 ? static_cast<double>(stats.masked_count) / total_count : 0.0;
    GlobalV::ofs_running
        << "PotCiderXC: potential diagnostic"
        << " call=" << eval_count
        << " max_abs_v_feature=" << stats.max_abs_v
        << " rms_v_feature=" << stats.rms_v
        << " max_abs_vtau_feature=" << stats.max_abs_vtau
        << " rms_vtau_feature=" << stats.rms_vtau
        << " max_abs_v_feature_lowrho=" << stats.max_abs_v_density
        << " max_abs_v_feature_lowgrad=" << stats.max_abs_v_gradient
        << " max_abs_vtau_feature_lowrho=" << stats.max_abs_vtau_density
        << " max_abs_vtau_feature_lowtau=" << stats.max_abs_vtau_tau
        << " masked_fraction=" << masked_fraction
        << " active_points=" << stats.active_count
        << " masked_points=" << stats.masked_count
        << " max_v_site=(" << stats.max_v_spin << "," << stats.max_v_ir << ")"
        << " max_v_rho=" << stats.min_rho_at_max_v
        << " max_v_grho=" << stats.min_grho_at_max_v
        << " max_v_tau=" << stats.min_tau_at_max_v
        << " max_vtau_site=(" << stats.max_vtau_spin << "," << stats.max_vtau_ir << ")"
        << " max_vtau_rho=" << stats.min_rho_at_max_vtau
        << " max_vtau_grho=" << stats.min_grho_at_max_vtau
        << " max_vtau_tau=" << stats.min_tau_at_max_vtau
        << " grid_weight=" << grid_weight
        << std::endl;
}

std::pair<double, ModuleBase::matrix> reconstruct_cider_vxc_with_native_helper(
    const int nspin,
    const std::size_t nrxx,
    const std::vector<double>& sgn,
    const std::vector<double>& rho_interleaved,
    const std::vector<std::vector<ModuleBase::Vector3<double>>>& gdr,
    const std::vector<double>& vrho_interleaved,
    const std::vector<double>& vsigma_interleaved,
    const double tpiba,
    const Charge* const chg)
{
    double vtxc = 0.0;
    ModuleBase::matrix v(nspin, nrxx);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:vtxc) schedule(static, 256)
#endif
    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            const std::size_t index = ir * nspin + is;
            const double v_tmp = ModuleBase::e2 * vrho_interleaved[index] * sgn[index];
            v(is, ir) += v_tmp;
            vtxc += v_tmp * rho_interleaved[index];
        }
    }

    if (!vsigma_interleaved.empty()) {
        const std::vector<std::vector<double>> dh = XC_Functional_Libxc::cal_dh(
            nspin, nrxx, sgn, gdr, vsigma_interleaved, tpiba, chg);

        double rvtxc = 0.0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:rvtxc) schedule(static, 256)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                rvtxc += dh[is][ir] * rho_interleaved[ir * nspin + is];
                v(is, ir) -= dh[is][ir];
            }
        }
        vtxc -= rvtxc;
    }

    return std::make_pair(vtxc, std::move(v));
}

std::vector<double> gather_spin_major_to_global(
    const ModulePW::PW_Basis* const rho_basis,
    const std::vector<double>& local,
    const int nchannels)
{
    const int nrxx_local = rho_basis->nrxx;
    const int nxyz = rho_basis->nxyz;
    if (nchannels <= 0) {
        return {};
    }

#ifdef __MPI
    if (GlobalV::NPROC_IN_POOL > 1 && POOL_WORLD != MPI_COMM_NULL) {
        const int ncxy = rho_basis->nx * rho_basis->ny;
        std::vector<double> global(nchannels * nxyz, 0.0);
        std::vector<double> gathered(nchannels * nxyz, 0.0);
        std::vector<int> rec(GlobalV::NPROC_IN_POOL);
        std::vector<int> dis(GlobalV::NPROC_IN_POOL);
        for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ++ip) {
            rec[ip] = rho_basis->numz[ip] * ncxy;
            dis[ip] = rho_basis->startz[ip] * ncxy;
        }

        for (int ich = 0; ich < nchannels; ++ich) {
            MPI_Allgatherv(local.data() + ich * nrxx_local,
                           nrxx_local,
                           MPI_DOUBLE,
                           gathered.data() + ich * nxyz,
                           rec.data(),
                           dis.data(),
                           MPI_DOUBLE,
                           POOL_WORLD);

            double* global_ch = global.data() + ich * nxyz;
            const double* gathered_ch = gathered.data() + ich * nxyz;
            for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ++ip) {
                for (int ixy = 0; ixy < ncxy; ++ixy) {
                    for (int iz = 0; iz < rho_basis->numz[ip]; ++iz) {
                        global_ch[rho_basis->nz * ixy + rho_basis->startz[ip] + iz] =
                            gathered_ch[rho_basis->numz[ip] * ixy
                                        + rho_basis->startz[ip] * ncxy
                                        + iz];
                    }
                }
            }
        }
        return global;
    }
#endif

    return local;
}

std::vector<double> scatter_spin_major_from_global(
    const ModulePW::PW_Basis* const rho_basis,
    const std::vector<double>& global,
    const int nchannels)
{
    const int nrxx_local = rho_basis->nrxx;
    std::vector<double> local(nchannels * nrxx_local, 0.0);
    if (nchannels <= 0) {
        return local;
    }

    const int ncxy = rho_basis->nx * rho_basis->ny;
    for (int ich = 0; ich < nchannels; ++ich) {
        const double* global_ch = global.data() + ich * rho_basis->nxyz;
        double* local_ch = local.data() + ich * nrxx_local;
        for (int ixy = 0; ixy < ncxy; ++ixy) {
            for (int iz = 0; iz < rho_basis->nplane; ++iz) {
                local_ch[ixy * rho_basis->nplane + iz] =
                    global_ch[rho_basis->nz * ixy + rho_basis->startz_current + iz];
            }
        }
    }
    return local;
}

std::vector<double> interleaved_to_spin_major(
    const std::vector<double>& interleaved,
    const int nchannels,
    const std::size_t nrxx)
{
    std::vector<double> spin_major(nchannels * nrxx, 0.0);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int ich = 0; ich < nchannels; ++ich) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            spin_major[ich * nrxx + ir] = interleaved[ir * nchannels + ich];
        }
    }
    return spin_major;
}

std::vector<double> spin_major_to_interleaved(
    const std::vector<double>& spin_major,
    const int nchannels,
    const std::size_t nrxx)
{
    std::vector<double> interleaved(nchannels * nrxx, 0.0);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int ich = 0; ich < nchannels; ++ich) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            interleaved[ir * nchannels + ich] = spin_major[ich * nrxx + ir];
        }
    }
    return interleaved;
}

void write_debug_array(std::ofstream& out, const std::string& name, const std::vector<double>& values)
{
    const std::uint32_t name_size = static_cast<std::uint32_t>(name.size());
    const std::uint64_t value_size = static_cast<std::uint64_t>(values.size());
    out.write(reinterpret_cast<const char*>(&name_size), sizeof(name_size));
    out.write(name.data(), name.size());
    out.write(reinterpret_cast<const char*>(&value_size), sizeof(value_size));
    if (!values.empty()) {
        out.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(double)));
    }
}

void maybe_write_abacus_cider_debug_dump(
    const int eval_count,
    const int nspin,
    const int nsigma,
    const std::size_t nrxx,
    const double grid_weight,
    const double xmix,
    const double feature_density_policy_code,
    const std::vector<double>& rho_baseline_sm,
    const std::vector<double>& sigma_baseline_sm,
    const std::vector<double>& tau_baseline_sm,
    const std::vector<double>& rho_feature_sm,
    const std::vector<double>& sigma_feature_sm,
    const std::vector<double>& tau_feature_sm,
    const std::vector<double>& exc_baseline,
    const std::vector<double>& exc_feature,
    const std::vector<double>& vrho_baseline_sm,
    const std::vector<double>& vsigma_baseline_sm,
    const std::vector<double>& vtau_baseline_sm,
    const std::vector<double>& vrho_feature_sm,
    const std::vector<double>& vsigma_feature_sm,
    const std::vector<double>& vtau_feature_sm,
    const ModuleBase::matrix& v_baseline,
    const ModuleBase::matrix& v_feature,
    const ModuleBase::matrix& v_total,
    const double etxc_baseline_ry,
    const double etxc_feature_ry,
    const double feature_exc_convention_code,
    const double etxc_feature_volume_ry,
    const double etxc_feature_rho_weighted_ry,
    const double vtxc_baseline_ry,
    const double vtxc_feature_ry)
{
    const char* prefix = std::getenv("ABACUS_CIDER_DEBUG_DUMP_PREFIX");
    if (prefix == nullptr || std::string(prefix).empty()) {
        return;
    }

    const std::string mode =
        std::getenv("ABACUS_CIDER_DEBUG_DUMP_MODE") == nullptr
            ? "all"
            : std::getenv("ABACUS_CIDER_DEBUG_DUMP_MODE");
    if (mode == "first" && eval_count != 1) {
        return;
    }

    char filename[4096];
    std::snprintf(filename, sizeof(filename), "%s_call%04d.bin", prefix, eval_count);
    std::ofstream out(filename, std::ios::binary);
    if (!out.good()) {
        GlobalV::ofs_warning << "PotCiderXC: failed to open debug dump " << filename << std::endl;
        return;
    }

    const char magic[8] = {'A', 'C', 'D', 'D', 'U', 'M', 'P', '1'};
    out.write(magic, sizeof(magic));
    const std::uint32_t version = 3;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    const std::uint32_t nspin_u = static_cast<std::uint32_t>(nspin);
    const std::uint32_t nsigma_u = static_cast<std::uint32_t>(nsigma);
    const std::uint64_t nrxx_u = static_cast<std::uint64_t>(nrxx);
    out.write(reinterpret_cast<const char*>(&nspin_u), sizeof(nspin_u));
    out.write(reinterpret_cast<const char*>(&nsigma_u), sizeof(nsigma_u));
    out.write(reinterpret_cast<const char*>(&nrxx_u), sizeof(nrxx_u));
    out.write(reinterpret_cast<const char*>(&grid_weight), sizeof(grid_weight));
    out.write(reinterpret_cast<const char*>(&xmix), sizeof(xmix));
    out.write(reinterpret_cast<const char*>(&feature_density_policy_code), sizeof(feature_density_policy_code));
    out.write(reinterpret_cast<const char*>(&etxc_baseline_ry), sizeof(etxc_baseline_ry));
    out.write(reinterpret_cast<const char*>(&etxc_feature_ry), sizeof(etxc_feature_ry));
    out.write(reinterpret_cast<const char*>(&feature_exc_convention_code), sizeof(feature_exc_convention_code));
    out.write(reinterpret_cast<const char*>(&etxc_feature_volume_ry), sizeof(etxc_feature_volume_ry));
    out.write(reinterpret_cast<const char*>(&etxc_feature_rho_weighted_ry), sizeof(etxc_feature_rho_weighted_ry));
    out.write(reinterpret_cast<const char*>(&vtxc_baseline_ry), sizeof(vtxc_baseline_ry));
    out.write(reinterpret_cast<const char*>(&vtxc_feature_ry), sizeof(vtxc_feature_ry));

    std::vector<std::pair<std::string, const std::vector<double>*>> arrays = {
        {"rho_baseline_sg", &rho_baseline_sm},
        {"sigma_baseline_xg", &sigma_baseline_sm},
        {"rho_feature_sg", &rho_feature_sm},
        {"sigma_feature_xg", &sigma_feature_sm},
        {"exc_baseline_g", &exc_baseline},
        {"exc_feature_g", &exc_feature},
        {"vrho_baseline_sg", &vrho_baseline_sm},
        {"vsigma_baseline_xg", &vsigma_baseline_sm},
        {"vrho_feature_sg", &vrho_feature_sm},
        {"vsigma_feature_xg", &vsigma_feature_sm},
    };
    if (!tau_baseline_sm.empty()) {
        arrays.push_back({"tau_baseline_sg", &tau_baseline_sm});
        arrays.push_back({"vtau_baseline_sg", &vtau_baseline_sm});
    }
    if (!tau_feature_sm.empty()) {
        arrays.push_back({"tau_feature_sg", &tau_feature_sm});
        arrays.push_back({"vtau_feature_sg", &vtau_feature_sm});
    }

    std::vector<double> v_baseline_sm(nspin * nrxx, 0.0);
    std::vector<double> v_feature_sm(nspin * nrxx, 0.0);
    std::vector<double> v_total_sm(nspin * nrxx, 0.0);
    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            v_baseline_sm[is * nrxx + ir] = v_baseline(is, ir);
            v_feature_sm[is * nrxx + ir] = v_feature(is, ir);
            v_total_sm[is * nrxx + ir] = v_total(is, ir);
        }
    }
    arrays.push_back({"v_baseline_sg", &v_baseline_sm});
    arrays.push_back({"v_feature_sg", &v_feature_sm});
    arrays.push_back({"v_total_sg", &v_total_sm});

    const std::uint32_t array_count = static_cast<std::uint32_t>(arrays.size());
    out.write(reinterpret_cast<const char*>(&array_count), sizeof(array_count));
    for (const auto& item : arrays) {
        write_debug_array(out, item.first, *item.second);
    }

    GlobalV::ofs_running << "PotCiderXC: wrote debug dump " << filename << std::endl;
}

double integrate_role_exc(
    const std::vector<double>& exc,
    const std::vector<double>& sgn,
    const std::vector<double>& rho_interleaved,
    const int nspin,
    const std::size_t nrxx)
{
    double etxc_local = 0.0;
    for (std::size_t ir = 0; ir < nrxx; ++ir) {
        double rho_total = 0.0;
        for (int is = 0; is < nspin; ++is) {
            const std::size_t index = ir * nspin + is;
            rho_total += rho_interleaved[index] * sgn[index];
        }
        etxc_local += ModuleBase::e2 * exc[ir] * rho_total;
    }
    return etxc_local;
}

double integrate_volume_exc(
    const std::vector<double>& exc,
    const std::size_t nrxx)
{
    double etxc_local = 0.0;
    for (std::size_t ir = 0; ir < nrxx; ++ir) {
        etxc_local += ModuleBase::e2 * exc[ir];
    }
    return etxc_local;
}

} // namespace

namespace elecstate
{

const ModuleBase::matrix* PotCiderXC::debug_last_feature_potential()
{
    return g_last_cider_feature_v_valid ? &g_last_cider_feature_v : nullptr;
}

const PotCiderXC::EnergyComponents& PotCiderXC::last_energy_components()
{
    return g_last_energy_components;
}

PotCiderXC::PotCiderXC(
    const ModulePW::PW_Basis* rho_basis_in,
    const UnitCell* ucell_in,
    double* etxc_in,
    double* vtxc_in,
    ModuleBase::matrix* vofk_in)
    : etxc_(etxc_in), vtxc_(vtxc_in), vofk_(vofk_in)
{
    this->rho_basis_ = rho_basis_in;
    this->dynamic_mode = true;
    this->fixed_mode = false;

    // Grid dimensions
    const std::size_t grid_shape[3] = {
        static_cast<std::size_t>(rho_basis_in->nx),
        static_cast<std::size_t>(rho_basis_in->ny),
        static_cast<std::size_t>(rho_basis_in->nz)};

    // Lattice vectors in row-major order (Bohr units). ABACUS stores
    // latvec in units of lat0, so scale explicitly before passing to
    // the native FFT/NLDF backend.
    const auto& lat = ucell_in->latvec;
    const double lat0 = ucell_in->lat0;
    const double cell_cv[9] = {
        lat0 * lat.e11, lat0 * lat.e12, lat0 * lat.e13,
        lat0 * lat.e21, lat0 * lat.e22, lat0 * lat.e23,
        lat0 * lat.e31, lat0 * lat.e32, lat0 * lat.e33
    };

    const std::string& model_path = PARAM.inp.cider_model;
    if (model_path.empty()) {
        ModuleBase::WARNING_QUIT("PotCiderXC", "cider_model path is empty");
    }

    const std::string raw_feature_density = PARAM.inp.cider_feature_density;
    const std::string requested_feature_density =
        canonical_cider_feature_density(raw_feature_density);

    const ModuleIO::CiderBackendCapabilities capabilities(
        true,
#ifdef ENABLE_CIDER_EXTERNAL
        true
#else
        false
#endif
    );
    std::string resolved_backend;
    try
    {
        resolved_backend = ModuleIO::resolve_cider_backend(
            PARAM.inp.cider_backend,
            capabilities);
    }
    catch (const std::exception& error)
    {
        ModuleBase::WARNING_QUIT("PotCiderXC", error.what());
        throw;
    }

    GlobalV::ofs_running
        << "PotCiderXC: constructing XC runtime"
        << " backend=" << resolved_backend
        << " model=" << model_path
        << " xmix=" << PARAM.inp.cider_xmix
        << " requested_feature_density=" << requested_feature_density
        << " nspin=" << PARAM.inp.nspin
        << " grid=(" << grid_shape[0] << "," << grid_shape[1] << ","
        << grid_shape[2] << ")"
        << std::endl;

    try
    {
        runtime_ = make_pot_cider_runtime(resolved_backend,
                                          model_path,
                                          grid_shape,
                                          cell_cv,
                                          PARAM.inp.nspin,
                                          PARAM.inp.cider_xmix);
    }
    catch (const std::exception& error)
    {
        ModuleBase::WARNING_QUIT(
            "PotCiderXC",
            std::string("failed to construct CIDER runtime backend=")
                + resolved_backend + " for model '"
                + model_path + "': " + error.what());
        throw;
    }
    const PotCiderRuntimeMetadata& metadata = runtime_->metadata();
    is_mgga_ = metadata.requires_tau;
    const std::string model_expected_policy = metadata.feature_density_policy;
    const std::string& model_policy_source
        = metadata.feature_density_policy_source;
    feature_exc_convention_ = metadata.feature_exc_convention;
    if (requested_feature_density == "auto") {
        feature_density_policy_ =
            model_expected_policy.empty() ? "valence" : model_expected_policy;
    } else {
        feature_density_policy_ = requested_feature_density;
        if (!model_expected_policy.empty()
            && model_expected_policy != feature_density_policy_) {
            ModuleBase::WARNING_QUIT(
                "PotCiderXC",
                "cider_feature_density conflicts with loaded CIDER model metadata");
        }
    }
    GlobalV::ofs_running
        << "PotCiderXC: runtime ready"
        << " backend=" << runtime_->backend_name()
        << " is_mgga=" << (is_mgga_ ? "true" : "false")
        << " feature_density=" << feature_density_policy_
        << " model_expected_feature_density="
        << (model_expected_policy.empty() ? "<unset>" : model_expected_policy)
        << " model_feature_density_source="
        << (model_policy_source.empty() ? "<unset>" : model_policy_source)
        << " feature_exc_convention=" << feature_exc_convention_
        << std::endl;
}

PotCiderXC::~PotCiderXC() = default;

void PotCiderXC::cal_v_eff(
    const Charge* const chg,
    const UnitCell* const ucell,
    ModuleBase::matrix& v_eff)
{
    ModuleBase::TITLE("PotCiderXC", "cal_v_eff");
    ModuleBase::timer::start("PotCiderXC", "cal_v_eff");

    const int nspin = chg->nspin;
    const std::size_t nrxx = chg->nrxx;
    const std::size_t nxyz = chg->rhopw->nxyz;
    const double tpiba = ucell->tpiba;

    if (nspin != 1 && nspin != 2) {
        ModuleBase::WARNING_QUIT(
            "PotCiderXC",
            "native CIDER evaluator supports only nspin=1 or nspin=2");
    }
    if (is_mgga_ && chg->kin_r == nullptr) {
        ModuleBase::WARNING_QUIT(
            "PotCiderXC",
            "tau-dependent CIDER model requires kinetic-energy density chg->kin_r");
    }

    static int eval_count = 0;
    eval_count += 1;
    GlobalV::ofs_running
        << "PotCiderXC: cal_v_eff call=" << eval_count
        << " nrxx=" << nrxx
        << " nspin=" << nspin
        << " tpiba=" << tpiba
        << std::endl;

    // === 1. Build role-specific densities in ABACUS interleaved layout ===
    // Baseline XC follows the normal NLCC path: valence + pseudo-core.
    // CIDER ML exchange features are selectable: valence for new ABACUS
    // models, or valence plus NLCC pseudo-core for legacy CIDER23x models.
    const bool feature_uses_pseudo_core =
        cider_features_use_pseudo_core(feature_density_policy_);
    std::vector<double> rho_baseline_interleaved(nrxx * nspin);
    std::vector<double> rho_feature_interleaved(nrxx * nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            const double rho_valence = chg->rho[is][ir];
            const double rho_baseline =
                rho_valence + chg->rho_core[ir] / static_cast<double>(nspin);
            rho_baseline_interleaved[ir * nspin + is] = rho_baseline;
            rho_feature_interleaved[ir * nspin + is] =
                feature_uses_pseudo_core ? rho_baseline : rho_valence;
        }
    }

    // === 2. Build role-specific sigma using ABACUS gradient machinery ===
    auto gdr_baseline = XC_Functional_Libxc::cal_gdr(
        nspin, nrxx, rho_baseline_interleaved, tpiba, chg);
    auto sigma_baseline_interleaved = XC_Functional_Libxc::convert_sigma(gdr_baseline);
    auto gdr_feature = XC_Functional_Libxc::cal_gdr(
        nspin, nrxx, rho_feature_interleaved, tpiba, chg);
    auto sigma_feature_interleaved = XC_Functional_Libxc::convert_sigma(gdr_feature);

    // === 3. Build role-specific tau in interleaved layout ===
    std::vector<double> tau_baseline_interleaved;
    std::vector<double> tau_feature_interleaved;
    if (is_mgga_ && chg->kin_r != nullptr) {
        tau_baseline_interleaved.resize(nrxx * nspin);
        tau_feature_interleaved.resize(nrxx * nspin);
        const bool use_tf_core = PARAM.inp.cider_tf_tau;
        constexpr double TF_FACTOR = (3.0 / 10.0) * std::pow(3.0 * M_PI * M_PI, 2.0 / 3.0);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                const double tau_valence = chg->kin_r[is][ir] / 2.0;
                double tau_baseline = tau_valence;
                if (use_tf_core) {
                    const double rho_cps = std::max(chg->rho_core[ir] / nspin, 0.0);
                    tau_baseline += TF_FACTOR * std::pow(rho_cps, 5.0 / 3.0);
                }
                tau_baseline_interleaved[ir * nspin + is] = tau_baseline;
                tau_feature_interleaved[ir * nspin + is] =
                    feature_uses_pseudo_core ? tau_baseline : tau_valence;
            }
        }
    }

    static bool density_policy_logged = false;
    if (!density_policy_logged) {
        GlobalV::ofs_running
            << "PotCiderXC: dual-density policy"
            << " raw_feature_density=" << PARAM.inp.cider_feature_density
            << " resolved_feature_density=" << feature_density_policy_
            << " baseline_rho_sigma=valence_plus_core"
            << " feature_rho_sigma="
            << (feature_uses_pseudo_core ? "valence_plus_core" : "valence_only")
            << " tau_feature="
            << (feature_uses_pseudo_core
                    ? (PARAM.inp.cider_tf_tau ? "valence_plus_tf_core" : "valence_only")
                    : "valence_only")
            << " tau_baseline="
            << (PARAM.inp.cider_tf_tau ? "valence_plus_tf_core" : "valence_only")
            << std::endl;
        density_policy_logged = true;
    }

    // === 4. Convert to the native evaluator's spin-major layout ===
    // rho[is*ngrids+ir], sigma[isig*ngrids+ir], tau[is*ngrids+ir]
    const int nsigma = (nspin == 1) ? 1 : 3;

    const std::vector<double> rho_baseline_sm =
        interleaved_to_spin_major(rho_baseline_interleaved, nspin, nrxx);
    const std::vector<double> sigma_baseline_sm =
        interleaved_to_spin_major(sigma_baseline_interleaved, nsigma, nrxx);
    const std::vector<double> rho_feature_sm =
        interleaved_to_spin_major(rho_feature_interleaved, nspin, nrxx);
    const std::vector<double> sigma_feature_sm =
        interleaved_to_spin_major(sigma_feature_interleaved, nsigma, nrxx);

    std::vector<double> tau_baseline_sm;
    std::vector<double> tau_feature_sm;
    if (!tau_baseline_interleaved.empty()) {
        tau_baseline_sm = interleaved_to_spin_major(
            tau_baseline_interleaved, nspin, nrxx);
        tau_feature_sm = interleaved_to_spin_major(
            tau_feature_interleaved, nspin, nrxx);
    }

    // === 5. Allocate output arrays (spin-major) ===
    std::vector<double> exc_baseline(nrxx, 0.0);
    std::vector<double> exc_feature(nrxx, 0.0);
    std::vector<double> vrho_baseline_sm(nrxx * nspin, 0.0);
    std::vector<double> vrho_feature_sm(nrxx * nspin, 0.0);
    std::vector<double> vsigma_baseline_sm(nrxx * nsigma, 0.0);
    std::vector<double> vsigma_feature_sm(nrxx * nsigma, 0.0);
    std::vector<double> vtau_baseline_sm;
    std::vector<double> vtau_feature_sm;
    std::vector<double> vtau_total_sm;
    if (is_mgga_) {
        vtau_baseline_sm.resize(nrxx * nspin, 0.0);
        vtau_feature_sm.resize(nrxx * nspin, 0.0);
        vtau_total_sm.resize(nrxx * nspin, 0.0);
    }

    // === 6. Call the native CIDER evaluator ===
    const std::vector<double> rho_baseline_sm_global =
        gather_spin_major_to_global(this->rho_basis_, rho_baseline_sm, nspin);
    const std::vector<double> sigma_baseline_sm_global =
        gather_spin_major_to_global(this->rho_basis_, sigma_baseline_sm, nsigma);
    const std::vector<double> rho_feature_sm_global =
        gather_spin_major_to_global(this->rho_basis_, rho_feature_sm, nspin);
    const std::vector<double> sigma_feature_sm_global =
        gather_spin_major_to_global(this->rho_basis_, sigma_feature_sm, nsigma);
    const std::vector<double> tau_baseline_sm_global =
        tau_baseline_sm.empty()
            ? std::vector<double>()
            : gather_spin_major_to_global(this->rho_basis_, tau_baseline_sm, nspin);
    const std::vector<double> tau_feature_sm_global =
        tau_feature_sm.empty()
            ? std::vector<double>()
            : gather_spin_major_to_global(this->rho_basis_, tau_feature_sm, nspin);

    std::vector<double> exc_baseline_global(nxyz, 0.0);
    std::vector<double> exc_feature_global(nxyz, 0.0);
    std::vector<double> vrho_baseline_sm_global(nxyz * nspin, 0.0);
    std::vector<double> vrho_feature_sm_global(nxyz * nspin, 0.0);
    std::vector<double> vsigma_baseline_sm_global(nxyz * nsigma, 0.0);
    std::vector<double> vsigma_feature_sm_global(nxyz * nsigma, 0.0);
    std::vector<double> vtau_baseline_sm_global;
    std::vector<double> vtau_feature_sm_global;
    double* vtau_baseline_sm_global_ptr = nullptr;
    double* vtau_feature_sm_global_ptr = nullptr;
    if (is_mgga_) {
        vtau_baseline_sm_global.resize(nxyz * nspin, 0.0);
        vtau_feature_sm_global.resize(nxyz * nspin, 0.0);
        vtau_baseline_sm_global_ptr = vtau_baseline_sm_global.data();
        vtau_feature_sm_global_ptr = vtau_feature_sm_global.data();
    }

    const double* tau_baseline_sm_global_ptr =
        tau_baseline_sm_global.empty() ? nullptr : tau_baseline_sm_global.data();
    const double* tau_feature_sm_global_ptr =
        tau_feature_sm_global.empty() ? nullptr : tau_feature_sm_global.data();

    try
    {
        const hamilt::cider::DensityTupleInput baseline_input(
            rho_baseline_sm_global.data(),
            rho_baseline_sm_global.size(),
            sigma_baseline_sm_global.data(),
            sigma_baseline_sm_global.size(),
            tau_baseline_sm_global_ptr,
            tau_baseline_sm_global.size());
        const hamilt::cider::DensityTupleInput feature_input(
            rho_feature_sm_global.data(),
            rho_feature_sm_global.size(),
            sigma_feature_sm_global.data(),
            sigma_feature_sm_global.size(),
            tau_feature_sm_global_ptr,
            tau_feature_sm_global.size());
        const hamilt::cider::EvaluationInput input(
            static_cast<std::size_t>(nspin), nxyz, baseline_input, feature_input);
        const hamilt::cider::ComponentOutput baseline_output(
            exc_baseline_global.data(),
            exc_baseline_global.size(),
            vrho_baseline_sm_global.data(),
            vrho_baseline_sm_global.size(),
            vsigma_baseline_sm_global.data(),
            vsigma_baseline_sm_global.size(),
            vtau_baseline_sm_global_ptr,
            vtau_baseline_sm_global.size());
        const hamilt::cider::ComponentOutput feature_output(
            exc_feature_global.data(),
            exc_feature_global.size(),
            vrho_feature_sm_global.data(),
            vrho_feature_sm_global.size(),
            vsigma_feature_sm_global.data(),
            vsigma_feature_sm_global.size(),
            vtau_feature_sm_global_ptr,
            vtau_feature_sm_global.size());
        runtime_->evaluate(input, baseline_output, feature_output);
    }
    catch (const std::exception& error)
    {
        ModuleBase::WARNING_QUIT(
            "PotCiderXC",
            std::string("native CIDER evaluation failed for model '")
                + PARAM.inp.cider_model + "': " + error.what());
        throw;
    }

    exc_baseline = scatter_spin_major_from_global(this->rho_basis_, exc_baseline_global, 1);
    exc_feature = scatter_spin_major_from_global(this->rho_basis_, exc_feature_global, 1);
    vrho_baseline_sm = scatter_spin_major_from_global(
        this->rho_basis_, vrho_baseline_sm_global, nspin);
    vrho_feature_sm = scatter_spin_major_from_global(
        this->rho_basis_, vrho_feature_sm_global, nspin);
    vsigma_baseline_sm = scatter_spin_major_from_global(
        this->rho_basis_, vsigma_baseline_sm_global, nsigma);
    vsigma_feature_sm = scatter_spin_major_from_global(
        this->rho_basis_, vsigma_feature_sm_global, nsigma);
    if (is_mgga_) {
        vtau_baseline_sm = scatter_spin_major_from_global(
            this->rho_basis_, vtau_baseline_sm_global, nspin);
        vtau_feature_sm = scatter_spin_major_from_global(
            this->rho_basis_, vtau_feature_sm_global, nspin);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (std::size_t i = 0; i < vtau_total_sm.size(); ++i) {
            vtau_total_sm[i] = vtau_baseline_sm[i] + vtau_feature_sm[i];
        }
    }

    GlobalV::ofs_running
        << "PotCiderXC: native dual evaluation returned"
        << " exc_baseline0=" << (exc_baseline.empty() ? 0.0 : exc_baseline[0])
        << " exc_feature0=" << (exc_feature.empty() ? 0.0 : exc_feature[0])
        << " vrho_baseline0=" << (vrho_baseline_sm.empty() ? 0.0 : vrho_baseline_sm[0])
        << " vrho_feature0=" << (vrho_feature_sm.empty() ? 0.0 : vrho_feature_sm[0])
        << std::endl;

    const std::vector<double> sgn_baseline = build_cider_sgn(
        nspin,
        nrxx,
        rho_baseline_interleaved,
        sigma_baseline_interleaved,
        tau_baseline_interleaved,
        is_mgga_);
    const std::vector<double> sgn_feature = build_cider_sgn(
        nspin,
        nrxx,
        rho_feature_interleaved,
        sigma_feature_interleaved,
        tau_feature_interleaved,
        is_mgga_);
    const bool disable_feature_post_mask =
        env_flag_enabled("ABACUS_CIDER_DISABLE_FEATURE_POST_MASK");
    const std::vector<double> sgn_feature_potential(
        disable_feature_post_mask ? nrxx * static_cast<std::size_t>(nspin) : 0,
        1.0);
    const std::vector<double>& sgn_feature_for_potential =
        disable_feature_post_mask ? sgn_feature_potential : sgn_feature;
    const double feature_vrho_scale =
        env_double_or_default("ABACUS_CIDER_FEATURE_VRHO_SCALE", 1.0);
    const double feature_vsigma_scale =
        env_double_or_default("ABACUS_CIDER_FEATURE_VSIGMA_SCALE", 1.0);
    const double feature_vtau_scale =
        env_double_or_default("ABACUS_CIDER_FEATURE_VTAU_SCALE", 1.0);
    if (disable_feature_post_mask
        || feature_vrho_scale != 1.0
        || feature_vsigma_scale != 1.0
        || feature_vtau_scale != 1.0) {
        GlobalV::ofs_running
            << "PotCiderXC: feature diagnostic switches"
            << " call=" << eval_count
            << " disable_feature_post_mask=" << (disable_feature_post_mask ? 1 : 0)
            << " vrho_scale=" << feature_vrho_scale
            << " vsigma_scale=" << feature_vsigma_scale
            << " vtau_scale=" << feature_vtau_scale
            << std::endl;
    }

    // === 7. Integrate energy components using each model's convention ===
    double etxc_baseline_local = integrate_role_exc(
        exc_baseline, sgn_baseline, rho_baseline_interleaved, nspin, nrxx);
    const double etxc_feature_volume_local =
        integrate_volume_exc(exc_feature, nrxx);
    const double etxc_feature_rho_weighted_local =
        integrate_role_exc(
            exc_feature, sgn_feature, rho_feature_interleaved, nspin, nrxx);
    double etxc_feature_local =
        feature_exc_convention_ == "energy_density"
            ? etxc_feature_volume_local
            : etxc_feature_rho_weighted_local;

    // === 8. Convert vrho/vsigma back to interleaved and use native ABACUS
    // helper machinery to reconstruct the local XC potential. This keeps the
    // divergence/gradient conventions aligned with the normal libxc path.
    const std::vector<double> vrho_baseline_int =
        spin_major_to_interleaved(vrho_baseline_sm, nspin, nrxx);
    const std::vector<double> vsigma_baseline_int =
        spin_major_to_interleaved(vsigma_baseline_sm, nsigma, nrxx);
    const std::vector<double> vrho_feature_int =
        scaled_vector_copy(
            spin_major_to_interleaved(vrho_feature_sm, nspin, nrxx),
            feature_vrho_scale);
    const std::vector<double> vsigma_feature_int =
        scaled_vector_copy(
            spin_major_to_interleaved(vsigma_feature_sm, nsigma, nrxx),
            feature_vsigma_scale);

    auto vtxc_v_baseline = reconstruct_cider_vxc_with_native_helper(
        nspin,
        nrxx,
        sgn_baseline,
        rho_baseline_interleaved,
        gdr_baseline,
        vrho_baseline_int,
        vsigma_baseline_int,
        tpiba,
        chg);
    auto vtxc_v_feature = reconstruct_cider_vxc_with_native_helper(
        nspin,
        nrxx,
        sgn_feature_for_potential,
        rho_feature_interleaved,
        gdr_feature,
        vrho_feature_int,
        vsigma_feature_int,
        tpiba,
        chg);
    const double vtxc_baseline_local = std::get<0>(vtxc_v_baseline);
    const double vtxc_feature_local = std::get<0>(vtxc_v_feature);
    double vtxc_local = vtxc_baseline_local + vtxc_feature_local;
    const ModuleBase::matrix& v_baseline = std::get<1>(vtxc_v_baseline);
    const ModuleBase::matrix& v_feature = std::get<1>(vtxc_v_feature);
    const double feature_v_cap = positive_env_double("ABACUS_CIDER_FEATURE_V_CAP");
    const double feature_vtau_cap = positive_env_double("ABACUS_CIDER_FEATURE_VTAU_CAP");
    ModuleBase::matrix v_feature_hamiltonian = v_feature;
    const CiderPotentialClipStats v_clip_stats =
        clip_feature_matrix_in_place(v_feature_hamiltonian, feature_v_cap);
    ModuleBase::matrix v_cider = v_baseline;
    g_last_cider_feature_v = v_feature_hamiltonian;
    g_last_cider_feature_v_valid = true;
    for (int is = 0; is < v_cider.nr; ++is) {
        for (int ir = 0; ir < v_cider.nc; ++ir) {
            v_cider(is, ir) += v_feature_hamiltonian(is, ir);
        }
    }

#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc_baseline_local);
    Parallel_Reduce::reduce_pool(etxc_feature_local);
    Parallel_Reduce::reduce_pool(vtxc_local);
#endif

    const double grid_weight = ucell->omega / chg->rhopw->nxyz;
    const double etxc_baseline_ry = etxc_baseline_local * grid_weight;
    const double etxc_feature_ry = etxc_feature_local * grid_weight;
    const double etxc_feature_volume_ry = etxc_feature_volume_local * grid_weight;
    const double etxc_feature_rho_weighted_ry =
        etxc_feature_rho_weighted_local * grid_weight;
    const double etxc_total_ry = etxc_baseline_ry + etxc_feature_ry;
    *(this->etxc_) = etxc_total_ry;
    *(this->vtxc_) = vtxc_local * grid_weight;
    const double ry_to_ha = 0.5;
    g_last_energy_components.valid = true;
    g_last_energy_components.baseline_Ha = etxc_baseline_ry * ry_to_ha;
    g_last_energy_components.feature_scaled_Ha = etxc_feature_ry * ry_to_ha;
    g_last_energy_components.feature_unmixed_Ha =
        std::abs(PARAM.inp.cider_xmix) > 1.0e-14
            ? (etxc_feature_ry / PARAM.inp.cider_xmix) * ry_to_ha
            : 0.0;
    g_last_energy_components.total_Ha = etxc_total_ry * ry_to_ha;
    g_last_energy_components.feature_volume_Ha = etxc_feature_volume_ry * ry_to_ha;
    g_last_energy_components.feature_rho_weighted_Ha =
        etxc_feature_rho_weighted_ry * ry_to_ha;
    g_last_energy_components.feature_density_policy = feature_density_policy_;
    g_last_energy_components.feature_exc_convention = feature_exc_convention_;

    maybe_write_abacus_cider_debug_dump(
        eval_count,
        nspin,
        nsigma,
        nrxx,
        grid_weight,
        PARAM.inp.cider_xmix,
        cider_feature_policy_code(feature_density_policy_),
        rho_baseline_sm,
        sigma_baseline_sm,
        tau_baseline_sm,
        rho_feature_sm,
        sigma_feature_sm,
        tau_feature_sm,
        exc_baseline,
        exc_feature,
        vrho_baseline_sm,
        vsigma_baseline_sm,
        vtau_baseline_sm,
        vrho_feature_sm,
        vsigma_feature_sm,
        vtau_feature_sm,
        v_baseline,
        v_feature_hamiltonian,
        v_cider,
        etxc_baseline_ry,
        etxc_feature_ry,
        cider_exc_convention_code(feature_exc_convention_),
        etxc_feature_volume_ry,
        etxc_feature_rho_weighted_ry,
        vtxc_baseline_local * grid_weight,
        vtxc_feature_local * grid_weight);

    const double xmix = PARAM.inp.cider_xmix;
    GlobalV::ofs_running
        << "PotCiderXC: component energy diagnostic"
        << " baseline_ry=" << etxc_baseline_ry
        << " baseline_Ha=" << (etxc_baseline_ry * ry_to_ha)
        << " feature_scaled_ry=" << etxc_feature_ry
        << " feature_scaled_Ha=" << (etxc_feature_ry * ry_to_ha);
    if (std::abs(xmix) > 1.0e-14) {
        const double etxc_feature_unmixed_ry = etxc_feature_ry / xmix;
        GlobalV::ofs_running
            << " feature_unmixed_ry=" << etxc_feature_unmixed_ry
            << " feature_unmixed_Ha=" << (etxc_feature_unmixed_ry * ry_to_ha);
    } else {
        GlobalV::ofs_running
            << " feature_unmixed_ry=nan"
            << " feature_unmixed_Ha=nan";
    }
    GlobalV::ofs_running
        << " total_ry=" << etxc_total_ry
        << " total_Ha=" << (etxc_total_ry * ry_to_ha)
        << " xmix=" << xmix
        << std::endl;

    std::vector<double> vtau_baseline_int;
    std::vector<double> vtau_feature_int;
    std::vector<double> vtau_feature_contrib_int;
    CiderPotentialClipStats vtau_clip_stats;
    if (is_mgga_ && !vtau_total_sm.empty())
    {
        vtau_baseline_int =
            spin_major_to_interleaved(vtau_baseline_sm, nspin, nrxx);
        vtau_feature_int =
            spin_major_to_interleaved(vtau_feature_sm, nspin, nrxx);
        vtau_feature_contrib_int.resize(nrxx * nspin, 0.0);
        for (int is = 0; is < nspin; ++is)
        {
            for (std::size_t ir = 0; ir < nrxx; ++ir)
            {
                const std::size_t index = ir * nspin + is;
                vtau_feature_contrib_int[index] =
                    vtau_feature_int[index]
                    * sgn_feature_for_potential[index]
                    * feature_vtau_scale;
            }
        }
        vtau_clip_stats =
            clip_vector_in_place(vtau_feature_contrib_int, feature_vtau_cap);
    }

    log_cider_potential_clipping(
        eval_count,
        feature_v_cap,
        v_clip_stats,
        feature_vtau_cap,
        vtau_clip_stats);

    ModuleBase::matrix cider_vofk;
    if (is_mgga_ && !vtau_total_sm.empty())
    {
        cider_vofk.create(nspin, nrxx);
        for (int is = 0; is < nspin; ++is)
        {
            for (std::size_t ir = 0; ir < nrxx; ++ir)
            {
                const std::size_t index = ir * nspin + is;
                cider_vofk(is, ir) =
                    vtau_baseline_int[index] * sgn_baseline[index]
                    + vtau_feature_contrib_int[index];
            }
        }
    }

    if (std::abs(PARAM.inp.cider_xmix) < 1e-14) {
        log_same_density_xc_comparison(
            chg,
            ucell,
            *(this->etxc_),
            *(this->vtxc_),
            v_cider,
            cider_vofk.nc > 0 ? &cider_vofk : nullptr);
    }

    if (std::getenv("ABACUS_CIDER_POTENTIAL_DIAGNOSTICS") != nullptr) {
        std::vector<double> vtau_feature_int;
        if (is_mgga_ && !vtau_feature_sm.empty()) {
            vtau_feature_int =
                spin_major_to_interleaved(vtau_feature_sm, nspin, nrxx);
        }
        const CiderPotentialDiagnostics potential_stats =
            collect_cider_potential_diagnostics(
                nspin,
                nrxx,
                sgn_feature,
                rho_feature_interleaved,
                sigma_feature_interleaved,
                tau_feature_interleaved,
                v_feature,
                vtau_feature_int);
        log_cider_potential_diagnostics(
            eval_count,
            grid_weight,
            potential_stats);
    }

    v_eff += v_cider;

    GlobalV::ofs_running
        << "PotCiderXC: completed call=" << eval_count
        << " etxc=" << *(this->etxc_)
        << " vtxc=" << *(this->vtxc_)
        << " grid_weight=" << grid_weight
        << std::endl;

    // === 9. MGGA: assign summed role derivatives into vofk ===
    if (is_mgga_ && vofk_ != nullptr && !vtau_total_sm.empty()) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                const std::size_t index = ir * nspin + is;
                (*vofk_)(is, ir) =
                    vtau_baseline_int[index] * sgn_baseline[index]
                    + vtau_feature_contrib_int[index];
            }
        }
    }

    ModuleBase::timer::end("PotCiderXC", "cal_v_eff");
}

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC
