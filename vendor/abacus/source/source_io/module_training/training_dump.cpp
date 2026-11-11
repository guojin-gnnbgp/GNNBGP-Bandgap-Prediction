#if defined(USE_LIBXC)

#include "training_dump.h"

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_comm.h"
#include "source_base/parallel_global.h"
#include "source_base/timer.h"
#include "source_estate/module_charge/charge.h"
#if defined(ENABLE_CIDER)
#include "source_estate/module_pot/pot_cider_xc.h"
#endif
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_hamilt/module_xc/libxc_abacus.h"
#include "source_io/module_parameter/parameter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef __HDF5
#include <hdf5.h>
#endif

#ifdef __MPI
#include <mpi.h>
#endif

namespace
{

struct TrainingArray
{
    std::string key;
    std::vector<unsigned long> shape;
    std::vector<double> data;
};

constexpr double RY_TO_HA = 0.5;

double ry_to_ha(const double energy_ry)
{
    return RY_TO_HA * energy_ry;
}

bool is_norm_conserving_pseudo(const UnitCell& ucell)
{
    for (int it = 0; it < ucell.ntype; ++it)
    {
        if (ucell.atoms[it].ncpp.tvanp)
        {
            return false;
        }
        if (ucell.atoms[it].ncpp.pp_type != "NC")
        {
            return false;
        }
    }
    return true;
}

std::vector<double> gather_spin_major_to_global(const ModulePW::PW_Basis& rho_basis,
                                                const std::vector<double>& local,
                                                const int nchannels)
{
    const int nrxx_local = rho_basis.nrxx;
    const int nxyz = rho_basis.nxyz;
    if (nchannels <= 0)
    {
        return {};
    }

#ifdef __MPI
    if (GlobalV::NPROC_IN_POOL > 1 && POOL_WORLD != MPI_COMM_NULL)
    {
        const int ncxy = rho_basis.nx * rho_basis.ny;
        std::vector<double> global(nchannels * nxyz, 0.0);
        std::vector<double> gathered(nchannels * nxyz, 0.0);
        std::vector<int> rec(GlobalV::NPROC_IN_POOL);
        std::vector<int> dis(GlobalV::NPROC_IN_POOL);
        for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ++ip)
        {
            rec[ip] = rho_basis.numz[ip] * ncxy;
            dis[ip] = rho_basis.startz[ip] * ncxy;
        }

        for (int ich = 0; ich < nchannels; ++ich)
        {
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
            for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ++ip)
            {
                for (int ixy = 0; ixy < ncxy; ++ixy)
                {
                    for (int iz = 0; iz < rho_basis.numz[ip]; ++iz)
                    {
                        global_ch[rho_basis.nz * ixy + rho_basis.startz[ip] + iz] =
                            gathered_ch[rho_basis.numz[ip] * ixy + rho_basis.startz[ip] * ncxy + iz];
                    }
                }
            }
        }
        return global;
    }
#endif

    return local;
}

std::string escape_json(const std::string& value)
{
    std::ostringstream os;
    for (const char c : value)
    {
        switch (c)
        {
        case '\\':
            os << "\\\\";
            break;
        case '"':
            os << "\\\"";
            break;
        case '\n':
            os << "\\n";
            break;
        case '\r':
            os << "\\r";
            break;
        case '\t':
            os << "\\t";
            break;
        default:
            os << c;
        }
    }
    return os.str();
}

std::string json_string(const std::string& value)
{
    return "\"" + escape_json(value) + "\"";
}

void write_npy_header(std::ofstream& os, const std::vector<unsigned long>& shape)
{
    std::ostringstream dict;
    dict << "{'descr': '<f8', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
        {
            dict << ", ";
        }
        dict << shape[i];
    }
    if (shape.size() == 1)
    {
        dict << ",";
    }
    dict << "), }";

    std::string header = dict.str();
    const std::size_t prefix_size = 10;
    const std::size_t padding = 16 - ((prefix_size + header.size() + 1) % 16);
    header.append(padding, ' ');
    header.push_back('\n');

    os.write("\x93NUMPY", 6);
    const char version[2] = {1, 0};
    os.write(version, 2);
    const unsigned short header_len = static_cast<unsigned short>(header.size());
    os.put(static_cast<char>(header_len & 0xff));
    os.put(static_cast<char>((header_len >> 8) & 0xff));
    os.write(header.data(), static_cast<std::streamsize>(header.size()));
}

void write_npy(const std::string& filename,
               const std::vector<unsigned long>& shape,
               const std::vector<double>& data)
{
    std::ofstream os(filename.c_str(), std::ios::binary);
    if (!os)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "Failed to open " + filename);
    }
    write_npy_header(os, shape);
    if (!data.empty())
    {
        os.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size() * sizeof(double)));
    }
}

#ifdef __HDF5
void check_hdf5(const herr_t status, const std::string& context)
{
    if (status < 0)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "HDF5 failure while " + context);
    }
}

class H5Handle
{
  public:
    H5Handle() = default;
    H5Handle(hid_t id, herr_t (*closer)(hid_t)) : id_(id), closer_(closer) {}
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    H5Handle(H5Handle&& other) noexcept : id_(other.id_), closer_(other.closer_)
    {
        other.id_ = -1;
        other.closer_ = nullptr;
    }
    H5Handle& operator=(H5Handle&& other) noexcept
    {
        if (this != &other)
        {
            close();
            id_ = other.id_;
            closer_ = other.closer_;
            other.id_ = -1;
            other.closer_ = nullptr;
        }
        return *this;
    }
    ~H5Handle()
    {
        close();
    }
    hid_t get() const
    {
        return id_;
    }

  private:
    void close()
    {
        if (id_ >= 0 && closer_ != nullptr)
        {
            closer_(id_);
        }
        id_ = -1;
        closer_ = nullptr;
    }
    hid_t id_ = -1;
    herr_t (*closer_)(hid_t) = nullptr;
};

std::vector<hsize_t> hdf5_dims(const std::vector<unsigned long>& shape)
{
    std::vector<hsize_t> dims(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        dims[i] = static_cast<hsize_t>(shape[i]);
    }
    return dims;
}

void write_hdf5_string_dataset(const hid_t parent, const std::string& name, const std::string& value)
{
    H5Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    check_hdf5(type.get(), "creating string type");
    check_hdf5(H5Tset_size(type.get(), value.size() + 1), "setting string dataset size");
    check_hdf5(H5Tset_strpad(type.get(), H5T_STR_NULLTERM), "setting string padding");
    H5Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    check_hdf5(space.get(), "creating scalar string dataspace");
    H5Handle dataset(H5Dcreate2(parent, name.c_str(), type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                     H5Dclose);
    check_hdf5(dataset.get(), "creating string dataset " + name);
    check_hdf5(H5Dwrite(dataset.get(), type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, value.data()),
               "writing string dataset " + name);
}

void write_hdf5_string_attr(const hid_t parent, const std::string& name, const std::string& value)
{
    H5Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    check_hdf5(type.get(), "creating string attribute type");
    check_hdf5(H5Tset_size(type.get(), value.size() + 1), "setting string attribute size");
    H5Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    check_hdf5(space.get(), "creating scalar string attribute dataspace");
    H5Handle attr(H5Acreate2(parent, name.c_str(), type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    check_hdf5(attr.get(), "creating string attribute " + name);
    check_hdf5(H5Awrite(attr.get(), type.get(), value.c_str()), "writing string attribute " + name);
}

void write_hdf5_int_attr(const hid_t parent, const std::string& name, const int value)
{
    H5Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    check_hdf5(space.get(), "creating integer attribute dataspace");
    H5Handle attr(H5Acreate2(parent, name.c_str(), H5T_NATIVE_INT, space.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    check_hdf5(attr.get(), "creating integer attribute " + name);
    check_hdf5(H5Awrite(attr.get(), H5T_NATIVE_INT, &value), "writing integer attribute " + name);
}

void write_hdf5_array(const hid_t arrays_group,
                      const std::string& key,
                      const std::vector<unsigned long>& shape,
                      const std::vector<double>& data)
{
    const std::vector<hsize_t> dims = hdf5_dims(shape);
    H5Handle space(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
    check_hdf5(space.get(), "creating dataspace for " + key);
    H5Handle dataset(H5Dcreate2(arrays_group,
                                key.c_str(),
                                H5T_NATIVE_DOUBLE,
                                space.get(),
                                H5P_DEFAULT,
                                H5P_DEFAULT,
                                H5P_DEFAULT),
                     H5Dclose);
    check_hdf5(dataset.get(), "creating dataset " + key);
    check_hdf5(H5Dwrite(dataset.get(),
                        H5T_NATIVE_DOUBLE,
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        data.empty() ? nullptr : data.data()),
               "writing dataset " + key);
}

void write_hdf5_training_dump(const std::string& filename,
                              const std::string& record_json,
                              const std::vector<TrainingArray>& arrays)
{
    const std::string tmp = filename + ".tmp";
    std::remove(tmp.c_str());
    H5Handle file(H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    check_hdf5(file.get(), "creating " + tmp);
    write_hdf5_string_attr(file.get(), "schema_name", "abacus_pw_nldx_training_dump_hdf5");
    write_hdf5_int_attr(file.get(), "schema_version", 1);
    write_hdf5_string_dataset(file.get(), "record_json", record_json);
    H5Handle record_group(H5Gcreate2(file.get(), "record", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    check_hdf5(record_group.get(), "creating record group");
    write_hdf5_string_attr(record_group.get(), "schema_name", "abacus_pw_nldx_training_dump");
    write_hdf5_int_attr(record_group.get(), "schema_version", 2);
    H5Handle arrays_group(H5Gcreate2(file.get(), "arrays", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    check_hdf5(arrays_group.get(), "creating arrays group");
    for (const TrainingArray& array : arrays)
    {
        write_hdf5_array(arrays_group.get(), array.key, array.shape, array.data);
    }
    check_hdf5(H5Fflush(file.get(), H5F_SCOPE_GLOBAL), "flushing " + tmp);
    file = H5Handle();
    if (std::rename(tmp.c_str(), filename.c_str()) != 0)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "Failed to rename " + tmp + " to " + filename);
    }
}
#endif

std::vector<double> matrix_to_vector(const ModuleBase::matrix& matrix)
{
    const int size = matrix.nr * matrix.nc;
    std::vector<double> out(size, 0.0);
    for (int i = 0; i < size; ++i)
    {
        out[i] = matrix.c[i];
    }
    return out;
}

std::vector<double> compute_sigma_local(const int nspin,
                                        const int nsigma,
                                        const int nrxx,
                                        const std::vector<double>& rho_interleaved,
                                        const double tpiba,
                                        const Charge& chr)
{
    const auto gdr = XC_Functional_Libxc::cal_gdr(nspin, nrxx, rho_interleaved, tpiba, &chr);
    const auto sigma_interleaved = XC_Functional_Libxc::convert_sigma(gdr);
    std::vector<double> sigma_local(nsigma * nrxx, 0.0);
    for (int isig = 0; isig < nsigma; ++isig)
    {
        for (int ir = 0; ir < nrxx; ++ir)
        {
            sigma_local[isig * nrxx + ir] = sigma_interleaved[ir * nsigma + isig];
        }
    }
    return sigma_local;
}

std::vector<double> ry_matrix_to_ha_vector(const ModuleBase::matrix& matrix)
{
    std::vector<double> out = matrix_to_vector(matrix);
    for (double& value : out)
    {
        value = ry_to_ha(value);
    }
    return out;
}

void compute_frontier(const elecstate::ElecState& elec, double& vbm, double& cbm)
{
    vbm = -std::numeric_limits<double>::infinity();
    cbm = std::numeric_limits<double>::infinity();
    if (elec.ekb.nr == 0 || elec.ekb.nc == 0)
    {
        vbm = elec.eferm.ef;
        cbm = elec.eferm.ef;
        return;
    }

    const int nks = elec.klist == nullptr ? elec.ekb.nr : elec.klist->get_nks();
    constexpr double threshold = 1.0e-5;
    for (int ik = 0; ik < nks; ++ik)
    {
        for (int ib = 0; ib < elec.ekb.nc; ++ib)
        {
            const double eig = elec.ekb(ik, ib);
            if (eig <= elec.eferm.ef + threshold && eig > vbm)
            {
                vbm = eig;
            }
            if (eig > elec.eferm.ef + threshold && eig < cbm)
            {
                cbm = eig;
            }
        }
    }
    if (cbm == std::numeric_limits<double>::infinity())
    {
        cbm = elec.eferm.ef;
    }
    if (vbm == -std::numeric_limits<double>::infinity())
    {
        vbm = elec.eferm.ef;
    }
#ifdef __MPI
    if (POOL_WORLD != MPI_COMM_NULL)
    {
        MPI_Allreduce(MPI_IN_PLACE, &vbm, 1, MPI_DOUBLE, MPI_MAX, POOL_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &cbm, 1, MPI_DOUBLE, MPI_MIN, POOL_WORLD);
    }
#endif
}

std::vector<double> kvector_matrix(const elecstate::ElecState& elec, const bool cartesian)
{
    std::vector<double> out;
    if (elec.klist == nullptr)
    {
        return out;
    }
    const int nks = elec.klist->get_nks();
    out.resize(nks * 3, 0.0);
    for (int ik = 0; ik < nks; ++ik)
    {
        const auto& kvec = cartesian ? elec.klist->kvec_c[ik] : elec.klist->kvec_d[ik];
        out[3 * ik] = kvec.x;
        out[3 * ik + 1] = kvec.y;
        out[3 * ik + 2] = kvec.z;
    }
    return out;
}

std::vector<double> kweights(const elecstate::ElecState& elec)
{
    std::vector<double> out;
    if (elec.klist == nullptr)
    {
        return out;
    }
    const int nks = elec.klist->get_nks();
    out.resize(nks, 0.0);
    for (int ik = 0; ik < nks; ++ik)
    {
        out[ik] = elec.klist->wk[ik];
    }
    return out;
}

std::vector<double> kspins(const elecstate::ElecState& elec)
{
    std::vector<double> out;
    if (elec.klist == nullptr)
    {
        return out;
    }
    const int nks = elec.klist->get_nks();
    out.resize(nks, 0.0);
    for (int ik = 0; ik < nks; ++ik)
    {
        out[ik] = static_cast<double>(elec.klist->isk[ik]);
    }
    return out;
}

} // namespace

namespace ModuleIO
{

namespace
{
bool last_training_band_edge_label_valid = false;
TrainingBandEdgeLabel last_training_band_edge_label;
bool last_training_sdmx_wavefunction_valid = false;
TrainingSDMXWavefunction last_training_sdmx_wavefunction;
}

void set_last_training_band_edge_label(const TrainingBandEdgeLabel& label)
{
    last_training_band_edge_label = label;
    last_training_band_edge_label_valid = label.present;
}

void clear_last_training_band_edge_label()
{
    last_training_band_edge_label = TrainingBandEdgeLabel();
    last_training_band_edge_label_valid = false;
}

void set_last_training_sdmx_wavefunction(const TrainingSDMXWavefunction& wavefunction)
{
    last_training_sdmx_wavefunction = wavefunction;
    last_training_sdmx_wavefunction_valid = wavefunction.present;
}

void clear_last_training_sdmx_wavefunction()
{
    last_training_sdmx_wavefunction = TrainingSDMXWavefunction();
    last_training_sdmx_wavefunction_valid = false;
}

void write_training_dump(const UnitCell& ucell,
                         const elecstate::ElecState& elec,
                         const ModulePW::PW_Basis& rho_basis,
                         const Charge& chr,
                         const bool scf_converged,
                         const int istep,
                         const std::string& output_root,
                         const TrainingBandEdgeLabel* band_edge_label,
                         const TrainingSDMXWavefunction* sdmx_wavefunction)
{
    ModuleBase::TITLE("ModuleIO", "write_training_dump");
    ModuleBase::timer::start("ModuleIO", "write_training_dump");

    if (PARAM.inp.basis_type != "pw")
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data is only available for basis_type pw");
    }
    std::string dft_functional_lower = PARAM.inp.dft_functional;
    std::transform(dft_functional_lower.begin(),
                   dft_functional_lower.end(),
                   dft_functional_lower.begin(),
                   ::tolower);
    if (dft_functional_lower != "pbe")
    {
        ModuleBase::WARNING_QUIT("write_training_dump",
                                 "out_training_data currently requires dft_functional = PBE for the PBE-base training workflow");
    }
    if (PARAM.inp.nspin != 1 && PARAM.inp.nspin != 2)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data supports nspin 1 or 2");
    }
    if (PARAM.inp.kpar != 1)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data currently requires kpar = 1");
    }
    if (PARAM.inp.bndpar != 1)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data currently requires bndpar = 1");
    }
    if (!is_norm_conserving_pseudo(ucell))
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data currently requires NC pseudopotentials");
    }

    const int nspin = PARAM.inp.nspin;
    const int nsigma = (nspin == 1) ? 1 : 3;
    const int nrxx = rho_basis.nrxx;
    const int nxyz = rho_basis.nxyz;
    const double dvol = ucell.omega / static_cast<double>(nxyz);

    std::vector<double> rho_valence_local(nspin * nrxx, 0.0);
    std::vector<double> rho_total_local(nspin * nrxx, 0.0);
    std::vector<double> rho_valence_interleaved(nspin * nrxx, 0.0);
    std::vector<double> rho_interleaved(nspin * nrxx, 0.0);
    for (int is = 0; is < nspin; ++is)
    {
        for (int ir = 0; ir < nrxx; ++ir)
        {
            const double rho_valence = chr.rho[is][ir];
            const double rho_total = rho_valence + chr.rho_core[ir] / static_cast<double>(nspin);
            rho_valence_local[is * nrxx + ir] = rho_valence;
            rho_total_local[is * nrxx + ir] = rho_total;
            rho_valence_interleaved[ir * nspin + is] = rho_valence;
            rho_interleaved[ir * nspin + is] = rho_total;
        }
    }

    std::vector<double> rho_core_local(nrxx, 0.0);
    for (int ir = 0; ir < nrxx; ++ir)
    {
        rho_core_local[ir] = chr.rho_core[ir];
    }

    const std::vector<double> sigma_valence_local =
        compute_sigma_local(nspin, nsigma, nrxx, rho_valence_interleaved, ucell.tpiba, chr);
    const std::vector<double> sigma_local =
        compute_sigma_local(nspin, nsigma, nrxx, rho_interleaved, ucell.tpiba, chr);

    std::vector<double> tau_local;
    std::vector<double> tau_valence_local;
    if (chr.kin_r != nullptr)
    {
        tau_local.assign(nspin * nrxx, 0.0);
        tau_valence_local.assign(nspin * nrxx, 0.0);
        const double pi = std::acos(-1.0);
        const double tf_factor = (3.0 / 10.0) * std::pow(3.0 * pi * pi, 2.0 / 3.0);
        for (int is = 0; is < nspin; ++is)
        {
            for (int ir = 0; ir < nrxx; ++ir)
            {
                double tau = chr.kin_r[is][ir] / 2.0;
                tau_valence_local[is * nrxx + ir] = tau;
                if (PARAM.inp.cider_tf_tau)
                {
                    const double rho_cps = std::max(chr.rho_core[ir] / static_cast<double>(nspin), 0.0);
                    tau += tf_factor * std::pow(rho_cps, 5.0 / 3.0);
                }
                tau_local[is * nrxx + ir] = tau;
            }
        }
    }

    const std::vector<double> rho_valence = gather_spin_major_to_global(rho_basis, rho_valence_local, nspin);
    const std::vector<double> rho_total = gather_spin_major_to_global(rho_basis, rho_total_local, nspin);
    const std::vector<double> rho_core = gather_spin_major_to_global(rho_basis, rho_core_local, 1);
    const std::vector<double> sigma = gather_spin_major_to_global(rho_basis, sigma_local, nsigma);
    const std::vector<double> sigma_valence = gather_spin_major_to_global(rho_basis, sigma_valence_local, nsigma);
    const std::vector<double> tau = tau_local.empty() ? std::vector<double>()
                                                      : gather_spin_major_to_global(rho_basis, tau_local, nspin);
    const std::vector<double> tau_valence = tau_valence_local.empty()
                                                ? std::vector<double>()
                                                : gather_spin_major_to_global(rho_basis, tau_valence_local, nspin);
    double vbm = 0.0;
    double cbm = 0.0;
    compute_frontier(elec, vbm, cbm);
    const TrainingBandEdgeLabel* edge_label = band_edge_label;
    if (edge_label == nullptr && PARAM.inp.out_bandgap_label && last_training_band_edge_label_valid)
    {
        edge_label = &last_training_band_edge_label;
    }
    const std::size_t nedge = edge_label != nullptr && edge_label->present ? edge_label->states.size() : 0;
    int vbm_count = 0;
    int cbm_count = 0;
    double vbm_shift_sum_Ha = 0.0;
    double cbm_shift_sum_Ha = 0.0;
    bool direct_gap = false;
    if (edge_label != nullptr && edge_label->present)
    {
        if (nedge == 0)
        {
            ModuleBase::WARNING_QUIT("write_training_dump", "band-edge label contains no selected states");
        }
        for (const auto& state: edge_label->states)
        {
            if (state.spin < 0 || state.spin >= nspin || state.k < 0 || state.k >= elec.ekb.nr
                || state.band < 0 || state.band >= elec.ekb.nc || !std::isfinite(state.eigenvalue_Ha)
                || !std::isfinite(state.occupation) || !std::isfinite(state.exx_shift_Ha))
            {
                ModuleBase::WARNING_QUIT("write_training_dump", "band-edge state metadata is invalid");
            }
            if (state.role == TrainingBandEdgeRole::Vbm)
            {
                ++vbm_count;
                vbm_shift_sum_Ha += state.exx_shift_Ha;
            }
            else
            {
                ++cbm_count;
                cbm_shift_sum_Ha += state.exx_shift_Ha;
            }
        }
        if (vbm_count == 0 || cbm_count == 0)
        {
            ModuleBase::WARNING_QUIT("write_training_dump", "band-edge label lacks a VBM or CBM state");
        }
        for (const auto& vbm_state: edge_label->states)
        {
            if (vbm_state.role != TrainingBandEdgeRole::Vbm)
            {
                continue;
            }
            for (const auto& cbm_state: edge_label->states)
            {
                direct_gap = direct_gap
                             || (cbm_state.role == TrainingBandEdgeRole::Cbm && cbm_state.k == vbm_state.k);
            }
        }
    }
    const TrainingSDMXWavefunction* sdmx_label = sdmx_wavefunction;
    if (sdmx_label == nullptr && PARAM.inp.out_sdmx_descriptor_data && last_training_sdmx_wavefunction_valid)
    {
        sdmx_label = &last_training_sdmx_wavefunction;
    }
    if (PARAM.inp.out_sdmx_descriptor_data && (sdmx_label == nullptr || !sdmx_label->present))
    {
        ModuleBase::WARNING_QUIT("write_training_dump",
                                 "out_sdmx_descriptor_data requires PW wavefunction coefficients");
    }
    TrainingExxEnergyDensity exx_density_storage;
    const TrainingExxEnergyDensity* exx_density_label
        = PARAM.inp.out_exx_energy_density && take_last_training_exx_energy_density(exx_density_storage)
              ? &exx_density_storage
              : nullptr;
    if (PARAM.inp.out_exx_energy_density && (exx_density_label == nullptr || !exx_density_label->present))
    {
        ModuleBase::WARNING_QUIT("write_training_dump",
                                 "out_exx_energy_density requires a current post-SCF EXX energy-density payload");
    }

    std::vector<double> exx_energy_density;
    double exx_density_integrated_energy_Ha = 0.0;
    double exx_density_abs_residual_Ha = 0.0;
    double exx_density_rel_residual = 0.0;
    if (exx_density_label != nullptr)
    {
        if (exx_density_label->values_local_Ha_bohr3.size() != static_cast<std::size_t>(nrxx))
        {
            ModuleBase::WARNING_QUIT("write_training_dump", "EXX energy-density local grid size is inconsistent");
        }
        if (exx_density_label->nx != rho_basis.nx || exx_density_label->ny != rho_basis.ny
            || exx_density_label->nz != rho_basis.nz)
        {
            ModuleBase::WARNING_QUIT("write_training_dump",
                                     "EXX energy-density mesh does not match the charge-density FFT mesh");
        }
        if (!std::isfinite(exx_density_label->scalar_energy_Ha)
            || !std::isfinite(exx_density_label->max_imaginary_abs_Ha_bohr3))
        {
            ModuleBase::WARNING_QUIT("write_training_dump", "EXX energy-density diagnostics are non-finite");
        }
        const double elec_scalar_Ha = ry_to_ha(elec.f_en.exx);
        const double scalar_consistency_tolerance = std::max(1.0e-12, 1.0e-12 * std::abs(elec_scalar_Ha));
        if (std::abs(exx_density_label->scalar_energy_Ha - elec_scalar_Ha) > scalar_consistency_tolerance)
        {
            ModuleBase::WARNING_QUIT("write_training_dump",
                                     "EXX energy-density payload scalar does not match the training EXX label");
        }
        exx_energy_density = gather_spin_major_to_global(rho_basis,
                                                         exx_density_label->values_local_Ha_bohr3,
                                                         1);
        if (exx_energy_density.size() != static_cast<std::size_t>(nxyz)
            || !std::all_of(exx_energy_density.begin(), exx_energy_density.end(), [](double value) {
                   return std::isfinite(value);
               }))
        {
            ModuleBase::WARNING_QUIT("write_training_dump", "EXX energy-density global grid is invalid");
        }
        exx_density_integrated_energy_Ha
            = std::accumulate(exx_energy_density.begin(), exx_energy_density.end(), 0.0) * dvol;
        exx_density_abs_residual_Ha
            = std::abs(exx_density_integrated_energy_Ha - exx_density_label->scalar_energy_Ha);
        exx_density_rel_residual
            = exx_density_abs_residual_Ha
              / std::max(std::abs(exx_density_label->scalar_energy_Ha), std::numeric_limits<double>::min());
        const double integration_tolerance
            = std::max(1.0e-8, 1.0e-8 * std::abs(exx_density_label->scalar_energy_Ha));
        if (exx_density_abs_residual_Ha > integration_tolerance)
        {
            ModuleBase::WARNING_QUIT(
                "write_training_dump",
                "EXX energy-density integral does not reproduce the scalar EXX label: residual = "
                    + std::to_string(exx_density_abs_residual_Ha) + " Ha");
        }
    }
    std::vector<double> orbital_sigma_txg_local;
    if (edge_label != nullptr && edge_label->present && !edge_label->orbital_rho_tsg.empty()
        && edge_label->orbital_sigma_txg.empty())
    {
        const std::size_t expected_rho_size = nedge * nspin * nrxx;
        if (edge_label->orbital_rho_tsg.size() != expected_rho_size)
        {
            ModuleBase::WARNING_QUIT("write_training_dump",
                                     "band_edge_orbital_rho_tsg has inconsistent size");
        }
        constexpr double eps = 1.0e-6;
        orbital_sigma_txg_local.assign(nedge * nsigma * nrxx, 0.0);
        for (std::size_t iedge = 0; iedge < nedge; ++iedge)
        {
            std::vector<double> rho_plus = rho_valence_interleaved;
            std::vector<double> rho_minus = rho_valence_interleaved;
            for (int is = 0; is < nspin; ++is)
            {
                for (int ir = 0; ir < nrxx; ++ir)
                {
                    const double response = edge_label->orbital_rho_tsg[(iedge * nspin + is) * nrxx + ir];
                    rho_plus[ir * nspin + is] += eps * response;
                    rho_minus[ir * nspin + is] -= eps * response;
                }
            }
            const std::vector<double> sigma_plus =
                compute_sigma_local(nspin, nsigma, nrxx, rho_plus, ucell.tpiba, chr);
            const std::vector<double> sigma_minus =
                compute_sigma_local(nspin, nsigma, nrxx, rho_minus, ucell.tpiba, chr);
            for (int isig = 0; isig < nsigma; ++isig)
            {
                for (int ir = 0; ir < nrxx; ++ir)
                {
                    orbital_sigma_txg_local[(iedge * nsigma + isig) * nrxx + ir] =
                        (sigma_plus[isig * nrxx + ir] - sigma_minus[isig * nrxx + ir]) / (2.0 * eps);
                }
            }
        }
    }

#ifdef __MPI
    if (GlobalV::MY_RANK != 0)
    {
        ModuleBase::timer::end("ModuleIO", "write_training_dump");
        return;
    }
#endif

    const std::string base_dir = output_root + "training_dump";
    ModuleBase::GlobalFunc::MAKE_DIR(base_dir);
    const int dump_step = std::max(istep, 0);
    const std::string dump_path = base_dir + "/step_" + std::to_string(dump_step) + ".h5";

    std::vector<TrainingArray> arrays;
    arrays.push_back({"rho_valence_sg",
                      {static_cast<unsigned long>(nspin), static_cast<unsigned long>(nxyz)},
                      rho_valence});
    arrays.push_back({"rho_sg",
                      {static_cast<unsigned long>(nspin), static_cast<unsigned long>(nxyz)},
                      rho_total});
    arrays.push_back({"rho_core_g", {static_cast<unsigned long>(nxyz)}, rho_core});
    arrays.push_back({"sigma_xg",
                      {static_cast<unsigned long>(nsigma), static_cast<unsigned long>(nxyz)},
                      sigma});
    arrays.push_back({"sigma_valence_xg",
                      {static_cast<unsigned long>(nsigma), static_cast<unsigned long>(nxyz)},
                      sigma_valence});
    if (exx_density_label != nullptr)
    {
        arrays.push_back({"exx_energy_density_g",
                          {static_cast<unsigned long>(nxyz)},
                          exx_energy_density});
    }
    if (!tau.empty())
    {
        arrays.push_back({"tau_sg",
                          {static_cast<unsigned long>(nspin), static_cast<unsigned long>(nxyz)},
                          tau});
        arrays.push_back({"tau_valence_sg",
                          {static_cast<unsigned long>(nspin), static_cast<unsigned long>(nxyz)},
                          tau_valence});
    }
    arrays.push_back({"ekb_kb",
                      {static_cast<unsigned long>(elec.ekb.nr), static_cast<unsigned long>(elec.ekb.nc)},
                      ry_matrix_to_ha_vector(elec.ekb)});
    arrays.push_back({"occ_kb",
                      {static_cast<unsigned long>(elec.wg.nr), static_cast<unsigned long>(elec.wg.nc)},
                      matrix_to_vector(elec.wg)});
    if (elec.klist != nullptr)
    {
        const int nks = elec.klist->get_nks();
        arrays.push_back({"kpt_cart_kv",
                          {static_cast<unsigned long>(nks), 3},
                          kvector_matrix(elec, true)});
        arrays.push_back({"kpt_direct_kv",
                          {static_cast<unsigned long>(nks), 3},
                          kvector_matrix(elec, false)});
        arrays.push_back({"wk_k", {static_cast<unsigned long>(nks)}, kweights(elec)});
        arrays.push_back({"kspin_k", {static_cast<unsigned long>(nks)}, kspins(elec)});
    }
    if (edge_label != nullptr && edge_label->present)
    {
        std::vector<double> edge_indices;
        std::vector<double> edge_roles;
        std::vector<double> edge_eigenvalues_Ha;
        std::vector<double> edge_occupations;
        std::vector<double> edge_kpt_direct;
        std::vector<double> edge_kpt_cart;
        std::vector<double> edge_exx_shifts_Ha;
        edge_indices.reserve(nedge * 3);
        edge_roles.reserve(nedge);
        edge_eigenvalues_Ha.reserve(nedge);
        edge_occupations.reserve(nedge);
        edge_kpt_direct.reserve(nedge * 3);
        edge_kpt_cart.reserve(nedge * 3);
        edge_exx_shifts_Ha.reserve(nedge);
        for (const auto& state: edge_label->states)
        {
            edge_indices.insert(edge_indices.end(),
                                {static_cast<double>(state.spin),
                                 static_cast<double>(state.k),
                                 static_cast<double>(state.band)});
            edge_roles.push_back(state.role == TrainingBandEdgeRole::Vbm ? 0.0 : 1.0);
            edge_eigenvalues_Ha.push_back(state.eigenvalue_Ha);
            edge_occupations.push_back(state.occupation);
            edge_kpt_direct.insert(edge_kpt_direct.end(), state.kpt_direct.begin(), state.kpt_direct.end());
            edge_kpt_cart.insert(edge_kpt_cart.end(), state.kpt_cart.begin(), state.kpt_cart.end());
            edge_exx_shifts_Ha.push_back(state.exx_shift_Ha);
        }
        arrays.push_back({"band_edge_index_tb",
                          {static_cast<unsigned long>(nedge), 3},
                          std::move(edge_indices)});
        arrays.push_back({"band_edge_role_t", {static_cast<unsigned long>(nedge)}, std::move(edge_roles)});
        arrays.push_back({"band_edge_eigenvalue_Ha_t",
                          {static_cast<unsigned long>(nedge)},
                          std::move(edge_eigenvalues_Ha)});
        arrays.push_back({"band_edge_occupation_t",
                          {static_cast<unsigned long>(nedge)},
                          std::move(edge_occupations)});
        arrays.push_back({"band_edge_kpt_direct_tv",
                          {static_cast<unsigned long>(nedge), 3},
                          std::move(edge_kpt_direct)});
        arrays.push_back({"band_edge_kpt_cart_tv",
                          {static_cast<unsigned long>(nedge), 3},
                          std::move(edge_kpt_cart)});
        arrays.push_back({"band_edge_exx_shift_Ha_t",
                          {static_cast<unsigned long>(nedge)},
                          std::move(edge_exx_shifts_Ha)});
        if (!edge_label->orbital_rho_tsg.empty())
        {
            const std::size_t expected_size = nedge * nspin * nrxx;
            if (edge_label->orbital_rho_tsg.size() != expected_size)
            {
                ModuleBase::WARNING_QUIT("write_training_dump",
                                         "band_edge_orbital_rho_tsg has inconsistent size");
            }
            const std::vector<double> gathered =
                gather_spin_major_to_global(rho_basis, edge_label->orbital_rho_tsg, nedge * nspin);
            arrays.push_back({"band_edge_orbital_rho_tsg",
                              {static_cast<unsigned long>(nedge),
                               static_cast<unsigned long>(nspin),
                               static_cast<unsigned long>(nxyz)},
                              gathered});
        }
        const std::vector<double>* orbital_sigma_txg =
            !edge_label->orbital_sigma_txg.empty() ? &edge_label->orbital_sigma_txg : &orbital_sigma_txg_local;
        if (!orbital_sigma_txg->empty())
        {
            const std::size_t expected_size = nedge * nsigma * nrxx;
            if (orbital_sigma_txg->size() != expected_size)
            {
                ModuleBase::WARNING_QUIT("write_training_dump",
                                         "band_edge_orbital_sigma_txg has inconsistent size");
            }
            const std::vector<double> gathered =
                gather_spin_major_to_global(rho_basis, *orbital_sigma_txg, nedge * nsigma);
            arrays.push_back({"band_edge_orbital_sigma_txg",
                              {static_cast<unsigned long>(nedge),
                               static_cast<unsigned long>(nsigma),
                               static_cast<unsigned long>(nxyz)},
                              gathered});
        }
        if (!edge_label->orbital_tau_tsg.empty())
        {
            const std::size_t expected_size = nedge * nspin * nrxx;
            if (edge_label->orbital_tau_tsg.size() != expected_size)
            {
                ModuleBase::WARNING_QUIT("write_training_dump",
                                         "band_edge_orbital_tau_tsg has inconsistent size");
            }
            const std::vector<double> gathered =
                gather_spin_major_to_global(rho_basis, edge_label->orbital_tau_tsg, nedge * nspin);
            arrays.push_back({"band_edge_orbital_tau_tsg",
                              {static_cast<unsigned long>(nedge),
                               static_cast<unsigned long>(nspin),
                               static_cast<unsigned long>(nxyz)},
                              gathered});
        }
    }
    if (sdmx_label != nullptr && sdmx_label->present)
    {
        const std::size_t coeff_size = static_cast<std::size_t>(sdmx_label->nspin)
                                       * static_cast<std::size_t>(sdmx_label->nkpt)
                                       * static_cast<std::size_t>(sdmx_label->nband)
                                       * static_cast<std::size_t>(sdmx_label->npwk_max);
        const std::size_t kg_size = static_cast<std::size_t>(sdmx_label->nkpt)
                                    * static_cast<std::size_t>(sdmx_label->npwk_max);
        if (sdmx_label->nspin != nspin || sdmx_label->nkpt != elec.ekb.nr
            || sdmx_label->nband != elec.ekb.nc || sdmx_label->npwk_max <= 0)
        {
            ModuleBase::WARNING_QUIT("write_training_dump", "SDMX wavefunction metadata is inconsistent");
        }
        if (sdmx_label->coeffs_real_skbg.size() != coeff_size
            || sdmx_label->coeffs_imag_skbg.size() != coeff_size)
        {
            ModuleBase::WARNING_QUIT("write_training_dump", "SDMX wavefunction coefficient size is inconsistent");
        }
        if (sdmx_label->npw_k.size() != static_cast<std::size_t>(sdmx_label->nkpt)
            || sdmx_label->g_index_kg.size() != kg_size)
        {
            ModuleBase::WARNING_QUIT("write_training_dump", "SDMX wavefunction G-index metadata is inconsistent");
        }
        arrays.push_back({"sdmx_coeffs_real_skbg",
                          {static_cast<unsigned long>(sdmx_label->nspin),
                           static_cast<unsigned long>(sdmx_label->nkpt),
                           static_cast<unsigned long>(sdmx_label->nband),
                           static_cast<unsigned long>(sdmx_label->npwk_max)},
                          sdmx_label->coeffs_real_skbg});
        arrays.push_back({"sdmx_coeffs_imag_skbg",
                          {static_cast<unsigned long>(sdmx_label->nspin),
                           static_cast<unsigned long>(sdmx_label->nkpt),
                           static_cast<unsigned long>(sdmx_label->nband),
                           static_cast<unsigned long>(sdmx_label->npwk_max)},
                          sdmx_label->coeffs_imag_skbg});
        arrays.push_back({"sdmx_npw_k",
                          {static_cast<unsigned long>(sdmx_label->nkpt)},
                          sdmx_label->npw_k});
        arrays.push_back({"sdmx_g_index_kg",
                          {static_cast<unsigned long>(sdmx_label->nkpt),
                           static_cast<unsigned long>(sdmx_label->npwk_max)},
                          sdmx_label->g_index_kg});
    }

    const bool exx_label_present = PARAM.inp.out_exx_label;
#if defined(ENABLE_CIDER)
    const auto& cider_components = elecstate::PotCiderXC::last_energy_components();
    const bool cider_label_present = PARAM.inp.out_cider_label && cider_components.valid;
#else
    const bool cider_label_present = false;
#endif
    std::ostringstream js;
    js << std::setprecision(16);
    js << "{\n";
    js << "  \"schema_name\": \"abacus_pw_nldx_training_dump\",\n";
    js << "  \"schema_version\": 2,\n";
    js << "  \"record_kind\": \"pbe_base_state\",\n";
    std::string label_mode = "pbe_base_state_density_only";
    if (exx_label_present && cider_label_present)
    {
        label_mode = PARAM.inp.out_bandgap_label ? "pbe_base_state_one_shot_full_exx_band_edge_and_cider_label"
                                                 : "pbe_base_state_one_shot_full_exx_and_cider_label";
    }
    else if (exx_label_present)
    {
        label_mode = PARAM.inp.out_bandgap_label ? "pbe_base_state_one_shot_full_exx_band_edge_label"
                                                 : "pbe_base_state_one_shot_full_exx_label";
    }
    else if (edge_label != nullptr && edge_label->present && !cider_label_present)
    {
        label_mode = "pbe_base_state_frontier_exx_band_edge_label";
    }
    else if (cider_label_present)
    {
        label_mode = "pbe_base_state_post_scf_cider_label";
    }
    js << "  \"label_mode\": " << json_string(label_mode) << ",\n";
    js << "  \"energy_unit\": \"Ha\",\n";
    js << "  \"length_unit\": \"Bohr\",\n";
    js << "  \"density_unit\": \"Bohr^-3\",\n";
    js << "  \"sigma_unit\": \"Bohr^-8\",\n";
    js << "  \"tau_unit\": \"Bohr^-5\",\n";
    js << "  \"kpt_cart_unit\": \"Bohr^-1\",\n";
    js << "  \"basis_type\": " << json_string(PARAM.inp.basis_type) << ",\n";
    js << "  \"pseudo_type_required\": \"ncp\",\n";
    js << "  \"parallel_policy\": \"kpar_1_bndpar_1_required\",\n";
    js << "  \"source_state\": \"self_consistent_pbe_base\",\n";
    js << "  \"exx_energy_evaluated\": " << (exx_label_present ? "true" : "false") << ",\n";
    js << "  \"exx_energy_source\": "
       << (exx_label_present ? json_string("one_shot_full_range_exx_on_pbe_state") : "null") << ",\n";
    js << "  \"nspin\": " << nspin << ",\n";
    js << "  \"nsigma\": " << nsigma << ",\n";
    js << "  \"nbands\": " << elec.ekb.nc << ",\n";
    js << "  \"nelec\": " << PARAM.inp.nelec << ",\n";
    js << "  \"grid_shape\": [" << rho_basis.nx << ", " << rho_basis.ny << ", " << rho_basis.nz << "],\n";
    js << "  \"nxyz\": " << nxyz << ",\n";
    js << "  \"dV_bohr3\": " << dvol << ",\n";
    js << "  \"volume_bohr3\": " << ucell.omega << ",\n";
    js << "  \"cell_cv_bohr\": [";
    js << ucell.lat0 * ucell.latvec.e11 << ", " << ucell.lat0 * ucell.latvec.e12 << ", "
       << ucell.lat0 * ucell.latvec.e13 << ", " << ucell.lat0 * ucell.latvec.e21 << ", "
       << ucell.lat0 * ucell.latvec.e22 << ", " << ucell.lat0 * ucell.latvec.e23 << ", "
       << ucell.lat0 * ucell.latvec.e31 << ", " << ucell.lat0 * ucell.latvec.e32 << ", "
       << ucell.lat0 * ucell.latvec.e33 << "],\n";
    js << "  \"ecutwfc_Ha\": " << ry_to_ha(PARAM.inp.ecutwfc) << ",\n";
    js << "  \"ecutrho_Ha\": " << ry_to_ha(PARAM.inp.ecutrho) << ",\n";
    js << "  \"dft_functional\": " << json_string(PARAM.inp.dft_functional) << ",\n";
    js << "  \"scf_thr\": " << PARAM.inp.scf_thr << ",\n";
    js << "  \"scf_nmax\": " << PARAM.inp.scf_nmax << ",\n";
    js << "  \"scf_iter\": " << elec.iter << ",\n";
    js << "  \"converged\": " << (scf_converged ? "true" : "false") << ",\n";
    js << "  \"ionic_step\": " << istep << ",\n";
    js << "  \"cider_model\": " << json_string(PARAM.inp.cider_model) << ",\n";
    js << "  \"cider_xmix\": " << PARAM.inp.cider_xmix << ",\n";
    js << "  \"cider_tf_tau\": " << (PARAM.inp.cider_tf_tau ? "true" : "false") << ",\n";
    js << "  \"cider_feature_density_input\": " << json_string(PARAM.inp.cider_feature_density) << ",\n";
    js << "  \"feature_density_policy_default\": \"valence\",\n";
    js << "  \"feature_density_policy_available\": [\"valence\", \"valence_pseudo_core\"],\n";
    js << "  \"feature_density_roles\": {\n";
    js << "    \"valence\": {\"rho\": \"rho_valence_sg\", \"sigma\": \"sigma_valence_xg\", \"tau\": \"tau_valence_sg\"},\n";
    js << "    \"valence_pseudo_core\": {\"rho\": \"rho_sg\", \"sigma\": \"sigma_xg\", \"tau\": \"tau_sg\"}\n";
    js << "  },\n";
    js << "  \"tau_present\": " << (!tau.empty() ? "true" : "false") << ",\n";
    js << "  \"efermi_Ha\": " << ry_to_ha(elec.eferm.ef) << ",\n";
    js << "  \"vbm_Ha\": " << ry_to_ha(vbm) << ",\n";
    js << "  \"cbm_Ha\": " << ry_to_ha(cbm) << ",\n";
    js << "  \"bandgap_Ha\": " << ry_to_ha(cbm - vbm) << ",\n";
    js << "  \"energy_Ha\": {\n";
    js << "    \"etot\": " << ry_to_ha(elec.f_en.etot) << ",\n";
    js << "    \"eband\": " << ry_to_ha(elec.f_en.eband) << ",\n";
    js << "    \"deband\": " << ry_to_ha(elec.f_en.deband) << ",\n";
    js << "    \"etxc\": " << ry_to_ha(elec.f_en.etxc) << ",\n";
    js << "    \"vtxc\": " << ry_to_ha(elec.f_en.vtxc) << ",\n";
    js << "    \"hartree\": " << ry_to_ha(elec.f_en.hartree_energy) << ",\n";
    js << "    \"ewald\": " << ry_to_ha(elec.f_en.ewald_energy) << ",\n";
    js << "    \"local_pp\": " << ry_to_ha(elec.f_en.e_local_pp) << "\n";
    js << "  },\n";
    js << "  \"exx_label\": {\n";
    if (exx_label_present)
    {
        js << "    \"present\": true,\n";
        js << "    \"definition\": \"one-shot full-range exact exchange energy evaluated on the converged PBE density and plane-wave wavefunctions\",\n";
        js << "    \"energy_evaluated\": true,\n";
        js << "    \"energy_source\": \"one_shot_full_range_exx_on_pbe_state\",\n";
        js << "    \"energy_Ha\": " << ry_to_ha(elec.f_en.exx) << ",\n";
        js << "    \"raw_exx_energy_available\": true,\n";
        js << "    \"hybrid_scaled\": false,\n";
        js << "    \"hybrid_alpha\": 1,\n";
        js << "    \"range\": \"full\",\n";
        js << "    \"screening\": \"none\",\n";
        js << "    \"base_state\": \"pbe_scf_density_and_pw_wavefunctions\",\n";
        if (exx_density_label != nullptr)
        {
            js << "    \"self_consistent_hybrid_required\": false,\n";
            js << "    \"energy_density\": {\n";
            js << "      \"present\": true,\n";
            js << "      \"array\": \"exx_energy_density_g\",\n";
            js << "      \"unit\": \"Ha Bohr^-3\",\n";
            js << "      \"gauge\": \"conventional_fock_exchange_hole\",\n";
            js << "      \"spin\": \"summed\",\n";
            js << "      \"grid\": \"charge_density_fft_real_space\",\n";
            js << "      \"integrated_energy_Ha\": " << exx_density_integrated_energy_Ha << ",\n";
            js << "      \"scalar_energy_Ha\": " << exx_density_label->scalar_energy_Ha << ",\n";
            js << "      \"integration_abs_residual_Ha\": " << exx_density_abs_residual_Ha << ",\n";
            js << "      \"integration_rel_residual\": " << exx_density_rel_residual << ",\n";
            js << "      \"max_imaginary_abs_Ha_bohr3\": "
               << exx_density_label->max_imaginary_abs_Ha_bohr3 << ",\n";
            js << "      \"full_range\": true,\n";
            js << "      \"hybrid_scaled\": false,\n";
            js << "      \"base_state\": \"pbe_scf_density_and_pw_wavefunctions\",\n";
            js << "      \"k_policy\": " << json_string(exx_density_label->k_policy) << "\n";
            js << "    }\n";
        }
        else
        {
            js << "    \"self_consistent_hybrid_required\": false\n";
        }
    }
    else
    {
        js << "    \"present\": false,\n";
        js << "    \"energy_evaluated\": false\n";
    }
    js << "  },\n";
    js << "  \"band_edge_label\": {\n";
    if (edge_label != nullptr && edge_label->present)
    {
        js << "    \"present\": true,\n";
        js << "    \"requested\": true,\n";
        js << "    \"definition\": \"frozen-orbital VBM/CBM full-range EXX band-edge diagonal on the converged PBE state\",\n";
        js << "    \"base_state\": \"pbe_scf_density_and_pw_wavefunctions\",\n";
        js << "    \"target_kind\": \"band_gap_exx_shift\",\n";
        js << "    \"unit\": \"Ha\",\n";
        js << "    \"hybrid_scaled\": false,\n";
        js << "    \"hybrid_alpha\": 1,\n";
        js << "    \"index_base\": 0,\n";
        js << "    \"role_code\": {\"vbm\": 0, \"cbm\": 1},\n";
        js << "    \"direct_gap\": " << (direct_gap ? "true" : "false") << ",\n";
        js << "    \"edge_selection_policy\": \"degenerate_frontier_average\",\n";
        js << "    \"degenerate_tolerance_Ha\": " << PARAM.inp.band_edge_degeneracy_tolerance << ",\n";
        js << "    \"occupation_threshold\": 1e-08,\n";
        js << "    \"nedge\": " << nedge << ",\n";
        js << "    \"vbm_count\": " << vbm_count << ",\n";
        js << "    \"cbm_count\": " << cbm_count << ",\n";
        js << "    \"gap_ref_shift_Ha\": "
           << (cbm_shift_sum_Ha / cbm_count - vbm_shift_sum_Ha / vbm_count) << ",\n";
        js << "    \"response_present\": "
           << (!edge_label->orbital_rho_tsg.empty() || !edge_label->orbital_sigma_txg.empty()
                       || !edge_label->orbital_tau_tsg.empty() || !orbital_sigma_txg_local.empty()
                   ? "true"
                   : "false")
           << ",\n";
        js << "    \"response_convention\": \"d_density_ingredients_d_abacus_wg\",\n";
        js << "    \"status\": \"evaluated\"\n";
    }
    else if (PARAM.inp.out_bandgap_label)
    {
        js << "    \"present\": false,\n";
        js << "    \"requested\": true,\n";
        js << "    \"definition\": \"frozen-orbital VBM/CBM full-range EXX band-edge diagonal on the converged PBE state\",\n";
        js << "    \"base_state\": \"pbe_scf_density_and_pw_wavefunctions\",\n";
        js << "    \"target_kind\": \"band_gap_exx_shift\",\n";
        js << "    \"unit\": \"Ha\",\n";
        js << "    \"hybrid_scaled\": false,\n";
        js << "    \"hybrid_alpha\": 1,\n";
        js << "    \"index_base\": 0,\n";
        js << "    \"role_code\": {\"vbm\": 0, \"cbm\": 1},\n";
        js << "    \"edge_selection_policy\": \"degenerate_frontier_average\",\n";
        js << "    \"degenerate_tolerance_Ha\": " << PARAM.inp.band_edge_degeneracy_tolerance << ",\n";
        js << "    \"occupation_threshold\": 1e-08,\n";
        js << "    \"status\": \"not_evaluated\"\n";
    }
    else
    {
        js << "    \"present\": false,\n";
        js << "    \"requested\": false\n";
    }
    js << "  },\n";
    js << "  \"cider_label\": {\n";
    if (cider_label_present)
    {
#if defined(ENABLE_CIDER)
        js << "    \"present\": true,\n";
        js << "    \"definition\": \"post-SCF CIDER exchange evaluated on the converged PBE density ingredients\",\n";
        js << "    \"energy_evaluated\": true,\n";
        js << "    \"energy_source\": \"post_scf_cider_exchange_on_pbe_state\",\n";
        js << "    \"model\": " << json_string(PARAM.inp.cider_model) << ",\n";
        js << "    \"xmix\": " << PARAM.inp.cider_xmix << ",\n";
        js << "    \"feature_density\": " << json_string(cider_components.feature_density_policy) << ",\n";
        js << "    \"exc_convention\": " << json_string(cider_components.feature_exc_convention) << ",\n";
        js << "    \"feature_scaled_energy_Ha\": " << cider_components.feature_scaled_Ha << ",\n";
        js << "    \"feature_unmixed_energy_Ha\": " << cider_components.feature_unmixed_Ha << ",\n";
        js << "    \"baseline_energy_Ha\": " << cider_components.baseline_Ha << ",\n";
        js << "    \"total_energy_Ha\": " << cider_components.total_Ha << ",\n";
        js << "    \"base_state\": \"pbe_scf_density_ingredients\",\n";
        js << "    \"self_consistent_cider_required\": false\n";
#endif
    }
    else
    {
        js << "    \"present\": false,\n";
        js << "    \"energy_evaluated\": false\n";
    }
    js << "  },\n";
    js << "  \"sdmx_wavefunction\": {\n";
    if (sdmx_label != nullptr && sdmx_label->present)
    {
        js << "    \"present\": true,\n";
        js << "    \"schema_name\": \"ciderpress_abacus_sdmx_wavefunction\",\n";
        js << "    \"schema_version\": 1,\n";
        js << "    \"status\": \"present\",\n";
        js << "    \"mesh\": [" << rho_basis.nx << ", " << rho_basis.ny << ", " << rho_basis.nz << "],\n";
        js << "    \"cell_cv\": [[";
        js << ucell.lat0 * ucell.latvec.e11 << ", " << ucell.lat0 * ucell.latvec.e12 << ", "
           << ucell.lat0 * ucell.latvec.e13 << "], ["
           << ucell.lat0 * ucell.latvec.e21 << ", " << ucell.lat0 * ucell.latvec.e22 << ", "
           << ucell.lat0 * ucell.latvec.e23 << "], ["
           << ucell.lat0 * ucell.latvec.e31 << ", " << ucell.lat0 * ucell.latvec.e32 << ", "
           << ucell.lat0 * ucell.latvec.e33 << "]],\n";
        js << "    \"nspin\": " << sdmx_label->nspin << ",\n";
        js << "    \"nkpt\": " << sdmx_label->nkpt << ",\n";
        js << "    \"nband\": " << sdmx_label->nband << ",\n";
        js << "    \"ngrid\": " << nxyz << ",\n";
        js << "    \"npwk_max\": " << sdmx_label->npwk_max << ",\n";
        js << "    \"coeff_layout\": \"s_k_b_g_abacus_g_index_order\",\n";
        js << "    \"coefficient_normalization\": \"sum_abs_c2_eq_1\",\n";
        js << "    \"fft_normalization\": \"numpy_ifft_times_ngrid_over_sqrt_omega\",\n";
        js << "    \"occupation_includes_k_weight\": true,\n";
        js << "    \"k_weights_normalization\": \"abacus_native\",\n";
        js << "    \"g_vector_convention\": \"abacus_igl2ig_k_index\",\n";
        js << "    \"coefficient_storage\": \"split_real_imag_float64_padded_to_npwk_max\",\n";
        js << "    \"band_selection\": \"all_bands\",\n";
        js << "    \"kpar\": " << PARAM.inp.kpar << ",\n";
        js << "    \"bndpar\": " << PARAM.inp.bndpar << ",\n";
        js << "    \"arrays\": {\n";
        js << "      \"coeffs_real_skbg\": \"sdmx_coeffs_real_skbg\",\n";
        js << "      \"coeffs_imag_skbg\": \"sdmx_coeffs_imag_skbg\",\n";
        js << "      \"npw_k\": \"sdmx_npw_k\",\n";
        js << "      \"g_index_kg\": \"sdmx_g_index_kg\",\n";
        js << "      \"k_weights\": \"wk_k\",\n";
        js << "      \"occupations\": \"occ_kb\",\n";
        js << "      \"kpts_cart\": \"kpt_cart_kv\"\n";
        js << "    }\n";
    }
    else
    {
        js << "    \"present\": false,\n";
        js << "    \"schema_name\": \"ciderpress_abacus_sdmx_wavefunction\",\n";
        js << "    \"schema_version\": 1,\n";
        js << "    \"status\": \"not_requested\",\n";
        js << "    \"reason\": \"out_sdmx_descriptor_data is false\",\n";
        js << "    \"required_coeff_layout\": \"s_k_b_g_abacus_g_index_order\",\n";
        js << "    \"required_coefficient_normalization\": \"sum_abs_c2_eq_1\",\n";
        js << "    \"required_fft_normalization\": \"numpy_ifft_times_ngrid_over_sqrt_omega\",\n";
        js << "    \"required_occupation_includes_k_weight\": true,\n";
        js << "    \"required_k_weights_normalization\": \"abacus_native\",\n";
        js << "    \"required_g_vector_convention\": \"abacus_igl2ig_k_index\"\n";
    }
    js << "  },\n";
    js << "  \"arrays\": {\n";
    js << "    \"rho_valence_sg\": \"rho_valence_sg\",\n";
    js << "    \"rho_sg\": \"rho_sg\",\n";
    js << "    \"rho_core_g\": \"rho_core_g\",\n";
    js << "    \"sigma_xg\": \"sigma_xg\",\n";
    js << "    \"sigma_valence_xg\": \"sigma_valence_xg\",\n";
    if (!tau.empty())
    {
        js << "    \"tau_sg\": \"tau_sg\",\n";
        js << "    \"tau_valence_sg\": \"tau_valence_sg\",\n";
    }
    js << "    \"ekb_kb\": \"ekb_kb\",\n";
    js << "    \"occ_kb\": \"occ_kb\",\n";
    js << "    \"kpt_cart_kv\": \"kpt_cart_kv\",\n";
    js << "    \"kpt_direct_kv\": \"kpt_direct_kv\",\n";
    js << "    \"wk_k\": \"wk_k\",\n";
    js << "    \"kspin_k\": \"kspin_k\"";
    if (exx_density_label != nullptr)
    {
        js << ",\n";
        js << "    \"exx_energy_density_g\": \"exx_energy_density_g\"";
    }
    if (sdmx_label != nullptr && sdmx_label->present)
    {
        js << ",\n";
        js << "    \"sdmx_coeffs_real_skbg\": \"sdmx_coeffs_real_skbg\",\n";
        js << "    \"sdmx_coeffs_imag_skbg\": \"sdmx_coeffs_imag_skbg\",\n";
        js << "    \"sdmx_npw_k\": \"sdmx_npw_k\",\n";
        js << "    \"sdmx_g_index_kg\": \"sdmx_g_index_kg\"";
    }
    if (edge_label != nullptr && edge_label->present)
    {
        js << ",\n";
        js << "    \"band_edge_index_tb\": \"band_edge_index_tb\",\n";
        js << "    \"band_edge_role_t\": \"band_edge_role_t\",\n";
        js << "    \"band_edge_eigenvalue_Ha_t\": \"band_edge_eigenvalue_Ha_t\",\n";
        js << "    \"band_edge_occupation_t\": \"band_edge_occupation_t\",\n";
        js << "    \"band_edge_kpt_direct_tv\": \"band_edge_kpt_direct_tv\",\n";
        js << "    \"band_edge_kpt_cart_tv\": \"band_edge_kpt_cart_tv\",\n";
        js << "    \"band_edge_exx_shift_Ha_t\": \"band_edge_exx_shift_Ha_t\"";
        if (!edge_label->orbital_rho_tsg.empty())
        {
            js << ",\n";
            js << "    \"band_edge_orbital_rho_tsg\": \"band_edge_orbital_rho_tsg\"";
        }
        if (!edge_label->orbital_sigma_txg.empty() || !orbital_sigma_txg_local.empty())
        {
            js << ",\n";
            js << "    \"band_edge_orbital_sigma_txg\": \"band_edge_orbital_sigma_txg\"";
        }
        if (!edge_label->orbital_tau_tsg.empty())
        {
            js << ",\n";
            js << "    \"band_edge_orbital_tau_tsg\": \"band_edge_orbital_tau_tsg\"";
        }
        js << "\n";
    }
    else
    {
        js << "\n";
    }
    js << "  },\n";
    js << "  \"pseudopotentials\": [\n";
    for (int it = 0; it < ucell.ntype; ++it)
    {
        js << "    {\"label\": " << json_string(ucell.atoms[it].label)
           << ", \"file\": " << json_string(it < static_cast<int>(ucell.pseudo_fn.size()) ? ucell.pseudo_fn[it] : "")
           << ", \"type\": " << json_string(it < static_cast<int>(ucell.pseudo_type.size()) ? ucell.pseudo_type[it] : "")
           << ", \"nlcc\": " << (ucell.atoms[it].ncpp.nlcc ? "true" : "false") << "}";
        js << (it + 1 == ucell.ntype ? "\n" : ",\n");
    }
    js << "  ]\n";
    js << "}\n";

#ifdef __HDF5
    write_hdf5_training_dump(dump_path, js.str(), arrays);
#else
    ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data requires an ABACUS build with ENABLE_HDF5=ON");
#endif

    if (exx_density_label != nullptr)
    {
        GlobalV::ofs_running << "write_training_dump: EXX energy density integral = "
                             << exx_density_integrated_energy_Ha << " Ha, scalar = "
                             << exx_density_label->scalar_energy_Ha << " Ha, abs residual = "
                             << exx_density_abs_residual_Ha << " Ha, max imaginary = "
                             << exx_density_label->max_imaginary_abs_Ha_bohr3 << " Ha/Bohr^3, k policy = "
                             << exx_density_label->k_policy << std::endl;
    }
    GlobalV::ofs_running << "write_training_dump: wrote " << dump_path << std::endl;
    ModuleBase::timer::end("ModuleIO", "write_training_dump");
}

} // namespace ModuleIO

#endif
