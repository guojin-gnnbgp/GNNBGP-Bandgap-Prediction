#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#define private public
#define protected public
#include "source_estate/elecstate_pw.h"
#ifdef __LCAO
#include "source_basis/module_ao/ORB_gaunt_table.h"
#endif
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_pw/module_pwdft/vl_pw.h"
#include "source_pw/module_pwdft/vnl_pw.h"
#include "source_pw/module_pwdft/soc.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/module_training/training_band_edge.h"
// mock functions for testing
int XC_Functional::func_type = 1;
bool XC_Functional::ked_flag = false;
bool XC_Functional::forced_ked_flag = false;
namespace elecstate
{
void Potential::init_pot(Charge const*)
{
}
void Potential::cal_v_eff(const Charge* chg, const UnitCell* ucell, ModuleBase::matrix& v_eff)
{
}
void Potential::cal_fixed_v(double* vl_pseudo)
{
}
Potential::~Potential()
{
}
} // namespace elecstate
Charge::Charge()
{
}
Charge::~Charge()
{
}
UnitCell::UnitCell()
{
}
UnitCell::~UnitCell()
{
}
Magnetism::Magnetism()
{
}
Magnetism::~Magnetism()
{
}
SepPot::SepPot(){}
SepPot::~SepPot(){}
Sep_Cell::Sep_Cell() noexcept {}
Sep_Cell::~Sep_Cell() noexcept {}
#ifdef __LCAO
InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}
ORB_gaunt_table::ORB_gaunt_table()
{
}
ORB_gaunt_table::~ORB_gaunt_table()
{
}
#endif
pseudopot_cell_vl::pseudopot_cell_vl()
{
}
pseudopot_cell_vl::~pseudopot_cell_vl()
{
}
pseudopot_cell_vnl::pseudopot_cell_vnl()
{
}
pseudopot_cell_vnl::~pseudopot_cell_vnl()
{
}
template <>
void pseudopot_cell_vnl::radial_fft_q<float, base_device::DEVICE_CPU>(base_device::DEVICE_CPU* ctx,
                                                                      const int ng,
                                                                      const int ih,
                                                                      const int jh,
                                                                      const int itype,
                                                                      const float* qnorm,
                                                                      const float* ylm,
                                                                      std::complex<float>* qg) const
{
}
template <>
void pseudopot_cell_vnl::radial_fft_q<double, base_device::DEVICE_CPU>(base_device::DEVICE_CPU* ctx,
                                                                       const int ng,
                                                                       const int ih,
                                                                       const int jh,
                                                                       const int itype,
                                                                       const double* qnorm,
                                                                       const double* ylm,
                                                                       std::complex<double>* qg) const
{
}
template <>
std::complex<float>* pseudopot_cell_vnl::get_vkb_data<float>() const
{
    return nullptr;
}
template <>
std::complex<double>* pseudopot_cell_vnl::get_vkb_data<double>() const
{
    return nullptr;
}
template <>
void pseudopot_cell_vnl::getvnl<float, base_device::DEVICE_CPU>(base_device::DEVICE_CPU*,
                                                                const UnitCell&,
                                                                int const&,
                                                                std::complex<float>*) const
{
}
template <>
void pseudopot_cell_vnl::getvnl<double, base_device::DEVICE_CPU>(base_device::DEVICE_CPU*,
                                                                 const UnitCell&,
                                                                 int const&,
                                                                 std::complex<double>*) const
{
}
Soc::~Soc()
{
}
Fcoef::~Fcoef()
{
}
#include "source_cell/klist.h"

void Charge::set_rho_core(const UnitCell& ucell, ModuleBase::ComplexMatrix const&, const bool*)
{
}
void Charge::init_rho(const UnitCell&,
                      const Parallel_Grid&,
                      ModuleBase::ComplexMatrix const&,
                      ModuleSymmetry::Symmetry& symm,
                      const void*,
                      const void*)
{
}
void Charge::set_rhopw(ModulePW::PW_Basis*)
{
}
void Charge::renormalize_rho()
{
}
void Charge::check_rho()
{
}

void Set_GlobalV_Default()
{
    PARAM.input.device = "cpu";
    PARAM.input.precision = "double";
    PARAM.sys.domag = false;
    PARAM.sys.domag_z = false;
    // Base class dependent
    PARAM.input.nspin = 1;
    PARAM.input.nelec = 10.0;
    PARAM.input.nupdown  = 0.0;
    PARAM.sys.two_fermi = false;
    PARAM.input.nbands = 6;
    PARAM.sys.nlocal = 6;
    PARAM.input.esolver_type = "ksdft";
    PARAM.input.lspinorb = false;
    PARAM.input.basis_type = "pw";
    GlobalV::KPAR = 1;
    GlobalV::NPROC_IN_POOL = 1;
    PARAM.sys.use_uspp = false;
}

/************************************************
 *  unit test of elecstate_pw.cpp
 ***********************************************/

/**
 * - Tested Functions:
 *   - Constructor: elecstate::ElecStatePW constructor and destructor
 *      - including double and single precision versions
 *   - InitRhoData: elecstate::ElecStatePW::init_rho_data()
 *      - get rho and kin_r for ElecStatePW
 *   - ParallelK: elecstate::ElecStatePW::parallelK()
 *      - trivial call due to removing of __MPI
 *   - todo: psiToRho: elecstate::ElecStatePW::psiToRho()
 */

class ElecStatePWTest : public ::testing::Test
{
  protected:
    elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_CPU>* elecstate_pw_d = nullptr;
    elecstate::ElecStatePW<std::complex<float>, base_device::DEVICE_CPU>* elecstate_pw_s = nullptr;
    ModulePW::PW_Basis_K* wfcpw = nullptr;
    Charge* chg = nullptr;
    K_Vectors* klist = nullptr;
    UnitCell* ucell = nullptr;
    pseudopot_cell_vnl* ppcell = nullptr;
    ModulePW::PW_Basis* rhodpw = nullptr;
    ModulePW::PW_Basis* rhopw = nullptr;
    ModulePW::PW_Basis_Big* bigpw = nullptr;
    void SetUp() override
    {
        Set_GlobalV_Default();
        wfcpw = new ModulePW::PW_Basis_K;
        chg = new Charge;
        klist = new K_Vectors;
        klist->set_nks(5);
        ucell = new UnitCell;
        ucell->omega = 500.0;
        ucell->tpiba = 2.0;
        ppcell = new pseudopot_cell_vnl;
        rhodpw = new ModulePW::PW_Basis;
        rhopw = new ModulePW::PW_Basis;
        bigpw = new ModulePW::PW_Basis_Big;
    }

    void TearDown() override
    {
        delete wfcpw;
        delete chg;
        delete klist;
        delete ucell;
        delete ppcell;
        delete rhodpw;
        delete rhopw;
        if (elecstate_pw_d != nullptr)
        {
            delete elecstate_pw_d;
        }
        if (elecstate_pw_s != nullptr)
        {
            delete elecstate_pw_s;
        }
    }
};

TEST_F(ElecStatePWTest, ConstructorDouble)
{
    elecstate_pw_d = new elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_CPU>(wfcpw,
                                                                                               chg,
                                                                                               klist,
                                                                                               ucell,
                                                                                               ppcell,
                                                                                               rhopw,
                                                                                               bigpw);
    EXPECT_EQ(elecstate_pw_d->classname, "ElecStatePW");
    EXPECT_EQ(elecstate_pw_d->charge, chg);
    EXPECT_EQ(elecstate_pw_d->klist, klist);
    EXPECT_EQ(elecstate_pw_d->bigpw, bigpw);
}

TEST_F(ElecStatePWTest, ConstructorSingle)
{
    elecstate_pw_s = new elecstate::ElecStatePW<std::complex<float>, base_device::DEVICE_CPU>(wfcpw,
                                                                                              chg,
                                                                                              klist,
                                                                                              ucell,
                                                                                              ppcell,
                                                                                              rhopw,
                                                                                              bigpw);
    EXPECT_EQ(elecstate_pw_s->classname, "ElecStatePW");
    EXPECT_EQ(elecstate_pw_s->charge, chg);
    EXPECT_EQ(elecstate_pw_s->klist, klist);
    EXPECT_EQ(elecstate_pw_s->bigpw, bigpw);
}

TEST_F(ElecStatePWTest, InitRhoDataDouble)
{
    XC_Functional::func_type = 3;
    chg->nrxx = 1000;
    elecstate_pw_d = new elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_CPU>(wfcpw,
                                                                                               chg,
                                                                                               klist,
                                                                                               ucell,
                                                                                               ppcell,
                                                                                               rhopw,
                                                                                               bigpw);
    elecstate_pw_d->init_rho_data();
    EXPECT_EQ(elecstate_pw_d->init_rho, true);
    EXPECT_EQ(elecstate_pw_d->rho, chg->rho);
    EXPECT_EQ(elecstate_pw_d->kin_r, chg->kin_r);
}

TEST_F(ElecStatePWTest, InitRhoDataSingle)
{
    PARAM.input.precision = "single";
    XC_Functional::func_type = 3;
    chg->nspin = PARAM.input.nspin;
    chg->nrxx = 1000;
    elecstate_pw_s = new elecstate::ElecStatePW<std::complex<float>, base_device::DEVICE_CPU>(wfcpw,
                                                                                              chg,
                                                                                              klist,
                                                                                              ucell,
                                                                                              ppcell,
                                                                                              rhopw,
                                                                                              bigpw);
    elecstate_pw_s->init_rho_data();
    EXPECT_EQ(elecstate_pw_s->init_rho, true);
    EXPECT_NE(elecstate_pw_s->rho, nullptr);
    EXPECT_NE(elecstate_pw_s->kin_r, nullptr);
}

TEST_F(ElecStatePWTest, ParallelKDouble)
{
    //this is a trivial call due to removing of __MPI
    elecstate_pw_d = new elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_CPU>(wfcpw,
                                                                                               chg,
                                                                                               klist,
                                                                                               ucell,
                                                                                               ppcell,
                                                                                               rhopw,
                                                                                               bigpw);
    EXPECT_NO_THROW(elecstate_pw_d->parallelK());
}

TEST_F(ElecStatePWTest, ParallelKSingle)
{
    //this is a trivial call due to removing of __MPI
    elecstate_pw_s = new elecstate::ElecStatePW<std::complex<float>, base_device::DEVICE_CPU>(wfcpw,
                                                                                              chg,
                                                                                              klist,
                                                                                              ucell,
                                                                                              ppcell,
                                                                                              rhopw,
                                                                                              bigpw);
    EXPECT_NO_THROW(elecstate_pw_s->parallelK());
}

TEST_F(ElecStatePWTest, BandEdgeOrbitalResponsesFollowDynamicStateSpinChannels)
{
    PARAM.input.nspin = 2;
    chg->nspin = 2;
    klist->set_nks(2);
    klist->isk = {0, 1};

    const ModuleBase::Matrix3 unit_lattice(1.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0,
                                           0.0, 0.0, 1.0);
    ModuleBase::Vector3<double> kvec_d[2] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    wfcpw->initgrids(4.0, unit_lattice, 4, 4, 4);
    wfcpw->initparameters(false, 10.0, 2, kvec_d, 1, false);
    wfcpw->setuptransform();
    wfcpw->collect_local_pw();
    chg->nrxx = wfcpw->nrxx;

    elecstate_pw_d = new elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_CPU>(wfcpw,
                                                                                               chg,
                                                                                               klist,
                                                                                               ucell,
                                                                                               ppcell,
                                                                                               rhopw,
                                                                                               bigpw);
    elecstate_pw_d->init_rho_data();

    constexpr int nbands = 3;
    std::vector<int> ngk = {wfcpw->npwk[0], wfcpw->npwk[1]};
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU> psi(2,
                                                                nbands,
                                                                wfcpw->npwk_max,
                                                                ngk,
                                                                true);
    psi.zero_out();
    for (int ik = 0; ik < 2; ++ik)
    {
        for (int ib = 0; ib < nbands; ++ib)
        {
            psi(ik, ib, 0) = std::complex<double>(1.0 + ib, 0.0);
        }
    }

    std::vector<ModuleIO::TrainingBandEdgeState> states(5);
    states[0].k = 0;
    states[0].band = 0;
    states[0].spin = 0;
    states[1].k = 1;
    states[1].band = 0;
    states[1].spin = 1;
    states[2].k = 0;
    states[2].band = 1;
    states[2].spin = 0;
    states[3].k = 1;
    states[3].band = 1;
    states[3].spin = 1;
    states[4].k = 0;
    states[4].band = 2;
    states[4].spin = 0;

    std::vector<double> rho_tsg;
    std::vector<double> tau_tsg;
    elecstate_pw_d->cal_band_edge_orbital_responses(psi, states, rho_tsg, tau_tsg);

    const int nrxx = chg->nrxx;
    ASSERT_EQ(rho_tsg.size(), static_cast<std::size_t>(5 * 2 * nrxx));
    ASSERT_EQ(tau_tsg.size(), static_cast<std::size_t>(5 * 2 * nrxx));
    for (std::size_t iedge = 0; iedge < states.size(); ++iedge)
    {
        double selected_spin_norm = 0.0;
        double other_spin_norm = 0.0;
        for (int ir = 0; ir < nrxx; ++ir)
        {
            selected_spin_norm += std::abs(rho_tsg[(iedge * 2 + states[iedge].spin) * nrxx + ir]);
            other_spin_norm += std::abs(rho_tsg[(iedge * 2 + 1 - states[iedge].spin) * nrxx + ir]);
        }
        EXPECT_GT(selected_spin_norm, 0.0);
        EXPECT_DOUBLE_EQ(other_spin_norm, 0.0);
    }
}

#undef protected
