#ifndef POT_CIDER_XC_H
#define POT_CIDER_XC_H

#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "pot_base.h"

#include <memory>
#include <string>

namespace elecstate
{

class PotCiderRuntime;

class PotCiderXC : public PotBase
{
  public:
    struct EnergyComponents
    {
        bool valid = false;
        double baseline_Ha = 0.0;
        double feature_scaled_Ha = 0.0;
        double feature_unmixed_Ha = 0.0;
        double total_Ha = 0.0;
        double feature_volume_Ha = 0.0;
        double feature_rho_weighted_Ha = 0.0;
        std::string feature_density_policy;
        std::string feature_exc_convention;
    };

    PotCiderXC(const ModulePW::PW_Basis* rho_basis_in,
               const UnitCell* ucell_in,
               double* etxc_in,
               double* vtxc_in,
               ModuleBase::matrix* vofk_in = nullptr);

    ~PotCiderXC() override;

    void cal_v_eff(const Charge* const chg,
                   const UnitCell* const ucell,
                   ModuleBase::matrix& v_eff) override;

    static const ModuleBase::matrix* debug_last_feature_potential();
    static const EnergyComponents& last_energy_components();

  private:
    std::unique_ptr<PotCiderRuntime> runtime_;
    double* etxc_ = nullptr;
    double* vtxc_ = nullptr;
    ModuleBase::matrix* vofk_ = nullptr;
    bool is_mgga_ = false;
    std::string feature_density_policy_ = "valence";
    std::string feature_exc_convention_ = "energy_density";
};

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC
#endif // POT_CIDER_XC_H
