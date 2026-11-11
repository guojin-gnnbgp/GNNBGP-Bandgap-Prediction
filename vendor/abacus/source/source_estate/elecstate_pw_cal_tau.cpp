#include "elecstate_pw.h"

namespace elecstate {

template<typename T, typename Device>
void ElecStatePW<T, Device>::cal_tau(const psi::Psi<T, Device>& psi)
{
    ModuleBase::TITLE("ElecStatePW", "cal_tau");
    for(int is=0; is<PARAM.inp.nspin; is++)
	{
        setmem_var_op()(this->kin_r[is], 0,  this->charge->nrxx);
	}

    for (int ik = 0; ik < psi.get_nk(); ++ik)
    {
        psi.fix_k(ik);
        int npw = psi.get_current_ngk();
        int current_spin = 0;
        if (PARAM.inp.nspin == 2)
        {
            current_spin = this->klist->isk[ik];
        }
        int nbands = psi.get_nbands();
        for (int ibnd = 0; ibnd < nbands; ibnd++)
        {
            this->basis->recip_to_real(this->ctx, &psi(ibnd,0), this->wfcr, ik);

            const auto w1 = static_cast<Real>(this->wg(ik, ibnd) / ucell->omega);

            // kinetic energy density
            for (int j = 0; j < 3; j++)
            {
                setmem_complex_op()(this->wfcr, 0,  this->charge->nrxx);

                meta_op()(this->ctx,
                            ik,
                            j,
                            npw,
                            this->basis->npwk_max,
                            static_cast<Real>(ucell->tpiba),
                            this->basis->template get_gcar_data<Real>(),
                            this->basis->template get_kvec_c_data<Real>(),
                            &psi(ibnd, 0),
                            this->wfcr);

                this->basis->recip_to_real(this->ctx, this->wfcr, this->wfcr, ik);

                elecstate_pw_op()(this->ctx, current_spin, this->charge->nrxx, w1, this->kin_r, this->wfcr);
            }
        }
    }
    if (PARAM.inp.device == "gpu" || PARAM.inp.precision == "single") {
        for (int ii = 0; ii < PARAM.inp.nspin; ii++) {
            castmem_var_d2h_op()(this->charge->kin_r[ii], this->kin_r[ii], this->charge->nrxx);
        }
    }
#ifdef __MPI
    this->charge->kin_r_mpi();
#endif
    ModuleBase::TITLE("ElecStatePW", "cal_tau");
}

template<typename T, typename Device>
void ElecStatePW<T, Device>::cal_band_edge_orbital_responses(const psi::Psi<T, Device>& psi,
                                                             const std::vector<ModuleIO::TrainingBandEdgeState>& states,
                                                             std::vector<double>& rho_tsg,
                                                             std::vector<double>& tau_tsg)
{
    ModuleBase::TITLE("ElecStatePW", "cal_band_edge_orbital_responses");
    if (PARAM.inp.nspin != 1 && PARAM.inp.nspin != 2)
    {
        ModuleBase::WARNING_QUIT("cal_band_edge_orbital_responses",
                                 "band-edge orbital responses support nspin 1 or 2");
    }
    const int nspin = PARAM.inp.nspin;
    const int nrxx = this->charge->nrxx;
    const int nedge = static_cast<int>(states.size());
    rho_tsg.assign(nedge * nspin * nrxx, 0.0);
    tau_tsg.assign(nedge * nspin * nrxx, 0.0);
    Real* rho_edge_device = nullptr;
    Real* tau_edge_device = nullptr;
    resmem_var_op()(rho_edge_device, nrxx, "ElecSPW::edge_rho");
    resmem_var_op()(tau_edge_device, nrxx, "ElecSPW::edge_tau");
    for (int iedge = 0; iedge < nedge; ++iedge)
    {
        const int ik = states[iedge].k;
        const int ibnd = states[iedge].band;
        if (ik < 0 || ik >= psi.get_nk() || ibnd < 0 || ibnd >= psi.get_nbands())
        {
            delmem_var_op()(rho_edge_device);
            delmem_var_op()(tau_edge_device);
            ModuleBase::WARNING_QUIT("cal_band_edge_orbital_responses",
                                     "selected band-edge state is outside the wavefunction dimensions");
        }
        psi.fix_k(ik);
        const int npw = psi.get_current_ngk();
        const int current_spin = nspin == 2 ? this->klist->isk[ik] : 0;
        if (states[iedge].spin != current_spin)
        {
            delmem_var_op()(rho_edge_device);
            delmem_var_op()(tau_edge_device);
            ModuleBase::WARNING_QUIT("cal_band_edge_orbital_responses",
                                     "selected band-edge spin does not match the k-point spin channel");
        }
        const auto unit_weight = static_cast<Real>(1.0 / this->ucell->omega);
        setmem_var_op()(rho_edge_device, 0, nrxx);
        setmem_var_op()(tau_edge_device, 0, nrxx);

        this->basis->recip_to_real(this->ctx, &psi(ibnd,0), this->wfcr, ik);
        Real* rho_edge_ptrs[1] = {rho_edge_device};
        elecstate_pw_op()(this->ctx, 0, nrxx, unit_weight, rho_edge_ptrs, this->wfcr);

        for (int j = 0; j < 3; ++j)
        {
            setmem_complex_op()(this->wfcr, 0, this->charge->nrxx);
            meta_op()(this->ctx,
                      ik,
                      j,
                      npw,
                      this->basis->npwk_max,
                      static_cast<Real>(this->ucell->tpiba),
                      this->basis->template get_gcar_data<Real>(),
                      this->basis->template get_kvec_c_data<Real>(),
                      &psi(ibnd, 0),
                      this->wfcr);
            this->basis->recip_to_real(this->ctx, this->wfcr, this->wfcr, ik);
            Real* tau_edge_ptrs[1] = {tau_edge_device};
            elecstate_pw_op()(this->ctx, 0, nrxx, unit_weight, tau_edge_ptrs, this->wfcr);
        }

        std::vector<Real> rho_edge_host(nrxx, 0);
        std::vector<Real> tau_edge_host(nrxx, 0);
        syncmem_var_d2h_op()(rho_edge_host.data(), rho_edge_device, nrxx);
        syncmem_var_d2h_op()(tau_edge_host.data(), tau_edge_device, nrxx);
        for (int ir = 0; ir < nrxx; ++ir)
        {
            const int base = (iedge * nspin + current_spin) * nrxx + ir;
            rho_tsg[base] = static_cast<double>(rho_edge_host[ir]);
            tau_tsg[base] = 0.5 * static_cast<double>(tau_edge_host[ir]);
        }
    }
    delmem_var_op()(rho_edge_device);
    delmem_var_op()(tau_edge_device);
    ModuleBase::TITLE("ElecStatePW", "cal_band_edge_orbital_responses");
}

template class ElecStatePW<std::complex<float>, base_device::DEVICE_CPU>;
template class ElecStatePW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ElecStatePW<std::complex<float>, base_device::DEVICE_GPU>;
template class ElecStatePW<std::complex<double>, base_device::DEVICE_GPU>;
#endif 
} // namespace elecstate
