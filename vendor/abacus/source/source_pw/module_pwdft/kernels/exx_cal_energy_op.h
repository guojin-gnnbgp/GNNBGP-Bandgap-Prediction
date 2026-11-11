#ifndef CAL_VEC_NORM_OP_H
#define CAL_VEC_NORM_OP_H
#include "source_base/macros.h"
namespace hamilt{
template <typename T, typename Device>
struct exx_cal_energy_op
{

    using FPTYPE = typename GetTypeReal<T>::type;
    FPTYPE operator()(const T *den, const FPTYPE *pot, FPTYPE scala, int npw);
};


// Operator to compute a batch of psi_in[i] * conj(psi_in[i]) and multiples the potential
// returns the total energy of an entire batch of bands
template <typename T, typename Device>
struct exx_density_potential_mul_op
{

    using FPTYPE = typename GetTypeReal<T>::type;
    FPTYPE operator()(const T *vector_in,
                     FPTYPE *vector_buffer,
                     const FPTYPE *pot,
                    FPTYPE *vec_temp,
                    FPTYPE *weight,
                     int npw,
                     int batch_idx);
} ;

template <typename T, typename Device>
struct exx_cal_energy_accumulate_op
{
    using FPTYPE = typename GetTypeReal<T>::type;
    void operator()(const T* den, const FPTYPE* pot, FPTYPE scalar, int npw, FPTYPE* result);
};

template <typename T, typename Device>
struct exx_energy_density_accumulate_op
{
    using FPTYPE = typename GetTypeReal<T>::type;

    void operator()(const T* psi_n,
                    const T* psi_m,
                    const T* pair_potential,
                    FPTYPE weighted_half_over_omega,
                    T* energy_density,
                    int nrxx);

    void operator_batch(const T* psi_n,
                        const T* psi_m_batch,
                        const T* pair_potential_batch,
                        const FPTYPE* pair_weights,
                        FPTYPE half_over_omega,
                        T* energy_density,
                        int nrxx,
                        int batch_size);
};

} // namespace hamilt
#endif //CAL_VEC_NORM_OP_H
