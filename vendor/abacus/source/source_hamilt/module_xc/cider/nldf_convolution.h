#ifndef MODULE_HAMILT_CIDER_NLDF_CONVOLUTION_H
#define MODULE_HAMILT_CIDER_NLDF_CONVOLUTION_H

#ifdef __cplusplus
extern "C" {
#endif

void abacus_cider_nldf_compute_kernel(double reciprocal_norm_squared,
                                       double* kernel,
                                       const double* norms,
                                       const double* exponents,
                                       int element_count);

void abacus_cider_nldf_compute_kernel_symmetric(
    double reciprocal_norm_squared,
    double* kernel,
    const double* norms,
    const double* exponents,
    int alpha_count);

#ifdef __cplusplus
}
#endif

#endif // MODULE_HAMILT_CIDER_NLDF_CONVOLUTION_H
