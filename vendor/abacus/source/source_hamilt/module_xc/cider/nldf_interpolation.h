#ifndef MODULE_HAMILT_CIDER_NLDF_INTERPOLATION_H
#define MODULE_HAMILT_CIDER_NLDF_INTERPOLATION_H

#ifdef __cplusplus
extern "C" {
#endif

void abacus_cider_nldf_ind_etb(double* indexes,
                               double* derivatives,
                               const double* exponents,
                               int grid_count,
                               double alpha0,
                               double lambda);
void abacus_cider_nldf_ind_clip(double* indexes,
                                double* derivatives,
                                int size_minus_one,
                                int grid_count);
void abacus_cider_nldf_coefs_spline_gq(double* coefficients,
                                        double* derivatives,
                                        const double* indexes,
                                        const double* table,
                                        int grid_count,
                                        int alpha_count);

#ifdef __cplusplus
}
#endif

#endif // MODULE_HAMILT_CIDER_NLDF_INTERPOLATION_H
