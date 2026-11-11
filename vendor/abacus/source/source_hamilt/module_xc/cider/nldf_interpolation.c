// CiderPress: Machine-learning based density functional theory calculations
// Copyright (C) 2024 The President and Fellows of Harvard College
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>
//
// Author: Kyle Bystrom <kylebystrom@gmail.com>
//
// Prefix-private minimal migration from ciderpress/lib/mod_cider/cider_coefs.c
// at CiderPress commit 6b2945ef9aea1494396e15b6f61389821d0e8549.

#include "nldf_interpolation.h"

#include <math.h>

void abacus_cider_nldf_ind_etb(double* indexes,
                               double* derivatives,
                               const double* exponents,
                               int grid_count,
                               double alpha0,
                               double lambda)
{
    const double ratio = 1.0 / log(lambda);
    const double inverse_alpha = 1.0 / alpha0;
    int grid;
    for (grid = 0; grid < grid_count; ++grid)
    {
        derivatives[grid] = ratio / exponents[grid];
        indexes[grid] = log(exponents[grid] * inverse_alpha) * ratio;
    }
}

void abacus_cider_nldf_ind_clip(double* indexes,
                                double* derivatives,
                                int size_minus_one,
                                int grid_count)
{
    const double upper = size_minus_one - 1.0e-10;
    int grid;
    for (grid = 0; grid < grid_count; ++grid)
    {
        double index = indexes[grid];
        const int above_lower = index > 0.0;
        derivatives[grid] = above_lower ? derivatives[grid] : 0.0;
        indexes[grid] = above_lower ? index : 0.0;
        index = indexes[grid];
        {
            const int below_upper = index < size_minus_one;
            derivatives[grid] = below_upper ? derivatives[grid] : 0.0;
            indexes[grid] = below_upper ? index : upper;
        }
    }
}

void abacus_cider_nldf_coefs_spline_gq(double* coefficients,
                                        double* derivatives,
                                        const double* indexes,
                                        const double* table,
                                        int grid_count,
                                        int alpha_count)
{
    int grid;
    for (grid = 0; grid < grid_count; ++grid)
    {
        const int interval = (int)indexes[grid];
        const double delta = indexes[grid] - interval;
        const double* interval_table
            = table + interval * 4 * alpha_count;
        int alpha;
        for (alpha = 0; alpha < alpha_count; ++alpha)
        {
            const double* polynomial = interval_table + 4 * alpha;
            coefficients[grid * alpha_count + alpha]
                = polynomial[0]
                  + delta
                        * (polynomial[1]
                           + delta * (polynomial[2] + delta * polynomial[3]));
            derivatives[grid * alpha_count + alpha]
                = polynomial[1]
                  + delta
                        * (2.0 * polynomial[2] + delta * 3.0 * polynomial[3]);
        }
    }
}
