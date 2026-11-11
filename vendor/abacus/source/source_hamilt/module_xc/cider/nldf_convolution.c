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
// Prefix-private minimal migration of ciderpw_compute_kernels_helper from
// ciderpress/lib/pwutil/nldf_fft_core.c at CiderPress commit
// 6b2945ef9aea1494396e15b6f61389821d0e8549.

#include "nldf_convolution.h"

#include <math.h>

void abacus_cider_nldf_compute_kernel(double reciprocal_norm_squared,
                                       double* kernel,
                                       const double* norms,
                                       const double* exponents,
                                       int element_count)
{
    int element;
    for (element = 0; element < element_count; ++element)
    {
        kernel[element]
            = norms[element] * exp(-exponents[element] * reciprocal_norm_squared);
    }
}

void abacus_cider_nldf_compute_kernel_symmetric(
    double reciprocal_norm_squared,
    double* kernel,
    const double* norms,
    const double* exponents,
    int alpha_count)
{
    int row;
    for (row = 0; row < alpha_count; ++row)
    {
        int column;
        for (column = 0; column <= row; ++column)
        {
            const int index = row * alpha_count + column;
            const int transpose = column * alpha_count + row;
            const double value
                = norms[index]
                  * exp(-exponents[index] * reciprocal_norm_squared);
            kernel[index] = value;
            kernel[transpose] = value;
        }
    }
}
