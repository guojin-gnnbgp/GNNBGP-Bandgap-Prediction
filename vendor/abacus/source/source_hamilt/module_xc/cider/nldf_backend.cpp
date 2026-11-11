#include "nldf_backend.h"

#include "nldf_convolution.h"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace hamilt
{
namespace cider
{
namespace
{
const double pi = 3.141592653589793238462643383279502884;

std::size_t checked_product(std::size_t first,
                            std::size_t second,
                            const char* message)
{
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
    {
        throw std::invalid_argument(message);
    }
    return first * second;
}

struct BufferRange
{
    std::uintptr_t begin;
    std::uintptr_t end;
};

BufferRange checked_buffer_range(const double* pointer, std::size_t count)
{
    const std::size_t bytes
        = checked_product(count, sizeof(double), "NLDF buffer byte size overflows");
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    {
        throw std::invalid_argument("NLDF buffer address range overflows");
    }
    return BufferRange{begin, begin + bytes};
}

bool overlaps(const BufferRange& first, const BufferRange& second)
{
    return first.begin < second.end && second.begin < first.end;
}

std::mutex& fftw_planner_mutex()
{
    // Planning and destruction use FFTW process-global state.  Serialize all
    // such operations across independently constructed native backends.
    static std::mutex mutex;
    return mutex;
}

class FftwPlanOwner
{
  public:
    explicit FftwPlanOwner(fftw_plan plan) : plan_(plan) {}

    ~FftwPlanOwner()
    {
        if (plan_ != nullptr)
        {
            std::lock_guard<std::mutex> guard(fftw_planner_mutex());
            fftw_destroy_plan(plan_);
        }
    }

    fftw_plan get() const { return plan_; }

  private:
    FftwPlanOwner(const FftwPlanOwner&);
    FftwPlanOwner& operator=(const FftwPlanOwner&);
    fftw_plan plan_;
};
}

NldfGrid::NldfGrid(const std::size_t shape[3], const double cell[9])
    : shape_{0, 0, 0}, cell_{0.0}, grid_count_(0)
{
    if (shape == nullptr || cell == nullptr)
    {
        throw std::invalid_argument("NLDF grid shape and cell are required");
    }
    std::size_t count = 1;
    for (std::size_t i = 0; i < 3; ++i)
    {
        if (shape[i] == 0 || shape[i] > static_cast<std::size_t>(std::numeric_limits<int>::max())
            || count > std::numeric_limits<std::size_t>::max() / shape[i])
        {
            throw std::invalid_argument("NLDF grid shape is invalid or overflows");
        }
        shape_[i] = shape[i];
        count *= shape[i];
    }
    for (std::size_t i = 0; i < 9; ++i)
    {
        if (!std::isfinite(cell[i]))
        {
            throw std::invalid_argument("NLDF cell must be finite");
        }
        cell_[i] = cell[i];
    }
    const double determinant
        = cell_[0] * (cell_[4] * cell_[8] - cell_[5] * cell_[7])
          - cell_[1] * (cell_[3] * cell_[8] - cell_[5] * cell_[6])
          + cell_[2] * (cell_[3] * cell_[7] - cell_[4] * cell_[6]);
    if (!(determinant > 0.0) || !std::isfinite(determinant))
    {
        throw std::invalid_argument("NLDF cell must have positive finite volume");
    }
    grid_count_ = count;
}

const std::size_t* NldfGrid::shape() const { return shape_; }
const double* NldfGrid::cell() const { return cell_; }
std::size_t NldfGrid::grid_count() const { return grid_count_; }

class NldfBackend::Impl
{
  public:
    Impl(const NldfGrid& input_grid,
         const std::vector<double>& input_alphas,
         const std::vector<double>& input_norms)
        : grid(input_grid), alphas(input_alphas), norms(input_norms)
    {
        if (alphas.size() < 2 || norms.size() != alphas.size()
            || alphas.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument("invalid NLDF backend alpha grid");
        }
        for (std::size_t i = 0; i < alphas.size(); ++i)
        {
            if (!(alphas[i] > 0.0) || !std::isfinite(alphas[i])
                || !(norms[i] > 0.0) || !std::isfinite(norms[i]))
            {
                throw std::invalid_argument(
                    "NLDF backend alphas and norms must be finite and positive");
            }
        }
        const std::size_t matrix_size = checked_product(
            alphas.size(), alphas.size(), "NLDF kernel matrix size overflows");
        kernel_norms.resize(matrix_size);
        kernel_exponents.resize(matrix_size);
        const double inverse_grid = 1.0 / static_cast<double>(grid.grid_count());
        for (std::size_t a = 0; a < alphas.size(); ++a)
        {
            for (std::size_t b = 0; b < alphas.size(); ++b)
            {
                const std::size_t index = b * alphas.size() + a;
                const double sum = alphas[a] + alphas[b];
                kernel_norms[index]
                    = inverse_grid * std::pow(pi / sum, 1.5) * norms[a] * norms[b];
                kernel_exponents[index] = 1.0 / (4.0 * sum);
                if (!std::isfinite(kernel_norms[index])
                    || !std::isfinite(kernel_exponents[index])
                    || !(kernel_norms[index] > 0.0)
                    || !(kernel_exponents[index] > 0.0))
                {
                    throw std::invalid_argument(
                        "NLDF derived Gaussian kernel data must be finite and positive");
                }
            }
        }
        setup_reciprocal_norms();
    }

    void setup_reciprocal_norms()
    {
        const double* c = grid.cell();
        const double c00 = c[0], c01 = c[3], c02 = c[6];
        const double c10 = c[1], c11 = c[4], c12 = c[7];
        const double c20 = c[2], c21 = c[5], c22 = c[8];
        const double determinant
            = c00 * (c11 * c22 - c21 * c12)
              - c01 * (c10 * c22 - c12 * c20)
              + c02 * (c10 * c21 - c11 * c20);
        const double factor = 2.0 * pi / determinant;
        const double reciprocal[9] = {
            (c11 * c22 - c21 * c12) * factor,
            -(c01 * c22 - c02 * c21) * factor,
            (c01 * c12 - c02 * c11) * factor,
            -(c10 * c22 - c12 * c20) * factor,
            (c00 * c22 - c02 * c20) * factor,
            -(c00 * c12 - c10 * c02) * factor,
            (c10 * c21 - c20 * c11) * factor,
            -(c00 * c21 - c20 * c01) * factor,
            (c00 * c11 - c10 * c01) * factor};
        const std::size_t* shape = grid.shape();
        const std::size_t nz_recip = shape[2] / 2 + 1;
        reciprocal_norms.resize(checked_product(
            checked_product(shape[0], shape[1], "NLDF reciprocal size overflows"),
            nz_recip,
            "NLDF reciprocal size overflows"));
        // Preserve the accepted upstream serial association exactly: the
        // linear FFT coefficient index is interpreted with y outer, x middle.
        for (std::size_t y = 0; y < shape[1]; ++y)
        {
            for (std::size_t x = 0; x < shape[0]; ++x)
            {
                for (std::size_t z = 0; z < nz_recip; ++z)
                {
                    const int k0 = static_cast<int>((x + shape[0] / 2) % shape[0])
                                   - static_cast<int>(shape[0] / 2);
                    const int k1 = static_cast<int>((y + shape[1] / 2) % shape[1])
                                   - static_cast<int>(shape[1] / 2);
                    const int k2 = static_cast<int>((z + shape[2] / 2) % shape[2])
                                   - static_cast<int>(shape[2] / 2);
                    const double kx = reciprocal[0] * k0 + reciprocal[3] * k1
                                      + reciprocal[6] * k2;
                    const double ky = reciprocal[1] * k0 + reciprocal[4] * k1
                                      + reciprocal[7] * k2;
                    const double kz = reciprocal[2] * k0 + reciprocal[5] * k1
                                      + reciprocal[8] * k2;
                    const std::size_t index = z + nz_recip * (x + shape[0] * y);
                    reciprocal_norms[index] = kx * kx + ky * ky + kz * kz;
                    if (!std::isfinite(reciprocal_norms[index]))
                    {
                        throw std::invalid_argument(
                            "NLDF reciprocal grid must be finite");
                    }
                }
            }
        }
    }

    NldfGrid grid;
    std::vector<double> alphas;
    std::vector<double> norms;
    std::vector<double> kernel_norms;
    std::vector<double> kernel_exponents;
    std::vector<double> reciprocal_norms;
    mutable std::mutex mutex;
};

NldfBackend::NldfBackend(const NldfGrid& grid,
                         const std::vector<double>& alphas,
                         const std::vector<double>& alpha_norms)
    : impl_(new Impl(grid, alphas, alpha_norms))
{
}

NldfBackend::~NldfBackend() { delete impl_; }

void NldfBackend::convolve(const double* input,
                           std::size_t input_size,
                           double* output,
                           std::size_t output_size) const
{
    const std::size_t alpha_count = impl_->alphas.size();
    const std::size_t expected = checked_product(
        impl_->grid.grid_count(), alpha_count, "NLDF convolution size overflows");
    if (input == nullptr || output == nullptr)
    {
        throw std::invalid_argument("invalid or aliased NLDF convolution buffers");
    }
    const BufferRange input_range = checked_buffer_range(input, input_size);
    const BufferRange output_range = checked_buffer_range(output, output_size);
    if (input_size != expected || output_size != expected)
    {
        throw std::invalid_argument("invalid NLDF convolution buffer size");
    }
    if (overlaps(input_range, output_range))
    {
        throw std::invalid_argument("NLDF convolution buffers must be disjoint");
    }
    for (std::size_t i = 0; i < expected; ++i)
    {
        if (!std::isfinite(input[i]))
        {
            throw std::invalid_argument("NLDF convolution input must be finite");
        }
    }
    std::lock_guard<std::mutex> guard(impl_->mutex);
    const std::size_t* shape = impl_->grid.shape();
    const std::size_t padded_z = 2 * (shape[2] / 2 + 1);
    const std::size_t padded_grid = checked_product(
        checked_product(shape[0], shape[1], "NLDF padded grid size overflows"),
        padded_z,
        "NLDF padded grid size overflows");
    std::vector<double> work(
        checked_product(padded_grid, alpha_count, "NLDF FFT work size overflows"),
        0.0);
    for (std::size_t x = 0; x < shape[0]; ++x)
    {
        for (std::size_t y = 0; y < shape[1]; ++y)
        {
            for (std::size_t z = 0; z < shape[2]; ++z)
            {
                const std::size_t grid_index = z + shape[2] * (y + shape[1] * x);
                const std::size_t padded_index = z + padded_z * (y + shape[1] * x);
                std::copy(input + grid_index * alpha_count,
                          input + (grid_index + 1) * alpha_count,
                          work.begin() + padded_index * alpha_count);
            }
        }
    }
    int dimensions[3] = {static_cast<int>(shape[0]),
                         static_cast<int>(shape[1]),
                         static_cast<int>(shape[2])};
    fftw_plan raw_forward = nullptr;
    {
        std::lock_guard<std::mutex> planner_guard(fftw_planner_mutex());
        raw_forward = fftw_plan_many_dft_r2c(
            3, dimensions, static_cast<int>(alpha_count), work.data(), nullptr,
            static_cast<int>(alpha_count), 1,
            reinterpret_cast<fftw_complex*>(work.data()), nullptr,
            static_cast<int>(alpha_count), 1, FFTW_ESTIMATE);
    }
    FftwPlanOwner forward(raw_forward);
    if (forward.get() == nullptr)
    {
        throw std::runtime_error("failed to create serial NLDF forward FFT plan");
    }

    fftw_plan raw_backward = nullptr;
    {
        std::lock_guard<std::mutex> planner_guard(fftw_planner_mutex());
        raw_backward = fftw_plan_many_dft_c2r(
            3, dimensions, static_cast<int>(alpha_count),
            reinterpret_cast<fftw_complex*>(work.data()), nullptr,
            static_cast<int>(alpha_count), 1, work.data(), nullptr,
            static_cast<int>(alpha_count), 1, FFTW_ESTIMATE);
    }
    FftwPlanOwner backward(raw_backward);
    if (backward.get() == nullptr)
    {
        throw std::runtime_error("failed to create serial NLDF backward FFT plan");
    }

    fftw_execute(forward.get());
    std::complex<double>* reciprocal
        = reinterpret_cast<std::complex<double>*>(work.data());
    std::vector<double> kernel(alpha_count * alpha_count);
    std::vector<std::complex<double> > transformed(alpha_count);
    for (std::size_t k = 0; k < impl_->reciprocal_norms.size(); ++k)
    {
        abacus_cider_nldf_compute_kernel_symmetric(
            impl_->reciprocal_norms[k],
            kernel.data(),
            impl_->kernel_norms.data(),
            impl_->kernel_exponents.data(),
            static_cast<int>(alpha_count));
        for (std::size_t b = 0; b < alpha_count; ++b)
        {
            transformed[b] = 0.0;
            for (std::size_t a = 0; a < alpha_count; ++a)
            {
                transformed[b] += reciprocal[k * alpha_count + a]
                                  * kernel[b * alpha_count + a];
            }
        }
        std::copy(transformed.begin(), transformed.end(), reciprocal + k * alpha_count);
    }
    fftw_execute(backward.get());
    std::vector<double> result(expected);
    for (std::size_t x = 0; x < shape[0]; ++x)
    {
        for (std::size_t y = 0; y < shape[1]; ++y)
        {
            for (std::size_t z = 0; z < shape[2]; ++z)
            {
                const std::size_t grid_index = z + shape[2] * (y + shape[1] * x);
                const std::size_t padded_index = z + padded_z * (y + shape[1] * x);
                std::copy(work.begin() + padded_index * alpha_count,
                          work.begin() + (padded_index + 1) * alpha_count,
                          result.begin() + grid_index * alpha_count);
            }
        }
    }
    for (std::size_t i = 0; i < expected; ++i)
    {
        if (!std::isfinite(result[i]))
        {
            throw std::runtime_error("NLDF convolution result is not finite");
        }
    }
    std::copy(result.begin(), result.end(), output);
}

} // namespace cider
} // namespace hamilt
