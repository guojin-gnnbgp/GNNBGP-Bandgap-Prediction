#ifndef MODULE_HAMILT_CIDER_NLDF_BACKEND_H
#define MODULE_HAMILT_CIDER_NLDF_BACKEND_H

#include <cstddef>
#include <vector>

namespace hamilt
{
namespace cider
{

// A complete serial real-space grid.  shape uses C-order (x,y,z), z-fast
// indexing.  cell is a C-order 3x3 matrix whose rows are lattice vectors in
// Bohr, matching the accepted CiderPress standalone bridge convention.
class NldfGrid
{
  public:
    NldfGrid(const std::size_t shape[3], const double cell[9]);

    const std::size_t* shape() const;
    const double* cell() const;
    std::size_t grid_count() const;

  private:
    std::size_t shape_[3];
    double cell_[9];
    std::size_t grid_count_;
};

class NldfBackend
{
  public:
    NldfBackend(const NldfGrid& grid,
                const std::vector<double>& alphas,
                const std::vector<double>& alpha_norms);
    ~NldfBackend();

    // Both arrays are grid-major, alpha-fast, with grid_count*alpha_count
    // doubles.  The operation is the symmetric Gaussian convolution used by
    // version J, so the same method is its Euclidean adjoint.
    void convolve(const double* input,
                  std::size_t input_size,
                  double* output,
                  std::size_t output_size) const;

  private:
    NldfBackend(const NldfBackend&);
    NldfBackend& operator=(const NldfBackend&);
    class Impl;
    Impl* impl_;
};

} // namespace cider
} // namespace hamilt

#endif // MODULE_HAMILT_CIDER_NLDF_BACKEND_H
