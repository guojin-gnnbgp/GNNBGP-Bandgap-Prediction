#ifndef MODULE_HAMILT_CIDER_MODEL_H
#define MODULE_HAMILT_CIDER_MODEL_H

#include <cstddef>
#include <string>
#include <vector>

namespace hamilt
{
namespace cider
{

enum class SemilocalLevel
{
    Gga,
    Mgga
};

enum class SemilocalMode
{
    Np,
    Npa
};

enum class FeatureDensityPolicy
{
    Valence,
    ValencePseudoCore
};

enum class EnergyConvention
{
    EnergyDensity
};

enum class RuntimePolicy
{
    DensityBridge
};

enum class OuterBaseline
{
    GgaXPbe,
    GgaCPbe
};

enum class MultiplicativeBaseline
{
    LdaX
};

enum class AdditiveBaseline
{
    GgaXChachiyo,
    GgaXPbe
};

enum class NormalizerKind
{
    Identity,
    Constant
};

enum class FeatureMapKind
{
    U,
    V,
    Z
};

class ParameterTriple
{
  public:
    ParameterTriple(double a0, double gradient_multiplier, double tau_multiplier);

    double a0() const;
    double gradient_multiplier() const;
    double tau_multiplier() const;

  private:
    double a0_;
    double gradient_multiplier_;
    double tau_multiplier_;
};

class NldfSettings
{
  public:
    NldfSettings(SemilocalLevel semilocal_level,
                 const ParameterTriple& theta_parameters,
                 const std::vector<ParameterTriple>& feature_parameters);

    SemilocalLevel semilocal_level() const;
    const ParameterTriple& theta_parameters() const;
    const std::vector<ParameterTriple>& feature_parameters() const;
    std::size_t feature_count() const;

  private:
    SemilocalLevel semilocal_level_;
    ParameterTriple theta_parameters_;
    std::vector<ParameterTriple> feature_parameters_;
};

class Normalizer
{
  public:
    Normalizer(NormalizerKind kind, double constant);

    NormalizerKind kind() const;
    double constant() const;

  private:
    NormalizerKind kind_;
    double constant_;
};

class FeatureMap
{
  public:
    FeatureMap(FeatureMapKind kind,
               std::size_t raw_feature_index,
               double gamma,
               double scale,
               double center,
               int exponent,
               double lower_bound,
               double upper_bound);

    FeatureMapKind kind() const;
    std::size_t raw_feature_index() const;
    double gamma() const;
    double scale() const;
    double center() const;
    int exponent() const;
    double lower_bound() const;
    double upper_bound() const;

  private:
    FeatureMapKind kind_;
    std::size_t raw_feature_index_;
    double gamma_;
    double scale_;
    double center_;
    int exponent_;
    double lower_bound_;
    double upper_bound_;
};

class SplineGrid
{
  public:
    SplineGrid(double lower, double upper, std::size_t order);

    double lower() const;
    double upper() const;
    std::size_t order() const;

  private:
    double lower_;
    double upper_;
    std::size_t order_;
};

class SplineTerm
{
  public:
    SplineTerm(double scale,
               const std::vector<std::size_t>& descriptor_indices,
               const std::vector<SplineGrid>& grids,
               const std::vector<std::size_t>& coefficient_extents,
               const std::vector<double>& coefficients);

    double scale() const;
    std::size_t dimension() const;
    const std::vector<std::size_t>& descriptor_indices() const;
    const std::vector<SplineGrid>& grids() const;
    const std::vector<std::size_t>& coefficient_extents() const;
    const std::vector<double>& coefficients() const;

  private:
    double scale_;
    std::vector<std::size_t> descriptor_indices_;
    std::vector<SplineGrid> grids_;
    std::vector<std::size_t> coefficient_extents_;
    std::vector<double> coefficients_;
};

class TrainingLabelMetadata
{
  public:
    TrainingLabelMetadata();
    TrainingLabelMetadata(const std::string& convention,
                          bool hybrid_scaled,
                          double scale,
                          const std::vector<double>& source_scales);

    bool specified() const;
    const std::string& convention() const;
    bool hybrid_scaled() const;
    double scale() const;
    const std::vector<double>& source_scales() const;

  private:
    bool specified_;
    std::string convention_;
    bool hybrid_scaled_;
    double scale_;
    std::vector<double> source_scales_;
};

class ModelMetadata
{
  public:
    ModelMetadata(SemilocalLevel semilocal_level,
                  bool has_nldf,
                  std::size_t semilocal_feature_count,
                  std::size_t nldf_feature_count,
                  std::size_t transformed_feature_count,
                  FeatureDensityPolicy feature_density_policy,
                  const std::string& feature_density_policy_source,
                  double kernel_output_scale,
                  double ml_exchange_scale,
                  const TrainingLabelMetadata& training_label_metadata);

    SemilocalLevel semilocal_level() const;
    bool is_mgga() const;
    bool requires_tau() const;
    bool has_nldf() const;
    std::size_t semilocal_feature_count() const;
    std::size_t nldf_feature_count() const;
    std::size_t raw_feature_count() const;
    std::size_t normalized_feature_count() const;
    std::size_t transformed_feature_count() const;
    FeatureDensityPolicy feature_density_policy() const;
    const std::string& feature_density_policy_source() const;
    EnergyConvention energy_convention() const;
    RuntimePolicy runtime_policy() const;
    OuterBaseline exchange_baseline() const;
    OuterBaseline correlation_baseline() const;
    double kernel_output_scale() const;
    double ml_exchange_scale() const;
    double runtime_exchange_scale(double requested_exchange_mix) const;
    const TrainingLabelMetadata& training_label_metadata() const;

  private:
    SemilocalLevel semilocal_level_;
    bool has_nldf_;
    std::size_t semilocal_feature_count_;
    std::size_t nldf_feature_count_;
    std::size_t transformed_feature_count_;
    FeatureDensityPolicy feature_density_policy_;
    std::string feature_density_policy_source_;
    double kernel_output_scale_;
    double ml_exchange_scale_;
    TrainingLabelMetadata training_label_metadata_;
};

// Public read-only interface. The concrete storage is private to the loader.
class Model
{
  public:
    virtual ~Model() = 0;

    virtual const ModelMetadata& metadata() const = 0;
    virtual SemilocalMode semilocal_mode() const = 0;
    virtual const NldfSettings* nldf_settings() const = 0;
    virtual double normalizer_cutoff() const = 0;
    virtual const std::vector<Normalizer>& normalizers() const = 0;
    virtual const std::vector<FeatureMap>& feature_maps() const = 0;
    virtual MultiplicativeBaseline multiplicative_baseline() const = 0;
    virtual AdditiveBaseline additive_baseline() const = 0;
    virtual double spline_constant() const = 0;
    virtual const std::vector<SplineTerm>& spline_terms() const = 0;

  protected:
    Model() = default;
};

} // namespace cider
} // namespace hamilt

#endif // MODULE_HAMILT_CIDER_MODEL_H
