#include "model.h"

namespace hamilt
{
namespace cider
{

ParameterTriple::ParameterTriple(double a0, double gradient_multiplier, double tau_multiplier)
    : a0_(a0), gradient_multiplier_(gradient_multiplier), tau_multiplier_(tau_multiplier)
{
}

double ParameterTriple::a0() const { return a0_; }
double ParameterTriple::gradient_multiplier() const { return gradient_multiplier_; }
double ParameterTriple::tau_multiplier() const { return tau_multiplier_; }

NldfSettings::NldfSettings(SemilocalLevel semilocal_level,
                           const ParameterTriple& theta_parameters,
                           const std::vector<ParameterTriple>& feature_parameters)
    : semilocal_level_(semilocal_level),
      theta_parameters_(theta_parameters),
      feature_parameters_(feature_parameters)
{
}

SemilocalLevel NldfSettings::semilocal_level() const { return semilocal_level_; }
const ParameterTriple& NldfSettings::theta_parameters() const { return theta_parameters_; }
const std::vector<ParameterTriple>& NldfSettings::feature_parameters() const
{
    return feature_parameters_;
}
std::size_t NldfSettings::feature_count() const { return feature_parameters_.size(); }

Normalizer::Normalizer(NormalizerKind kind, double constant) : kind_(kind), constant_(constant) {}
NormalizerKind Normalizer::kind() const { return kind_; }
double Normalizer::constant() const { return constant_; }

FeatureMap::FeatureMap(FeatureMapKind kind,
                       std::size_t raw_feature_index,
                       double gamma,
                       double scale,
                       double center,
                       int exponent,
                       double lower_bound,
                       double upper_bound)
    : kind_(kind),
      raw_feature_index_(raw_feature_index),
      gamma_(gamma),
      scale_(scale),
      center_(center),
      exponent_(exponent),
      lower_bound_(lower_bound),
      upper_bound_(upper_bound)
{
}

FeatureMapKind FeatureMap::kind() const { return kind_; }
std::size_t FeatureMap::raw_feature_index() const { return raw_feature_index_; }
double FeatureMap::gamma() const { return gamma_; }
double FeatureMap::scale() const { return scale_; }
double FeatureMap::center() const { return center_; }
int FeatureMap::exponent() const { return exponent_; }
double FeatureMap::lower_bound() const { return lower_bound_; }
double FeatureMap::upper_bound() const { return upper_bound_; }

SplineGrid::SplineGrid(double lower, double upper, std::size_t order)
    : lower_(lower), upper_(upper), order_(order)
{
}
double SplineGrid::lower() const { return lower_; }
double SplineGrid::upper() const { return upper_; }
std::size_t SplineGrid::order() const { return order_; }

SplineTerm::SplineTerm(double scale,
                       const std::vector<std::size_t>& descriptor_indices,
                       const std::vector<SplineGrid>& grids,
                       const std::vector<std::size_t>& coefficient_extents,
                       const std::vector<double>& coefficients)
    : scale_(scale),
      descriptor_indices_(descriptor_indices),
      grids_(grids),
      coefficient_extents_(coefficient_extents),
      coefficients_(coefficients)
{
}
double SplineTerm::scale() const { return scale_; }
std::size_t SplineTerm::dimension() const { return descriptor_indices_.size(); }
const std::vector<std::size_t>& SplineTerm::descriptor_indices() const
{
    return descriptor_indices_;
}
const std::vector<SplineGrid>& SplineTerm::grids() const { return grids_; }
const std::vector<std::size_t>& SplineTerm::coefficient_extents() const
{
    return coefficient_extents_;
}
const std::vector<double>& SplineTerm::coefficients() const { return coefficients_; }

TrainingLabelMetadata::TrainingLabelMetadata()
    : specified_(false), hybrid_scaled_(false), scale_(1.0)
{
}
TrainingLabelMetadata::TrainingLabelMetadata(const std::string& convention,
                                             bool hybrid_scaled,
                                             double scale,
                                             const std::vector<double>& source_scales)
    : specified_(true),
      convention_(convention),
      hybrid_scaled_(hybrid_scaled),
      scale_(scale),
      source_scales_(source_scales)
{
}
bool TrainingLabelMetadata::specified() const { return specified_; }
const std::string& TrainingLabelMetadata::convention() const { return convention_; }
bool TrainingLabelMetadata::hybrid_scaled() const { return hybrid_scaled_; }
double TrainingLabelMetadata::scale() const { return scale_; }
const std::vector<double>& TrainingLabelMetadata::source_scales() const
{
    return source_scales_;
}

ModelMetadata::ModelMetadata(SemilocalLevel semilocal_level,
                             bool has_nldf,
                             std::size_t semilocal_feature_count,
                             std::size_t nldf_feature_count,
                             std::size_t transformed_feature_count,
                             FeatureDensityPolicy feature_density_policy,
                             const std::string& feature_density_policy_source,
                             double kernel_output_scale,
                             double ml_exchange_scale,
                             const TrainingLabelMetadata& training_label_metadata)
    : semilocal_level_(semilocal_level),
      has_nldf_(has_nldf),
      semilocal_feature_count_(semilocal_feature_count),
      nldf_feature_count_(nldf_feature_count),
      transformed_feature_count_(transformed_feature_count),
      feature_density_policy_(feature_density_policy),
      feature_density_policy_source_(feature_density_policy_source),
      kernel_output_scale_(kernel_output_scale),
      ml_exchange_scale_(ml_exchange_scale),
      training_label_metadata_(training_label_metadata)
{
}

SemilocalLevel ModelMetadata::semilocal_level() const { return semilocal_level_; }
bool ModelMetadata::is_mgga() const { return semilocal_level_ == SemilocalLevel::Mgga; }
bool ModelMetadata::requires_tau() const { return is_mgga(); }
bool ModelMetadata::has_nldf() const { return has_nldf_; }
std::size_t ModelMetadata::semilocal_feature_count() const { return semilocal_feature_count_; }
std::size_t ModelMetadata::nldf_feature_count() const { return nldf_feature_count_; }
std::size_t ModelMetadata::raw_feature_count() const
{
    return semilocal_feature_count_ + nldf_feature_count_;
}
std::size_t ModelMetadata::normalized_feature_count() const { return raw_feature_count(); }
std::size_t ModelMetadata::transformed_feature_count() const
{
    return transformed_feature_count_;
}
FeatureDensityPolicy ModelMetadata::feature_density_policy() const
{
    return feature_density_policy_;
}
const std::string& ModelMetadata::feature_density_policy_source() const
{
    return feature_density_policy_source_;
}
EnergyConvention ModelMetadata::energy_convention() const { return EnergyConvention::EnergyDensity; }
RuntimePolicy ModelMetadata::runtime_policy() const { return RuntimePolicy::DensityBridge; }
OuterBaseline ModelMetadata::exchange_baseline() const { return OuterBaseline::GgaXPbe; }
OuterBaseline ModelMetadata::correlation_baseline() const { return OuterBaseline::GgaCPbe; }
double ModelMetadata::kernel_output_scale() const { return kernel_output_scale_; }
double ModelMetadata::ml_exchange_scale() const { return ml_exchange_scale_; }
double ModelMetadata::runtime_exchange_scale(double requested_exchange_mix) const
{
    return ml_exchange_scale_ == 0.0 ? requested_exchange_mix
                                     : requested_exchange_mix / ml_exchange_scale_;
}
const TrainingLabelMetadata& ModelMetadata::training_label_metadata() const
{
    return training_label_metadata_;
}

Model::~Model() = default;

} // namespace cider
} // namespace hamilt
