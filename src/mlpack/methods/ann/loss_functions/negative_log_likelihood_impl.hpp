/**
 * @file methods/ann/loss_functions/negative_log_likelihood_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the NegativeLogLikelihood class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_IMPL_HPP

// In case it hasn't yet been included.
#include "negative_log_likelihood.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
NegativeLogLikelihood<InputDataType, OutputDataType>::NegativeLogLikelihood(
    const bool reduction) : reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType>
typename PredictionType::elem_type
NegativeLogLikelihood<InputDataType, OutputDataType>::Forward(
    const PredictionType& prediction,
    const TargetType& target)
{
  PredictionType loss;
  loss.zeros(size(target));
  for (size_t i = 0; i < target.n_cols; ++i)
  {
    size_t currentTarget = target(i);
    Log::Assert(currentTarget >= 0 && currentTarget < prediction.n_cols,
        "Target class out of range.");
    loss(i) = -prediction(i, currentTarget);
  }
  typename PredictionType::elem_type lossSum = arma::accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType, typename LossType>
void NegativeLogLikelihood<InputDataType, OutputDataType>::Backward(
      const PredictionType& prediction,
      const TargetType& target,
      LossType& loss)
{
  loss.zeros(size(prediction));
  for (size_t i = 0; i < target.n_cols; ++i)
  {
    size_t currentTarget = target(i);
    Log::Assert(currentTarget >= 0 && currentTarget < prediction.n_cols,
        "Target class out of range.");
    loss(i, currentTarget) = -1;
  }

  if (!reduction)
    loss = loss / target.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void NegativeLogLikelihood<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace ann
} // namespace mlpack

#endif
