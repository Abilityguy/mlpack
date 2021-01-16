/**
 * @file methods/ann/loss_functions/squared_hinge_loss_impl.hpp
 * @author Anush Kini
 *
 * Implementation of the Squared Hinge loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SQUARED_HINGE_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_SQUARED_HINGE_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "squared_hinge_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
SquaredHingeLoss<InputDataType, OutputDataType>::SquaredHingeLoss()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType>
typename PredictionType::elem_type
SquaredHingeLoss<InputDataType, OutputDataType>::Forward(
    const PredictionType& prediction,
    const TargetType& target)
{
  TargetType temp = target - (target == 0);
  return (arma::accu(arma::square(arma::max(1 - prediction % temp, 0.))))
      / target.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType, typename LossType>
void SquaredHingeLoss<InputDataType, OutputDataType>::Backward(
    const PredictionType& prediction,
    const TargetType& target,
    LossType& loss)
{
  TargetType temp = target - (target == 0);
  loss = (prediction < 1 / temp) % -temp;
}
