#include "src/enzyme_ad/jax/Implementations/ReferenceOps.h"

#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/reference/Ops.h"

namespace mlir {
namespace enzyme {

stablehlo::Tensor reluOp(const stablehlo::Tensor &input,
                         ShapedType resultType) {
  auto zero =
      stablehlo::constantOp(cast<ElementsAttr>(makeAttr(resultType, 0)));
  return stablehlo::maxOp(input, zero, resultType);
}

stablehlo::Tensor softplusOp(const stablehlo::Tensor &input,
                             ShapedType resultType) {
  auto zero =
      stablehlo::constantOp(cast<ElementsAttr>(makeAttr(resultType, 0)));
  auto maxTerm = stablehlo::maxOp(input, zero, resultType);
  auto absInput = stablehlo::absOp(input, resultType);
  auto negAbsInput = stablehlo::negOp(absInput, resultType);
  auto expTerm = stablehlo::exponentialOp(negAbsInput, resultType);
  auto logTerm = stablehlo::log1pOp(expTerm, resultType);
  return stablehlo::addOp(maxTerm, logTerm, resultType);
}

stablehlo::Tensor geluTanhOp(const stablehlo::Tensor &input,
                             ShapedType resultType) {
  // x * (0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))))
  constexpr double kHalf = 0.5;
  constexpr double kOne = 1.0;
  constexpr double kSqrt2OverPi = 0.7978845608028654;
  constexpr double kCoeff = 0.044715;

  auto cHalf =
      stablehlo::constantOp(cast<ElementsAttr>(makeAttr(resultType, kHalf)));
  auto cOne =
      stablehlo::constantOp(cast<ElementsAttr>(makeAttr(resultType, kOne)));
  auto cSqrt2OverPi = stablehlo::constantOp(
      cast<ElementsAttr>(makeAttr(resultType, kSqrt2OverPi)));
  auto cCoeff =
      stablehlo::constantOp(cast<ElementsAttr>(makeAttr(resultType, kCoeff)));

  auto x2 = stablehlo::multiplyOp(input, input, resultType);
  auto x3 = stablehlo::multiplyOp(input, x2, resultType);
  auto coeffTimesX3 = stablehlo::multiplyOp(cCoeff, x3, resultType);
  auto xPlusPoly = stablehlo::addOp(input, coeffTimesX3, resultType);
  auto scaled = stablehlo::multiplyOp(cSqrt2OverPi, xPlusPoly, resultType);
  auto tanhTerm = stablehlo::tanhOp(scaled, resultType);
  auto onePlusTanh = stablehlo::addOp(cOne, tanhTerm, resultType);
  auto halfTimesOnePlusTanh =
      stablehlo::multiplyOp(cHalf, onePlusTanh, resultType);
  return stablehlo::multiplyOp(input, halfTimesOnePlusTanh, resultType);
}

stablehlo::Tensor geluSigmoidOp(const stablehlo::Tensor &input,
                                ShapedType resultType) {
  // x * sigmoid(sqrt(8 / pi) * x * (1 + 0.044715 * x^2))
  constexpr double kSqrt8OverPi = 1.5957691216057308;
  constexpr double kOne = 1.0;
  constexpr double kCoeff = 0.044715;

  auto cSqrt8OverPi = stablehlo::constantOp(
      cast<ElementsAttr>(makeAttr(resultType, kSqrt8OverPi)));
  auto cOne =
      stablehlo::constantOp(cast<ElementsAttr>(makeAttr(resultType, kOne)));
  auto cCoeff =
      stablehlo::constantOp(cast<ElementsAttr>(makeAttr(resultType, kCoeff)));

  auto x2 = stablehlo::multiplyOp(input, input, resultType);
  auto coeffTimesX2 = stablehlo::multiplyOp(cCoeff, x2, resultType);
  auto onePlusCoeffX2 = stablehlo::addOp(cOne, coeffTimesX2, resultType);
  auto xTimesInner = stablehlo::multiplyOp(input, onePlusCoeffX2, resultType);
  auto scaled = stablehlo::multiplyOp(cSqrt8OverPi, xTimesInner, resultType);
  auto sigmoid = stablehlo::logisticOp(scaled, resultType);
  return stablehlo::multiplyOp(input, sigmoid, resultType);
}

} // namespace enzyme
} // namespace mlir
