#include "src/enzyme_ad/jax/Implementations/ReferenceOps.h"

#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/reference/Ops.h"

#include <cassert>
#include <cmath>
#include <complex>

namespace mlir {
namespace enzyme {

// Applies `floatFn` or `complexFn` elementwise to `input`, computing in double
// precision and rounding back to the element type of `resultType`.
template <typename FloatFn, typename ComplexFn>
static stablehlo::Tensor
mapWithUpcastToDouble(const stablehlo::Tensor &input, ShapedType resultType,
                      FloatFn floatFn, ComplexFn complexFn) {
  Type elemType = resultType.getElementType();
  stablehlo::Tensor result(resultType);

  if (stablehlo::isSupportedFloatType(elemType)) {
    for (auto it = result.index_begin(); it != result.index_end(); ++it) {
      double x = input.get(*it).getFloatValue().convertToDouble();
      result.set(*it, stablehlo::convert(elemType, floatFn(x)));
    }
    return result;
  }

  assert(stablehlo::isSupportedComplexType(elemType) &&
         "expected a float or complex element type");
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    auto z = input.get(*it).getComplexValue();
    std::complex<double> zd(z.real().convertToDouble(),
                            z.imag().convertToDouble());
    result.set(*it, stablehlo::convert(elemType, complexFn(zd)));
  }
  return result;
}

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

stablehlo::Tensor acosOp(const stablehlo::Tensor &input,
                         ShapedType resultType) {
  return mapWithUpcastToDouble(
      input, resultType, [](double x) { return std::acos(x); },
      [](std::complex<double> z) { return std::acos(z); });
}

stablehlo::Tensor acoshOp(const stablehlo::Tensor &input,
                          ShapedType resultType) {
  return mapWithUpcastToDouble(
      input, resultType, [](double x) { return std::acosh(x); },
      [](std::complex<double> z) { return std::acosh(z); });
}

stablehlo::Tensor asinOp(const stablehlo::Tensor &input,
                         ShapedType resultType) {
  return mapWithUpcastToDouble(
      input, resultType, [](double x) { return std::asin(x); },
      [](std::complex<double> z) { return std::asin(z); });
}

stablehlo::Tensor asinhOp(const stablehlo::Tensor &input,
                          ShapedType resultType) {
  return mapWithUpcastToDouble(
      input, resultType, [](double x) { return std::asinh(x); },
      [](std::complex<double> z) { return std::asinh(z); });
}

stablehlo::Tensor atanOp(const stablehlo::Tensor &input,
                         ShapedType resultType) {
  return mapWithUpcastToDouble(
      input, resultType, [](double x) { return std::atan(x); },
      [](std::complex<double> z) { return std::atan(z); });
}

stablehlo::Tensor atanhOp(const stablehlo::Tensor &input,
                          ShapedType resultType) {
  return mapWithUpcastToDouble(
      input, resultType, [](double x) { return std::atanh(x); },
      [](std::complex<double> z) { return std::atanh(z); });
}

stablehlo::Tensor coshOp(const stablehlo::Tensor &input,
                         ShapedType resultType) {
  return mapWithUpcastToDouble(
      input, resultType, [](double x) { return std::cosh(x); },
      [](std::complex<double> z) { return std::cosh(z); });
}

stablehlo::Tensor sinhOp(const stablehlo::Tensor &input,
                         ShapedType resultType) {
  return mapWithUpcastToDouble(
      input, resultType, [](double x) { return std::sinh(x); },
      [](std::complex<double> z) { return std::sinh(z); });
}

} // namespace enzyme
} // namespace mlir
