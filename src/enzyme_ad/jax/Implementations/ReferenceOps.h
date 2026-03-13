#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/reference/Ops.h"

namespace mlir {
namespace enzyme {

stablehlo::Tensor reluOp(const stablehlo::Tensor &input, ShapedType resultType);

stablehlo::Tensor softplusOp(const stablehlo::Tensor &input,
                             ShapedType resultType);

stablehlo::Tensor geluTanhOp(const stablehlo::Tensor &input,
                             ShapedType resultType);

stablehlo::Tensor geluSigmoidOp(const stablehlo::Tensor &input,
                                ShapedType resultType);

} // namespace enzyme
} // namespace mlir
