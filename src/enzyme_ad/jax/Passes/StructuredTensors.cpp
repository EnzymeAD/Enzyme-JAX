#include "src/enzyme_ad/jax/Passes/StructuredTensors.h"

#include "absl/status/status.h"
#include "mlir/IR/Matchers.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {

absl::Status detectConstantSetindexScatterOp(stablehlo::ScatterOp scatterOp,
                                             bool allowedMultipleUses,
                                             bool onlyConstantZerosAllowed,
                                             DenseElementsAttr *constAttr) {
  if (scatterOp.getInputs().size() != 1) {
    return absl::UnimplementedError(
        "Detection not implemented for scatter op with >1 input.");
  }

  if (!scatterOp.getResult(0).hasOneUse() && !allowedMultipleUses) {
    return absl::InvalidArgumentError(
        "ScatterOp has multiple uses, not supported.");
  }

  if (!isScatterSetindexOp(scatterOp)) {
    return absl::InvalidArgumentError("ScatterOp is not a setindex op.");
  }

  auto input = scatterOp.getInputs()[0];
  if (onlyConstantZerosAllowed) {
    if (matchPattern(input, m_AnyZeroFloat()) ||
        matchPattern(input, m_Zero())) {
      return absl::OkStatus();
    }
  } else {
    if (matchPattern(input, m_Constant(constAttr))) {
      return absl::OkStatus();
    }
  }

  return absl::InvalidArgumentError(
      "Scatter Op is not a constant setindex op.");
}

// TODO: detect batched diagonal tensors
absl::Status detectDiagonalTensor(stablehlo::ScatterOp scatterOp,
                                  mlir::Value *outUpdates) {
  auto status = detectConstantSetindexScatterOp(scatterOp, true, true, nullptr);
  if (!status.ok())
    return status;

  auto input = scatterOp.getInputs()[0];
  auto inputShape = cast<ShapedType>(input.getType()).getShape();
  // TODO: support the non-square case
  if (inputShape.size() != 2 || inputShape[0] != inputShape[1])
    return absl::InvalidArgumentError("Input is not a diagonal tensor.");

  auto indices = scatterOp.getScatterIndices();
  auto indicesShape = cast<ShapedType>(indices.getType()).getShape();
  if (indicesShape.size() != 2 || indicesShape[0] != inputShape[0] ||
      indicesShape[1] != 2)
    return absl::InvalidArgumentError("Indices are not for a diagonal tensor.");

  auto updates = scatterOp.getUpdates()[0];
  if (cast<RankedTensorType>(updates.getType()).getRank() != 1)
    return absl::InvalidArgumentError("Updates are not a vector.");

  auto scatterDimNumbers = scatterOp.getScatterDimensionNumbers();
  auto validScatterDimNumbers = stablehlo::ScatterDimensionNumbersAttr::get(
      scatterOp.getContext(), ArrayRef<int64_t>(), ArrayRef<int64_t>({0, 1}),
      ArrayRef<int64_t>(), ArrayRef<int64_t>(), ArrayRef<int64_t>({0, 1}), 1);
  if (scatterDimNumbers != validScatterDimNumbers)
    return absl::InvalidArgumentError(
        "Scatter dimension numbers are not valid for a diagonal tensor.");

  if (auto iotaOp = dyn_cast<stablehlo::IotaOp>(indices.getDefiningOp())) {
    if (iotaOp.getIotaDimension() == 0) {
      *outUpdates = updates;
      return absl::OkStatus();
    }
  }

  return absl::InvalidArgumentError("Not a diagonal tensor.");
}

} // namespace enzyme
} // namespace mlir
