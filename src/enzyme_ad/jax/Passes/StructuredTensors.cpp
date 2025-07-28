#include "src/enzyme_ad/jax/Passes/StructuredTensors.h"

#include "absl/status/statusor.h"
#include "mlir/IR/Matchers.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {

absl::StatusOr<DenseElementsAttr>
detectConstantSetindexScatterOp(stablehlo::ScatterOp scatterOp,
                                bool allowedMultipleUses,
                                bool onlyConstantZerosAllowed) {
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
    DenseElementsAttr attr;
    if (matchPattern(input, m_Constant(&attr))) {
      return attr;
    }
  }

  return absl::InvalidArgumentError(
      "Scatter Op is not a constant setindex op.");
}

} // namespace enzyme
} // namespace mlir
