#pragma once

#include "absl/status/statusor.h"
#include "mlir/IR/Matchers.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {

absl::StatusOr<DenseElementsAttr>
detectConstantSetindexScatterOp(stablehlo::ScatterOp scatterOp,
                                bool allowedMultipleUses,
                                bool onlyConstantZerosAllowed);

} // namespace enzyme
} // namespace mlir
