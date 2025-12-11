#pragma once

#include "absl/status/status.h"
#include "mlir/IR/Matchers.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/SetVector.h"

#include <optional>

namespace mlir {
namespace enzyme {

absl::Status detectConstantSetindexScatterOp(stablehlo::ScatterOp scatterOp,
                                             bool allowedMultipleUses,
                                             bool onlyConstantZerosAllowed,
                                             DenseElementsAttr *constAttr);

absl::Status detectDiagonalTensor(stablehlo::ScatterOp scatterOp,
                                  mlir::Value *outUpdates);

struct IotaLikeTensor {
  int64_t start;
  int64_t dimension;
  int64_t scale = 1; // multiplicative factor applied to the iota
  mlir::RankedTensorType tensorType;
};

std::optional<IotaLikeTensor> detectIotaLikeTensor(mlir::Value tensor);

// TODO: we can do a full analysis and return if the access is on a specific set
// of diagonals. Checks that all accesses for this Op and its users thereoff are
// along the diagonal.
bool allAccessesAreOnMainDiagonal(
    mlir::Operation *op, llvm::SetVector<mlir::Operation *> &opsToReplace);
bool allAccessesAreOnMainDiagonal(
    stablehlo::ReshapeOp op, llvm::SetVector<mlir::Operation *> &opsToReplace);
bool allAccessesAreOnMainDiagonal(
    stablehlo::GatherOp op, llvm::SetVector<mlir::Operation *> &opsToReplace);

} // namespace enzyme
} // namespace mlir
