#pragma once

#include "absl/status/status.h"
#include "mlir/IR/Matchers.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

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
  int64_t limit;
  int64_t dimension;
  mlir::RankedTensorType tensorType;
};

std::optional<IotaLikeTensor> detectIotaLikeTensor(mlir::Value tensor);

} // namespace enzyme
} // namespace mlir
