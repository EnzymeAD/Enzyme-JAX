#include "src/enzyme_ad/jax/Passes/StructuredTensors.h"

#include "absl/status/status.h"
#include "mlir/IR/Matchers.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/SetVector.h"

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

  auto checkCommonScatterOp = mlir::stablehlo::CheckCommonScatterOp(scatterOp);

  if (!checkCommonScatterOp.isSetindexScatter) {
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

  auto isIotaLikeTensor = detectIotaLikeTensor(indices);
  if (isIotaLikeTensor) {
    auto iotaLikeTensor = isIotaLikeTensor.value();
    if (iotaLikeTensor.dimension == 0 && iotaLikeTensor.start == 0 &&
        iotaLikeTensor.scale == 1) {
      *outUpdates = updates;
      return absl::OkStatus();
    }
  }

  return absl::InvalidArgumentError("Not a diagonal tensor.");
}

std::optional<IotaLikeTensor> detectIotaLikeTensor(mlir::Value tensor) {
  if (!tensor)
    return std::nullopt;

  auto elemType =
      cast<mlir::RankedTensorType>(tensor.getType()).getElementType();
  if (!isa<mlir::IntegerType>(elemType))
    return std::nullopt;

  struct ChainItem {
    mlir::Operation *op;
    int64_t offset; // only populated for AddOp/SubtractOp
    int64_t scale;  // only populated for MulOp
  };

  // Build a chain of operations from startOp to the base case
  SmallVector<ChainItem> chain;
  llvm::DenseSet<mlir::Operation *> visited;
  mlir::Operation *currentOp = tensor.getDefiningOp();

  // Traverse to find base case
  while (currentOp && !visited.contains(currentOp)) {
    visited.insert(currentOp);

    // check if we found a base case
    if (isa<stablehlo::IotaOp, stablehlo::ConstantOp>(currentOp)) {
      chain.push_back({currentOp, 0, 1});
      break;
    }

    // navigate to the next op. If any unsupported intermediate op is found,
    // then return std::nullopt
    Operation *nextOp;

    // TODO: we might want to support broadcast_in_dim / insert_dims / drop_dims
    // as well
    if (isa<stablehlo::TransposeOp>(currentOp)) {
      chain.push_back({currentOp, 0, 1});
      nextOp = currentOp->getOperand(0).getDefiningOp();
    } else if (auto convertOp = dyn_cast<stablehlo::ConvertOp>(currentOp)) {
      // if operand of convertOp is not a integer, then return std::nullopt
      if (!isa<mlir::IntegerType>(
              cast<TensorType>(convertOp.getOperand().getType())
                  .getElementType()))
        return std::nullopt;
      chain.push_back({currentOp, 0, 1});
      nextOp = convertOp.getOperand().getDefiningOp();
    } else if (auto addOp = dyn_cast<stablehlo::AddOp>(currentOp)) {
      APInt offsetVal;
      if (matchPattern(addOp.getRhs(), m_ConstantInt(&offsetVal))) {
        chain.push_back({currentOp, offsetVal.getSExtValue(), 1});
        nextOp = addOp.getLhs().getDefiningOp();
      } else if (matchPattern(addOp.getLhs(), m_ConstantInt(&offsetVal))) {
        chain.push_back({currentOp, offsetVal.getSExtValue(), 1});
        nextOp = addOp.getRhs().getDefiningOp();
      } else {
        return std::nullopt;
      }
    } else if (auto subOp = dyn_cast<stablehlo::SubtractOp>(currentOp)) {
      APInt offsetVal;
      if (matchPattern(subOp.getRhs(), m_ConstantInt(&offsetVal))) {
        chain.push_back({currentOp, -offsetVal.getSExtValue(), 1});
        nextOp = subOp.getLhs().getDefiningOp();
      } else {
        return std::nullopt;
      }
    } else if (auto mulOp = dyn_cast<stablehlo::MulOp>(currentOp)) {
      APInt scaleVal;
      if (matchPattern(mulOp.getRhs(), m_ConstantInt(&scaleVal))) {
        chain.push_back({currentOp, 0, scaleVal.getSExtValue()});
        nextOp = mulOp.getLhs().getDefiningOp();
      } else if (matchPattern(mulOp.getLhs(), m_ConstantInt(&scaleVal))) {
        chain.push_back({currentOp, 0, scaleVal.getSExtValue()});
        nextOp = mulOp.getRhs().getDefiningOp();
      } else {
        return std::nullopt;
      }
    } else { // unsupported op
      return std::nullopt;
    }

    currentOp = nextOp;
  }

  if (chain.empty())
    return std::nullopt;

  // process the base case
  IotaLikeTensor result;
  if (auto iotaOp = dyn_cast<stablehlo::IotaOp>(chain.back().op)) {
    auto iotaType = cast<RankedTensorType>(iotaOp.getResult().getType());
    auto iotaDim = static_cast<int64_t>(iotaOp.getIotaDimension());
    result = IotaLikeTensor{0, iotaDim, 1, iotaType};
  } else if (auto constantOp =
                 dyn_cast<stablehlo::ConstantOp>(chain.back().op)) {
    auto denseAttr = cast<DenseElementsAttr>(constantOp.getValue());
    auto constType = cast<RankedTensorType>(constantOp.getResult().getType());
    auto shape = constType.getShape();

    if (denseAttr.isSplat())
      return std::nullopt;

    // Calculate strides for indexing
    SmallVector<int64_t> strides(constType.getRank(), 1);
    for (int64_t i = constType.getRank() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    bool isIotaLike = false;
    auto denseAttrValues = denseAttr.getValues<APInt>();

    for (int64_t dim = 0; dim < constType.getRank(); dim++) {
      bool isIotaAlongDim = true;
      std::optional<int64_t> detectedStart;
      std::optional<int64_t> detectedScale;

      SmallVector<int64_t> indices(constType.getRank(), 0);
      int64_t numElements = constType.getNumElements();

      for (int64_t idx = 0; idx < numElements && isIotaAlongDim; idx++) {
        int64_t temp = idx;
        // linear to cartesian indexing
        for (int64_t d = 0; d < constType.getRank(); d++) {
          indices[d] = temp / strides[d];
          temp = temp % strides[d];
        }

        int64_t actualValue = denseAttrValues[idx].getSExtValue();

        if (!detectedStart) {
          detectedStart = actualValue;
        } else if (!detectedScale && indices[dim] == 1) {
          // Detect scale from the second element along this dimension
          detectedScale = actualValue - detectedStart.value();
          if (detectedScale.value() == 0) {
            // Scale of 0 means all values are the same, not an iota
            isIotaAlongDim = false;
            break;
          }
        }

        int64_t expectedValue =
            detectedStart.value() + indices[dim] * detectedScale.value_or(1);
        if (actualValue != expectedValue) {
          isIotaAlongDim = false;
          break;
        }
      }

      if (isIotaAlongDim && detectedStart) {
        isIotaLike = true;
        int64_t scale = detectedScale.value_or(1);
        result = IotaLikeTensor{detectedStart.value(), dim, scale, constType};
        break;
      }
    }

    if (!isIotaLike)
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  // traverse the chain in reverse order
  for (int64_t i = chain.size() - 2; i >= 0; i--) {
    auto item = chain[i];

    if (isa<stablehlo::ConvertOp>(item.op)) {
      continue;
    } else if (auto transposeOp = dyn_cast<stablehlo::TransposeOp>(item.op)) {
      auto permutation = transposeOp.getPermutation();
      for (int64_t idx = 0; idx < permutation.size(); idx++) {
        if (permutation[idx] == result.dimension) {
          result.dimension = idx;
          break;
        }
      }
      continue;
    } else if (isa<stablehlo::AddOp, stablehlo::SubtractOp>(item.op)) {
      result.start += item.offset;
      continue;
    } else if (isa<stablehlo::MulOp>(item.op)) {
      result.start *= item.scale;
      result.scale *= item.scale;
      continue;
    }

    assert(false && "reached unreachable case...");
  }

  result.tensorType = cast<RankedTensorType>(tensor.getType());
  return result;
}

bool allAccessesAreOnMainDiagonalPostReshape(stablehlo::ReshapeOp op,
                                             stablehlo::SliceOp sliceOp) {
  auto reshapeInTy = cast<RankedTensorType>(op.getOperand().getType());
  auto reshapeOutTy = cast<RankedTensorType>(op.getType());

  if (reshapeOutTy.getRank() != 1 ||
      reshapeInTy.getRank() != 2) // [M, N] -> [M * N] vector
    return false;

  auto M = reshapeInTy.getDimSize(0);
  auto N = reshapeInTy.getDimSize(1);
  auto diagLen = std::min(M, N);
  auto diagStride = N + 1;

  int64_t start = sliceOp.getStartIndices()[0];
  int64_t limit = sliceOp.getLimitIndices()[0];
  int64_t stride = sliceOp.getStrides()[0];

  if (stride % diagStride != 0)
    return false;

  // start can be on any of the diagonal elements
  if (start % diagStride != 0)
    return false;

  if (limit > M * N)
    return false; // technically this is illegal

  // sanity check
  int64_t count = (limit - start + stride - 1) / stride;
  if (count <= 0 || count > diagLen)
    return false;

  return true;
}

bool allAccessesAreOnMainDiagonalPostReshape(
    stablehlo::ReshapeOp op, Operation *user,
    llvm::SetVector<Operation *> &opsToReplace) {
  if (auto sliceOp = dyn_cast<stablehlo::SliceOp>(user)) {
    if (allAccessesAreOnMainDiagonalPostReshape(op, sliceOp)) {
      opsToReplace.insert(sliceOp);
      return true;
    }
    return false;
  }
  return false;
}

bool allAccessesAreOnMainDiagonal(Operation *op,
                                  llvm::SetVector<Operation *> &opsToReplace) {
  if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(op)) {
    return allAccessesAreOnMainDiagonal(reshapeOp, opsToReplace);
  } else if (auto gatherOp = dyn_cast<stablehlo::GatherOp>(op)) {
    return allAccessesAreOnMainDiagonal(gatherOp, opsToReplace);
  }
  return false;
}

bool allAccessesAreOnMainDiagonal(stablehlo::ReshapeOp op,
                                  llvm::SetVector<Operation *> &opsToReplace) {
  auto reshapeInTy = cast<RankedTensorType>(op.getOperand().getType());
  if (reshapeInTy.getRank() != 2) // [M, N] matrix
    return false;                 // quick exit

  llvm::SmallPtrSet<Operation *, 4> seenOps;
  for (auto user : op->getUsers()) {
    if (seenOps.count(user))
      continue;

    if (!allAccessesAreOnMainDiagonalPostReshape(op, user, opsToReplace))
      return false;

    seenOps.insert(user);
  }

  return true;
}

bool allAccessesAreOnMainDiagonal(stablehlo::GatherOp op,
                                  llvm::SetVector<Operation *> &opsToReplace) {
  return false; // TODO: implement this where we are doing gather with iota
}

} // namespace enzyme
} // namespace mlir
