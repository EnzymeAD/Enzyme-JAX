#include "src/enzyme_ad/jax/Passes/AutoBatching.h"

#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>

#define DEBUG_TYPE "auto-batching"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_AUTOBATCHINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

static int64_t batchCounter = 0;

namespace utils {

// This function checks if any 2 ops in the list are data-dependent on each
// other. We exploit the fact that while traversing the dep graph if we are at a
// position before the other ops in the set, we know that the other ops are not
// data dependent.
bool anyOpsAreDataDependent(ArrayRef<Operation *> ops) {
  if (ops.size() <= 1) {
    return false;
  }

  Block *parentBlock = ops[0]->getBlock();

  SmallVector<Operation *> todo;

  for (auto op : ops) {
    // dependency analysis for ops in different blocks is hard. conservatively
    // assume that all ops are data dependent
    if (op->getBlock() != parentBlock) {
      return true;
    }
    todo.push_back(op);
  }

  SmallPtrSet<Operation *, 1> toCheck(ops.begin(), ops.end());

  SmallPtrSet<Operation *, 1> done;

  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (done.contains(cur))
      continue;
    done.insert(cur);
    // Only consider operations whose values are isolated form above
    if (cur->getNumRegions() != 0 &&
        !cur->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
      SmallVector<Region *> rtodo;
      for (auto &reg : cur->getRegions()) {
        rtodo.push_back(&reg);
      }
      bool legal = true;
      while (!rtodo.empty()) {
        auto reg = rtodo.pop_back_val();
        for (auto &b : *reg) {
          for (auto &o : b) {
            for (auto v : o.getOperands()) {
              if (!cur->isAncestor(v.getParentBlock()->getParentOp())) {
                legal = false;
                goto endCheck;
              }
            }
            for (auto &reg : o.getRegions()) {
              rtodo.push_back(&reg);
            }
          }
        }
      }
    endCheck:;
      if (!legal)
        return true;
    }
    for (auto v : cur->getOperands()) {
      if (auto op2 = v.getDefiningOp()) {
        if (op2->getBlock() != parentBlock) {
          continue;
        }
        if (toCheck.contains(op2))
          return true;
        todo.push_back(op2);
      } else {
        // No blockargument can be in the list of ops, since it is
        // definitionally defined outside the block
        assert(isa<BlockArgument>(v));
        continue;
      }
    }
  }

  return false;
}

func::FuncOp CreateWrapperUnbatchedFunction(
    mlir::ModuleOp modOp, PatternRewriter &rewriter, std::string funcName,
    SmallVectorImpl<BatchLiftingMode> &batchLiftingModes, Operation *op,
    std::optional<SmallVector<int64_t>> outShape) {
  assert(op->getNumResults() == 1 && "only support single result ops");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(modOp.getBody());

  SmallVector<Type> argTypes;
  for (auto [i, v] : llvm::enumerate(op->getOperands())) {
    if (batchLiftingModes[i] == BatchLiftingMode::CONSTANT) {
      continue;
    }
    argTypes.push_back(v.getType());
  }

  FunctionType calleeType =
      rewriter.getFunctionType(argTypes, op->getResultTypes());
  func::FuncOp funcOp = func::FuncOp::create(
      rewriter, op->getLoc(), funcName + std::to_string(batchCounter++),
      calleeType);
  funcOp.setPrivate();

  auto &entryBlock = *funcOp.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  IRMapping mapper;
  size_t argIdx = 0;
  for (auto [batchLiftMode, operand] :
       llvm::zip(batchLiftingModes, op->getOperands())) {
    if (batchLiftMode == BatchLiftingMode::CONSTANT) { // clone into fn body
      auto clonedConst = rewriter.clone(*operand.getDefiningOp());
      mapper.map(operand, clonedConst->getResult(0));
      continue;
    }
    mapper.map(operand, entryBlock.getArguments()[argIdx++]);
  }

  Value result = rewriter.clone(*op, mapper)->getResult(0);
  if (outShape.has_value()) {
    result = stablehlo::ReshapeOpCreate(rewriter, op->getLoc(), result,
                                        outShape.value());
  }
  func::ReturnOp::create(rewriter, op->getLoc(), ValueRange(result));

  return funcOp;
}

void ConstructAndExtractBatchOperands(
    PatternRewriter &rewriter, ArrayRef<Operation *> batchOps, Location loc,
    std::optional<BatchOperandConstructionInfo<stablehlo::SliceOp>> batchInfo,
    SmallVectorImpl<Value> &operands,
    SmallVectorImpl<BatchLiftingMode> &liftingModes) {
  for (int i = 0; i < batchOps[0]->getNumOperands(); i++) {
    SmallVector<Value> currentOperands(batchOps.size());
    for (auto [idx, op] : llvm::enumerate(batchOps)) {
      currentOperands[idx] = op->getOperand(i);
    }

    // all the values are same
    SplatElementsAttr firstSplat;
    if (matchPattern(currentOperands[0], m_Constant(&firstSplat))) {
      if (llvm::all_of(currentOperands, [&](Value v) {
            SplatElementsAttr splatAttr;
            if (matchPattern(v, m_Constant(&splatAttr))) {
              return splatAttr == firstSplat;
            }
            return false;
          })) {
        liftingModes.push_back(BatchLiftingMode::CONSTANT);
        continue;
      }
    }

    if (batchInfo.has_value() && i == batchInfo->sliceOperandIndex) {
      SmallVector<Value> concatOperands(batchInfo->slices.size());
      for (auto [idx, slice] : llvm::enumerate(batchInfo->slices)) {
        assert(slice.dimensions.size() == 1);
        concatOperands[idx] = slice.sliceOp.getResult();
      }
      auto sliceDim = batchInfo->slices[0].dimensions[0];

      SmallVector<Value> newConcatOperands;
      auto canSimplify = stablehlo::concatSliceSimplify(
          rewriter, concatOperands, sliceDim, newConcatOperands);
      Value concatResult = stablehlo::ConcatenateOpCreate(
          rewriter, loc,
          canSimplify.succeeded() ? newConcatOperands : concatOperands,
          sliceDim);

      auto nbatchSize = static_cast<int64_t>(batchInfo->slices.size());
      auto resTy = cast<RankedTensorType>(concatResult.getType());
      assert(resTy.getDimSize(sliceDim) == nbatchSize);

      SmallVector<int64_t> permutation(resTy.getRank());
      permutation[0] = sliceDim;
      for (int64_t i = 0; i < sliceDim; i++) {
        permutation[i + 1] = i;
      }
      for (int64_t i = sliceDim + 1; i < resTy.getRank(); i++) {
        permutation[i] = i;
      }

      Value newOperand = stablehlo::TransposeOpCreate(
          rewriter, loc, concatResult, permutation);

      bool needsReshape = false;
      SmallVector<int64_t> outShape;
      if (batchInfo->intermediateReshape) {
        if (batchInfo->slices[0].explicitReshapeShape.has_value()) {
          needsReshape = true;
          outShape = batchInfo->slices[0].explicitReshapeShape.value();
          outShape.insert(outShape.begin(), nbatchSize);
        }
      } else {
        needsReshape = true;
        outShape =
            llvm::to_vector(cast<ShapedType>(newOperand.getType()).getShape());
        outShape.insert(outShape.begin() + sliceDim + 1, 1);
      }

      if (needsReshape) {
        newOperand =
            stablehlo::ReshapeOpCreate(rewriter, loc, newOperand, outShape);
      }

      liftingModes.push_back(BatchLiftingMode::DEFINED_OUTSIDE_WHILE);
      operands.push_back(newOperand);
    } else { // Non-equivalent operands - need to concatenate them
      SmallVector<Value> newConcatOperands;
      for (Value operand : currentOperands) {
        auto inputType = cast<RankedTensorType>(operand.getType());
        auto inputShape = inputType.getShape();

        SmallVector<int64_t> outputShape(inputShape.begin(), inputShape.end());
        outputShape.insert(outputShape.begin(), 1);

        // expand the batch dim (== 0) to `1`
        newConcatOperands.push_back(
            stablehlo::ReshapeOpCreate(rewriter, loc, operand, outputShape));
      }

      liftingModes.push_back(BatchLiftingMode::DEFINED_OUTSIDE_WHILE);
      operands.push_back(
          stablehlo::ConcatenateOpCreate(rewriter, loc, newConcatOperands, 0));
    }
  }
}

bool IsEquivalentToIgnoringValueEquivalence(Operation *op1, Operation *op2) {
  return OperationEquivalence::isEquivalentTo(
      op1, op2, OperationEquivalence::ignoreValueEquivalence, nullptr,
      OperationEquivalence::IgnoreLocations, nullptr);
}

bool allOpsAreUnique(const SmallVector<Operation *> &ops) {
  SmallPtrSet<Operation *, 8> seen;
  return llvm::all_of(ops,
                      [&](Operation *op) { return seen.insert(op).second; });
}

// dim == -1 => ignore the dimension check
bool CheckIsValidForBatching(
    stablehlo::ReshapeOp op, int64_t dim,
    llvm::SmallVectorImpl<int64_t> &intermediateInsertions,
    bool checkInsertion) {
  auto inTy = cast<RankedTensorType>(op.getOperand().getType());
  auto outTy = cast<RankedTensorType>(op.getType());

  if (!checkInsertion) {
    std::swap(inTy, outTy);
  }

  auto insertionDims = findReshapeInsertionDims(inTy, outTy);
  if (insertionDims.empty()) {
    return false;
  }

  if (dim == -1 || llvm::is_contained(insertionDims, dim)) {
    for (auto i : insertionDims) {
      if (i != dim) {
        intermediateInsertions.push_back(i);
      }
    }
    return true;
  }
  return false;
}

bool CheckIsValidForBatching(
    stablehlo::BroadcastInDimOp op, int64_t dim,
    llvm::SmallVectorImpl<int64_t> &intermediateInsertions,
    bool checkInsertion) {
  if (!checkInsertion) {
    return false; // bcast in dim cannot perform deletion
  }

  auto inputType = cast<RankedTensorType>(op.getOperand().getType());
  auto outputType = cast<RankedTensorType>(op.getType());

  // If concat dim is present in broadcast dims, then it is not a valid insert
  if ((dim != -1 && llvm::is_contained(op.getBroadcastDimensions(), dim)) ||
      !llvm::is_sorted(op.getBroadcastDimensions())) {
    return false;
  }

  // all bcasted dim but preserve size
  for (auto [i, bDim] : llvm::enumerate(op.getBroadcastDimensions())) {
    if (outputType.getDimSize(bDim) != inputType.getDimSize(i)) {
      return false;
    }
  }

  bool found = false;
  for (size_t i = 0; i < outputType.getRank(); i++) {
    if (!llvm::is_contained(op.getBroadcastDimensions(), i) &&
        outputType.getDimSize(i) == 1) {
      if (i == dim) {
        found = true;
        continue;
      }
      intermediateInsertions.push_back(i);
    }
  }
  return dim == -1 || found;
}

} // namespace utils

SliceInfo<stablehlo::SliceOp> constructSliceInfo(stablehlo::SliceOp sliceOp) {
  auto startIndices = llvm::to_vector(sliceOp.getStartIndices());
  auto limitIndices = llvm::to_vector(sliceOp.getLimitIndices());

  SmallVector<int64_t> dimensions;
  for (size_t i = 0; i < startIndices.size(); ++i) {
    if (startIndices[i] == limitIndices[i] - 1) {
      dimensions.push_back(i);
    }
  }

  if (dimensions.empty()) {
    return SliceInfo<stablehlo::SliceOp>();
  }

  return SliceInfo<stablehlo::SliceOp>{sliceOp, dimensions, false,
                                       std::nullopt};
}

void ComputeSliceDimension(ArrayRef<SliceInfo<mlir::stablehlo::SliceOp>> slices,
                           int64_t *sliceDim) {
  // find potential slice dimensions by constructing the intersection of all
  llvm::SmallSetVector<int64_t, 4> dimensions(slices[0].dimensions.begin(),
                                              slices[0].dimensions.end());
  for (auto &slice : llvm::drop_begin(slices)) {
    llvm::SmallSetVector<int64_t, 4> curDims(slice.dimensions.begin(),
                                             slice.dimensions.end());
    llvm::set_intersect(dimensions, curDims);
  }
  if (dimensions.empty()) {
    *sliceDim = -1;
    return;
  }

  // find the first dimension for which all conditions on the slice ops are
  // satisfied
  for (auto dim : dimensions) {
    auto baseSliceOp = slices[0].sliceOp;
    auto explicitReshapeShape = slices[0].explicitReshapeShape;

    for (auto slice : llvm::drop_begin(slices)) {
      auto curSliceOp = slice.sliceOp;

      if (explicitReshapeShape.has_value()) {
        if (!slice.explicitReshapeShape.has_value() ||
            explicitReshapeShape.value() !=
                slice.explicitReshapeShape.value()) {
          *sliceDim = -1;
          return;
        }
      }

      for (int64_t i = 0; i < baseSliceOp.getStartIndices().size(); ++i) {
        if (i == dim) {
          continue;
        }

        if (baseSliceOp.getStartIndices()[i] !=
                curSliceOp.getStartIndices()[i] ||
            baseSliceOp.getLimitIndices()[i] !=
                curSliceOp.getLimitIndices()[i] ||
            baseSliceOp.getStrides()[i] != curSliceOp.getStrides()[i]) {
          *sliceDim = -1;
          return;
        }
      }
    }

    *sliceDim = dim;
    return;
  }

  *sliceDim = -1;
  return;
}

LogicalResult ConcatInsertDimToBatchBase::matchAndRewriteImpl(
    stablehlo::ConcatenateOp concatOp, PatternRewriter &rewriter) const {
  if (concatOp.getNumOperands() <= 1) {
    return failure();
  }

  auto concatDim = concatOp.getDimension();
  auto concatType = cast<RankedTensorType>(concatOp.getResult().getType());
  auto concatShape = concatType.getShape();

  SmallVector<Operation *> concatOpOperands;
  SmallVector<int64_t> extraIntermediateInsertions;

  for (auto [i, v] : llvm::enumerate(concatOp.getOperands())) {
    auto definingOp = v.getDefiningOp();
    if (!definingOp) {
      return rewriter.notifyMatchFailure(concatOp, "operand is not a valid op");
    }

    SmallVector<int64_t> intermediateInsertions;
    bool validIntermediate =
        TypeSwitch<Operation *, bool>(definingOp)
            .Case<stablehlo::ReshapeOp, stablehlo::BroadcastInDimOp>(
                [&](auto op) {
                  return ::utils::CheckIsValidForBatching(
                      op, concatDim, intermediateInsertions, true);
                })
            .Default([](auto op) { return false; });

    if (i == 0) {
      extraIntermediateInsertions = std::move(intermediateInsertions);
    } else if (extraIntermediateInsertions != intermediateInsertions) {
      return rewriter.notifyMatchFailure(concatOp,
                                         "not all ops have same intermediate "
                                         "reshape");
    }

    if (!validIntermediate) {
      return rewriter.notifyMatchFailure(concatOp, "operand is not a valid op");
    }

    auto vdefOp = isValidTargetOp(definingOp->getOperand(0).getDefiningOp());
    if (!vdefOp) {
      return rewriter.notifyMatchFailure(concatOp, "not a valid target op");
    }

    if (concatOpOperands.size() != 0) {
      if (!::utils::IsEquivalentToIgnoringValueEquivalence(concatOpOperands[0],
                                                           vdefOp)) {
        return rewriter.notifyMatchFailure(concatOp,
                                           "op is not equivalent to first");
      }
    }

    if (!isOnlyUsedInOperation(vdefOp, definingOp)) {
      return rewriter.notifyMatchFailure(concatOp,
                                         "op is not only used in reshape op");
    }

    concatOpOperands.push_back(vdefOp);
  }

  SmallVector<Value> batchOpOperands;
  SmallVector<BatchLiftingMode> liftingModes;
  ::utils::ConstructAndExtractBatchOperands(rewriter, concatOpOperands,
                                            concatOp.getLoc(), std::nullopt,
                                            batchOpOperands, liftingModes);

  SmallVector<int64_t> outputShape;
  for (int i = 0; i < concatShape.size(); i++) {
    if (i == concatDim) {
      continue;
    }
    outputShape.push_back(concatShape[i]);
  }

  std::optional<SmallVector<int64_t>> explicitReshapeShape;
  if (!extraIntermediateInsertions.empty()) {
    explicitReshapeShape = outputShape;
  }
  auto moduleOp = concatOp->getParentOfType<ModuleOp>();
  func::FuncOp func = ::utils::CreateWrapperUnbatchedFunction(
      moduleOp, rewriter, "enzymexla_unbatched_ConcatInsertDimToBatch_",
      liftingModes, concatOpOperands[0], explicitReshapeShape);

  outputShape.insert(outputShape.begin(), concatShape[concatDim]);
  auto batchOp = enzyme::BatchOp::create(
      rewriter, concatOp.getLoc(),
      RankedTensorType::get(outputShape, concatType.getElementType()),
      mlir::FlatSymbolRefAttr::get(concatOp.getContext(), func.getName()),
      ValueRange(batchOpOperands),
      rewriter.getDenseI64ArrayAttr({concatShape[concatDim]}));

  SmallVector<int64_t> permutation;
  for (int i = 1; i <= concatDim; i++) {
    permutation.push_back(i);
  }
  permutation.push_back(0);
  for (int i = concatDim + 1; i < concatShape.size(); i++) {
    permutation.push_back(i);
  }

  rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
      concatOp, batchOp->getResult(0), permutation);

  enzyme::batchutils::batchOperationInline(
      rewriter, batchOp, cast<FunctionOpInterface>(func.getOperation()));
  return success();
}

LogicalResult
SliceToBatchBase::matchAndRewriteImpl(stablehlo::SliceOp sliceOp,
                                      PatternRewriter &rewriter) const {
  Value sliceInput = sliceOp.getOperand();
  auto block = sliceOp->getBlock();

  // Find all slices of the same input that feed into equivalent operations
  SmallVector<SliceInfo<stablehlo::SliceOp>> relatedSlices;
  SmallVector<Operation *> relatedOps;
  SmallVector<bool> allHaveIntermediateReshapes;
  Operation *targetOp = nullptr;
  int64_t sliceOperandIndex = -1;

  // Build worklist of all slice operations on the same input
  for (auto [idx, user] : llvm::enumerate(sliceInput.getUsers())) {
    auto candidateSlice = dyn_cast<stablehlo::SliceOp>(user);
    if (!candidateSlice || !candidateSlice.getResult().hasOneUse()) {
      continue;
    }

    auto sliceInfo = constructSliceInfo(candidateSlice);

    Operation *onlyUser = *candidateSlice.getResult().getUsers().begin();
    Operation *candidateTargetOp = isValidTargetOp(onlyUser);

    bool isIntermediateReshape = false;
    Operation *preceedingOp = candidateSlice;
    if (!candidateTargetOp) {
      SmallVector<int64_t> intermediateInsertions;
      isIntermediateReshape =
          TypeSwitch<Operation *, bool>(onlyUser)
              .Case<stablehlo::ReshapeOp, stablehlo::BroadcastInDimOp>(
                  [&](auto op) {
                    auto res = op.getResult();
                    preceedingOp = op;
                    if (res.hasOneUse() &&
                        ::utils::CheckIsValidForBatching(
                            op, -1, intermediateInsertions, false)) {
                      candidateTargetOp =
                          isValidTargetOp(*res.getUsers().begin());
                      if (candidateTargetOp) {
                        return true;
                      }
                      return true;
                    }
                    return false;
                  })
              .Default([](auto op) { return false; });

      if (isIntermediateReshape) {
        // resolve the intermediate reshape
        sliceInfo.intermediateReshape = true;
        sliceInfo.explicitReshapeShape = llvm::to_vector(
            cast<ShapedType>(preceedingOp->getResult(0).getType()).getShape());
      }
    }

    if (!candidateTargetOp || candidateTargetOp->getBlock() != block) {
      continue; // only consider ops in the same block
    }

    // check that all of the ops are equivalent and that the slice operand is
    // at the same location
    if (targetOp) {
      if (!::utils::IsEquivalentToIgnoringValueEquivalence(targetOp,
                                                           candidateTargetOp)) {
        continue;
      }
      if (candidateTargetOp->getOperand(sliceOperandIndex) !=
          preceedingOp->getResult(0)) {
        continue;
      }
    } else {
      for (auto [i, opOperand] :
           llvm::enumerate(candidateTargetOp->getOperands())) {
        if (opOperand == preceedingOp->getResult(0)) {
          sliceOperandIndex = i;
          break;
        }
      }
      targetOp = candidateTargetOp;
    }

    relatedSlices.push_back(sliceInfo);
    relatedOps.push_back(candidateTargetOp);
    allHaveIntermediateReshapes.push_back(isIntermediateReshape);
  }

  if (relatedSlices.size() <= 1) {
    return rewriter.notifyMatchFailure(sliceOp, "no related slices found");
  }

  if (!::utils::allOpsAreUnique(relatedOps)) {
    return rewriter.notifyMatchFailure(sliceOp, "ops are not unique");
  }

  if (sliceOperandIndex < 0) {
    return rewriter.notifyMatchFailure(sliceOp, "slice operand not found");
  }

  if (llvm::any_of(allHaveIntermediateReshapes, [=](bool b) {
        return b != allHaveIntermediateReshapes[0];
      })) {
    return rewriter.notifyMatchFailure(
        sliceOp, "slices have different intermediate reshape");
  }

  int64_t sliceDim = -1;
  ComputeSliceDimension(relatedSlices, &sliceDim);

  if (sliceDim < 0) {
    return rewriter.notifyMatchFailure(sliceOp, "slice dimension not found");
  }

  // Sort all vectors together based on sliceStart to ensure better locality
  SmallVector<size_t> indices(relatedSlices.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
    return relatedSlices[i].sliceOp.getStartIndices()[sliceDim] <
           relatedSlices[j].sliceOp.getStartIndices()[sliceDim];
  });

  // Reorder all three vectors according to the sorted indices
  SmallVector<SliceInfo<stablehlo::SliceOp>> sortedSlices;
  SmallVector<Operation *> sortedOps;

  for (size_t idx : indices) {
    sortedSlices.push_back(relatedSlices[idx]);
    sortedOps.push_back(relatedOps[idx]);
  }

  relatedSlices = std::move(sortedSlices);
  relatedOps = std::move(sortedOps);

  // quite an expensive check, so run at the very end
  if (::utils::anyOpsAreDataDependent(relatedOps)) {
    return rewriter.notifyMatchFailure(sliceOp, "ops are data dependent");
  }

  // Linear time algorithm to find first and last related ops:
  // - Build a set for O(1) lookup
  // - Single pass through block to find first and last
  llvm::SmallPtrSet<Operation *, 8> relatedOpsSet(relatedOps.begin(),
                                                  relatedOps.end());
  Operation *firstRelatedOp = nullptr;
  Operation *lastRelatedOp = nullptr;
  for (auto &op : *block) {
    if (relatedOpsSet.contains(&op)) {
      if (!firstRelatedOp) {
        firstRelatedOp = &op;
      }
      lastRelatedOp = &op;
    }
  }
  assert(firstRelatedOp && lastRelatedOp && "No related ops found in block");

  auto rangeBegin = firstRelatedOp->getIterator();
  auto rangeEnd = std::next(lastRelatedOp->getIterator());

  // First pass: collect all ops in the range and identify which are related
  llvm::SmallPtrSet<Operation *, 16> opsInRange;
  llvm::SetVector<Operation *> relatedOpsInRange;
  llvm::SmallVector<Operation *> nonRelatedOps;

  for (auto it = rangeBegin; it != rangeEnd; ++it) {
    opsInRange.insert(&*it);
    if (relatedOpsSet.contains(&*it)) {
      relatedOpsInRange.insert(&*it);
    } else {
      nonRelatedOps.push_back(&*it);
    }
  }

  // Use worklist to compute transitive "depends on related op" set
  // Start with ops that directly depend on related ops, then propagate to users
  llvm::SmallPtrSet<Operation *, 16> dependsOnRelated;
  llvm::SmallVector<Operation *> worklist;

  // Initialize worklist with non-related ops that directly depend on related
  // ops
  for (Operation *op : nonRelatedOps) {
    for (Value operand : op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (relatedOpsSet.contains(defOp)) {
          dependsOnRelated.insert(op);
          worklist.push_back(op);
          break;
        }
      }
    }
  }

  // Propagate: if op depends on related, all its users in range also depend
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Operation *user : op->getUsers()) {
      if (opsInRange.contains(user) && !relatedOpsSet.contains(user) &&
          !dependsOnRelated.contains(user)) {
        dependsOnRelated.insert(user);
        worklist.push_back(user);
      }
    }
  }

  // Partition non-related ops into preOps and postOps
  llvm::SetVector<Operation *> preOps;
  llvm::SetVector<Operation *> postOps;

  for (Operation *op : nonRelatedOps) {
    if (dependsOnRelated.contains(op)) {
      postOps.insert(op);
    } else {
      preOps.insert(op);
    }
  }

  // Sort each group topologically
  auto sortedPreOps = mlir::topologicalSort(preOps);
  auto sortedRelated = mlir::topologicalSort(relatedOpsInRange);
  auto sortedPostOps = mlir::topologicalSort(postOps);

  // Move all ops in order: preOps, then relatedOps, then postOps
  // Strategy: insert all ops just before rangeEnd in reverse order
  Operation *insertionPoint = &*rangeEnd;
  for (Operation *op : llvm::reverse(sortedPostOps)) {
    op->moveBefore(insertionPoint);
    insertionPoint = op;
  }
  for (Operation *op : llvm::reverse(sortedRelated)) {
    op->moveBefore(insertionPoint);
    insertionPoint = op;
  }
  for (Operation *op : llvm::reverse(sortedPreOps)) {
    op->moveBefore(insertionPoint);
    insertionPoint = op;
  }

  // The last related op is now the insertion point
  rewriter.setInsertionPoint(lastRelatedOp);

  for (auto &slice : relatedSlices) {
    slice.dimensions = {sliceDim};
  }

  SmallVector<Value> batchOpOperands;
  SmallVector<BatchLiftingMode> liftingModes;
  ::utils::ConstructAndExtractBatchOperands(
      rewriter, relatedOps, sliceOp.getLoc(),
      BatchOperandConstructionInfo<stablehlo::SliceOp>{
          relatedSlices, static_cast<int32_t>(sliceOperandIndex),
          allHaveIntermediateReshapes[0]},
      batchOpOperands, liftingModes);

  auto moduleOp = sliceOp->getParentOfType<ModuleOp>();
  func::FuncOp func = ::utils::CreateWrapperUnbatchedFunction(
      moduleOp, rewriter, "enzymexla_unbatched_SliceToBatch_", liftingModes,
      relatedOps[0], std::nullopt);

  SmallVector<int64_t> outputShape;
  outputShape.push_back(relatedSlices.size());
  auto relatedOpsType =
      cast<RankedTensorType>(relatedOps[0]->getResult(0).getType());
  auto funcRetShape = relatedOpsType.getShape();
  outputShape.append(funcRetShape.begin(), funcRetShape.end());

  auto batchOp = enzyme::BatchOp::create(
      rewriter, sliceOp.getLoc(),
      RankedTensorType::get(outputShape, relatedOpsType.getElementType()),
      mlir::FlatSymbolRefAttr::get(sliceOp.getContext(), func.getName()),
      ValueRange(batchOpOperands),
      rewriter.getDenseI64ArrayAttr(
          {static_cast<int64_t>(relatedSlices.size())}));

  SmallVector<int64_t> startIndices(outputShape.size(), 0);
  SmallVector<int64_t> endIndices;
  endIndices.append(outputShape.begin(), outputShape.end());
  SmallVector<int64_t> strides(outputShape.size(), 1);
  for (auto [idx, sliceInfoAndOp] :
       llvm::enumerate(llvm::zip(relatedSlices, relatedOps))) {
    auto &[sliceInfo, otherOp] = sliceInfoAndOp;
    startIndices[0] = idx;
    endIndices[0] = idx + 1;

    auto slicedOp = stablehlo::SliceOp::create(
        rewriter, sliceOp.getLoc(), batchOp->getResult(0),
        rewriter.getDenseI64ArrayAttr(startIndices),
        rewriter.getDenseI64ArrayAttr(endIndices),
        rewriter.getDenseI64ArrayAttr(strides));
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
        otherOp, otherOp->getResult(0).getType(), slicedOp);
  }

  enzyme::batchutils::batchOperationInline(
      rewriter, batchOp, cast<FunctionOpInterface>(func.getOperation()));
  return success();
}

static bool definedOutside(Value v, Operation *op) {
  return !op->isAncestor(v.getParentBlock()->getParentOp());
}

LogicalResult GreedyWhileLoopBatchFission::matchAndRewriteImpl(
    stablehlo::WhileOp whileOp, PatternRewriter &rewriter) const {
  auto info = WhileLoopInfo(whileOp);
  auto computeInfoSuccess = info.computeInfo();
  if (computeInfoSuccess.failed())
    return computeInfoSuccess;

  // TODO: the only actual restriction is that the total loop iterations be
  // constant and the indexing must be affine
  if (!info.isValid() || !info.isConstant())
    return failure();

  auto &whileBody = whileOp.getBody().front();

  auto affineIndexInfoMap = info.getAffineIndexInfo();

  auto parentFunc = whileOp->getParentOp();
  if (!parentFunc)
    return rewriter.notifyMatchFailure(whileOp, "parent function not found");

  // Find all dynamic slices in the loop body that meet the criteria:
  // 1. All slice variables are constant across iterations
  // 2. Only one variable in the body is a direct descendant of the induction
  // variable
  SmallPtrSet<Operation *, 8> seenOps;
  SmallVector<SliceInfo<stablehlo::DynamicSliceOp>> candidateSlices;

  for (auto [value, affineIndexInfo] : affineIndexInfoMap) {
    for (auto user : value.getUsers()) {
      if (user->getBlock() != &whileBody || seenOps.contains(user)) {
        continue;
      }

      seenOps.insert(user);

      if (auto sliceOp = dyn_cast<stablehlo::DynamicSliceOp>(user)) {
        auto result = isDynamicSliceValidForBatching(
            sliceOp, affineIndexInfoMap, whileBody, whileOp);

        if (isValidForBatchingResult(result.result)) {
          candidateSlices.push_back(SliceInfo<stablehlo::DynamicSliceOp>{
              sliceOp, result.dimensions, false, std::nullopt});
        }
      }
    }
  }

  // Create a map of user operations to their corresponding dynamic slices
  llvm::MapVector<Operation *,
                  SmallVector<SliceInfo<stablehlo::DynamicSliceOp>>>
      userOpToSlicesMap;
  for (auto ds : candidateSlices) {
    for (auto op : ds.sliceOp->getUsers()) {
      userOpToSlicesMap[op].push_back(ds);

      if (isa<stablehlo::ReshapeOp>(op)) {
        auto operandTy = cast<RankedTensorType>(op->getOperand(0).getType());
        auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());

        std::optional<SmallVector<int64_t>> reshapeShape;
        if (!areValidInsertionDims(resultTy, operandTy, ds.dimensions)) {
          reshapeShape = llvm::to_vector(resultTy.getShape());
        }

        for (auto user : op->getUsers()) {
          userOpToSlicesMap[user].push_back(
              SliceInfo<stablehlo::DynamicSliceOp>{ds.sliceOp, ds.dimensions,
                                                   true, reshapeShape});
        }
      }
    }
  }

  // for certain operations on index variables it is more efficient to hoist
  // those out of the loop and then perform indirect indexing
  for (auto &[val, slices] : affineIndexInfoMap) {
    for (auto user : val.getUsers()) {
      if (isa<stablehlo::CompareOp, stablehlo::BroadcastInDimOp>(user)) {
        userOpToSlicesMap[user].push_back(
            SliceInfo<stablehlo::DynamicSliceOp>{});
      }
    }
  }

  if (userOpToSlicesMap.empty())
    return failure();

  bool anyOpRewritten = false;

  for (auto &[op, slices] : userOpToSlicesMap) {
    bool avoidBatching =
        llvm::TypeSwitch<Operation *, bool>(op)
            .Case<stablehlo::DynamicSliceOp, stablehlo::DynamicUpdateSliceOp,
                  stablehlo::ReshapeOp>([=](auto op) { return true; })
            .Case<stablehlo::BroadcastInDimOp>(
                [=](auto op) { return stablehlo::broadcastInDimIsReshape(op); })
            .Default([](auto op) { return false; });
    if (avoidBatching) {
      continue;
    }

    if ((dyn_cast<BatchOpInterface>(op) ||
         stablehlo::hasTraitElementwise(op)) &&
        op->getNumResults() == 1) {
      if (liftOperationByBatching(rewriter, whileOp, slices, op, info)) {
        anyOpRewritten = true;
      } else if (liftReduceLikeOperation(rewriter, whileOp, slices, op, info)) {
        anyOpRewritten = true;
      }
    }
  }

  return anyOpRewritten ? success() : failure();
};

GreedyWhileLoopBatchFission::ValidBatchingInfo
GreedyWhileLoopBatchFission::isDynamicSliceValidForBatching(
    stablehlo::DynamicSliceOp sliceOp,
    llvm::MapVector<Value, mlir::enzyme::WhileLoopInfo::AffineIndexInfo>
        &affineIndexInfoMap,
    Block &whileBody, stablehlo::WhileOp whileOp) const {
  auto operand = sliceOp.getOperand();

  if (operand.getParentBlock() == &whileBody) {
    auto failureRetVal = ValidBatchingInfo{
        IsValidForBatchingResult::OPERAND_NOT_ACCESSIBLE_FROM_PARENT, {}};
    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      auto terminator = whileBody.getTerminator();
      if (!terminator ||
          terminator->getOperand(blockArg.getArgNumber()) != operand) {
        return failureRetVal;
      }
    } else {
      return failureRetVal;
    }
  }

  SmallVector<int64_t> dimensions;
  auto sliceSizes = sliceOp.getSliceSizes();

  for (auto [i, startIndex] : llvm::enumerate(sliceOp.getStartIndices())) {
    if (affineIndexInfoMap.contains(startIndex) && sliceSizes[i] == 1) {
      dimensions.push_back(i);
      continue;
    }

    if (definedOutside(startIndex, whileOp))
      continue;

    return ValidBatchingInfo{IsValidForBatchingResult::DYNAMIC_START_INDEX, {}};
  }

  if (dimensions.empty())
    return ValidBatchingInfo{
        IsValidForBatchingResult::NO_INDUCTION_VARIABLE_DETECTED, {}};

  // We should have exactly one index from the body, and it should be
  // a descendant of the induction variable
  return ValidBatchingInfo{IsValidForBatchingResult::VALID, dimensions};
}

bool traverseOperandsForHoisting(
    ArrayRef<Value> operands, stablehlo::WhileOp whileOp,
    ArrayRef<SliceInfo<stablehlo::DynamicSliceOp>> slices, WhileLoopInfo &info,
    SmallVectorImpl<BatchLiftingMode> &batchLiftingModes,
    SmallVectorImpl<Value> &batchOperands,
    SmallVectorImpl<SmallVector<int64_t>> &sliceDims,
    SmallVectorImpl<int64_t> &hoistedDims,
    SmallVectorImpl<SliceInfo<stablehlo::DynamicSliceOp>> &mappedSliceInfos,
    DenseMap<Value, SmallVector<Operation *>> &hoistMap) {
  auto affineIndexInfoMap = info.getAffineIndexInfo();

  batchLiftingModes.resize(operands.size());
  batchOperands.resize(operands.size());
  sliceDims.resize(operands.size());
  hoistedDims.resize(operands.size());
  mappedSliceInfos.resize(operands.size());

  for (auto [i, operand] : llvm::enumerate(operands)) {
    Value outerValue;
    SmallVector<Operation *> canBeHoisted;
    if (info.isConstantAcrossIterations(operand, outerValue, canBeHoisted)) {
      if (outerValue) {
        SplatElementsAttr splat;
        if (matchPattern(operand, m_Constant(&splat))) {
          batchLiftingModes[i] = BatchLiftingMode::CONSTANT;
        } else {
          batchLiftingModes[i] = BatchLiftingMode::DEFINED_OUTSIDE_WHILE;
        }
        batchOperands[i] = outerValue;
      } else {
        hoistMap[operand] = canBeHoisted;
        hoistedDims[i] = cast<mlir::OpResult>(operand).getResultNumber();
        batchLiftingModes[i] = BatchLiftingMode::NEEDS_HOISTING_OUTSIDE_WHILE;
        batchOperands[i] = operand;
      }
      continue;
    }

    if (affineIndexInfoMap.contains(operand) &&
        !cast<RankedTensorType>(operand.getType())
             .getElementType()
             .isInteger(1)) {
      batchLiftingModes[i] = BatchLiftingMode::AFFINE_INDEX;
      batchOperands[i] = operand;
      continue;
    }

    auto defOp = operand.getDefiningOp();
    if (!defOp) {
      return false;
    }

    Operation *dsOp;
    bool mustBeIntermediateReshape = false;
    if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(defOp)) {
      mustBeIntermediateReshape = true;
      dsOp = reshapeOp.getOperand().getDefiningOp();
    } else {
      dsOp = defOp;
    }

    if (auto ds = dyn_cast<stablehlo::DynamicSliceOp>(dsOp)) {
      auto itr = llvm::find_if(
          slices, [&](const SliceInfo<stablehlo::DynamicSliceOp> &info) {
            return info.sliceOp == ds;
          });
      if (itr != slices.end()) {
        batchLiftingModes[i] = BatchLiftingMode::DYNAMIC_SLICE;
        sliceDims[i] = itr->dimensions;

        auto dsOperand = ds->getOperand(0);
        if (definedOutside(dsOperand, whileOp)) {
          batchOperands[i] = dsOperand;
        } else {
          auto blockArg = dyn_cast<BlockArgument>(dsOperand);
          assert(blockArg && "expected block arg");
          batchOperands[i] = whileOp->getOperand(blockArg.getArgNumber());
        }

        mappedSliceInfos[i] = *itr;
        if (mustBeIntermediateReshape && !itr->intermediateReshape) {
          return false;
        }
        continue;
      } else {
        return false;
      }
    }

    return false;
  }

  return true;
}

void hoistChainOfOps(DenseMap<Value, SmallVector<Operation *>> &hoistMap,
                     PatternRewriter &rewriter, stablehlo::WhileOp whileOp,
                     WhileLoopInfo &info,
                     DenseMap<Value, Value> &hoistedValues) {
  IRMapping mapper;
  for (auto &[val, ops] : hoistMap) {
    llvm::SetVector<Operation *> toHoist(ops.begin(), ops.end());
    auto sorted = mlir::topologicalSort(toHoist);

    for (auto &op : sorted) {
      if (llvm::all_of(op->getResults(),
                       [&](Value v) { return mapper.contains(v); }))
        continue;

      for (auto operand : op->getOperands()) {
        if (mapper.contains(operand)) {
          continue;
        }

        if (!definedOutside(operand, whileOp)) {
          Value outerValue;
          SmallVector<Operation *> canBeHoisted;
          if (info.isConstantAcrossIterations(operand, outerValue, canBeHoisted,
                                              false)) {
            mapper.map(operand, outerValue);
          }
        }
      }
      auto hoisted = rewriter.clone(*op, mapper);
      for (auto [origRes, newRes] :
           llvm::zip(op->getResults(), hoisted->getResults())) {
        mapper.map(origRes, newRes);
      }
    }

    hoistedValues[val] = mapper.lookup(val);
  }
}

void constructNewOperandsForHoistedOp(
    PatternRewriter &rewriter, stablehlo::WhileOp whileOp, WhileLoopInfo &info,
    SmallVectorImpl<BatchLiftingMode> &batchLiftingModes,
    SmallVectorImpl<Value> &batchOperands,
    SmallVectorImpl<SmallVector<int64_t>> &sliceDims,
    SmallVectorImpl<int64_t> &hoistedDims,
    SmallVectorImpl<SliceInfo<stablehlo::DynamicSliceOp>> &mappedSliceInfos,
    DenseMap<Value, Value> &hoistedValues,
    SmallVectorImpl<Value> &newOperands) {
  newOperands.clear();

  for (auto [consType, baseOp, sliceDim, sliceInfo, hoistDim] :
       llvm::zip(batchLiftingModes, batchOperands, sliceDims, mappedSliceInfos,
                 hoistedDims)) {
    auto operandType = cast<RankedTensorType>(baseOp.getType());
    int operandRank = cast<RankedTensorType>(baseOp.getType()).getRank();

    switch (consType) {
    case BatchLiftingMode::DYNAMIC_SLICE: {
      // hoist the dynamic slice out of the loop and replace the sliceDim
      // with full slice.
      Value newSlice;
      bool successfulHoist = info.hoistOperationFromLoop(
          rewriter, baseOp, sliceInfo.sliceOp, sliceDim, newSlice);
      assert(successfulHoist && "Expected DS hoist to succeed");
      (void)successfulHoist;
      auto originalShape =
          cast<RankedTensorType>(sliceInfo.sliceOp.getType()).getShape();

      auto DSType = cast<RankedTensorType>(newSlice.getType());
      SmallVector<int64_t> permutation(DSType.getRank());
      permutation[0] = sliceDim[0];
      for (size_t i = 0; i < sliceDim[0]; i++)
        permutation[i + 1] = i;
      for (size_t i = sliceDim[0] + 1; i < DSType.getRank(); i++)
        permutation[i] = i;

      Value newOperand = stablehlo::TransposeOpCreate(
          rewriter, whileOp->getLoc(), newSlice, permutation);

      bool applyReshape = true;
      SmallVector<int64_t> reshapeShape;
      if (sliceInfo.intermediateReshape) {
        if (sliceInfo.explicitReshapeShape.has_value()) {
          reshapeShape = sliceInfo.explicitReshapeShape.value();
          reshapeShape.insert(reshapeShape.begin(), info.getConstantNumIters());
        } else {
          applyReshape = false;
        }
      } else {
        reshapeShape = llvm::to_vector(originalShape);
        reshapeShape.insert(reshapeShape.begin(), info.getConstantNumIters());
        for (auto dim : sliceDim)
          reshapeShape[dim + 1] = 1;
      }

      if (applyReshape) {
        newOperand = stablehlo::ReshapeOpCreate(rewriter, whileOp->getLoc(),
                                                newOperand, reshapeShape);
      }

      newOperands.push_back(newOperand);
      break;
    }
    case BatchLiftingMode::NEEDS_HOISTING_OUTSIDE_WHILE: {
      baseOp = hoistedValues[baseOp];
      LLVM_FALLTHROUGH;
    }
    case BatchLiftingMode::DEFINED_OUTSIDE_WHILE: {
      auto operandShape = operandType.getShape();
      SmallVector<int64_t> newOperandShape(operandRank + 1);
      newOperandShape[0] = info.getConstantNumIters();
      for (int i = 0; i < operandRank; i++)
        newOperandShape[i + 1] = operandShape[i];

      SmallVector<int64_t> mapping(operandRank);
      std::iota(mapping.begin(), mapping.end(), 1);

      auto broadcastedOperand = stablehlo::BroadcastInDimOp::create(
          rewriter, whileOp->getLoc(),
          RankedTensorType::get(newOperandShape, operandType.getElementType()),
          baseOp, rewriter.getDenseI64ArrayAttr(mapping));
      newOperands.push_back(broadcastedOperand->getResult(0));
      break;
    }
    case BatchLiftingMode::CONSTANT: {
      break; // copied into the function body no need to include in operands
    }
    case BatchLiftingMode::AFFINE_INDEX: {
      SmallVector<int64_t> loopIndicesShape(operandRank + 1, 1);
      loopIndicesShape[0] = info.getConstantNumIters();

      auto hoistedTy =
          RankedTensorType::get(loopIndicesShape, operandType.getElementType());
      Value loopIndices =
          stablehlo::IotaOp::create(rewriter, whileOp->getLoc(), hoistedTy, 0);

      auto createConst = [&](int64_t val) {
        return stablehlo::ConstantOp::create(
            rewriter, whileOp->getLoc(), hoistedTy,
            cast<ElementsAttr>(makeAttr(hoistedTy, val)));
      };

      auto startVal = createConst(info.getConstantStart().value());
      auto stepVal = createConst(info.getConstantStep().value());
      loopIndices = stablehlo::AddOp::create(
          rewriter, whileOp->getLoc(), loopIndices,
          stablehlo::MulOp::create(rewriter, whileOp->getLoc(), stepVal,
                                   startVal));

      auto affineIndexInfo = info.getAffineIndexInfo()[baseOp];
      auto scale = createConst(affineIndexInfo.scale.getSExtValue());
      auto offset = createConst(affineIndexInfo.offset.getSExtValue());
      auto res = stablehlo::AddOp::create(
          rewriter, whileOp->getLoc(),
          stablehlo::MulOp::create(rewriter, whileOp->getLoc(), scale,
                                   loopIndices),
          offset);
      newOperands.push_back(res);
      break;
    }
    }
  }
}

bool liftReduceLikeOperation(
    PatternRewriter &rewriter, stablehlo::WhileOp whileOp,
    ArrayRef<SliceInfo<stablehlo::DynamicSliceOp>> slices, Operation *op,
    WhileLoopInfo info) {
  // we can hoist `sub` / `div` by emitting a `neg` / `reciprocal` and then
  // apply the hoisting. note that this only applies if the LHS is the loop
  // caried dependency
  bool specialOps = isa<stablehlo::SubtractOp, stablehlo::DivOp>(op);
  if (!specialOps && !stablehlo::canFuseIntoReduce(op)) {
    return false;
  }

  auto result = op->getResult(0);
  if (!llvm::hasSingleElement(result.getUsers())) {
    return false;
  }

  auto returnOp = dyn_cast<stablehlo::ReturnOp>(*result.getUsers().begin());
  if (!returnOp || returnOp != whileOp.getBody().front().getTerminator()) {
    return false;
  }

  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);

  bool isLhsLoopCarriedDep = false, isRhsLoopCarriedDep = false;
  int64_t argIdx;
  if (auto lhsBlockArg = dyn_cast<BlockArgument>(lhs)) {
    if (lhsBlockArg.getOwner() == &whileOp.getBody().front() &&
        returnOp->getOperand(lhsBlockArg.getArgNumber()) == result) {
      argIdx = lhsBlockArg.getArgNumber();
      isLhsLoopCarriedDep = true;
    }
  }
  if (auto rhsBlockArg = dyn_cast<BlockArgument>(rhs)) {
    if (rhsBlockArg.getOwner() == &whileOp.getBody().front() &&
        returnOp->getOperand(rhsBlockArg.getArgNumber()) == result) {
      argIdx = rhsBlockArg.getArgNumber();
      isRhsLoopCarriedDep = true;
    }
  }

  // while dead args is needed to clean this up
  if (whileOp->getResult(argIdx).getUsers().empty()) {
    return false;
  }

  if (isLhsLoopCarriedDep == isRhsLoopCarriedDep) {
    return false; // atmost one of lhs/rhs must be loop carried dep
  }
  if (specialOps && isRhsLoopCarriedDep) { // only lhs can be loop carried dep
    return false;
  }

  Value otherOperand = isLhsLoopCarriedDep ? rhs : lhs;

  SmallVector<BatchLiftingMode> batchLiftingModes;
  SmallVector<Value> batchOperands;
  SmallVector<SmallVector<int64_t>> sliceDims;
  SmallVector<int64_t> hoistedDims;
  SmallVector<SliceInfo<stablehlo::DynamicSliceOp>> mappedSliceInfos;
  DenseMap<Value, SmallVector<Operation *>> hoistMap;

  SmallVector<Value> opOperands = {otherOperand};

  if (!traverseOperandsForHoisting(opOperands, whileOp, slices, info,
                                   batchLiftingModes, batchOperands, sliceDims,
                                   hoistedDims, mappedSliceInfos, hoistMap)) {
    return false;
  }

  rewriter.setInsertionPoint(whileOp);
  DenseMap<Value, Value> hoistedValues;
  hoistChainOfOps(hoistMap, rewriter, whileOp, info, hoistedValues);

  SmallVector<Value> newOperands;
  constructNewOperandsForHoistedOp(
      rewriter, whileOp, info, batchLiftingModes, batchOperands, sliceDims,
      hoistedDims, mappedSliceInfos, hoistedValues, newOperands);

  auto whileOperand = whileOp->getOperand(argIdx);

  auto elemType =
      cast<RankedTensorType>(whileOperand.getType()).getElementType();

  Value reduceInput = newOperands[0];
  OperationName opName = op->getName();
  if (specialOps) {
    if (isa<stablehlo::SubtractOp>(op)) {
      reduceInput =
          stablehlo::NegOp::create(rewriter, op->getLoc(), reduceInput);
      opName = OperationName("stablehlo.add", op->getContext());
    } else if (isa<stablehlo::DivOp>(op)) {
      auto numerator = stablehlo::ConstantOp::create(
          rewriter, op->getLoc(), rewriter.getOneAttr(reduceInput.getType()));
      reduceInput = stablehlo::DivOp::create(rewriter, op->getLoc(), numerator,
                                             reduceInput);
      opName = OperationName("stablehlo.multiply", op->getContext());
    } else {
      llvm_unreachable("unhandled special op");
    }
  }

  Value initVal;

  if (specialOps) {
    TypeSwitch<Operation *>(op)
        .Case<stablehlo::SubtractOp>([&](auto op) {
          initVal = stablehlo::getIdentityValueForOp<stablehlo::AddOp>(
              rewriter, op->getLoc(), elemType);
        })
        .Case<stablehlo::DivOp>([&](auto op) {
          initVal = stablehlo::getIdentityValueForOp<stablehlo::MulOp>(
              rewriter, op->getLoc(), elemType);
        });
  } else {
    initVal = stablehlo::getIdentityValue(
        rewriter, op->getLoc(),
        cast<RankedTensorType>(otherOperand.getType()).getElementType(), op);
  }

  auto reduceOp = stablehlo::ReduceOp::create(
      rewriter, whileOp->getLoc(),
      TypeRange{whileOp->getResult(argIdx).getType()}, ValueRange{reduceInput},
      ValueRange{initVal}, rewriter.getDenseI64ArrayAttr({0}));

  auto scalarType = RankedTensorType::get({}, elemType);
  Block *block = rewriter.createBlock(&reduceOp.getBody());
  block->addArgument(scalarType, whileOp->getLoc());
  block->addArgument(scalarType, whileOp->getLoc());

  {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    OperationState state(op->getLoc(), opName);
    state.addTypes(TypeRange{scalarType});
    state.addOperands(ValueRange{block->getArgument(0), block->getArgument(1)});
    // Create the operation from the state.
    auto *newOp = mlir::Operation::create(state);
    rewriter.insert(newOp);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), newOp->getResults());
  }

  rewriter.setInsertionPointAfter(reduceOp);
  OperationState finalResState(whileOp->getLoc(), opName);
  finalResState.addTypes(TypeRange{whileOp->getResult(argIdx).getType()});
  finalResState.addOperands(ValueRange{reduceOp->getResult(0), whileOperand});
  auto *finalResOp = mlir::Operation::create(finalResState);
  rewriter.insert(finalResOp);

  rewriter.replaceAllUsesWith(whileOp->getResult(argIdx),
                              finalResOp->getResult(0));
  return true;
}

bool liftOperationByBatching(
    PatternRewriter &rewriter, stablehlo::WhileOp whileOp,
    ArrayRef<SliceInfo<stablehlo::DynamicSliceOp>> slices, Operation *op,
    WhileLoopInfo info) {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  auto affineIndexInfoMap = info.getAffineIndexInfo();

  SmallVector<BatchLiftingMode> batchLiftingModes;
  SmallVector<Value> batchOperands;
  SmallVector<SmallVector<int64_t>> sliceDims;
  SmallVector<int64_t> hoistedDims;
  SmallVector<SliceInfo<stablehlo::DynamicSliceOp>> mappedSliceInfos;
  DenseMap<Value, SmallVector<Operation *>> hoistMap;

  auto opOperands = llvm::to_vector(op->getOperands());
  if (!traverseOperandsForHoisting(opOperands, whileOp, slices, info,
                                   batchLiftingModes, batchOperands, sliceDims,
                                   hoistedDims, mappedSliceInfos, hoistMap)) {
    return false;
  }

  func::FuncOp func = ::utils::CreateWrapperUnbatchedFunction(
      moduleOp, rewriter, "enzymexla_unbatched_WhileLoopBatchFission_",
      batchLiftingModes, op, std::nullopt);

  rewriter.setInsertionPoint(whileOp);

  // hoist any operations that can be hoisted
  DenseMap<Value, Value> hoistedValues;
  hoistChainOfOps(hoistMap, rewriter, whileOp, info, hoistedValues);

  SmallVector<Value> newOperands;
  constructNewOperandsForHoistedOp(
      rewriter, whileOp, info, batchLiftingModes, batchOperands, sliceDims,
      hoistedDims, mappedSliceInfos, hoistedValues, newOperands);

  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  auto resultShape = resultType.getShape();
  SmallVector<int64_t> outputShape(resultShape.size() + 1);
  outputShape[0] = info.getConstantNumIters();
  for (int i = 0; i < resultShape.size(); i++)
    outputShape[i + 1] = resultShape[i];

  auto inductionVar = info.getInductionVariable();
  auto inductionVarType = cast<RankedTensorType>(inductionVar.getType());

  auto constZero = stablehlo::ConstantOp::create(
      rewriter, whileOp->getLoc(), inductionVarType,
      cast<ElementsAttr>(makeAttr(inductionVarType, 0)));

  auto batchOp = enzyme::BatchOp::create(
      rewriter, whileOp->getLoc(),
      RankedTensorType::get(outputShape, resultType.getElementType()),
      mlir::FlatSymbolRefAttr::get(func.getContext(), func.getName()),
      ValueRange(newOperands),
      rewriter.getDenseI64ArrayAttr({info.getConstantNumIters()}));

  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> dynamicSliceStarts(outputShape.size(), constZero);
  Value resIndex = stablehlo::SubtractOp::create(rewriter, whileOp->getLoc(),
                                                 info.getInductionVariable(),
                                                 info.getStart());
  if (!info.isStepOne()) {
    resIndex = stablehlo::DivOp::create(rewriter, whileOp->getLoc(), resIndex,
                                        info.getStep(rewriter));
  }
  dynamicSliceStarts[0] = resIndex;

  SmallVector<int64_t> dynamicSliceSizes(outputShape.begin(),
                                         outputShape.end());
  dynamicSliceSizes[0] = 1;

  auto dynamicSlice = stablehlo::DynamicSliceOp::create(
      rewriter, whileOp->getLoc(), batchOp->getResult(0), dynamicSliceStarts,
      dynamicSliceSizes);
  rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
      op, op->getResult(0).getType(), dynamicSlice);

  enzyme::batchutils::batchOperationInline(
      rewriter, batchOp, cast<FunctionOpInterface>(func.getOperation()));

  return true;
}

mlir::LogicalResult WhileElementwiseReductionToReduce::matchAndRewriteImpl(
    stablehlo::WhileOp whileOp, PatternRewriter &rewriter) const {
  auto &body = whileOp.getBody().front();
  auto term = body.getTerminator();
  if (!term) {
    return failure();
  }
  auto returnOp = dyn_cast<stablehlo::ReturnOp>(term);
  if (!returnOp) {
    return failure();
  }

  WhileLoopInfo info(whileOp);
  auto computedInfo = info.computeInfo();
  (void)computedInfo;
  if (!info.isValid()) {
    return failure();
  }

  bool anyRewritten = false;
  SmallVector<SliceInfo<stablehlo::DynamicSliceOp>> slices; // dummy

  for (size_t i = 0; i < whileOp.getNumOperands(); i++) {
    auto iterArg = body.getArgument(i);
    if (!llvm::hasSingleElement(iterArg.getUsers())) {
      continue;
    }
    auto user = *iterArg.getUsers().begin();

    if (!stablehlo::hasTraitElementwise(user) && user->getNumOperands() != 2 &&
        user->getNumResults() != 1) {
      continue;
    }

    if (user->getResult(0) != returnOp.getOperand(i)) {
      continue;
    }

    anyRewritten |=
        liftReduceLikeOperation(rewriter, whileOp, slices, user, info);
  }

  return success(anyRewritten);
}

struct AutoBatchingPass
    : public enzyme::impl::AutoBatchingPassBase<AutoBatchingPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    if (concat_insert_dim_passes) {
      patterns.add<ConcatInsertDimToBatch<stablehlo::DotGeneralOp>,
                   ConcatInsertDimToBatch<stablehlo::GatherOp>,
                   ConcatInsertDimToBatch<stablehlo::IotaOp>,
                   ConcatInsertDimToBatch<stablehlo::ReduceOp>,
                   // ConcatInsertDimToBatch<stablehlo::ScatterOp>, after batch
                   // op interface is implemented
                   ConcatInsertDimToBatch<stablehlo::SortOp>,
                   ConcatInsertDimToBatch<stablehlo::ReduceWindowOp>,
                   ConcatInsertDimToBatch<stablehlo::ConcatenateOp>,
                   ConcatInsertDimToBatch<stablehlo::GetDimensionSizeOp>,
                   ConcatInsertDimToBatch<stablehlo::ReverseOp>,
                   ConcatInsertDimToBatch<stablehlo::ReduceWindowOp>,
                   ConcatInsertDimToBatch<stablehlo::ConvolutionOp>,
                   ConcatInsertDimElementwiseToBatch>(context);
    }

    if (slice_to_batch_passes) {
      patterns.add<
          SliceToBatch<stablehlo::DotGeneralOp>,
          SliceToBatch<stablehlo::GatherOp>, SliceToBatch<stablehlo::IotaOp>,
          SliceToBatch<stablehlo::ReduceOp>, SliceToBatch<stablehlo::SortOp>,
          SliceToBatch<stablehlo::TransposeOp>,
          SliceToBatch<stablehlo::ReduceWindowOp>,
          SliceToBatch<stablehlo::ConcatenateOp>,
          SliceToBatch<stablehlo::GetDimensionSizeOp>,
          SliceToBatch<stablehlo::ReverseOp>,
          SliceToBatch<stablehlo::ReduceWindowOp>,
          SliceToBatch<stablehlo::ConvolutionOp>, SliceToBatchBroadcastInDim,
          SliceToBatchElementwise>(context);
    }

    if (while_loop_batching_mode == "greedy") {
      patterns.add<GreedyWhileLoopBatchFission>(context);
    } else if (while_loop_batching_mode != "none") {
      llvm::errs() << "Unknown while loop batching mode: "
                   << while_loop_batching_mode << "\n";
      signalPassFailure();
    }

    patterns.add<WhileElementwiseReductionToReduce>(context);

    GreedyRewriteConfig config;
    config.setMaxIterations(max_iterations);
    config.setUseTopDownTraversal(top_down);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
