#include "src/enzyme_ad/jax/Passes/AutoBatching.h"

#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
#include <llvm/ADT/STLExtras.h>
#include <numeric>
#include <tuple>

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

// The values the regions of `ops` read from outside `ops`. A clone of the
// ops into a wrapper function keeps those references, which then reach
// across the function boundary: a constant can be cloned alongside, anything
// else cannot be taken.
static void collectRegionCaptures(ArrayRef<Operation *> ops,
                                  SmallVectorImpl<Value> &captures) {
  llvm::SmallPtrSet<Operation *, 4> roots(ops.begin(), ops.end());
  DenseSet<Value> seen;
  for (Operation *op : ops) {
    for (Region &region : op->getRegions()) {
      region.walk([&](Operation *inner) {
        for (Value v : inner->getOperands()) {
          if (!seen.insert(v).second)
            continue;
          Operation *def = v.getDefiningOp();
          Region *home = def ? def->getParentRegion() : v.getParentRegion();
          bool inside = false;
          for (Region *r = home; r; r = r->getParentRegion())
            if (roots.contains(r->getParentOp())) {
              inside = true;
              break;
            }
          if (!inside)
            captures.push_back(v);
        }
      });
    }
  }
}

bool regionsCaptureOnlyConstants(ArrayRef<Operation *> ops) {
  SmallVector<Value> captures;
  collectRegionCaptures(ops, captures);
  return llvm::all_of(captures, [](Value v) {
    return v.getDefiningOp<stablehlo::ConstantOp>() != nullptr;
  });
}

bool regionsCaptureOnlyConstants(Operation *op) {
  return regionsCaptureOnlyConstants(ArrayRef<Operation *>(op));
}

func::FuncOp CreateWrapperUnbatchedFunction(
    mlir::ModuleOp modOp, PatternRewriter &rewriter, std::string funcName,
    std::optional<SmallVector<BatchLiftingMode>> batchLiftingModes,
    ArrayRef<Operation *> ops, std::optional<SmallVector<int64_t>> inShape,
    std::optional<SmallVector<int64_t>> outShape, FunctionType calleeType) {
  // A region that reads a value from outside the ops keeps reading it from
  // inside the wrapper: a constant is cloned in, anything else refuses the
  // wrapper, and the callers decide this before they build anything.
  SmallVector<Value> captures;
  collectRegionCaptures(ops, captures);
  for (Value v : captures)
    if (!v.getDefiningOp<stablehlo::ConstantOp>())
      return nullptr;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(modOp.getBody());

  auto lastOp = ops.back();
  auto firstOp = ops.front();

  func::FuncOp funcOp = func::FuncOp::create(
      rewriter, lastOp->getLoc(), funcName + std::to_string(batchCounter++),
      calleeType);
  funcOp.setPrivate();

  auto &entryBlock = *funcOp.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  // Wrapper arguments correspond positionally to the non-CONSTANT operand
  // slots of `firstOp`. If a value occupies several slots (`multiply %x, %x`)
  // the last slot's argument wins; that is sound because the matcher only
  // batches ops with the same repetition structure, so those slots receive
  // identical batched operands.
  IRMapping mapper;
  size_t argIdx = 0;
  for (auto [i, operand] : llvm::enumerate(firstOp->getOperands())) {
    Value mapped;
    if (batchLiftingModes.has_value() &&
        batchLiftingModes.value()[i] ==
            BatchLiftingMode::CONSTANT) { // clone into fn body
      mapped = rewriter.clone(*operand.getDefiningOp())->getResult(0);
    } else {
      mapped = entryBlock.getArguments()[argIdx++];
    }
    if (inShape.has_value()) {
      mapped = stablehlo::ReshapeOpCreate(rewriter, firstOp->getLoc(), mapped,
                                          inShape.value());
    }
    mapper.map(operand, mapped);
  }

  for (auto op : ops) {
    auto clonedOp = rewriter.clone(*op, mapper);
    // A constant the regions read from outside is cloned into the block
    // that reads it, so the region stands on its own: a batch interface
    // clones a region with a mapping of its own, and a value the region
    // took from the wrapper's body would be left behind when the wrapper
    // is inlined and erased.
    for (Region &region : clonedOp->getRegions()) {
      for (Block &block : region) {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(&block);
        DenseMap<Value, Value> local;
        block.walk([&](Operation *inner) {
          for (OpOperand &use : inner->getOpOperands()) {
            Value v = use.get();
            if (!llvm::is_contained(captures, v))
              continue;
            auto it = local.find(v);
            if (it == local.end())
              it =
                  local
                      .insert(
                          {v, rewriter.clone(*v.getDefiningOp())->getResult(0)})
                      .first;
            use.set(it->second);
          }
        });
      }
    }
    for (size_t i = 0; i < op->getNumResults(); i++) {
      mapper.map(op->getResult(i), clonedOp->getResult(i));
    }
  }

  SmallVector<Value> results;
  for (auto result : lastOp->getResults()) {
    results.push_back(mapper.lookup(result));
  }

  if (outShape.has_value()) {
    for (size_t i = 0; i < results.size(); i++) {
      results[i] = stablehlo::ReshapeOpCreate(rewriter, lastOp->getLoc(),
                                              results[i], outShape.value());
    }
  }
  func::ReturnOp::create(rewriter, lastOp->getLoc(), results);

  return funcOp;
}

func::FuncOp CreateWrapperUnbatchedFunction(
    mlir::ModuleOp modOp, PatternRewriter &rewriter, std::string funcName,
    SmallVector<BatchLiftingMode> batchLiftingModes, Operation *op,
    std::optional<SmallVector<int64_t>> outShape) {
  SmallVector<Type> argTypes;
  for (auto [i, v] : llvm::enumerate(op->getOperands())) {
    if (batchLiftingModes[i] == BatchLiftingMode::CONSTANT) {
      continue;
    }
    argTypes.push_back(v.getType());
  }

  FunctionType calleeType =
      rewriter.getFunctionType(argTypes, op->getResultTypes());

  SmallVector<Operation *> opList = {op};
  return CreateWrapperUnbatchedFunction(modOp, rewriter, funcName,
                                        batchLiftingModes, opList, std::nullopt,
                                        outShape, calleeType);
}

func::FuncOp
CreateWrapperUnbatchedFunction(mlir::ModuleOp modOp, PatternRewriter &rewriter,
                               std::string funcName, ArrayRef<Operation *> ops,
                               RankedTensorType inTy, RankedTensorType outTy,
                               std::optional<SmallVector<int64_t>> inShape,
                               std::optional<SmallVector<int64_t>> outShape) {
  FunctionType calleeType =
      rewriter.getFunctionType(TypeRange(inTy), TypeRange(outTy));

  return CreateWrapperUnbatchedFunction(modOp, rewriter, funcName, std::nullopt,
                                        ops, inShape, outShape, calleeType);
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

// Two ops are equivalent if they differ only by a consistent renaming of their
// operands: whenever one op uses the same value in two slots, so must the
// other. `multiply %z, %z` and `multiply %y, %z` are therefore *not*
// equivalent. The batching wrapper clones the first op through a value-keyed
// IRMapping, so a value repeated across slots can only be read from one
// wrapper argument; that is only correct if every op in the batch repeats the
// same slots, i.e. those slots receive identical batched operands.
bool IsEquivalentUpToOperandRenaming(Operation *op1, Operation *op2) {
  if (!OperationEquivalence::isEquivalentTo(
          op1, op2, OperationEquivalence::ignoreValueEquivalence, nullptr,
          OperationEquivalence::IgnoreLocations, nullptr))
    return false;
  DenseMap<Value, Value> forward, backward;
  for (auto [a, b] : llvm::zip_equal(op1->getOperands(), op2->getOperands())) {
    if (forward.try_emplace(a, b).first->second != b ||
        backward.try_emplace(b, a).first->second != a)
      return false;
  }
  return true;
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

  auto outputType = cast<RankedTensorType>(op.getType());

  // If concat dim is present in broadcast dims, then it is not a valid insert
  if (dim != -1 && llvm::is_contained(op.getBroadcastDimensions(), dim)) {
    return false;
  }

  if (!stablehlo::OpIsReshapeLike(op)) {
    return false;
  }

  if (dim == -1) {
    return true;
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
  return found;
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
    if (vdefOp && !::utils::regionsCaptureOnlyConstants(vdefOp))
      vdefOp = nullptr;
    if (!vdefOp) {
      return rewriter.notifyMatchFailure(concatOp, "not a valid target op");
    }

    if (concatOpOperands.size() != 0) {
      if (!::utils::IsEquivalentUpToOperandRenaming(concatOpOperands[0],
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
  if (!func)
    return failure();

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
    if (candidateTargetOp &&
        !::utils::regionsCaptureOnlyConstants(candidateTargetOp))
      candidateTargetOp = nullptr;

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
      if (!::utils::IsEquivalentUpToOperandRenaming(targetOp,
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

  // The pieces the ops read are stacked into the batched operand. Pieces that
  // follow one another merge into one slice; any others can only be stacked
  // by concatenating them. For a convert that is not a form the
  // simplifications keep: ConvertConcat distributes a convert over a
  // concatenate's inputs unconditionally, a slice of the result then selects
  // one of them, and the converts this pattern batched stand again for it to
  // batch once more. Other ops over such a stack are left as they are
  // (ConcatElementwise refuses converts for the same reason). It is decided
  // here, before anything is built: a pattern that builds and then declines
  // is offered its own leavings for as long as the driver runs.
  if (isa<stablehlo::ConvertOp>(relatedOps[0])) {
    for (size_t i = 1, e = relatedSlices.size(); i < e; ++i) {
      if (!stablehlo::canMergeSlicesAlongAxis(
              sliceDim, relatedSlices[i - 1].sliceOp, relatedSlices[i].sliceOp))
        return rewriter.notifyMatchFailure(
            sliceOp, "converts of pieces that do not follow one another");
    }
  }

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

  // The ops of the range that read a related op's result, directly or
  // through another such op. A value is read by an op when the op takes it as
  // an operand and when an op nested in one of its regions captures it, so
  // ask about every op the range op contains. Definitions precede their uses
  // in the block, so one pass in program order reaches them all.
  llvm::SmallPtrSet<Operation *, 16> dependsOnRelated;
  for (auto it = rangeBegin; it != rangeEnd; ++it) {
    Operation *op = &*it;
    if (relatedOpsSet.contains(op))
      continue;
    bool reads = false;
    op->walk([&](Operation *inner) {
      for (Value v : inner->getOperands()) {
        Operation *defOp = v.getDefiningOp();
        if (defOp &&
            (relatedOpsSet.contains(defOp) || dependsOnRelated.contains(defOp)))
          reads = true;
      }
    });
    if (reads)
      dependsOnRelated.insert(op);
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
  if (!func)
    return failure();

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
       llvm::enumerate(llvm::zip_equal(relatedSlices, relatedOps))) {
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

// traverse a chain of dynamic update slices and extract the broadest slice of
// data that is being updated
static bool extractDynamicUpdateSliceUpdate(
    Operation *op, BlockArgument blockArg, SmallVectorImpl<Value> &startIndices,
    SmallVectorImpl<int64_t> &sliceSizes, WhileLoopInfo &info) {
  bool firstCheck = true;

  // For dimensions that are fully updated, we don't need to repeatedly check
  // those
  SmallVector<bool> fullUpdate(startIndices.size(), false);

  auto fullDimUpdated = [&](Value operand, Value update, Value start,
                            int64_t dim) {
    if (!matchPattern(start, m_Zero())) {
      return false;
    }

    auto operandTy = cast<RankedTensorType>(operand.getType());
    auto updateTy = cast<RankedTensorType>(operand.getType());
    return operandTy.getDimSize(dim) == updateTy.getDimSize(dim);
  };

  while (op) {
    auto dusOp = dyn_cast<stablehlo::DynamicUpdateSliceOp>(op);
    if (!dusOp) {
      return false;
    }

    auto dusOperand = dusOp.getOperand();
    auto dusUpdate = dusOp.getUpdate();
    RankedTensorType dusUpdateTy = dusUpdate.getType();
    auto curStartIndices = dusOp.getStartIndices();

    if (firstCheck) {
      for (size_t i = 0; i < dusOp.getStartIndices().size(); i++) {
        startIndices[i] = curStartIndices[i];
        sliceSizes[i] = dusUpdateTy.getDimSize(i);
        fullUpdate[i] =
            fullDimUpdated(dusOperand, dusUpdate, curStartIndices[i], i);
      }
      firstCheck = false;
    } else {
      for (size_t i = 0; i < dusOp.getStartIndices().size(); i++) {
        if (fullUpdate[i]) {
          continue;
        } else {
          bool wasFullDimUpdated =
              fullDimUpdated(dusOperand, dusUpdate, curStartIndices[i], i);
          if (wasFullDimUpdated) {
            fullUpdate[i] = true;
            startIndices[i] = curStartIndices[i];
            sliceSizes[i] = dusUpdateTy.getDimSize(i);
            continue;
          }
        }

        if (startIndices[i] == curStartIndices[i]) {
          // take the maximum slice size
          sliceSizes[i] = std::max(dusUpdateTy.getDimSize(i), sliceSizes[i]);
        } else {
          LLVM_DEBUG(dusOp->emitError(
              "TODO: support the case where we need to resolve "
              "the starts correctly"));
          return false;
        }
      }
    }

    if (dusOperand == blockArg) {
      return true;
    }

    op = dusOperand.getDefiningOp();
  }

  return false;
}

// The widest region of a loop-carried buffer that the chain of
// dynamic_update_slices feeding the terminator writes back on each iteration.
struct CarriedStoreInfo {
  SmallVector<Value> startIndices;
  SmallVector<int64_t> sliceSizes;
  bool valid = false;
};

// Keyed by the block argument number of the carried buffer. Walking the update
// chain is the expensive part of isSelfLaneCarriedLoad and the answer only
// depends on the argument, so callers that ask about many loads share one.
using CarriedStoreCache = DenseMap<unsigned, CarriedStoreInfo>;

// Recognizes a load of a loop-carried buffer that reads back exactly the lane
// this iteration stores to -- `buf[iv] = f(buf[iv], ...)`. Iterations then
// touch disjoint lanes, so the read carries no cross-iteration dependence and
// the loop is a map over the indexed dimension. `buf[iv + 1]` is the opposite
// case: a genuine recurrence, which this rejects.
//
// On success `permutationOut` receives the mapping from the slice's dimensions
// to the buffer's (the loads may sit behind a chain of transposes) and
// `argNumOut` the buffer's argument number.
static bool isSelfLaneCarriedLoad(stablehlo::DynamicSliceOp dsOp,
                                  stablehlo::WhileOp whileOp,
                                  WhileLoopInfo &info,
                                  SmallVectorImpl<int64_t> *permutationOut,
                                  unsigned *argNumOut,
                                  CarriedStoreCache *cache) {
  auto &whileBody = whileOp.getBody().front();
  auto *term = whileBody.getTerminator();
  if (!term) {
    return false;
  }

  SmallVector<stablehlo::TransposeOp> transposeChain;
  Value operand = dsOp.getOperand();
  while (auto transposeOp = operand.getDefiningOp<stablehlo::TransposeOp>()) {
    transposeChain.push_back(transposeOp);
    operand = transposeOp.getOperand();
  }

  auto blockArg = dyn_cast<BlockArgument>(operand);
  if (!blockArg || blockArg.getOwner() != &whileBody) {
    return false;
  }

  SmallVector<int64_t> permutation(
      cast<RankedTensorType>(operand.getType()).getRank());
  std::iota(permutation.begin(), permutation.end(), 0);
  for (auto transposeOp : transposeChain) {
    auto perm = transposeOp.getPermutation();
    for (auto &dim : permutation) {
      dim = perm[dim];
    }
  }

  unsigned argNum = blockArg.getArgNumber();
  if (argNum >= term->getNumOperands()) {
    return false;
  }

  CarriedStoreInfo local;
  CarriedStoreInfo *store = &local;
  bool needsCompute = true;
  if (cache) {
    auto it = cache->find(argNum);
    if (it != cache->end()) {
      store = &it->second;
      needsCompute = false;
    } else {
      store = &(*cache)[argNum];
    }
  }

  if (needsCompute) {
    auto res = term->getOperand(argNum);
    auto resRank = cast<RankedTensorType>(res.getType()).getRank();
    store->startIndices.assign(resRank, Value());
    store->sliceSizes.assign(resRank, 0);
    store->valid = extractDynamicUpdateSliceUpdate(
        res.getDefiningOp(), blockArg, store->startIndices, store->sliceSizes,
        info);
  }

  if (!store->valid ||
      store->startIndices.size() != dsOp.getStartIndices().size()) {
    return false;
  }

  // we can generalize this but for now we are extremely restrictive (most
  // usecases will generally satisfy these constraints)
  //   1. all start indices must be the same
  //   2. atleast one of the indices must be dependent on the induction
  //      variable
  //   3. all start indices dependent on the induction should have slice sizes
  //      of 1. this can be extended to ensure that each step > step size
  //      (currently not implemented).
  SmallVector<Value> dsStartIndices(store->startIndices.size());
  SmallVector<int64_t> dsSliceSizes(store->sliceSizes.size());
  for (auto [dsDim, argDim] : llvm::enumerate(permutation)) {
    dsStartIndices[argDim] = dsOp.getStartIndices()[dsDim];
    dsSliceSizes[argDim] = dsOp.getSliceSizes()[dsDim];
  }

  auto affineIndexInfo = info.getAffineIndexInfo();
  bool foundDepIndex = false;
  for (auto [dsStart, dusStart, dsSliceSize, dusSliceSize] :
       llvm::zip_equal(dsStartIndices, store->startIndices, dsSliceSizes,
                       store->sliceSizes)) {
    if (dsStart != dusStart || dsSliceSize != dusSliceSize) {
      return false;
    }

    if (!info.isConstantAcrossIterations(dsStart, false) &&
        affineIndexInfo.contains(dsStart)) {
      foundDepIndex = true;
      if (dsSliceSize != 1) {
        return false;
      }
    }
  }

  if (!foundDepIndex) {
    return false;
  }

  if (permutationOut) {
    *permutationOut = std::move(permutation);
  }
  if (argNumOut) {
    *argNumOut = argNum;
  }
  return true;
}

static bool isSelfLaneCarriedLoad(stablehlo::DynamicSliceOp dsOp,
                                  stablehlo::WhileOp whileOp,
                                  WhileLoopInfo &info) {
  return isSelfLaneCarriedLoad(dsOp, whileOp, info, /*permutationOut=*/nullptr,
                               /*argNumOut=*/nullptr, /*cache=*/nullptr);
}

// Ops that must not be handed to the batcher: either they are pure metadata
// (a reshape costs nothing inside the loop) or their batched form lowers back
// to a loop, which defeats the point.
static bool avoidBatching(Operation *op) {
  if (!op) {
    return true;
  }

  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<stablehlo::ReshapeOp, stablehlo::SliceOp, stablehlo::ReturnOp,
            // avoid ops that use SHLOGenericBatchOpInterface since that
            // lowers to loop
            stablehlo::IfOp, stablehlo::CaseOp, stablehlo::WhileOp,
            stablehlo::CustomCallOp>([](auto op) { return true; })
      .Case<stablehlo::BroadcastInDimOp, stablehlo::TransposeOp>(
          [](auto op) { return stablehlo::OpIsReshapeLike(op); })
      .Default([](auto op) { return false; });
}

// Reshapes are transparent to batching: they are skipped when the slices
// reaching an op are collected, and they carry no cost inside the loop.
static bool isTransparentForBatching(Operation *op) {
  return isa<stablehlo::ReshapeOp>(op);
}

// The computation a candidate dynamic_slice feeds, grown as far as it can be
// batched, so that the decision to hoist can be taken over the whole thing
// rather than one op at a time.
struct HoistableSliceComputation {
  stablehlo::WhileOp whileOp;
  WhileLoopInfo *info;
  ArrayRef<SliceInfo<stablehlo::DynamicSliceOp>> candidateSlices;
  const llvm::MapVector<Operation *,
                        SmallVector<SliceInfo<stablehlo::DynamicSliceOp>>>
      *userOpToSlicesMap;

  SliceInfo<stablehlo::DynamicSliceOp> sInfo;
  SetVector<Operation *> ops; // topologically ordered
  SetVector<Value> results;   // frontier: values escaping `ops`
  DenseMap<Value, bool> batchableCache;

  // The last state whose frontier was profitable; grow() falls back to it
  // rather than leaving the computation forked.
  SetVector<Operation *> checkpointedOps;
  SetVector<Value> checkpointedResults;

  HoistableSliceComputation(
      stablehlo::WhileOp whileOp, WhileLoopInfo &info,
      ArrayRef<SliceInfo<stablehlo::DynamicSliceOp>> candidateSlices,
      const llvm::MapVector<Operation *,
                            SmallVector<SliceInfo<stablehlo::DynamicSliceOp>>>
          &userOpToSlicesMap,
      SliceInfo<stablehlo::DynamicSliceOp> sInfo)
      : whileOp(whileOp), info(&info), candidateSlices(candidateSlices),
        userOpToSlicesMap(&userOpToSlicesMap), sInfo(sInfo) {
    ops.insert(sInfo.sliceOp);
    results.insert(sInfo.sliceOp.getResult());
    checkpoint();
  }

  Block &body() { return whileOp.getBody().front(); }

  void grow() {
    while (absorbOneLayer()) {
      if (isProfitable()) {
        checkpoint();
      }
    }

    restoreCheckpoint();
  }

  bool isProfitable() {
    // A DAG of nothing but the slice and some reshapes has nothing to batch.
    if (!llvm::any_of(ops, [&](Operation *op) {
          return op != sInfo.sliceOp.getOperation() &&
                 !isTransparentForBatching(op);
        })) {
      return false;
    }

    // A single value is always worth hoisting: whatever stays behind in the
    // loop, the batched computation replaces a per-iteration one.
    if (results.size() == 1) {
      return true;
    }

    // Every value of a forked frontier is materialized across the whole trip
    // count and sliced back in the loop, so it is only worth it when every
    // branch goes on to empty the loop. Otherwise grow() falls back to the
    // last single value.
    return llvm::all_of(results,
                        [&](Value result) { return endsWell(result); });
  }

  // Does hoisting the computation that produces `result` actually shrink the
  // loop?
  bool endsWell(Value result) {
    // The whole body is dead once this is hoisted.
    if (llvm::all_of(result.getUsers(),
                     [&](Operation *user) { return isTerminator(user); })) {
      return true;
    }

    // ... or it lands in another array at the lane this iteration owns, so the
    // loop is a map over that dimension.
    if (reachesCarriedStore(result)) {
      return true;
    }

    return false;
  }

  // Replay the per-op batching path over the ops we decided are worth it, in
  // topological order.
  LogicalResult hoistOut(PatternRewriter &rewriter,
                         SmallPtrSetImpl<Operation *> &alreadyHoisted) {
    bool anyOpRewritten = false;

    for (Operation *op : ops) {
      auto it = userOpToSlicesMap->find(op);
      if (it == userOpToSlicesMap->end() || !alreadyHoisted.insert(op).second) {
        continue;
      }
      ArrayRef<SliceInfo<stablehlo::DynamicSliceOp>> slices = it->second;

      if (auto dsOp = dyn_cast<stablehlo::DynamicSliceOp>(op)) {
        if (raiseDynamicSliceToGather(rewriter, whileOp, slices, dsOp, *info)) {
          anyOpRewritten = true;
        }
      } else if ((dyn_cast<BatchOpInterface>(op) ||
                  stablehlo::hasTraitElementwise(op)) &&
                 op->getNumResults() == 1) {
        if (liftOperationByBatching(rewriter, whileOp, slices, op, *info)) {
          anyOpRewritten = true;
        } else if (liftReduceLikeOperation(rewriter, whileOp, slices, op,
                                           *info)) {
          anyOpRewritten = true;
        }
      }
    }

    return success(anyOpRewritten);
  }

private:
  bool isTerminator(Operation *op) { return op == body().getTerminator(); }

  // Absorb every op that consumes the current frontier and can be batched.
  // Returns whether anything was added.
  bool absorbOneLayer() {
    SmallVector<Value> frontier(results.begin(), results.end());
    bool changed = false;

    for (Value v : frontier) {
      for (Operation *user : v.getUsers()) {
        if (ops.contains(user) || !canAbsorb(user)) {
          continue;
        }
        ops.insert(user);
        changed = true;
      }
    }

    if (changed) {
      recomputeResults();
    }
    return changed;
  }

  void checkpoint() {
    checkpointedOps = ops;
    checkpointedResults = results;
  }

  void restoreCheckpoint() {
    ops = checkpointedOps;
    results = checkpointedResults;
  }

  void recomputeResults() {
    results.clear();
    for (Operation *op : ops) {
      for (Value result : op->getResults()) {
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !ops.contains(user);
            })) {
          results.insert(result);
        }
      }
    }
  }

  // Can `op` join the computation? Every one of its operands has to be
  // something the batched form can be fed.
  bool canAbsorb(Operation *op) {
    // Growing into a nested region would batch the inner loop's trip count,
    // not this loop's.
    if (op->getBlock() != &body()) {
      return false;
    }

    if (isCarriedStore(op)) {
      return true;
    }

    if (!isTransparentForBatching(op) &&
        (avoidBatching(op) || op->getNumResults() != 1)) {
      return false;
    }

    return llvm::all_of(op->getOperands(), [&](Value v) {
      return isAdmissibleOperand(v) || isCarriedAccumulator(v, op);
    });
  }

  // The `acc = acc <op> x` shape: `v` is a loop-carried accumulator that `op`
  // updates and hands straight back to the terminator. The accumulator itself
  // is not a batchable value -- it is precisely the cross-iteration
  // dependency -- but the loop is then a reduction over the batched dimension,
  // which liftReduceLikeOperation turns into a stablehlo.reduce.
  bool isCarriedAccumulator(Value v, Operation *op) {
    auto blockArg = dyn_cast<BlockArgument>(v);
    if (!blockArg || blockArg.getOwner() != &body() ||
        op->getNumResults() != 1) {
      return false;
    }

    auto *term = body().getTerminator();
    return term && blockArg.getArgNumber() < term->getNumOperands() &&
           term->getOperand(blockArg.getArgNumber()) == op->getResult(0);
  }

  bool isAdmissibleOperand(Value v) {
    if (Operation *defOp = v.getDefiningOp()) {
      if (ops.contains(defOp)) {
        return true;
      }
    }
    return isBatchableValue(v);
  }

  // Can a batched form of this loop produce `v` for all iterations at once?
  //
  // This has to look past the computation being grown: an operand is just as
  // good when it comes from a sibling computation rooted at a different
  // candidate slice. A loop whose body is one big expression over several
  // loads would otherwise stall at the first op that joins two of them.
  bool isBatchableValue(Value v) {
    auto it = batchableCache.find(v);
    if (it != batchableCache.end()) {
      return it->second;
    }
    // Guard against cycles; a block argument fed by the terminator can bring
    // the walk back to where it started.
    batchableCache[v] = false;

    bool batchable = computeIsBatchableValue(v);
    batchableCache[v] = batchable;
    return batchable;
  }

  bool computeIsBatchableValue(Value v) {
    // Loop invariant, or hoistable to loop invariant.
    Value outerValue;
    SmallVector<Operation *> canBeHoisted;
    if (info->isConstantAcrossIterations(v, outerValue, canBeHoisted, true)) {
      return true;
    }

    if (info->getAffineIndexInfo().contains(v)) {
      return true;
    }

    Operation *defOp = v.getDefiningOp();
    if (!defOp || defOp->getBlock() != &body()) {
      return false;
    }

    // A load of some other array at an affine lane: either one of the slices
    // this pattern already knows how to batch, or a read of a carried buffer
    // at exactly the lane this iteration writes back. Anything else read off a
    // carried buffer is a genuine recurrence.
    if (auto dsOp = dyn_cast<stablehlo::DynamicSliceOp>(defOp)) {
      if (llvm::any_of(candidateSlices,
                       [&](const SliceInfo<stablehlo::DynamicSliceOp> &slice) {
                         return slice.sliceOp == dsOp;
                       })) {
        return true;
      }
      if (isSelfLaneCarriedLoad(dsOp, whileOp, *info)) {
        return true;
      }
    }

    if (!isTransparentForBatching(defOp) &&
        (avoidBatching(defOp) || defOp->getNumResults() != 1)) {
      return false;
    }

    return llvm::all_of(defOp->getOperands(), [&](Value operand) {
      return isBatchableValue(operand);
    });
  }

  // A dynamic_update_slice writing this iteration's lane of a loop-carried
  // buffer straight back to the terminator -- the sink of an array-to-array
  // map, absorbed so that growth reaches the terminator.
  bool isCarriedStore(Operation *op) {
    auto dusOp = dyn_cast<stablehlo::DynamicUpdateSliceOp>(op);
    if (!dusOp) {
      return false;
    }

    auto blockArg = dyn_cast<BlockArgument>(dusOp.getOperand());
    if (!blockArg || blockArg.getOwner() != &body()) {
      return false;
    }

    auto *term = body().getTerminator();
    if (!term || blockArg.getArgNumber() >= term->getNumOperands() ||
        term->getOperand(blockArg.getArgNumber()) != dusOp.getResult()) {
      return false;
    }

    auto affineIndexInfo = info->getAffineIndexInfo();
    bool foundDepIndex = false;
    for (Value startIndex : dusOp.getStartIndices()) {
      if (info->isConstantAcrossIterations(startIndex, false)) {
        continue;
      }
      if (!affineIndexInfo.contains(startIndex)) {
        return false;
      }
      foundDepIndex = true;
    }
    return foundDepIndex;
  }

  // Follow `v` through reshapes to see whether it is stored into a carried
  // buffer. Growth normally absorbs such a store, so this only fires when the
  // store itself could not be absorbed.
  bool reachesCarriedStore(Value v) {
    for (Operation *user : v.getUsers()) {
      if (isCarriedStore(user)) {
        return true;
      }
      if (isTransparentForBatching(user) && user->getNumResults() == 1 &&
          reachesCarriedStore(user->getResult(0))) {
        return true;
      }
    }
    return false;
  }
};

// Users of the induction variable are recorded with an empty SliceInfo -- they
// consume an affine index rather than a slice.
static bool
isAffineIndexOnlyUser(ArrayRef<SliceInfo<stablehlo::DynamicSliceOp>> slices) {
  return !slices.empty() &&
         llvm::all_of(slices,
                      [](const SliceInfo<stablehlo::DynamicSliceOp> &slice) {
                        return !slice.sliceOp;
                      });
}

LogicalResult GreedyWhileLoopBatchFission::matchAndRewriteImpl(
    stablehlo::WhileOp whileOp, PatternRewriter &rewriter) const {
  // Never fission a checkpoint segment loop. Batching trades memory for
  // parallelism by computing every iteration at once; for a loop that
  // checkpointed reverse-mode AD created specifically to keep only one segment
  // live, that rebuilds the very tape the recompute was paying to avoid. Such a
  // loop only becomes eligible in the first place when the segment length
  // divides the trip count evenly (LoopCheckpointing::segmentLength returns the
  // constant nInner rather than a select), so without this guard peak memory
  // silently depends on that divisibility.
  if (isOrContainsCheckpointSegmentLoop(whileOp))
    return rewriter.notifyMatchFailure(whileOp, "checkpoint segment loop");

  auto info = WhileLoopInfo(whileOp);
  auto computeInfoSuccess = info.computeInfo();
  if (computeInfoSuccess.failed())
    return computeInfoSuccess;

  // TODO: the only actual restriction is that the total loop iterations be
  // constant and the indexing must be affine
  if (!info.isValid() || !info.isConstant())
    return failure();

  // A loop that never runs (an inner loop whose bound became constant when
  // the outer loop was unrolled) has nothing to batch; leave it for
  // canonicalization instead of building zero-sized tensors.
  if (info.getConstantNumIters() <= 0)
    return rewriter.notifyMatchFailure(whileOp, "loop runs no iterations");

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
        auto result =
            isDynamicSliceValidForBatching(sliceOp, info, whileBody, whileOp);

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
      if (isa<stablehlo::ReshapeOp>(op)) {
        auto operandTy = cast<RankedTensorType>(op->getOperand(0).getType());
        auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());

        std::optional<SmallVector<int64_t>> reshapeShape;
        if (!areValidInsertionDims(resultTy, operandTy, ds.dimensions)) {
          reshapeShape = llvm::to_vector(resultTy.getShape());
        }

        for (auto user : op->getUsers()) {
          if (avoidBatching(user)) {
            continue;
          }

          userOpToSlicesMap[user].push_back(
              SliceInfo<stablehlo::DynamicSliceOp>{ds.sliceOp, ds.dimensions,
                                                   true, reshapeShape});
        }
      } else {
        if (avoidBatching(op)) {
          continue;
        }
        userOpToSlicesMap[op].push_back(ds);
      }
    }
  }

  // for certain operations on index variables it is more efficient to hoist
  // those out of the loop and then perform indirect indexing
  for (auto &[val, info] : affineIndexInfoMap) {
    for (auto user : val.getUsers()) {
      if (avoidBatching(user)) {
        continue;
      }

      if (isa<stablehlo::CompareOp, stablehlo::BroadcastInDimOp>(user)) {
        userOpToSlicesMap[user].push_back(
            SliceInfo<stablehlo::DynamicSliceOp>{});
      }
    }
  }

  if (userOpToSlicesMap.empty()) {
    return failure();
  }

  bool anyOpRewritten = false;
  SmallPtrSet<Operation *, 8> alreadyHoisted;

  SmallVector<HoistableSliceComputation, 2> profitable;
  for (auto &slice : candidateSlices) {
    HoistableSliceComputation computation(whileOp, info, candidateSlices,
                                          userOpToSlicesMap, slice);
    computation.grow();

    if (computation.isProfitable()) {
      profitable.push_back(std::move(computation));
    }
  }

  for (auto &computation : profitable) {
    if (computation.hoistOut(rewriter, alreadyHoisted).succeeded()) {
      anyOpRewritten = true;
    }
  }

  for (auto &[op, slices] : userOpToSlicesMap) {
    if (!alreadyHoisted.count(op) && isAffineIndexOnlyUser(slices)) {
      alreadyHoisted.insert(op);
      if (liftOperationByBatching(rewriter, whileOp, slices, op, info)) {
        anyOpRewritten = true;
      } else if (liftReduceLikeOperation(rewriter, whileOp, slices, op, info)) {
        anyOpRewritten = true;
      }
    }
  }

  return success(anyOpRewritten);
};

GreedyWhileLoopBatchFission::ValidBatchingInfo
GreedyWhileLoopBatchFission::isDynamicSliceValidForBatching(
    stablehlo::DynamicSliceOp sliceOp, mlir::enzyme::WhileLoopInfo &loopInfo,
    Block &whileBody, stablehlo::WhileOp whileOp) const {
  auto operand = sliceOp.getOperand();
  auto affineIndexInfoMap = loopInfo.getAffineIndexInfo();

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

    if (loopInfo.isConstantAcrossIterations(startIndex, true)) {
      continue;
    }

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

    if (!dsOp) {
      return false;
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

// `v` as the value of every iteration of a loop of `numIters` iterations:
// broadcast along a new leading dimension.
static Value broadcastToIterations(OpBuilder &builder, Location loc, Value v,
                                   int64_t numIters) {
  auto ty = cast<RankedTensorType>(v.getType());
  SmallVector<int64_t> shape{numIters};
  llvm::append_range(shape, ty.getShape());
  SmallVector<int64_t> mapping(ty.getRank());
  std::iota(mapping.begin(), mapping.end(), 1);
  return stablehlo::BroadcastInDimOp::create(
      builder, loc, RankedTensorType::get(shape, ty.getElementType()), v,
      builder.getDenseI64ArrayAttr(mapping));
}

LogicalResult constructNewOperandsForHoistedOp(
    PatternRewriter &rewriter, stablehlo::WhileOp whileOp, WhileLoopInfo &info,
    SmallVectorImpl<BatchLiftingMode> &batchLiftingModes,
    SmallVectorImpl<Value> &batchOperands,
    SmallVectorImpl<SmallVector<int64_t>> &sliceDims,
    SmallVectorImpl<int64_t> &hoistedDims,
    SmallVectorImpl<SliceInfo<stablehlo::DynamicSliceOp>> &mappedSliceInfos,
    DenseMap<Value, Value> &hoistedValues, SmallVectorImpl<Value> &newOperands,
    bool batchConstants = false) {
  newOperands.clear();

  for (auto [consType, baseOp, sliceDim, sliceInfo, hoistDim] :
       llvm::zip_equal(batchLiftingModes, batchOperands, sliceDims,
                       mappedSliceInfos, hoistedDims)) {
    auto operandType = cast<RankedTensorType>(baseOp.getType());
    int operandRank = cast<RankedTensorType>(baseOp.getType()).getRank();

    auto broadcastValue = [&](Value operand) {
      return broadcastToIterations(rewriter, whileOp->getLoc(), operand,
                                   info.getConstantNumIters());
    };

    switch (consType) {
    case BatchLiftingMode::DYNAMIC_SLICE: {
      // hoist the dynamic slice out of the loop and replace the sliceDim
      // with full slice.
      Value newSlice;
      bool successfulHoist = info.hoistOperationFromLoop(
          rewriter, baseOp, sliceInfo.sliceOp, sliceDim, newSlice);
      if (!successfulHoist) {
        return failure();
      }
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
      newOperands.push_back(broadcastValue(hoistedValues[baseOp]));
      break;
    }
    case BatchLiftingMode::DEFINED_OUTSIDE_WHILE: {
      newOperands.push_back(broadcastValue(baseOp));
      break;
    }
    case BatchLiftingMode::CONSTANT: {
      if (batchConstants) {
        newOperands.push_back(broadcastValue(baseOp));
      }
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

  return success();
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
  int64_t argIdx = -1;
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

  if (isLhsLoopCarriedDep == isRhsLoopCarriedDep) {
    return false; // atmost one of lhs/rhs must be loop carried dep
  }
  if (specialOps && isRhsLoopCarriedDep) { // only lhs can be loop carried dep
    return false;
  }

  // Only now does argIdx hold the position the carried value came from: it is
  // set by whichever of the two branches above found one, and asking about it
  // before they have agreed on exactly one reads it unset.
  // while dead args is needed to clean this up
  if (argIdx >= whileOp->getNumResults() ||
      whileOp->getResult(argIdx).getUsers().empty()) {
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
  if (!constructNewOperandsForHoistedOp(
           rewriter, whileOp, info, batchLiftingModes, batchOperands, sliceDims,
           hoistedDims, mappedSliceInfos, hoistedValues, newOperands, true)
           .succeeded()) {
    return false;
  }

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
    stablehlo::ReturnOp::create(rewriter, op->getLoc(), newOp->getResults());
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

bool raiseDynamicSliceToGather(
    PatternRewriter &rewriter, stablehlo::WhileOp whileOp,
    ArrayRef<SliceInfo<stablehlo::DynamicSliceOp>> slices,
    stablehlo::DynamicSliceOp dsOp, WhileLoopInfo info) {
  // Pattern: x[..., y[idx], z[idx], ...] where idx is an affine function of
  // the loop induction variable. We need to:
  // 1. Hoist the computation of y[idx], z[idx], etc. for all loop iterations
  // 2. Create a gather operation from x using those indices
  // 3. Replace dsOp uses with a dynamic slice into the gather result

  // Find all start indices that are dependent on the loop and come from inner
  // dynamic slices (through possible reshape)
  SmallVector<int64_t> dependentDims;
  SmallVector<Value> innerSliceOperands;
  SmallVector<SliceInfo<stablehlo::DynamicSliceOp>> innerSliceInfos;

  for (auto [i, startIndex] : llvm::enumerate(dsOp.getStartIndices())) {
    // Use traverseOperandsForHoisting to classify this operand
    SmallVector<BatchLiftingMode> modes;
    SmallVector<Value> operands;
    SmallVector<SmallVector<int64_t>> dims;
    SmallVector<int64_t> hoisted;
    SmallVector<SliceInfo<stablehlo::DynamicSliceOp>> mapped;
    DenseMap<Value, SmallVector<Operation *>> hoistMap;

    SmallVector<Value> singleOperand = {startIndex};
    if (!traverseOperandsForHoisting(singleOperand, whileOp, slices, info,
                                     modes, operands, dims, hoisted, mapped,
                                     hoistMap)) {
      return false;
    }

    if (modes[0] == BatchLiftingMode::DYNAMIC_SLICE) {
      dependentDims.push_back(i);
      innerSliceOperands.push_back(operands[0]);
      innerSliceInfos.push_back(mapped[0]);
    }
  }

  if (dependentDims.empty()) {
    return false;
  }

  // Get outer operand - it must be constant across iterations (loop invariant)
  Value outerOperand;
  Value dsOperand = dsOp.getOperand();
  SmallVector<Operation *> canBeHoisted;
  if (!info.isConstantAcrossIterations(dsOperand, outerOperand, canBeHoisted,
                                       true)) {
    return false;
  }

  // Decide every hoist before creating anything (see liftOperationByBatching).
  for (auto [operand, sliceInfo] :
       llvm::zip_equal(innerSliceOperands, innerSliceInfos)) {
    if (!info.canHoistOperationFromLoop(operand, sliceInfo.sliceOp,
                                        sliceInfo.dimensions))
      return false;
  }

  rewriter.setInsertionPoint(whileOp);

  if (!outerOperand) {
    // The operand is defined inside the loop but is hoistable - hoist it
    DenseMap<Value, SmallVector<Operation *>> hoistMap;
    hoistMap[dsOperand] = canBeHoisted;
    DenseMap<Value, Value> hoistedValues;
    hoistChainOfOps(hoistMap, rewriter, whileOp, info, hoistedValues);
    outerOperand = hoistedValues[dsOperand];
  }

  // Verify all non-dependent start indices are constant across iterations
  for (auto [i, startIndex] : llvm::enumerate(dsOp.getStartIndices())) {
    if (llvm::is_contained(dependentDims, i)) {
      continue;
    }
    if (!info.isConstantAcrossIterations(startIndex, true)) {
      return false;
    }
  }

  int64_t numIters = info.getConstantNumIters();
  Location loc = dsOp.getLoc();

  // Step 1: Hoist each inner slice operand and construct the gather indices
  // We need to gather all indices for all loop iterations and concatenate them.
  SmallVector<Value> hoistedIndicesList;
  Type hoistedIndicesElemTy;

  for (size_t idx = 0; idx < dependentDims.size(); idx++) {
    Value hoistedIndices;
    if (!info.hoistOperationFromLoop(
            rewriter, innerSliceOperands[idx], innerSliceInfos[idx].sliceOp,
            innerSliceInfos[idx].dimensions, hoistedIndices)) {
      return false;
    }

    auto hoistedTy = cast<RankedTensorType>(hoistedIndices.getType());
    if (idx == 0) {
      hoistedIndicesElemTy = hoistedTy.getElementType();
    }

    // Reshape to [numIters, 1] for use as gather indices
    SmallVector<int64_t> reshapeShape = {numIters, 1};

    // Convert type if needed
    if (hoistedTy.getElementType() != hoistedIndicesElemTy) {
      hoistedIndices = stablehlo::ConvertOp::create(
          rewriter, loc,
          RankedTensorType::get(hoistedTy.getShape(), hoistedIndicesElemTy),
          hoistedIndices);
    }

    Value reshaped =
        stablehlo::ReshapeOpCreate(rewriter, loc, hoistedIndices, reshapeShape);
    hoistedIndicesList.push_back(reshaped);
  }

  // Concatenate all hoisted indices along the last dimension
  Value gatherIndices;
  if (hoistedIndicesList.size() == 1) {
    gatherIndices = hoistedIndicesList[0];
  } else {
    // Result shape: [numIters, numDependentDims]
    SmallVector<int64_t> concatShape = {numIters,
                                        (int64_t)dependentDims.size()};
    auto concatTy = RankedTensorType::get(concatShape, hoistedIndicesElemTy);
    gatherIndices = stablehlo::ConcatenateOp::create(
        rewriter, loc, concatTy, hoistedIndicesList, /*dimension=*/1);
  }

  // Step 2: Create the gather operation from the outer operand
  auto outerOperandTy = cast<RankedTensorType>(outerOperand.getType());
  auto dsSliceSizes = dsOp.getSliceSizes();

  // The gather slice sizes: dependent dimensions get 1, others get original
  SmallVector<int64_t> gatherSliceSizes;
  for (size_t i = 0; i < dsSliceSizes.size(); i++) {
    if (llvm::is_contained(dependentDims, i)) {
      gatherSliceSizes.push_back(1);
    } else {
      gatherSliceSizes.push_back(dsSliceSizes[i]);
    }
  }

  // offsetDims: output dimensions corresponding to non-collapsed slice dims
  // Start at 1 since batch dim is at position 0, then consecutive for each
  // non-collapsed dimension
  SmallVector<int64_t> offsetDims;
  int64_t offsetDimIdx = 1; // Start after the batch dimension
  for (size_t i = 0; i < outerOperandTy.getRank(); i++) {
    if (!llvm::is_contained(dependentDims, i)) {
      offsetDims.push_back(offsetDimIdx);
      offsetDimIdx++;
    }
  }

  // collapsedSliceDims: the dimensions we're indexing into
  SmallVector<int64_t> collapsedSliceDims = llvm::to_vector(dependentDims);

  // startIndexMap: maps index vector dimensions to operand dimensions
  SmallVector<int64_t> startIndexMap = llvm::to_vector(dependentDims);

  // Calculate output shape: [numIters, ...sliceSizes for non-dependent dims...]
  SmallVector<int64_t> gatherOutputShape;
  gatherOutputShape.push_back(numIters);
  for (size_t i = 0; i < dsSliceSizes.size(); i++) {
    if (!llvm::is_contained(dependentDims, i)) {
      gatherOutputShape.push_back(dsSliceSizes[i]);
    }
  }

  auto gatherResultTy =
      RankedTensorType::get(gatherOutputShape, outerOperandTy.getElementType());

  auto gatherOp = stablehlo::GatherOp::create(
      rewriter, loc, gatherResultTy, outerOperand, gatherIndices,
      stablehlo::GatherDimensionNumbersAttr::get(
          rewriter.getContext(),
          /*offsetDims=*/offsetDims,
          /*collapsedSliceDims=*/collapsedSliceDims,
          /*operandBatchingDims=*/{},
          /*startIndicesBatchingDims=*/{},
          /*startIndexMap=*/startIndexMap,
          /*indexVectorDim=*/1),
      gatherSliceSizes);

  // Step 3: Replace the dsOp with a dynamic slice into the gather result
  // The dynamic slice will index using the loop induction variable
  rewriter.setInsertionPointAfter(dsOp);

  auto inductionVar = info.getInductionVariable();
  auto inductionVarType = cast<RankedTensorType>(inductionVar.getType());

  // Compute the index for the dynamic slice
  Value sliceIndex;
  if (info.isConstantStart() && info.getConstantStart() == 0) {
    sliceIndex = inductionVar;
  } else {
    sliceIndex = stablehlo::SubtractOp::create(rewriter, loc, inductionVar,
                                               info.getStart());
  }
  if (!info.isStepOne()) {
    sliceIndex = stablehlo::DivOp::create(rewriter, loc, sliceIndex,
                                          info.getStep(rewriter));
  }

  // Convert sliceIndex to the same type as gather indices if needed
  if (inductionVarType.getElementType() != hoistedIndicesElemTy) {
    auto newIndexTy = RankedTensorType::get({}, hoistedIndicesElemTy);
    sliceIndex =
        stablehlo::ConvertOp::create(rewriter, loc, newIndexTy, sliceIndex);
  }

  // Create constZero with the same type as sliceIndex (after conversion)
  auto sliceIndexTy = cast<RankedTensorType>(sliceIndex.getType());
  auto constZero = stablehlo::ConstantOp::create(
      rewriter, loc, sliceIndexTy,
      cast<ElementsAttr>(makeAttr(sliceIndexTy, 0)));
  // Build the start indices for dynamic slice
  SmallVector<Value> dynSliceStarts;
  dynSliceStarts.push_back(sliceIndex);
  for (size_t i = 0; i < dsSliceSizes.size(); i++) {
    if (!llvm::is_contained(dependentDims, i)) {
      dynSliceStarts.push_back(constZero);
    }
  }

  // Build the slice sizes (1 for the batch dim, original sizes for others)
  SmallVector<int64_t> dynSliceSizes;
  dynSliceSizes.push_back(1);
  for (size_t i = 0; i < dsSliceSizes.size(); i++) {
    if (!llvm::is_contained(dependentDims, i)) {
      dynSliceSizes.push_back(dsSliceSizes[i]);
    }
  }

  auto dynSlice = stablehlo::DynamicSliceOpCreate(
      rewriter, loc, gatherOp.getResult(), dynSliceStarts, dynSliceSizes);

  // Reshape to match the original dsOp output type
  auto replacement = stablehlo::ReshapeOpCreate(
      rewriter, loc, dynSlice, cast<ShapedType>(dsOp.getType()).getShape());
  rewriter.replaceOp(dsOp, replacement);
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

  // The op's regions may read the loop body, as a scatter that writes a
  // constant made there does. The wrapper the lift clones the op into can
  // take a constant along and nothing else; decided here, before anything is
  // built.
  if (!::utils::regionsCaptureOnlyConstants(op))
    return false;

  auto opOperands = llvm::to_vector(op->getOperands());
  if (!traverseOperandsForHoisting(opOperands, whileOp, slices, info,
                                   batchLiftingModes, batchOperands, sliceDims,
                                   hoistedDims, mappedSliceInfos, hoistMap)) {
    return false;
  }

  // Decide every hoist before creating anything: a failed attempt must not
  // leave ops behind, or the greedy driver revisits the loop forever.
  for (auto [mode, baseOp, sliceDim, sliceInfo] : llvm::zip_equal(
           batchLiftingModes, batchOperands, sliceDims, mappedSliceInfos)) {
    if (mode == BatchLiftingMode::DYNAMIC_SLICE &&
        !info.canHoistOperationFromLoop(baseOp, sliceInfo.sliceOp, sliceDim))
      return false;
  }

  func::FuncOp func = ::utils::CreateWrapperUnbatchedFunction(
      moduleOp, rewriter, "enzymexla_unbatched_WhileLoopBatchFission_",
      batchLiftingModes, op, std::nullopt);
  if (!func)
    return false;

  rewriter.setInsertionPoint(whileOp);

  // hoist any operations that can be hoisted
  DenseMap<Value, Value> hoistedValues;
  hoistChainOfOps(hoistMap, rewriter, whileOp, info, hoistedValues);

  SmallVector<Value> newOperands;
  if (!constructNewOperandsForHoistedOp(
           rewriter, whileOp, info, batchLiftingModes, batchOperands, sliceDims,
           hoistedDims, mappedSliceInfos, hoistedValues, newOperands)
           .succeeded()) {
    return false;
  }

  auto inductionVar = info.getInductionVariable();

  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  auto resultShape = resultType.getShape();
  SmallVector<int64_t> outputShape(resultShape.size() + 1);
  outputShape[0] = info.getConstantNumIters();
  for (int i = 0; i < resultShape.size(); i++)
    outputShape[i + 1] = resultShape[i];

  auto batchOp = enzyme::BatchOp::create(
      rewriter, whileOp->getLoc(),
      RankedTensorType::get(outputShape, resultType.getElementType()),
      mlir::FlatSymbolRefAttr::get(func.getContext(), func.getName()),
      ValueRange(newOperands),
      rewriter.getDenseI64ArrayAttr({info.getConstantNumIters()}));

  rewriter.setInsertionPointAfter(op);

  auto inductionVarType = cast<RankedTensorType>(inductionVar.getType());
  auto constZero = stablehlo::ConstantOp::create(
      rewriter, whileOp->getLoc(), inductionVarType,
      cast<ElementsAttr>(makeAttr(inductionVarType, 0)));
  SmallVector<Value> dynamicSliceStarts(outputShape.size(), constZero);
  Value resIndex;
  if (info.isConstantStart() && info.getConstantStart() == 0) {
    resIndex = info.getInductionVariable();
  } else {
    resIndex = stablehlo::SubtractOp::create(rewriter, whileOp->getLoc(),
                                             info.getInductionVariable(),
                                             info.getStart());
  }
  if (!info.isStepOne()) {
    resIndex = stablehlo::DivOp::create(rewriter, whileOp->getLoc(), resIndex,
                                        info.getStep(rewriter));
  }
  dynamicSliceStarts[0] = resIndex;

  SmallVector<int64_t> dynamicSliceSizes(outputShape.begin(),
                                         outputShape.end());
  dynamicSliceSizes[0] = 1;

  auto dynamicSlice = stablehlo::DynamicSliceOpCreate(
      rewriter, whileOp->getLoc(), batchOp->getResult(0), dynamicSliceStarts,
      dynamicSliceSizes);
  auto newReshape = stablehlo::ReshapeOpCreate(
      rewriter, whileOp->getLoc(), dynamicSlice,
      cast<ShapedType>(op->getResult(0).getType()).getShape());
  rewriter.replaceOp(op, newReshape);

  enzyme::batchutils::batchOperationInline(
      rewriter, batchOp, cast<FunctionOpInterface>(func.getOperation()));

  return true;
}

mlir::LogicalResult WhileElementwiseReductionToReduce::matchAndRewriteImpl(
    stablehlo::WhileOp whileOp, PatternRewriter &rewriter) const {
  // Same reasoning as GreedyWhileLoopBatchFission: lifting the reduction
  // materializes every iteration of the segment at once.
  if (isOrContainsCheckpointSegmentLoop(whileOp))
    return rewriter.notifyMatchFailure(whileOp, "checkpoint segment loop");

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
  if (!info.isValid() || !info.isConstant() ||
      info.getConstantNumIters() <= 0) {
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

    if (!stablehlo::hasTraitElementwise(user) || user->getNumOperands() != 2 ||
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

mlir::LogicalResult
RemoveLoopCarriedDependenciesFromWhileLoadOperations::matchAndRewriteImpl(
    stablehlo::WhileOp whileOp, PatternRewriter &rewriter) const {
  auto info = WhileLoopInfo(whileOp);
  auto computeInfoSuccess = info.computeInfo();
  if (computeInfoSuccess.failed()) {
    return computeInfoSuccess;
  }

  auto &whileBody = whileOp.getBody().front();
  if (!whileBody.getTerminator()) {
    return failure();
  }

  bool anyOpRewritten = false;
  CarriedStoreCache carriedStores;

  whileBody.walk([&](stablehlo::DynamicSliceOp dsOp) {
    SmallVector<int64_t> permutation;
    unsigned argNum;
    if (!isSelfLaneCarriedLoad(dsOp, whileOp, info, &permutation, &argNum,
                               &carriedStores)) {
      return WalkResult::advance();
    }

    // The load only ever sees the value the buffer came into the loop with, so
    // read it straight off the loop operand and drop the carried dependency.
    Value newOperand = whileOp->getOperand(argNum);
    bool isIdentityPermutation =
        llvm::all_of(llvm::enumerate(permutation), [](auto pair) {
          return (int64_t)pair.index() == pair.value();
        });
    if (!isIdentityPermutation) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(whileOp);
      newOperand = stablehlo::TransposeOp::create(rewriter, dsOp.getLoc(),
                                                  newOperand, permutation);
    }
    rewriter.modifyOpInPlace(dsOp, [&]() { dsOp.setOperand(0, newOperand); });
    anyOpRewritten = true;

    return WalkResult::advance();
  });

  return success(anyOpRewritten);
}

LogicalResult
WhileIsCopySimplify::matchAndRewriteImpl(stablehlo::WhileOp whileOp,
                                         PatternRewriter &rewriter) const {
  WhileLoopInfo info(whileOp);
  auto computedInfo = info.computeInfo();
  (void)computedInfo;
  if (!info.isValid() || !info.isConstant() ||
      info.getConstantNumIters() <= 0) {
    return failure();
  }

  auto &whileBody = whileOp.getBody().front();
  auto affineIndexInfo = info.getAffineIndexInfo();

  auto returnOp = dyn_cast<stablehlo::ReturnOp>(whileBody.getTerminator());
  if (!returnOp) {
    return failure();
  }

  auto modOp = whileOp->getParentOfType<mlir::ModuleOp>();
  if (!modOp) {
    return failure();
  }

  bool anyOpRewritten = false;
  SmallVector<std::tuple<enzyme::BatchOp, func::FuncOp>> batchOps;

  for (auto [idx, returnValue] : llvm::enumerate(returnOp.getOperands())) {
    auto dusOp = returnValue.getDefiningOp<stablehlo::DynamicUpdateSliceOp>();
    if (!dusOp) {
      continue;
    }

    // check if operand is a loop caried variable
    auto blockArg = dyn_cast<BlockArgument>(dusOp.getOperand());
    if (!blockArg || blockArg.getOwner() != &whileBody ||
        blockArg.getArgNumber() != idx) {
      continue;
    }

    if (whileOp.getResult(idx).getUses().empty()) {
      continue;
    }

    auto dusInductionVarDims =
        getInductionVariableDimension(dusOp, affineIndexInfo, whileOp, info);
    Value indirectScatterIndices;
    if (dusInductionVarDims.empty()) {
      // Handle a common non-contiguous copy idiom:
      //
      //   dst[index_table[iv]] = src[iv]
      //
      // The destination index is not affine, so it cannot be raised as one
      // large dynamic_update_slice.  If the index table is a constant set of
      // unique indices, hoist its per-iteration load and use it to build a
      // scatter instead.  Requiring uniqueness preserves the sequential
      // overwrite semantics of the original loop.
      int64_t indirectDim = -1;
      stablehlo::DynamicSliceOp indexSlice;
      for (auto [dim, startIndex] : llvm::enumerate(dusOp.getStartIndices())) {
        if (info.isConstantAcrossIterations(startIndex, false))
          continue;
        if (indirectDim != -1) {
          indirectDim = -1;
          break;
        }

        Value indexValue = startIndex;
        if (auto reshape = indexValue.getDefiningOp<stablehlo::ReshapeOp>())
          indexValue = reshape.getOperand();
        indexSlice = indexValue.getDefiningOp<stablehlo::DynamicSliceOp>();
        if (!indexSlice)
          break;
        indirectDim = dim;
      }

      DenseElementsAttr indexTable;
      if (indirectDim >= 0 && indexSlice &&
          matchPattern(indexSlice.getOperand(), m_Constant(&indexTable)) &&
          indexTable.getNumElements() ==
              static_cast<size_t>(info.getConstantNumIters())) {
        llvm::SmallDenseSet<APInt> seenIndices;
        bool allUniqueAndInBounds = true;
        int64_t maxIndex = cast<RankedTensorType>(dusOp.getOperand().getType())
                               .getDimSize(indirectDim) -
                           cast<RankedTensorType>(dusOp.getUpdate().getType())
                               .getDimSize(indirectDim);
        for (APInt index : indexTable.getValues<APInt>()) {
          if (index.isNegative() || index.getSExtValue() > maxIndex ||
              !seenIndices.insert(index).second) {
            allUniqueAndInBounds = false;
            break;
          }
        }

        auto indexInductionDims = getInductionVariableDimension(
            indexSlice, affineIndexInfo, whileOp, info);
        if (allUniqueAndInBounds && indexInductionDims.size() == 1 &&
            info.canHoistOperationFromLoop(indexSlice, indexInductionDims)) {
          rewriter.setInsertionPoint(whileOp);
          if (info.hoistOperationFromLoop(rewriter, indexSlice.getOperand(),
                                          indexSlice, indexInductionDims,
                                          indirectScatterIndices)) {
            indirectScatterIndices = stablehlo::ReshapeOpCreate(
                rewriter, whileOp.getLoc(), indirectScatterIndices,
                {info.getConstantNumIters(), 1});
            dusInductionVarDims.push_back(indirectDim);
          }
        }
      }
    }

    if (dusInductionVarDims.empty() ||
        (!indirectScatterIndices &&
         !info.canHoistOperationFromLoop(dusOp, dusInductionVarDims))) {
      continue;
    }

    Value newDUSUpdate;
    SmallVector<Operation *> canBeHoisted;
    Value outerValue;
    auto origUpdate = dusOp.getUpdate();
    if (info.isConstantAcrossIterations(origUpdate, outerValue, canBeHoisted,
                                        true)) {
      RankedTensorType origUpdateTy = origUpdate.getType();
      if (llvm::any_of(dusInductionVarDims, [&](auto i) {
            return origUpdateTy.getDimSize(i) != 1;
          })) {
        continue;
      }

      anyOpRewritten = true;
      if (outerValue) {
        newDUSUpdate = outerValue;
      } else {
        DenseMap<Value, SmallVector<Operation *>> hoistMap;
        hoistMap[origUpdate] = canBeHoisted;
        DenseMap<Value, Value> hoistedValues;
        hoistChainOfOps(hoistMap, rewriter, whileOp, info, hoistedValues);
        newDUSUpdate = hoistedValues[origUpdate];
      }

      SmallVector<int64_t> newUpdateShapeWithoutIndVarDims;
      for (auto [i, sz] : llvm::enumerate(
               cast<RankedTensorType>(newDUSUpdate.getType()).getShape())) {
        if (llvm::is_contained(dusInductionVarDims, i)) {
          continue;
        }
        newUpdateShapeWithoutIndVarDims.push_back(sz);
      }

      newDUSUpdate =
          stablehlo::ReshapeOpCreate(rewriter, whileOp.getLoc(), newDUSUpdate,
                                     newUpdateShapeWithoutIndVarDims);

      auto newDusUpdateTy = cast<RankedTensorType>(newDUSUpdate.getType());
      SmallVector<int64_t> mapping(newDusUpdateTy.getRank());
      std::iota(mapping.begin(), mapping.end(), 1);

      auto newUpdateShape = llvm::to_vector(newDusUpdateTy.getShape());
      newUpdateShape.insert(newUpdateShape.begin(), info.getConstantNumIters());

      newDUSUpdate = stablehlo::BroadcastInDimOp::create(
          rewriter, whileOp.getLoc(),
          RankedTensorType::get(newUpdateShape,
                                newDusUpdateTy.getElementType()),
          newDUSUpdate, rewriter.getDenseI64ArrayAttr(mapping));
    } else {
      auto updateChainOrNone = extractValidUpdateChain(rewriter, dusOp, whileOp,
                                                       affineIndexInfo, info);
      if (!updateChainOrNone.has_value()) {
        continue;
      }

      auto updateChain = updateChainOrNone.value();
      if (updateChain.empty()) {
        continue;
      }

      auto dsOp = dyn_cast<stablehlo::DynamicSliceOp>(updateChain.back());
      auto dsInductionVarDims =
          getInductionVariableDimension(dsOp, affineIndexInfo, whileOp, info);
      if (dsInductionVarDims.empty() ||
          !info.canHoistOperationFromLoop(dsOp, dsInductionVarDims)) {
        continue;
      }

      anyOpRewritten = true;

      rewriter.setInsertionPoint(whileOp);
      Value dsResult;
      auto successfulHoist = info.hoistOperationFromLoop(
          rewriter, dsOp.getOperand(), dsOp, dsInductionVarDims, dsResult);
      if (!successfulHoist) {
        return failure(); // can happen if user is doing an out of bounds access
      }
      auto dsResTy = cast<RankedTensorType>(dsResult.getType());

      // move the induction var to the front
      SmallVector<int64_t> dsPerm;
      dsPerm.push_back(dsInductionVarDims[0]);
      for (size_t i = 0; i < dsResTy.getRank(); i++) {
        if (dsInductionVarDims[0] != i) {
          dsPerm.push_back(i);
        }
      }

      dsResult =
          stablehlo::TransposeOp::create(rewriter, whileOp.getLoc(), dsResult,
                                         rewriter.getDenseI64ArrayAttr(dsPerm));

      SmallVector<Operation *> opList;
      for (size_t i = updateChain.size() - 2; i >= 1; i--) {
        opList.push_back(updateChain[i]);
      }

      auto fnInTy =
          removeBatchedDims(dsOp.getResult().getType(), dsInductionVarDims);
      auto fnOutTy =
          removeBatchedDims(dusOp.getUpdate().getType(), dusInductionVarDims);

      if (opList.empty()) {
        auto resShape = llvm::to_vector(fnOutTy.getShape());
        resShape.insert(resShape.begin(), info.getConstantNumIters());
        newDUSUpdate = stablehlo::ReshapeOpCreate(rewriter, whileOp.getLoc(),
                                                  dsResult, resShape);
      } else {
        auto funcOp = ::utils::CreateWrapperUnbatchedFunction(
            modOp, rewriter, "__enzymexla__copy_inside_while_loop_", opList,
            fnInTy, fnOutTy,
            llvm::to_vector(
                cast<RankedTensorType>(dsOp.getResult().getType()).getShape()),
            llvm::to_vector(fnOutTy.getShape()));
        if (!funcOp) {
          continue;
        }

        auto dusOpUpdateTy =
            cast<RankedTensorType>(dusOp.getUpdate().getType());

        SmallVector<int64_t> dusOpUpdateShapePreTranspose = {
            info.getConstantNumIters()};
        for (auto [i, sz] : llvm::enumerate(dusOpUpdateTy.getShape())) {
          if (!llvm::is_contained(dusInductionVarDims, i)) {
            dusOpUpdateShapePreTranspose.push_back(sz);
          }
        }
        auto batchOpOutTy = RankedTensorType::get(
            dusOpUpdateShapePreTranspose, dusOpUpdateTy.getElementType());

        auto batchOp = enzyme::BatchOp::create(
            rewriter, whileOp.getLoc(), batchOpOutTy, funcOp.getSymName(),
            ValueRange(dsResult),
            rewriter.getDenseI64ArrayAttr({info.getConstantNumIters()}));
        batchOps.push_back({batchOp, funcOp});

        newDUSUpdate = batchOp.getResult(0);
      }
    }

    Value newDUS;
    if (indirectScatterIndices) {
      auto operandTy =
          cast<RankedTensorType>(whileOp.getOperands()[idx].getType());
      auto updateTy = cast<RankedTensorType>(newDUSUpdate.getType());
      SmallVector<int64_t> updateWindowDims(updateTy.getRank() - 1);
      std::iota(updateWindowDims.begin(), updateWindowDims.end(), 1);

      auto scatter = stablehlo::ScatterOp::create(
          rewriter, dusOp.getLoc(), ValueRange{whileOp.getOperands()[idx]},
          indirectScatterIndices, ValueRange{newDUSUpdate},
          stablehlo::ScatterDimensionNumbersAttr::get(
              dusOp.getContext(), updateWindowDims, dusInductionVarDims,
              /*inputBatchingDims=*/{}, /*scatterIndicesBatchingDims=*/{},
              dusInductionVarDims, /*indexVectorDim=*/1),
          /*indicesAreSorted=*/false, /*uniqueIndices=*/true);
      Block *updateBody = rewriter.createBlock(
          &scatter.getUpdateComputation(), {},
          {RankedTensorType::get({}, operandTy.getElementType()),
           RankedTensorType::get({}, operandTy.getElementType())},
          {dusOp.getLoc(), dusOp.getLoc()});
      rewriter.setInsertionPointToStart(updateBody);
      stablehlo::ReturnOp::create(rewriter, dusOp.getLoc(),
                                  updateBody->getArgument(1));
      newDUS = scatter.getResult(0);
    } else {
      bool successfulHoist = info.hoistOperationFromLoop(
          rewriter, whileOp.getOperands()[idx], newDUSUpdate, dusOp,
          dusInductionVarDims, newDUS);
      if (!successfulHoist)
        return failure();
    }

    whileOp.getResult(idx).replaceAllUsesWith(newDUS);
  }

  if (!batchOps.empty()) {
    for (auto &[batchOp, funcOp] : batchOps) {
      enzyme::batchutils::batchOperationInline(
          rewriter, batchOp, cast<FunctionOpInterface>(funcOp.getOperation()));
    }
  }

  return success(anyOpRewritten);
}

std::optional<SmallVector<Operation *>>
WhileIsCopySimplify::extractValidUpdateChain(
    PatternRewriter &rewriter, stablehlo::DynamicUpdateSliceOp dusOp,
    stablehlo::WhileOp whileOp,
    llvm::MapVector<Value, WhileLoopInfo::AffineIndexInfo> &affineIndexInfo,
    WhileLoopInfo &info) const {
  SmallVector<Operation *> updateChain;
  updateChain.push_back(dusOp);

  bool success = extractValidUpdateChainInner(
      rewriter, dusOp.getUpdate().getDefiningOp(), whileOp, updateChain);

  if (!success) {
    return std::nullopt;
  }
  return updateChain;
}

bool WhileIsCopySimplify::extractValidUpdateChainInner(
    PatternRewriter &rewriter, Operation *op, stablehlo::WhileOp whileOp,
    SmallVectorImpl<Operation *> &updateChain) const {
  if (!op) {
    return false;
  }

  if (auto sliceOp = dyn_cast<stablehlo::DynamicSliceOp>(op)) {
    // base case, we have reached the dynamic slice
    updateChain.push_back(sliceOp);
    return definedOutside(sliceOp.getOperand(), whileOp);
  }

  if (isa<stablehlo::ConvertOp, stablehlo::BroadcastInDimOp,
          stablehlo::ReshapeOp, stablehlo::TransposeOp>(op)) {
    updateChain.push_back(op);
    return extractValidUpdateChainInner(
        rewriter, op->getOperand(0).getDefiningOp(), whileOp, updateChain);
  }

  return false;
}

template <typename OpTy>
SmallVector<int64_t> WhileIsCopySimplify::getInductionVariableDimension(
    OpTy op,
    llvm::MapVector<Value, WhileLoopInfo::AffineIndexInfo> &affineIndexInfo,
    stablehlo::WhileOp whileOp, WhileLoopInfo &info) const {
  return getInductionVariableDimension(op.getStartIndices(), affineIndexInfo,
                                       whileOp, info);
}

SmallVector<int64_t> WhileIsCopySimplify::getInductionVariableDimension(
    mlir::OperandRange startIndices,
    llvm::MapVector<Value, WhileLoopInfo::AffineIndexInfo> &affineIndexInfo,
    stablehlo::WhileOp whileOp, WhileLoopInfo &info) const {
  SmallVector<int64_t> inductionVarDimensions;

  for (auto [i, startIndex] : llvm::enumerate(startIndices)) {
    // we could hoist the other dimensions but licm should fix this
    if (info.isConstantAcrossIterations(startIndex, false)) {
      continue;
    }

    if (!affineIndexInfo.contains(startIndex)) {
      return {};
    }

    inductionVarDimensions.push_back(i);
  }
  return inductionVarDimensions;
}

static SmallVector<int64_t> prepend(int64_t n, ArrayRef<int64_t> shape) {
  SmallVector<int64_t> r{n};
  r.append(shape.begin(), shape.end());
  return r;
}
static SmallVector<int64_t> shifted(ArrayRef<int64_t> dims) {
  SmallVector<int64_t> r;
  for (int64_t d : dims)
    r.push_back(d + 1);
  return r;
}

namespace {
// The elements of a rank-1 buffer one iteration addresses: concrete offsets,
// all shifted by one loop-invariant base the iterations share. The base is the
// same value in every iteration, so it cancels when two iterations are
// compared and the offsets alone decide whether they overlap.
struct IterationIndices {
  SmallVector<int64_t> offsets;
  Value base;
};

// Evaluates an index of the loop body for one iteration, as far as constants,
// the induction variable and a single loop-invariant addend allow.
struct IndexEvaluator {
  Region &body;
  Value iv;
  int64_t start, step;

  bool invariant(Value v) const {
    return !body.isAncestor(v.getParentRegion());
  }

  std::optional<IterationIndices> eval(Value v, int64_t iter) {
    if (v == iv)
      return IterationIndices{{start + iter * step}, Value()};
    Operation *op = v.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (isa<stablehlo::ReshapeOp, stablehlo::ConvertOp,
            stablehlo::BroadcastInDimOp>(op))
      return eval(op->getOperand(0), iter);
    if (auto cst = dyn_cast<stablehlo::ConstantOp>(op)) {
      auto attr = dyn_cast<DenseIntElementsAttr>(cst.getValue());
      if (!attr)
        return std::nullopt;
      IterationIndices out;
      out.base = Value();
      for (const APInt &e : attr.getValues<APInt>())
        out.offsets.push_back(e.getSExtValue());
      return out;
    }
    if (auto ds = dyn_cast<stablehlo::DynamicSliceOp>(op))
      return evalTableRow(ds, iter);
    if (isa<stablehlo::AddOp, stablehlo::SubtractOp, stablehlo::MulOp>(op))
      return evalArith(op, iter);
    return std::nullopt;
  }

  // A row (or window) of a constant table, selected by this iteration.
  std::optional<IterationIndices> evalTableRow(stablehlo::DynamicSliceOp ds,
                                               int64_t iter) {
    auto cst = ds.getOperand().getDefiningOp<stablehlo::ConstantOp>();
    if (!cst)
      return std::nullopt;
    auto attr = dyn_cast<DenseIntElementsAttr>(cst.getValue());
    auto ty = dyn_cast<RankedTensorType>(ds.getOperand().getType());
    if (!attr || !ty || !ty.hasStaticShape())
      return std::nullopt;
    ArrayRef<int64_t> shape = ty.getShape();
    ArrayRef<int64_t> sizes = ds.getSliceSizes();
    SmallVector<int64_t> starts;
    for (Value s : ds.getStartIndices()) {
      auto e = eval(s, iter);
      if (!e || e->base || e->offsets.size() != 1)
        return std::nullopt;
      starts.push_back(e->offsets[0]);
    }
    if (starts.size() != shape.size() || sizes.size() != shape.size())
      return std::nullopt;
    // stablehlo clamps a start so the window stays inside the operand
    for (auto [i, s] : llvm::enumerate(starts))
      starts[i] = std::min(std::max<int64_t>(s, 0), shape[i] - sizes[i]);
    auto values = attr.getValues<APInt>();
    IterationIndices out;
    out.base = Value();
    SmallVector<int64_t> idx(shape.size(), 0);
    int64_t count = 1;
    for (int64_t n : sizes)
      count *= n;
    for (int64_t c = 0; c < count; ++c) {
      int64_t rest = c, flat = 0;
      for (int64_t d = shape.size() - 1; d >= 0; --d) {
        int64_t within = rest % sizes[d];
        rest /= sizes[d];
        idx[d] = starts[d] + within;
      }
      for (int64_t d = 0; d < (int64_t)shape.size(); ++d)
        flat = flat * shape[d] + idx[d];
      out.offsets.push_back(values[flat].getSExtValue());
    }
    return out;
  }

  // An index shifted or scaled by a constant, or shifted by a value the loop
  // does not change.
  std::optional<IterationIndices> evalArith(Operation *op, int64_t iter) {
    auto lhs = eval(op->getOperand(0), iter);
    auto rhs = eval(op->getOperand(1), iter);
    bool isAdd = isa<stablehlo::AddOp>(op);
    if (lhs && rhs) {
      if (lhs->base || rhs->base)
        return std::nullopt;
      // one side may be a scalar the other is taken against elementwise
      SmallVector<int64_t> &a = lhs->offsets, &b = rhs->offsets;
      if (a.size() != b.size() && a.size() != 1 && b.size() != 1)
        return std::nullopt;
      IterationIndices out;
      out.base = Value();
      size_t n = std::max(a.size(), b.size());
      for (size_t i = 0; i < n; ++i) {
        int64_t x = a[a.size() == 1 ? 0 : i], y = b[b.size() == 1 ? 0 : i];
        out.offsets.push_back(isa<stablehlo::MulOp>(op)  ? x * y
                              : isAdd                    ? x + y
                                                         : x - y);
      }
      return out;
    }
    // A loop-invariant addend shifts every iteration alike, so it cancels.
    if (!isAdd)
      return std::nullopt;
    Value other = lhs ? op->getOperand(1) : op->getOperand(0);
    auto known = lhs ? lhs : rhs;
    if (!known || known->base || !invariant(other))
      return std::nullopt;
    known->base = other;
    return known;
  }
};

// Iterations of a loop that is not tagged parallel may still run at once when
// no iteration can observe another's writes: every read and every write of a
// carried buffer addresses a set of its elements, and the sets of two
// iterations never meet. Only rank-1 buffers addressed one element at a time
// are proved here, which is the shape a raised kernel's dof loop has.
static LogicalResult
proveIterationsIndependent(stablehlo::WhileOp whileOp, Block &body, Value iv,
                           int64_t numIters, int64_t start, int64_t step,
                           const DenseMap<Value, unsigned> &chainRoot) {
  // The proof enumerates the elements of every iteration, so it is only run
  // for a loop short enough for that to be cheap.
  if (numIters > 64)
    return failure();
  IndexEvaluator eval{whileOp.getBody(), iv, start, step};
  // carried argument -> the elements each iteration touches
  DenseMap<unsigned, SmallVector<DenseSet<int64_t>>> touched;
  DenseMap<unsigned, Value> bases;
  // `clamps` says the access clamps an index that falls outside the buffer,
  // as a gather and a dynamic slice do: two indices that differ may then land
  // on the same element, so only indices proved to stay inside are admitted.
  // A scatter drops such an update instead, in the loop and in its batched
  // form alike, so its indices need no range.
  auto record = [&](unsigned arg, Value buffer, Value idx, int64_t window,
                    bool clamps) -> LogicalResult {
    auto &sets = touched[arg];
    if (sets.empty())
      sets.resize(numIters);
    int64_t dim = cast<RankedTensorType>(buffer.getType()).getDimSize(0);
    for (int64_t k = 0; k < numIters; ++k) {
      auto e = eval.eval(idx, k);
      if (!e)
        return failure();
      auto it = bases.find(arg);
      if (it == bases.end())
        bases[arg] = e->base;
      else if (it->second != e->base)
        return failure();
      for (int64_t o : e->offsets) {
        // An index out of the buffer's range is clamped by a gather and
        // dropped by a scatter, so two of them can meet where their values
        // say they do not. Only a proof over indices that stay inside holds.
        if (clamps && !e->base && (o < 0 || o + window > dim))
          return failure();
        for (int64_t w = 0; w < window; ++w)
          sets[k].insert(o + w);
      }
    }
    return success();
  };
  for (Operation &op : body.without_terminator()) {
    Value buffer = op.getNumOperands() ? op.getOperand(0) : Value();
    auto root = buffer ? chainRoot.find(buffer) : chainRoot.end();
    if (!buffer || root == chainRoot.end())
      continue;
    auto rank1 = [&](Value v) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      return t && t.getRank() == 1 && t.hasStaticShape();
    };
    if (auto sc = dyn_cast<stablehlo::ScatterOp>(&op)) {
      auto dn = sc.getScatterDimensionNumbers();
      if (!rank1(buffer) || sc.getInputs().size() != 1 ||
          dn.getScatterDimsToOperandDims() != ArrayRef<int64_t>{0} ||
          dn.getInsertedWindowDims() != ArrayRef<int64_t>{0} ||
          !dn.getUpdateWindowDims().empty() ||
          !dn.getInputBatchingDims().empty())
        return failure();
      if (failed(record(root->second, buffer, sc.getScatterIndices(), 1,
                        /*clamps=*/false)))
        return failure();
    } else if (auto g = dyn_cast<stablehlo::GatherOp>(&op)) {
      // one element per start index, whether the size-1 dimension is kept as
      // an offset dimension or collapsed away
      auto dn = g.getDimensionNumbers();
      if (!rank1(buffer) || dn.getStartIndexMap() != ArrayRef<int64_t>{0} ||
          !dn.getOperandBatchingDims().empty() ||
          g.getSliceSizes() != ArrayRef<int64_t>{1})
        return failure();
      if (failed(record(root->second, buffer, g.getStartIndices(), 1,
                        /*clamps=*/true)))
        return failure();
    } else if (auto dus = dyn_cast<stablehlo::DynamicUpdateSliceOp>(&op)) {
      auto ut = dyn_cast<RankedTensorType>(dus.getUpdate().getType());
      if (!rank1(buffer) || !ut || ut.getRank() != 1 || !ut.hasStaticShape() ||
          failed(record(root->second, buffer, dus.getStartIndices()[0],
                        ut.getDimSize(0), /*clamps=*/true)))
        return failure();
    } else if (auto ds = dyn_cast<stablehlo::DynamicSliceOp>(&op)) {
      if (!rank1(buffer) ||
          failed(record(root->second, buffer, ds.getStartIndices()[0],
                        ds.getSliceSizes()[0], /*clamps=*/true)))
        return failure();
    } else {
      return failure(); // some other reach into a carried buffer
    }
  }
  if (touched.empty())
    return failure();
  for (auto &[arg, sets] : touched)
    for (int64_t i = 0; i < numIters; ++i)
      for (int64_t j = i + 1; j < numIters; ++j)
        for (int64_t e : sets[i])
          if (sets[j].contains(e))
            return failure();
  return success();
}

} // namespace

namespace {
// Batches the body of an enzymexla.parallel while over its iterations: a value
// that varies with the iteration gains a leading dimension of the trip count,
// a write into a carried buffer becomes one scatter of every iteration's
// write, and a constant-trip loop nested in the body keeps running, over the
// batched values (the iterations of the parallel loop are independent, so it
// and the parallel loop interchange).
struct ParallelWhileBatcher {
  PatternRewriter &rewriter;
  Location loc;
  enzyme::WhileLoopInfo &info;
  int64_t numIters;
  DenseSet<Value> batched;
  DenseMap<Value, unsigned> chainRoot; // buffer value -> carried arg number
  IRMapping map; // body value -> value emitted in its place
  DenseMap<Operation *, unsigned> innerIv; // nested while -> its iv arg
  Region *loopBody;                        // the parallel loop's body

  ParallelWhileBatcher(PatternRewriter &rewriter, Location loc,
                       enzyme::WhileLoopInfo &info, int64_t numIters,
                       Region *loopBody)
      : rewriter(rewriter), loc(loc), info(info), numIters(numIters),
        loopBody(loopBody) {}

  bool isBatched(Value v) const { return batched.contains(v); }
  bool isChainLink(Operation *op) const {
    return isa<stablehlo::ScatterOp, stablehlo::DynamicUpdateSliceOp>(op) &&
           chainRoot.count(op->getOperand(0));
  }
  static int64_t invariantElements(Value v) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    return t && t.hasStaticShape() ? t.getNumElements() : -1;
  }
  // A region (a scatter's update computation, a reduce body) may capture
  // values of the loop body. An op that keeps its region as is (hoisted, or a
  // write into a carried buffer) takes loop-invariant ones out of the loop
  // along with it, but a per-iteration one cannot be batched inside a scalar
  // region; the batching interfaces take no capture along at all.
  bool captures(Operation *op, bool onlyBatched) const {
    bool found = false;
    for (Region &r : op->getRegions())
      r.walk([&](Operation *inner) {
        for (Value v : inner->getOperands())
          if (!r.isAncestor(v.getParentRegion()) &&
              (!onlyBatched || isBatched(v)))
            found = true;
      });
    return found;
  }

  bool capturesFromLoop(Operation *op) const {
    bool found = false;
    for (Region &r : op->getRegions())
      r.walk([&](Operation *inner) {
        for (Value v : inner->getOperands())
          if (!r.isAncestor(v.getParentRegion()) &&
              (isBatched(v) || loopBody->isAncestor(v.getParentRegion())))
            found = true;
      });
    return found;
  }

  // A carried buffer is the root of a chain of writes (scatter,
  // dynamic_update_slice, a nested loop it is carried through) that ends at
  // the yield, and its only other uses, at any point of the chain, are reads
  // (gather, dynamic_slice): the iterations are independent, so a read sees
  // this iteration's earlier writes over the buffer the loop started from and
  // nothing another iteration writes, which is what a read of the batched
  // writes up to that point returns.
  LogicalResult analyzeChain(BlockArgument arg, Operation *ret, unsigned k) {
    if (auto it = chainRoot.find(arg); it != chainRoot.end())
      return success(it->second == k);
    Value yielded = ret->getOperand(arg.getArgNumber());
    SmallVector<Value> chain{arg};
    for (Value cur = yielded; cur != arg;) {
      Operation *link = cur.getDefiningOp();
      Value prev;
      if (auto sc = dyn_cast_or_null<stablehlo::ScatterOp>(link)) {
        if (sc.getInputs().size() != 1)
          return failure();
        prev = sc.getInputs()[0];
      } else if (auto dus =
                     dyn_cast_or_null<stablehlo::DynamicUpdateSliceOp>(link)) {
        prev = dus.getOperand();
      } else if (auto w = dyn_cast_or_null<stablehlo::WhileOp>(link)) {
        prev = w->getOperand(cast<OpResult>(cur).getResultNumber());
      } else {
        return failure();
      }
      if (chainRoot.count(cur))
        return failure();
      chain.push_back(cur);
      cur = prev;
    }
    for (Value v : chain)
      chainRoot[v] = k;
    for (Value v : chain)
      for (Operation *user : v.getUsers()) {
        // The next write: the buffer goes in as the operand whose result
        // continues the chain.
        bool write = false;
        for (auto [i, o] : llvm::enumerate(user->getOperands()))
          write |= o == v && i < user->getNumResults() &&
                   chainRoot.count(user->getResult(i));
        bool read = isa<stablehlo::GatherOp, stablehlo::DynamicSliceOp>(user) &&
                    user->getOperand(0) == v;
        if (!write && !read && !(user == ret && v == yielded))
          return failure();
      }
    return success();
  }

  // A loop nested in the body runs the same number of times in every
  // iteration of the parallel loop, over values that may vary with it.
  LogicalResult analyzeWhile(stablehlo::WhileOp w) {
    enzyme::WhileLoopInfo wi(w);
    if (failed(wi.computeInfo()) || !wi.isValid() || !wi.isConstant() ||
        !isMemoryEffectFree(w))
      return failure();
    Value wiv = wi.getInductionVariable();
    if (!wiv)
      return failure();
    unsigned ivn = cast<BlockArgument>(wiv).getArgNumber();
    innerIv[w] = ivn;
    // Nothing the condition reads varies with the parallel iteration.
    for (Operation &c : w.getCond().front())
      for (Value v : c.getOperands()) {
        if (auto ca = dyn_cast<BlockArgument>(v);
            ca && ca.getOwner() == &w.getCond().front()) {
          if (ca.getArgNumber() != ivn)
            return failure();
        } else if (isBatched(v) || chainRoot.count(v)) {
          return failure();
        }
      }
    Block &b = w.getBody().front();
    Operation *wret = b.getTerminator();
    for (auto arg : b.getArguments()) {
      unsigned k = arg.getArgNumber();
      if (k == ivn)
        continue;
      Value init = w->getOperand(k);
      if (chainRoot.count(init)) {
        if (failed(analyzeChain(arg, wret, chainRoot.lookup(init))))
          return failure();
      } else if (isBatched(init)) {
        batched.insert(arg);
      }
    }
    // A carried value varies with the parallel iteration when its initial
    // value or the value the body yields does.
    size_t before;
    do {
      before = batched.size();
      if (failed(analyzeBlock(b)))
        return failure();
      for (auto arg : b.getArguments()) {
        unsigned k = arg.getArgNumber();
        if (k != ivn && !chainRoot.count(arg) && isBatched(wret->getOperand(k)))
          batched.insert(arg);
      }
    } while (before != batched.size());
    for (auto arg : b.getArguments())
      if (arg.getArgNumber() != ivn && isBatched(arg))
        batched.insert(w.getResult(arg.getArgNumber()));
    return success();
  }

  // Which body values vary with the iteration, and can every op that does be
  // batched.
  LogicalResult analyzeBlock(Block &body) {
    for (Operation &op : body.without_terminator()) {
      if (auto w = dyn_cast<stablehlo::WhileOp>(&op)) {
        if (failed(analyzeWhile(w)))
          return failure();
        continue;
      }
      if (isChainLink(&op)) {
        // A write into a carried buffer: its indices and update are batched
        // (or the same every iteration); the buffer itself is not.
        if (auto sc = dyn_cast<stablehlo::ScatterOp>(&op);
            sc && sc.getScatterIndices().getType().getRank() == 0)
          return failure();
        if (auto dus = dyn_cast<stablehlo::DynamicUpdateSliceOp>(&op);
            dus && (!dus.getOperand().getType().hasStaticShape() ||
                    !dus.getUpdate().getType().hasStaticShape()))
          return failure();
        if (captures(&op, /*onlyBatched=*/true))
          return failure();
        continue;
      }
      bool any =
          llvm::any_of(op.getOperands(), [&](Value v) { return isBatched(v); });
      if (!any) {
        // Loop-invariant computation: hoisted as is, so it runs once instead of
        // once per iteration. Region ops are only fine when they read nothing
        // carried.
        if (!isMemoryEffectFree(&op))
          return failure();
        if (op.getNumRegions() &&
            (!isa<stablehlo::ReduceOp, stablehlo::ScatterOp>(&op) ||
             captures(&op, /*onlyBatched=*/true)))
          return failure();
        continue;
      }
      // The batching interfaces clone a region as is: a capture of a value
      // of the loop (batched or not) would dangle.
      if (!isMemoryEffectFree(&op) || capturesFromLoop(&op))
        return failure();
      auto broadcastable = [&](Value v) {
        if (isBatched(v))
          return true;
        int64_t n = invariantElements(v);
        return n >= 0 &&
               n * numIters <=
                   ParallelWhileToBatchedScatter::kMaxBroadcastElements;
      };
      if ((op.hasTrait<OpTrait::Elementwise>() ||
           isa<stablehlo::SelectOp>(&op)) &&
          op.getNumResults() == 1) {
        // A broadcast feeding an elementwise op is fused by XLA; no budget.
      } else if (isa<stablehlo::ReshapeOp, stablehlo::BroadcastInDimOp,
                     stablehlo::TransposeOp, stablehlo::SliceOp,
                     stablehlo::ReverseOp, stablehlo::ConcatenateOp>(&op)) {
        // batched along the new leading dimension
      } else if (auto pad = dyn_cast<stablehlo::PadOp>(&op)) {
        if (isBatched(pad.getPaddingValue()))
          return failure();
      } else if (auto rw = dyn_cast<stablehlo::ReduceWindowOp>(&op)) {
        // the batch interface takes the init back to its scalar constant
        if (rw.getInputs().size() != 1 || isBatched(rw.getInitValues()[0]) ||
            !rw.getInitValues()[0].getDefiningOp<stablehlo::ConstantOp>() ||
            !broadcastable(rw.getInputs()[0]))
          return failure();
      } else if (auto red = dyn_cast<stablehlo::ReduceOp>(&op)) {
        // the batch interface takes each init back to its scalar; an input
        // that is the same every iteration is broadcast
        if (llvm::any_of(red.getInitValues(),
                         [&](Value v) { return isBatched(v); }) ||
            !llvm::all_of(red.getInputs(), broadcastable))
          return failure();
      } else if (auto dus = dyn_cast<stablehlo::DynamicUpdateSliceOp>(&op)) {
        // A write into a tensor of this iteration: every iteration's copy is
        // written (the batch interface makes a scatter with batching dims).
        if (!broadcastable(dus.getOperand()))
          return failure();
      } else if (isa<stablehlo::DynamicSliceOp>(&op)) {
        // batched along the new leading dimension, or gathered when the
        // start indices vary
      } else if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(&op)) {
        // the batch interface adds the leading dimension as a batching one
        if (!broadcastable(dot.getLhs()) || !broadcastable(dot.getRhs()))
          return failure();
      } else if (isa<stablehlo::GatherOp>(&op)) {
        // the indices gain the leading dimension (and the operand when it
        // varies), existing batching dimensions shift
      } else if (auto sc = dyn_cast<stablehlo::ScatterOp>(&op)) {
        if (sc.getInputs().size() != 1 || !broadcastable(sc.getInputs()[0]))
          return failure();
      } else if (isa<enzymexla::WrapOp, enzymexla::ExtendOp,
                     enzymexla::RotateOp>(&op)) {
        // along the same dimension past the batch one
        if (!broadcastable(op.getOperand(0)))
          return failure();
      } else {
        return failure();
      }
      for (Value r : op.getResults())
        batched.insert(r);
    }
    return success();
  }

  Type batchedType(Type t) const {
    auto rt = cast<RankedTensorType>(t);
    return RankedTensorType::get(prepend(numIters, rt.getShape()),
                                 rt.getElementType());
  }
  // A value as an operand of a batched op: itself when batched, otherwise
  // broadcast along the new leading dimension.
  Value operand(Value v) {
    Value m = map.lookupOrDefault(v);
    return isBatched(v) ? m : broadcastToIterations(rewriter, loc, m, numIters);
  }

  // The nested loop again, carrying the batched values with their leading
  // dimension and a buffer as the chain of scatters so far.
  void emitWhile(stablehlo::WhileOp w) {
    Block &b = w.getBody().front();
    Operation *wret = b.getTerminator();
    SmallVector<Value> inits;
    SmallVector<Type> types;
    for (auto arg : b.getArguments()) {
      Value init = w->getOperand(arg.getArgNumber());
      inits.push_back(isBatched(arg) ? operand(init)
                                     : map.lookupOrDefault(init));
      types.push_back(inits.back().getType());
    }
    auto nw = stablehlo::WhileOp::create(rewriter, loc, types, inits);
    {
      IRMapping cm = map;
      w.getCond().cloneInto(&nw.getCond(), cm);
      for (auto [na, t] : llvm::zip(nw.getCond().front().getArguments(), types))
        na.setType(t);
    }
    SmallVector<Location> locs(types.size(), loc);
    Block *nb = rewriter.createBlock(&nw.getBody(), {}, types, locs);
    for (auto [oa, na] : llvm::zip(b.getArguments(), nb->getArguments()))
      map.map(oa, na);
    emitBlock(b);
    SmallVector<Value> yields;
    for (auto arg : b.getArguments()) {
      Value y = wret->getOperand(arg.getArgNumber());
      yields.push_back(isBatched(arg) ? operand(y) : map.lookupOrDefault(y));
    }
    stablehlo::ReturnOp::create(rewriter, loc, yields);
    rewriter.setInsertionPointAfter(nw);
    for (auto [o, n] : llvm::zip(w.getResults(), nw.getResults()))
      map.map(o, n);
  }

  void emitBlock(Block &body) {
    ArrayRef<int64_t> batchSizes(numIters);
    for (Operation &op : body.without_terminator()) {
      if (auto w = dyn_cast<stablehlo::WhileOp>(&op)) {
        emitWhile(w);
        continue;
      }
      if (!llvm::any_of(op.getOperands(),
                        [&](Value v) { return isBatched(v); }) &&
          !isChainLink(&op)) {
        Operation *c = rewriter.clone(op, map);
        for (auto [o, n] : llvm::zip(op.getResults(), c->getResults()))
          map.map(o, n);
        continue;
      }
      // The operands of the batched op, for the batching interfaces. A carried
      // buffer written into, or the operand a dynamic_slice or gather reads, is
      // one tensor that every iteration shares.
      Value shared;
      if (isChainLink(&op) ||
          (isa<stablehlo::DynamicSliceOp, stablehlo::GatherOp>(&op) &&
           !isBatched(op.getOperand(0))))
        shared = op.getOperand(0);
      IRMapping bm;
      for (Value v : op.getOperands())
        bm.map(v, v == shared ? map.lookupOrDefault(v) : operand(v));
      if (auto dus = dyn_cast<stablehlo::DynamicUpdateSliceOp>(&op);
          dus && isChainLink(&op)) {
        // Every iteration's window, written by one scatter into the buffer.
        (void)stablehlo::batchDynamicUpdateSliceAsScatter(
            dus, rewriter, bm, batchSizes, /*operandIsBatched=*/false);
      } else if (auto sc = dyn_cast<stablehlo::ScatterOp>(&op);
                 sc && isChainLink(&op)) {
        // Every iteration's scatter into the buffer at once: its indices and
        // updates gain the leading dimension, the buffer does not.
        auto dn = sc.getScatterDimensionNumbers();
        auto ndn = stablehlo::ScatterDimensionNumbersAttr::get(
            op.getContext(), shifted(dn.getUpdateWindowDims()),
            dn.getInsertedWindowDims(), dn.getInputBatchingDims(),
            shifted(dn.getScatterIndicesBatchingDims()),
            dn.getScatterDimsToOperandDims(), dn.getIndexVectorDim() + 1);
        auto nsc = stablehlo::ScatterOp::create(
            rewriter, loc, ValueRange{bm.lookup(sc.getInputs()[0])},
            bm.lookup(sc.getScatterIndices()),
            ValueRange{bm.lookup(sc.getUpdates()[0])}, ndn,
            /*indices_are_sorted=*/false, /*unique_indices=*/false);
        IRMapping rmap = map;
        sc.getUpdateComputation().cloneInto(&nsc.getUpdateComputation(), rmap);
        bm.map(sc.getResult(0), nsc.getResult(0));
      } else if (auto ds = dyn_cast<stablehlo::DynamicSliceOp>(&op);
                 ds && !isBatched(ds.getOperand())) {
        // The operand is the same every iteration. With a single
        // start affine in the induction variable and the others the same every
        // iteration too, the windows of all iterations are a slice of it.
        Value src = bm.lookup(ds.getOperand());
        SmallVector<int64_t> dims;
        bool othersInvariant = true;
        for (auto [d, st] : llvm::enumerate(ds.getStartIndices())) {
          if (isBatched(st))
            dims.push_back(d);
          else
            othersInvariant &= info.isConstantAcrossIterations(st);
        }
        Value windows;
        if (othersInvariant && dims.size() == 1 &&
            info.canHoistOperationFromLoop(src, ds, dims) &&
            info.hoistOperationFromLoop(rewriter, src, ds, dims[0], windows)) {
          // The windows lie one after the other along the sliced dimension:
          // split it into (iteration, window), then put the iteration first.
          int64_t dim = dims[0];
          SmallVector<int64_t> split(ds.getSliceSizes());
          split.insert(split.begin() + dim, numIters);
          windows = stablehlo::ReshapeOpCreate(rewriter, loc, windows, split);
          SmallVector<int64_t> perm{dim};
          for (int64_t d = 0; d < (int64_t)split.size(); ++d)
            if (d != dim)
              perm.push_back(d);
          bm.map(ds.getResult(),
                 stablehlo::TransposeOpCreate(rewriter, loc, windows, perm));
        } else {
          // Otherwise every iteration's start indices, gathered at once.
          (void)stablehlo::batchDynamicSliceAsGather(
              ds, rewriter, bm, batchSizes,
              /*operandIsBatched=*/false);
        }
      } else if (auto g = dyn_cast<stablehlo::GatherOp>(&op);
                 g && !isBatched(g.getOperand())) {
        // Every iteration gathers from the same operand: only the indices gain
        // the leading dimension, rather than broadcasting the operand.
        auto dn = g.getDimensionNumbers();
        auto ng = stablehlo::GatherOp::create(
            rewriter, loc, bm.lookup(g.getOperand()),
            bm.lookup(g.getStartIndices()),
            stablehlo::GatherDimensionNumbersAttr::get(
                op.getContext(), shifted(dn.getOffsetDims()),
                dn.getCollapsedSliceDims(), dn.getOperandBatchingDims(),
                shifted(dn.getStartIndicesBatchingDims()),
                dn.getStartIndexMap(), dn.getIndexVectorDim() + 1),
            g.getSliceSizesAttr(), /*indices_are_sorted=*/false);
        bm.map(g.getResult(), ng.getResult());
      } else if (auto iface = dyn_cast<BatchOpInterface>(&op);
                 iface &&
                 succeeded(iface.createBatch(rewriter, bm, batchSizes))) {
        // Batched by the regular batching machinery.
      } else {
        // Elementwise ops and reshapes batch as themselves on batched types.
        Operation *n = rewriter.clone(op, bm);
        for (Value r : n->getResults())
          r.setType(batchedType(r.getType()));
      }
      for (Value r : op.getResults())
        map.map(r, bm.lookup(r));
    }
  }
};
} // namespace

LogicalResult ParallelWhileToBatchedScatter::matchAndRewriteImpl(
    stablehlo::WhileOp whileOp, PatternRewriter &rewriter) const {
  // The tag says the raiser made this loop out of a parallel dimension of a
  // kernel; an untagged loop has to earn the same conclusion from its indices.
  bool tagged = whileOp->hasAttr("enzymexla.parallel");
  if (!tagged && !prove_independent_iterations)
    return failure();
  enzyme::WhileLoopInfo info(whileOp);
  if (failed(info.computeInfo()) || !info.isValid() || !info.isConstant())
    return failure();
  int64_t numIters = info.getConstantNumIters();
  if (numIters <= 1)
    return failure();
  Value iv = info.getInductionVariable();
  if (!iv)
    return failure();
  Block &body = whileOp.getBody().front();
  auto ret = cast<stablehlo::ReturnOp>(body.getTerminator());
  unsigned ivNum = cast<BlockArgument>(iv).getArgNumber();

  ParallelWhileBatcher batcher(rewriter, whileOp.getLoc(), info, numIters,
                               &whileOp.getBody());
  batcher.batched.insert(iv);
  // Every carried value other than the induction variable is either
  // unchanged or a buffer written through a chain of writes.
  SmallVector<bool> unchanged(body.getNumArguments(), false);
  for (auto arg : body.getArguments()) {
    unsigned k = arg.getArgNumber();
    if (k == ivNum)
      continue;
    if (ret.getOperand(k) == arg)
      unchanged[k] = true;
    else if (failed(batcher.analyzeChain(arg, ret, k)))
      return failure();
  }
  int64_t start = *info.getConstantStart(), step = *info.getConstantStep();
  if (!tagged &&
      failed(proveIterationsIndependent(whileOp, body, iv, numIters, start,
                                        step, batcher.chainRoot)))
    return failure();
  if (failed(batcher.analyzeBlock(body)))
    return failure();

  // Emit before the loop.
  Location loc = whileOp.getLoc();
  rewriter.setInsertionPoint(whileOp);
  auto ivTy = cast<RankedTensorType>(iv.getType());
  Value iota = stablehlo::IotaOp::create(
      rewriter, loc, RankedTensorType::get({numIters}, ivTy.getElementType()),
      0);
  if (step != 1)
    iota = stablehlo::MulOp::create(
        rewriter, loc, iota,
        stablehlo::ConstantOp::create(
            rewriter, loc, cast<ElementsAttr>(makeAttr(iota.getType(), step))));
  if (start != 0)
    iota = stablehlo::AddOp::create(
        rewriter, loc, iota,
        stablehlo::ConstantOp::create(
            rewriter, loc,
            cast<ElementsAttr>(makeAttr(iota.getType(), start))));
  batcher.map.map(iv, iota);
  for (auto arg : body.getArguments())
    if (unchanged[arg.getArgNumber()] || batcher.chainRoot.count(arg))
      batcher.map.map(arg, whileOp->getOperand(arg.getArgNumber()));
  batcher.emitBlock(body);

  SmallVector<Value> results;
  for (auto arg : body.getArguments()) {
    unsigned k = arg.getArgNumber();
    if (k == ivNum) {
      results.push_back(stablehlo::ConstantOp::create(
          rewriter, loc,
          cast<ElementsAttr>(makeAttr(ivTy, start + numIters * step))));
    } else {
      results.push_back(batcher.map.lookup(ret.getOperand(k)));
    }
  }
  rewriter.replaceOp(whileOp, results);
  return success();
}

namespace mlir {
namespace enzyme {

void populateAutoBatchingPassPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx,
                                      AutoBatchingPassPipelineOptions options) {
  if (options.enableSliceToBatch) {
    patterns
        .add<SliceToBatch<stablehlo::DotGeneralOp>,
             SliceToBatch<stablehlo::GatherOp>, SliceToBatch<stablehlo::IotaOp>,
             SliceToBatch<stablehlo::SortOp>,
             SliceToBatchReduceLike<stablehlo::ReduceOp>,
             SliceToBatchReduceLike<stablehlo::ReduceWindowOp>,
             SliceToBatch<stablehlo::ConcatenateOp>,
             SliceToBatch<stablehlo::GetDimensionSizeOp>,
             SliceToBatch<stablehlo::ReverseOp>,
             SliceToBatch<stablehlo::ConvolutionOp>,
             SliceToBatch<stablehlo::ScatterOp>,
             SliceToBatchWithReshapeLikeCheck<stablehlo::BroadcastInDimOp>,
             SliceToBatchWithReshapeLikeCheck<stablehlo::TransposeOp>,
             SliceToBatchElementwise>(ctx);
  }

  if (options.enableConcatInsertDimToBatch) {
    patterns.add<ConcatInsertDimToBatch<stablehlo::DotGeneralOp>,
                 ConcatInsertDimToBatch<stablehlo::GatherOp>,
                 ConcatInsertDimToBatch<stablehlo::IotaOp>,
                 ConcatInsertDimToBatchReduceLike<stablehlo::ReduceOp>,
                 ConcatInsertDimToBatchReduceLike<stablehlo::ReduceWindowOp>,
                 ConcatInsertDimToBatch<stablehlo::ScatterOp>,
                 ConcatInsertDimToBatch<stablehlo::SortOp>,
                 ConcatInsertDimToBatch<stablehlo::ConcatenateOp>,
                 ConcatInsertDimToBatch<stablehlo::GetDimensionSizeOp>,
                 ConcatInsertDimToBatch<stablehlo::ReverseOp>,
                 ConcatInsertDimToBatch<stablehlo::ReduceWindowOp>,
                 ConcatInsertDimToBatch<stablehlo::ConvolutionOp>,
                 ConcatInsertDimElementwiseToBatch>(ctx);
  }

  if (options.whileLoopBatchingMode == "greedy") {
    patterns.add<GreedyWhileLoopBatchFission>(ctx);
  }

  if (options.enableWhileElementwiseReductionToReduce) {
    patterns.add<WhileElementwiseReductionToReduce>(ctx);
  }

  if (options.enableWhileIsCopySimplify) {
    patterns.add<WhileIsCopySimplify>(ctx);
  }

  if (options.enableRemoveLoopCarriedDependenciesFromWhileLoadOperations) {
    patterns.add<RemoveLoopCarriedDependenciesFromWhileLoadOperations>(ctx);
  }

  if (options.enableParallelWhileToBatchedScatter) {
    patterns.add<ParallelWhileToBatchedScatter>(ctx);
  }
}

} // namespace enzyme
} // namespace mlir

struct AutoBatchingPass
    : public enzyme::impl::AutoBatchingPassBase<AutoBatchingPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    mlir::enzyme::AutoBatchingPassPipelineOptions options{
        slice_to_batch_passes,
        concat_insert_dim_passes,
        while_loop_batching_mode,
        while_elementwise_reduction_to_reduce_passes,
        while_is_copy_simplify_passes,
        while_remove_loop_carried_dependencies_from_load_operations,
        parallel_while_to_batched_scatter_passes};
    mlir::enzyme::populateAutoBatchingPassPatterns(patterns, context, options);

    GreedyRewriteConfig config;
    config.setMaxIterations(max_iterations);
    config.setUseTopDownTraversal(top_down);
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};
