#include "mhlo/IR/hlo_ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "optimize-communication"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_OPTIMIZECOMMUNICATION
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::sdy;

using Index = SmallVector<int64_t, 6>;
using Pair = std::pair<int64_t, int64_t>;
using PairVec = SmallVector<Pair, 64>;

template <typename T> Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));
  if (isa<FloatType>(elemType))
    return FloatAttr::get(elemType, val);
  else
    return IntegerAttr::get(elemType, val);
}

SmallVector<int64_t, 6>
computeMeshStrides(const SmallVector<int64_t, 6> &shape) {
  int rank = shape.size();
  SmallVector<int64_t, 6> strides(rank);
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

SmallVector<int64_t, 16>
getShardingDevices(const sdy::TensorShardingAttr &shardingAttr, int dimension,
                   Operation *op) {
  TensorShardingAttr op_shardings[] = {shardingAttr};

  auto meshAttr = mlir::sdy::getCommonMesh(op_shardings, op_shardings, op);

  SmallVector<int64_t, 16> devices;
  for (auto dimSharding : shardingAttr.getDimShardings()) {
    int64_t totalSize = 1;
    for (auto axis : dimSharding.getAxes()) {
      totalSize *= meshAttr.getAxisSize(axis.getName());
    }
    devices.push_back(totalSize);
  }

  return devices;
}

int64_t getNumDevicesAlongDimension(const sdy::TensorShardingAttr &shardingAttr,
                                    int dimension, Operation *op) {
  TensorShardingAttr op_shardings[] = {shardingAttr};

  auto meshAttr = mlir::sdy::getCommonMesh(op_shardings, op_shardings, op);
  int64_t numDevices = 1;
  for (auto meshAxis : shardingAttr.getDimShardings()[dimension].getAxes()) {
    numDevices *= meshAttr.getAxisSize(meshAxis.getName());
  }
  return numDevices;
}

SmallVector<int64_t, 16>
generateShiftPairs(const sdy::TensorShardingAttr &shardingAttr, int dimension,
                   Operation *op, bool leftToRight, bool onlyEdges,
                   bool splitHalfComm = false) {
  TensorShardingAttr op_shardings[] = {shardingAttr};

  auto meshAttr = mlir::sdy::getCommonMesh(op_shardings, op_shardings, op);

  SmallVector<sdy::MeshAxisAttr> meshAxisAttrs =
      llvm::to_vector(meshAttr.getAxes());
  SmallVector<StringRef> meshAxisNames;
  SmallVector<int64_t, 6> meshShape;
  for (const auto &axis : meshAxisAttrs) {
    meshAxisNames.push_back(axis.getName());
    meshShape.push_back(axis.getSize());
  }

  int64_t numDevicesAlongDimension =
      getNumDevicesAlongDimension(shardingAttr, dimension, op);

  SmallVector<int64_t, 6> strides = computeMeshStrides(meshShape);

  auto meshAxes = shardingAttr.getDimShardings()[dimension].getAxes();
  assert(meshAxes.size() == 1); // TODO: support multiple mesh axes
  DenseMap<StringRef, int64_t> meshAxisToIndex;
  for (size_t i = 0; i < meshAxisNames.size(); ++i)
    meshAxisToIndex[meshAxisNames[i]] = i;
  int64_t axisIndex = meshAxisToIndex[meshAxes[0].getName()];

  SmallVector<int64_t, 16> flatPairs;
  for (int64_t srcId = 0; srcId < meshAttr.getTotalSize(); ++srcId) {
    SmallVector<int64_t, 6> idx(meshShape.size());
    int64_t tmp = srcId;
    for (size_t i = 0; i < meshShape.size(); ++i) {
      idx[i] = tmp / strides[i];
      tmp %= strides[i];
    }

    // Get current position along the axis we're shifting on
    int64_t axisSize = meshShape[axisIndex];
    int64_t srcCoord = idx[axisIndex];
    int64_t midPoint = axisSize / 2;

    // Calculate destination coordinate
    int64_t dstCoord;

    if (splitHalfComm) {
      // For split half communication:
      // - Left half (srcCoord < midPoint) communicates to the left
      // - Right half (srcCoord >= midPoint) communicates to the right
      // - No communication across the boundary

      if (srcCoord < midPoint) {
        // Left half always communicates to the left
        dstCoord = (srcCoord - 1 + axisSize) % axisSize;

        // Skip if this would cross the boundary to right half
        if (dstCoord >= midPoint)
          continue;
      } else {
        // Right half always communicates to the right
        dstCoord = (srcCoord + 1) % axisSize;

        // Skip if this would cross the boundary to left half
        if (dstCoord < midPoint)
          continue;
      }
    } else {
      if (leftToRight) {
        dstCoord = (srcCoord + 1) % axisSize;
      } else {
        dstCoord = (srcCoord - 1 + axisSize) % axisSize;
      }

      if (onlyEdges) {
        if (leftToRight) {
          if (!(srcCoord == axisSize - 1 && dstCoord == 0))
            continue;
        } else {
          if (!(srcCoord == 0 && dstCoord == axisSize - 1))
            continue;
        }
      }
    }

    // Calculate the full destination ID
    int64_t dstId = 0;
    for (size_t i = 0; i < meshShape.size(); ++i) {
      int64_t coord = (i == axisIndex) ? dstCoord : idx[i];
      dstId += coord * strides[i];
    }

    // Add the pair in the correct order based on leftToRight
    flatPairs.emplace_back(dstId);
    flatPairs.emplace_back(srcId);
  }

  return flatPairs;
}

void updateManualComputationAxesShape(TensorShardingAttr shardingAttr,
                                      PatternRewriter &rewriter, Operation *op,
                                      SmallVector<StringAttr> &manualAxes,
                                      SmallVector<int64_t> &localShape,
                                      int64_t dimension) {
  TensorShardingAttr op_shardings[] = {shardingAttr};

  auto meshAttr = mlir::sdy::getCommonMesh(op_shardings, op_shardings, op);

  for (auto meshAxis : meshAttr.getAxes()) {
    manualAxes.push_back(rewriter.getStringAttr(meshAxis.getName()));
  }

  auto dimShardings = shardingAttr.getDimShardings();
  auto ndevices = getShardingDevices(shardingAttr, dimension, op);

  for (int i = 0; i < localShape.size(); i++) {
    localShape[i] /= ndevices[i];
  }
}

struct PeriodicConcatSimplify
    : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concat,
                                PatternRewriter &rewriter) const override {
    if (concat.getNumOperands() != 3) {
      return failure();
    }

    auto allOperands = llvm::to_vector(concat.getOperands());

    auto leftSliceOp = allOperands[0].getDefiningOp<stablehlo::SliceOp>();
    if (!leftSliceOp)
      return failure();
    if (!leftSliceOp->hasOneUse())
      return failure();

    auto midOp = allOperands[1];
    if (!midOp.hasOneUse())
      return failure();

    auto rightSliceOp = allOperands[2].getDefiningOp<stablehlo::SliceOp>();
    if (!rightSliceOp)
      return failure();
    if (!rightSliceOp->hasOneUse())
      return failure();

    if (leftSliceOp.getOperand() != rightSliceOp.getOperand())
      return failure();

    Value superSliceOp;

    auto midSliceOp = midOp.getDefiningOp<stablehlo::SliceOp>();
    if (!midSliceOp) {
      superSliceOp = rewriter.create<stablehlo::SliceOp>(
          concat.getLoc(), leftSliceOp.getOperand(),
          rightSliceOp.getStartIndices(), leftSliceOp.getLimitIndices(),
          leftSliceOp.getStrides());
    } else {
      if ((leftSliceOp.getOperand() != midSliceOp.getOperand()) ||
          (rightSliceOp.getStartIndices()[concat.getDimension()] !=
           midSliceOp.getStartIndices()[concat.getDimension()]) ||
          (leftSliceOp.getLimitIndices()[concat.getDimension()] !=
           midSliceOp.getLimitIndices()[concat.getDimension()])) {
        // We need to compute the global slice
        superSliceOp = rewriter.create<stablehlo::SliceOp>(
            concat.getLoc(), leftSliceOp.getOperand(),
            rightSliceOp.getStartIndices(), leftSliceOp.getLimitIndices(),
            leftSliceOp.getStrides());

        if (cast<RankedTensorType>(superSliceOp.getType()).getShape() !=
            cast<RankedTensorType>(midSliceOp.getType()).getShape()) {
          return failure();
        }
      } else {
        superSliceOp = midSliceOp;
      }
    }

    for (int i = 0; i < concat.getType().getShape().size(); i++) {
      if (leftSliceOp.getStrides()[i] != 1 || rightSliceOp.getStrides()[i] != 1)
        return failure();
    }

    for (int i = 0; i < concat.getType().getShape().size(); i++) {
      if (i == concat.getDimension()) {
        continue;
      }
      if (rightSliceOp.getStartIndices()[i] !=
          leftSliceOp.getStartIndices()[i]) {
        return failure();
      }
      if (rightSliceOp.getLimitIndices()[i] !=
          leftSliceOp.getLimitIndices()[i]) {
        return failure();
      }
    }

    auto leftSliceSharding = mlir::sdy::getSharding(leftSliceOp);
    auto rightSliceSharding = mlir::sdy::getSharding(rightSliceOp);
    if (leftSliceSharding != rightSliceSharding) {
      return failure();
    }

    auto midOpSharding = mlir::sdy::getSharding(midOp);
    if (leftSliceSharding != midOpSharding) {
      return failure();
    }

    auto concatSharding = mlir::sdy::getSharding(concat);

    TensorShardingAttr op_shardings[] = {concatSharding};
    TensorShardingAttr op_shardings_in[] = {concatSharding, concatSharding};
    TensorShardingPerValueAttr in_shardings =
        TensorShardingPerValueAttr::get(concat.getContext(), op_shardings_in);
    TensorShardingPerValueAttr out_shardings =
        TensorShardingPerValueAttr::get(concat.getContext(), op_shardings);

    SmallVector<StringAttr> manual_axes;
    SmallVector<int64_t> localShape =
        llvm::to_vector(cast<RankedTensorType>(midOp.getType()).getShape());

    updateManualComputationAxesShape(concatSharding, rewriter, concat,
                                     manual_axes, localShape,
                                     concat.getDimension());

    auto ndevices =
        getShardingDevices(concatSharding, concat.getDimension(), concat);

    int64_t numDevicesAlongDimension = getNumDevicesAlongDimension(
        concatSharding, concat.getDimension(), concat);

    if (numDevicesAlongDimension % 2 != 0) {
      return failure();
    }

    int left_padding = 0;
    int right_padding = 0;
    auto N1 = cast<RankedTensorType>(midOp.getType())
                  .getShape()[concat.getDimension()];
    auto N2 = leftSliceOp.getType().getShape()[concat.getDimension()];
    auto N3 = rightSliceOp.getType().getShape()[concat.getDimension()];
    auto N = N2;
    if (N2 != N3) {
      if (N2 > N3) {
        right_padding = N2 - N3;
        N = N2;
      } else {
        left_padding = N3 - N2;
        N = N3;
      }
    }
    auto T = N1 + 2 * N;

    if (T % numDevicesAlongDimension != 0) {
      int extra =
          ((T / numDevicesAlongDimension) + 1) * numDevicesAlongDimension - T;

      if (extra % 2 == 0) {
        left_padding += extra / 2;
        right_padding += extra / 2;
        N += extra / 2;
        T += extra;
      } else {
        // TODO: handle this if we ever need it. basically we find the nearest
        //       multiple of 2 & numDevicesAlongDimension that is larger than T
        return failure();
      }
    }

    SmallVector<int64_t> localRetShape =
        llvm::to_vector(concat.getType().getShape());
    SmallVector<int64_t> manualOpRetShape =
        llvm::to_vector(concat.getType().getShape());
    for (int i = 0; i < localRetShape.size(); i++) {
      if (i == concat.getDimension()) {
        localRetShape[i] = T / numDevicesAlongDimension;
      } else {
        localRetShape[i] /= ndevices[i];
      }
    }
    manualOpRetShape[concat.getDimension()] = T;

    mlir::Type in_tys[2]{
        RankedTensorType::get(localShape, concat.getType().getElementType()),
        RankedTensorType::get(localShape, concat.getType().getElementType())};
    mlir::Location in_locs[] = {superSliceOp.getLoc(), midOp.getLoc()};

    Value manual_ops[] = {superSliceOp, midOp};
    Type manual_types[] = {RankedTensorType::get(
        manualOpRetShape, concat.getType().getElementType())};
    auto manual = rewriter.create<sdy::ManualComputationOp>(
        concat.getLoc(), manual_types, manual_ops, in_shardings, out_shardings,
        manual_axes);

    auto blk = rewriter.createBlock(&manual.getBody(), manual.getBody().begin(),
                                    in_tys, in_locs);
    auto superSliceInnerArg = blk->getArgument(0);
    auto midOpInnerArg = blk->getArgument(1);
    auto partition_id =
        rewriter.create<stablehlo::PartitionIdOp>(concat.getLoc());

    auto zero = rewriter.create<stablehlo::ConstantOp>(
        concat.getLoc(), rewriter.getZeroAttr(partition_id.getType()));

    Value leftSide = rewriter.create<stablehlo::RemOp>(
        concat.getLoc(), partition_id,
        rewriter.create<stablehlo::ConstantOp>(
            concat.getLoc(), partition_id.getType(),
            makeAttr(partition_id.getType(), numDevicesAlongDimension)
                .cast<ElementsAttr>()));
    Value isLeftSide = rewriter.create<stablehlo::CompareOp>(
        concat.getLoc(), leftSide, zero, stablehlo::ComparisonDirection::EQ);

    // partition_id == (numDevicesAlongDimension -1) (2ny - 1) ...
    Value rightSide = rewriter.create<stablehlo::AddOp>(
        concat.getLoc(), partition_id,
        rewriter.create<stablehlo::ConstantOp>(
            concat.getLoc(),
            makeAttr(partition_id.getType(), 1).cast<ElementsAttr>()));
    rightSide = rewriter.create<stablehlo::RemOp>(
        concat.getLoc(), rightSide,
        rewriter.create<stablehlo::ConstantOp>(
            concat.getLoc(), partition_id.getType(),
            makeAttr(partition_id.getType(), numDevicesAlongDimension)
                .cast<ElementsAttr>()));
    Value isRightSide = rewriter.create<stablehlo::CompareOp>(
        concat.getLoc(), rightSide, zero, stablehlo::ComparisonDirection::EQ);

    auto isNotLeftSide =
        rewriter.create<stablehlo::NotOp>(concat.getLoc(), isLeftSide);
    auto isNotRightSide =
        rewriter.create<stablehlo::NotOp>(concat.getLoc(), isRightSide);
    Type ifTypes[] = {RankedTensorType::get(localRetShape,
                                            concat.getType().getElementType())};
    auto if1 = rewriter.create<stablehlo::IfOp>(
        concat.getLoc(), ifTypes,
        rewriter.create<stablehlo::AndOp>(concat.getLoc(), isNotLeftSide,
                                          isNotRightSide));
    rewriter.create<sdy::ReturnOp>(concat.getLoc(), if1->getResults());

    rewriter.createBlock(&if1.getTrueBranch(), if1.getTrueBranch().begin());

    SmallVector<int64_t> inner_strides(concat.getType().getShape().size(), 1);

    // if ..... !leftSide  && !rightSide
    Value ny_2 = rewriter.create<stablehlo::ConstantOp>(
        concat.getLoc(), partition_id.getType(),
        makeAttr(partition_id.getType(), numDevicesAlongDimension / 2)
            .cast<ElementsAttr>());

    Value isLeftBlock = rewriter.create<stablehlo::CompareOp>(
        concat.getLoc(), leftSide, ny_2, stablehlo::ComparisonDirection::LT);

    int64_t commSize = N - 2 * N / numDevicesAlongDimension;
    auto if3 =
        rewriter.create<stablehlo::IfOp>(concat.getLoc(), ifTypes, isLeftBlock);

    auto sourceTargetPairsVec = generateShiftPairs(
        concatSharding, concat.getDimension(), concat, true, false, true);
    auto sourceTargetPairs = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {(int64_t)(sourceTargetPairsVec.size() / 2), (int64_t)2},
            rewriter.getI64Type()),
        sourceTargetPairsVec);

    Value alpha = rewriter.create<stablehlo::ConstantOp>(
        concat.getLoc(), partition_id.getType(),
        makeAttr(partition_id.getType(), 2 * N / numDevicesAlongDimension)
            .cast<ElementsAttr>());
    auto onePId = rewriter.create<stablehlo::ConstantOp>(
        concat.getLoc(), partition_id.getType(),
        makeAttr(partition_id.getType(), 1).cast<ElementsAttr>());

    // Case I: for the left part of the comm
    {
      rewriter.createBlock(&if3.getTrueBranch(), if3.getTrueBranch().begin());

      SmallVector<int64_t> inner_starts_from_left(
          concat.getType().getShape().size(), 0);
      SmallVector<int64_t> inner_limits_from_left = llvm::to_vector(
          cast<RankedTensorType>(midOpInnerArg.getType()).getShape());
      inner_starts_from_left[concat.getDimension()] =
          inner_limits_from_left[concat.getDimension()] - commSize;

      auto leftSlice = rewriter.create<stablehlo::SliceOp>(
          concat.getLoc(), midOpInnerArg, inner_starts_from_left,
          inner_limits_from_left, inner_strides);

      auto cperm = rewriter.create<stablehlo::CollectivePermuteOp>(
          concat.getLoc(), leftSlice, sourceTargetPairs,
          stablehlo::ChannelHandleAttr::get(concat.getContext(), /*handle*/ 1,
                                            /*type*/ 0));

      Value concat_args_inner[] = {leftSlice, midOpInnerArg};
      auto innerConcat = rewriter.create<stablehlo::ConcatenateOp>(
          concat.getLoc(), concat_args_inner, concat.getDimension());

      SmallVector<Value> dynamicSliceStartSlices;
      for (int i = 0; i < concat.getType().getShape().size(); i++) {
        if (i == concat.getDimension()) {
          auto diffIdx = rewriter.create<stablehlo::MulOp>(
              concat.getLoc(),
              rewriter.create<stablehlo::SubtractOp>(concat.getLoc(),
                                                     partition_id, onePId),
              alpha);
          dynamicSliceStartSlices.push_back(diffIdx);
        } else {
          dynamicSliceStartSlices.push_back(zero);
        }
      }

      auto slicedPart = rewriter.create<stablehlo::DynamicSliceOp>(
          concat.getLoc(), innerConcat, dynamicSliceStartSlices, localRetShape);

      rewriter.create<stablehlo::ReturnOp>(concat.getLoc(),
                                           slicedPart->getResults());
    }

    // Case II: for the right part of the comm
    {
      rewriter.createBlock(&if3.getFalseBranch(), if3.getFalseBranch().begin());

      SmallVector<int64_t> inner_starts_from_right(
          concat.getType().getShape().size(), 0);
      SmallVector<int64_t> inner_limits_from_right = llvm::to_vector(
          cast<RankedTensorType>(midOpInnerArg.getType()).getShape());
      inner_limits_from_right[concat.getDimension()] = commSize;

      auto rightSlice = rewriter.create<stablehlo::SliceOp>(
          concat.getLoc(), midOpInnerArg, inner_starts_from_right,
          inner_limits_from_right, inner_strides);

      auto cperm = rewriter.create<stablehlo::CollectivePermuteOp>(
          concat.getLoc(), rightSlice, sourceTargetPairs,
          stablehlo::ChannelHandleAttr::get(concat.getContext(), /*handle*/ 1,
                                            /*type*/ 0));

      Value concat_args_inner[] = {midOpInnerArg, rightSlice};
      auto innerConcat = rewriter.create<stablehlo::ConcatenateOp>(
          concat.getLoc(), concat_args_inner, concat.getDimension());

      SmallVector<Value> dynamicSliceStartSlices;
      for (int i = 0; i < concat.getType().getShape().size(); i++) {
        if (i == concat.getDimension()) {
          auto diffIdx = rewriter.create<stablehlo::MulOp>(
              concat.getLoc(),
              rewriter.create<stablehlo::AddOp>(concat.getLoc(), partition_id,
                                                onePId),
              alpha);
          auto startIdx = rewriter.create<stablehlo::SubtractOp>(
              concat.getLoc(),
              rewriter.create<stablehlo::ConstantOp>(
                  concat.getLoc(), partition_id.getType(),
                  makeAttr(partition_id.getType(), N).cast<ElementsAttr>()),
              diffIdx);
          dynamicSliceStartSlices.push_back(startIdx);
        } else {
          dynamicSliceStartSlices.push_back(zero);
        }
      }

      auto slicedPart = rewriter.create<stablehlo::DynamicSliceOp>(
          concat.getLoc(), innerConcat, dynamicSliceStartSlices, localRetShape);

      rewriter.create<stablehlo::ReturnOp>(concat.getLoc(),
                                           slicedPart->getResults());
    }

    rewriter.setInsertionPointAfter(if3);
    rewriter.create<stablehlo::ReturnOp>(concat.getLoc(), if3->getResults());

    // else

    rewriter.createBlock(&if1.getFalseBranch(), if1.getFalseBranch().begin());

    SmallVector<int64_t> inner_starts3(concat.getType().getShape().size(), 0);
    SmallVector<int64_t> inner_limits3 = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    inner_limits3[concat.getDimension()] = N;
    auto end_slice = rewriter.create<stablehlo::SliceOp>(
        concat.getLoc(), superSliceInnerArg, inner_starts3, inner_limits3,
        inner_strides);

    auto sourceTargetIdxsLeftEdges = generateShiftPairs(
        concatSharding, concat.getDimension(), concat, true, true);

    auto result_1 = rewriter.create<stablehlo::CollectivePermuteOp>(
        concat.getLoc(), end_slice,
        DenseIntElementsAttr::get(
            RankedTensorType::get(
                {(int64_t)(sourceTargetIdxsLeftEdges.size() / 2), (int64_t)2},
                rewriter.getI64Type()),
            sourceTargetIdxsLeftEdges),
        stablehlo::ChannelHandleAttr::get(concat.getContext(), /*handle*/ 1,
                                          /*type*/ 0));

    SmallVector<int64_t> inner_starts4(concat.getType().getShape().size(), 0);
    SmallVector<int64_t> inner_limits4 = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    inner_starts4[concat.getDimension()] =
        inner_limits4[concat.getDimension()] - N;
    auto start_slice = rewriter.create<stablehlo::SliceOp>(
        concat.getLoc(), superSliceInnerArg, inner_starts4, inner_limits4,
        inner_strides);

    auto sourceTargetIdxsRightEdges = generateShiftPairs(
        concatSharding, concat.getDimension(), concat, false, true);

    auto result_2 = rewriter.create<stablehlo::CollectivePermuteOp>(
        concat.getLoc(), start_slice,
        DenseIntElementsAttr::get(
            RankedTensorType::get(
                {(int64_t)(sourceTargetIdxsRightEdges.size() / 2), (int64_t)2},
                rewriter.getI64Type()),
            sourceTargetIdxsRightEdges),
        stablehlo::ChannelHandleAttr::get(concat.getContext(), /*handle*/ 1,
                                          /*type*/ 0));

    auto if2 =
        rewriter.create<stablehlo::IfOp>(concat.getLoc(), ifTypes, isLeftSide);
    rewriter.create<stablehlo::ReturnOp>(concat.getLoc(), if2->getResults());

    // if lhsSide
    {
      rewriter.createBlock(&if2.getTrueBranch(), if2.getTrueBranch().begin());

      SmallVector<int64_t> inner_starts5(concat.getType().getShape().size(), 0);
      SmallVector<int64_t> inner_limits5 = llvm::to_vector(
          cast<RankedTensorType>(midOpInnerArg.getType()).getShape());
      inner_limits5[concat.getDimension()] = (T / numDevicesAlongDimension) - N;

      auto lhsRightSlice = rewriter.create<stablehlo::SliceOp>(
          concat.getLoc(), midOpInnerArg, inner_starts5, inner_limits5,
          inner_strides);

      Value concat_args2[] = {result_1, lhsRightSlice};
      auto final_result = rewriter.create<stablehlo::ConcatenateOp>(
          concat.getLoc(), concat_args2, concat.getDimension());
      rewriter.create<stablehlo::ReturnOp>(concat.getLoc(),
                                           final_result->getResults());
    }

    // else rightSide
    {
      rewriter.createBlock(&if2.getFalseBranch(), if2.getFalseBranch().begin());

      SmallVector<int64_t> inner_starts6(concat.getType().getShape().size(), 0);
      SmallVector<int64_t> inner_limits6 = llvm::to_vector(
          cast<RankedTensorType>(midOpInnerArg.getType()).getShape());
      inner_starts6[concat.getDimension()] =
          N - (2 * N / numDevicesAlongDimension);

      auto rhsLeftSlice = rewriter.create<stablehlo::SliceOp>(
          concat.getLoc(), midOpInnerArg, inner_starts6, inner_limits6,
          inner_strides);

      Value concat_args2[] = {rhsLeftSlice, result_2};
      auto final_result = rewriter.create<stablehlo::ConcatenateOp>(
          concat.getLoc(), concat_args2, concat.getDimension());
      rewriter.create<stablehlo::ReturnOp>(concat.getLoc(),
                                           final_result->getResults());
    }

    rewriter.setInsertionPointAfter(manual);
    if (left_padding != 0 || right_padding != 0) {
      SmallVector<int64_t> sliceStartIndices(concat.getType().getShape().size(),
                                             0);
      SmallVector<int64_t> sliceLimits = llvm::to_vector(
          cast<RankedTensorType>(manual->getResults()[0].getType()).getShape());

      if (left_padding > 0) {
        sliceStartIndices[concat.getDimension()] = left_padding;
      }

      if (right_padding > 0) {
        sliceLimits[concat.getDimension()] -= right_padding;
      }

      rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
          concat, manual->getResults()[0], sliceStartIndices, sliceLimits,
          inner_strides);
    } else {
      rewriter.replaceOp(concat, manual);
    }

    return success();
  }
};

struct RotateCommOptimize : public OpRewritePattern<enzymexla::RotateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::RotateOp rotate,
                                PatternRewriter &rewriter) const override {
    int32_t ndims = rotate.getType().getRank();
    auto elType = rotate.getType().getElementType();
    auto rotateShape = cast<RankedTensorType>(rotate.getType()).getShape();

    auto rotateSharding = mlir::sdy::getSharding(rotate);

    TensorShardingAttr opShardings[] = {rotateSharding};
    TensorShardingPerValueAttr inShardings =
        TensorShardingPerValueAttr::get(rotate.getContext(), opShardings);
    TensorShardingPerValueAttr outShardings =
        TensorShardingPerValueAttr::get(rotate.getContext(), opShardings);

    SmallVector<StringAttr> manualAxes;
    SmallVector<int64_t> localShape = llvm::to_vector(rotateShape);

    updateManualComputationAxesShape(rotateSharding, rewriter, rotate,
                                     manualAxes, localShape,
                                     rotate.getDimension());

    int64_t numDevicesAlongDimension = getNumDevicesAlongDimension(
        rotateSharding, rotate.getDimension(), rotate);

    SmallVector<int64_t> outputShape = llvm::to_vector(rotateShape);

    int32_t amount = rotate.getAmount();
    int32_t rightPadding = 0;
    Value inputArg = rotate.getOperand();
    if (outputShape[rotate.getDimension()] % numDevicesAlongDimension != 0) {
      int32_t extra =
          ((outputShape[rotate.getDimension()] / numDevicesAlongDimension) +
           1) *
          numDevicesAlongDimension;
      rightPadding = extra - outputShape[rotate.getDimension()];
      amount += rightPadding;
      outputShape[rotate.getDimension()] = extra;

      SmallVector<int64_t> padLow(ndims, 0);
      SmallVector<int64_t> padHigh(ndims, 0);
      padHigh[rotate.getDimension()] = rightPadding;
      localShape[rotate.getDimension()] = extra / numDevicesAlongDimension;
      SmallVector<int64_t> padInner(ndims, 0);

      inputArg = rewriter.create<stablehlo::PadOp>(
          rotate.getLoc(), rotate.getOperand(),
          rewriter.create<stablehlo::ConstantOp>(rotate.getLoc(),
                                                 rewriter.getZeroAttr(elType)),
          padLow, padHigh, padInner);
    }

    SmallVector<int64_t> innerStrides(ndims, 1);
    mlir::Type inTyps[1]{RankedTensorType::get(localShape, elType)};
    mlir::Location inLocs[] = {rotate.getLoc()};

    Value manualOps[] = {inputArg};
    Type manualTypes[] = {RankedTensorType::get(outputShape, elType)};
    auto manual = rewriter.create<sdy::ManualComputationOp>(
        rotate.getLoc(), manualTypes, manualOps, inShardings, outShardings,
        manualAxes);

    {
      auto blk = rewriter.createBlock(&manual.getBody(),
                                      manual.getBody().begin(), inTyps, inLocs);
      auto innerArg = blk->getArgument(0);
      auto partitionId =
          rewriter.create<stablehlo::PartitionIdOp>(rotate.getLoc());

      auto zero = rewriter.create<stablehlo::ConstantOp>(
          rotate.getLoc(), rewriter.getZeroAttr(partitionId.getType()));

      SmallVector<int64_t> innerStarts(ndims, 0);
      SmallVector<int64_t> innerLimits = llvm::to_vector(
          cast<RankedTensorType>(innerArg.getType()).getShape());
      innerLimits[rotate.getDimension()] = amount;
      auto commSlice = rewriter.create<stablehlo::SliceOp>(
          rotate.getLoc(), innerArg, innerStarts, innerLimits, innerStrides);

      auto sourceTargetIdxs = generateShiftPairs(
          rotateSharding, rotate.getDimension(), rotate, true, false);

      auto commResult = rewriter.create<stablehlo::CollectivePermuteOp>(
          rotate.getLoc(), commSlice,
          DenseIntElementsAttr::get(
              RankedTensorType::get(
                  {(int64_t)(sourceTargetIdxs.size() / 2), (int64_t)2},
                  rewriter.getI64Type()),
              sourceTargetIdxs),
          stablehlo::ChannelHandleAttr::get(rotate.getContext(), /*handle*/ 1,
                                            /*type*/ 0));

      SmallVector<int64_t> innerStartsPresent(ndims, 0);
      SmallVector<int64_t> innerLimitsPresent = llvm::to_vector(
          cast<RankedTensorType>(innerArg.getType()).getShape());
      innerStartsPresent[rotate.getDimension()] = amount;
      auto remSlice = rewriter.create<stablehlo::SliceOp>(
          rotate.getLoc(), innerArg, innerStartsPresent, innerLimitsPresent,
          innerStrides);

      Value concatArgs[] = {remSlice, commResult};
      auto innerConcat = rewriter.create<stablehlo::ConcatenateOp>(
          rotate.getLoc(), concatArgs, rotate.getDimension());

      rewriter.create<sdy::ReturnOp>(rotate.getLoc(),
                                     innerConcat->getResults());
    }

    if (rightPadding != 0) {
      rewriter.setInsertionPointAfter(manual);

      SmallVector<int64_t> innerStarts(ndims, 0);
      SmallVector<int64_t> innerLimits = llvm::to_vector(outputShape);
      innerLimits[rotate.getDimension()] -= rightPadding;

      auto sliceRemovePadding = rewriter.create<stablehlo::SliceOp>(
          rotate.getLoc(), manual->getResults()[0], innerStarts, innerLimits,
          innerStrides);
      rewriter.replaceOp(rotate, sliceRemovePadding);
    } else {
      rewriter.replaceOp(rotate, manual);
    }

    return success();
  }
};

struct OptimizeCommunicationPass
    : public enzyme::impl::OptimizeCommunicationBase<
          OptimizeCommunicationPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<PeriodicConcatSimplify, RotateCommOptimize>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
