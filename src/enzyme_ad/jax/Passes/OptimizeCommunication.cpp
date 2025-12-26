#include "mhlo/IR/hlo_ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#endif
#include "shardy/dialect/sdy/ir/dialect.h"
#ifdef __clang__
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif
#include "shardy/dialect/sdy/ir/utils.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/SmallVector.h"

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

// From
// https://github.com/openxla/shardy/blob/0d88b5d25971bd66272195ceeb2288cde72997d0/shardy/dialect/sdy/ir/verifiers.cc#L765C1-L782C2
// Returns the accumulated axes size of a tensor sharding with respect to manual
// axes.
// If an axis in dimShardingAxes belongs to manualAxes, it's an axis
// the user is doing a manual computation on, thus the ManualComputationOp's
// body will have tensors smaller wrt this manual axis.
template <typename T>
int64_t accumulatedManualAxesSize(ArrayRef<AxisRefAttr> dimShardingAxes,
                                  const T &manualAxes, MeshAttr mesh) {
  int64_t axesFactor = 1;
  for (AxisRefAttr axisRef : dimShardingAxes) {
    if (llvm::is_contained(manualAxes, axisRef.getName())) {
      axesFactor *= axisRef.getSize(mesh);
    }
  }

  return axesFactor;
}

// From
// https://github.com/openxla/shardy/blob/0d88b5d25971bd66272195ceeb2288cde72997d0/shardy/dialect/sdy/ir/verifiers.cc#L869
template <typename T>
RankedTensorType getLocalType(RankedTensorType globalType,
                              TensorShardingAttr sharding,
                              const T &manualAxesSet, Operation *op) {
  SymbolTable symbolTable(op->getParentOfType<ModuleOp>());
  SmallVector<int64_t> newDimSizes;
  auto globalRankedType = mlir::cast<RankedTensorType>(globalType);
  for (auto [dimensionSize, dimSharding] : llvm::zip_equal(
           globalRankedType.getShape(), sharding.getDimShardings())) {
    if (dimensionSize == ShapedType::kDynamic) {
      newDimSizes.push_back(ShapedType::kDynamic);
    } else {
      // Safe to call `getMesh` because the sharding was already verified.
      newDimSizes.push_back(
          dimensionSize /
          accumulatedManualAxesSize(dimSharding.getAxes(), manualAxesSet,
                                    sharding.getMesh(symbolTable)));
    }
  }
  return RankedTensorType::get(newDimSizes, globalRankedType.getElementType());
}

using Index = SmallVector<int64_t, 6>;
using Pair = std::pair<int64_t, int64_t>;
using PairVec = SmallVector<Pair, 64>;

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
                                      SmallVectorImpl<int64_t> &localShape,
                                      int64_t dimension) {
  TensorShardingAttr op_shardings[] = {shardingAttr};

  auto meshAttr = mlir::sdy::getCommonMesh(op_shardings, op_shardings, op);
  assert(meshAttr);

  for (auto meshAxis : meshAttr.getAxes()) {
    manualAxes.push_back(rewriter.getStringAttr(meshAxis.getName()));
  }

  auto ndevices = getShardingDevices(shardingAttr, dimension, op);

  for (int i = 0; i < localShape.size(); i++) {
    localShape[i] /= ndevices[i];
  }
}

void generateCommPatternForNonEdges(
    PatternRewriter &rewriter, Operation *op,
    stablehlo::PartitionIdOp partitionId, stablehlo::ConstantOp zero,
    Value superSliceInnerArg, Value midOpInnerArg,
    TensorShardingAttr opSharding, int concatDim, int paddedBoundarySize,
    int numDevicesAlongDimension, int ndims, ArrayRef<int64_t> localRetShape,
    Value leftSide, int &channel_id) {
  auto sourceTargetPairsVec =
      generateShiftPairs(opSharding, concatDim, op, /*leftToRight*/ true,
                         /*onlyEdges*/ false, /*splitHalfComm*/ true);
  auto sourceTargetPairs = DenseIntElementsAttr::get(
      RankedTensorType::get(
          {(int64_t)(sourceTargetPairsVec.size() / 2), (int64_t)2},
          rewriter.getI64Type()),
      sourceTargetPairsVec);

  SmallVector<int64_t> innerStrides(ndims, 1);

  auto partitionIdType = partitionId.getType();
  auto alpha = stablehlo::ConstantOp::create(
      rewriter, op->getLoc(), partitionIdType,
      cast<ElementsAttr>(makeAttr(
          partitionIdType, 2 * paddedBoundarySize / numDevicesAlongDimension)));

  // pId % numDevicesAlongDimension
  auto pidRem = stablehlo::RemOp::create(
      rewriter, op->getLoc(), partitionId,
      stablehlo::ConstantOp::create(
          rewriter, op->getLoc(), partitionIdType,
          cast<ElementsAttr>(
              makeAttr(partitionIdType, numDevicesAlongDimension))));

  auto numDevicesAlongDimension_2 = stablehlo::ConstantOp::create(
      rewriter, op->getLoc(), partitionIdType,
      cast<ElementsAttr>(
          makeAttr(partitionIdType, numDevicesAlongDimension / 2)));
  auto isLeftBlock = stablehlo::CompareOp::create(
      rewriter, op->getLoc(), leftSide, numDevicesAlongDimension_2,
      stablehlo::ComparisonDirection::LT);

  auto commShape = llvm::to_vector(
      cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
  commShape[concatDim] = paddedBoundarySize;
  Type ifTypesCommSelect[] = {RankedTensorType::get(
      commShape,
      cast<RankedTensorType>(superSliceInnerArg.getType()).getElementType())};
  auto ifCondCommSelect = stablehlo::IfOp::create(
      rewriter, op->getLoc(), ifTypesCommSelect, isLeftBlock);

  {
    rewriter.createBlock(&ifCondCommSelect.getTrueBranch(),
                         ifCondCommSelect.getTrueBranch().begin());

    SmallVector<int64_t> innerStarts(ndims, 0);
    SmallVector<int64_t> innerLimits = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    innerStarts[concatDim] = innerLimits[concatDim] - paddedBoundarySize;

    auto leftSlice =
        stablehlo::SliceOp::create(rewriter, op->getLoc(), superSliceInnerArg,
                                   innerStarts, innerLimits, innerStrides);
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                leftSlice->getResults());
  }

  {
    rewriter.createBlock(&ifCondCommSelect.getFalseBranch(),
                         ifCondCommSelect.getFalseBranch().begin());

    SmallVector<int64_t> innerStarts(ndims, 0);
    SmallVector<int64_t> innerLimits = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    innerLimits[concatDim] = paddedBoundarySize;

    auto rightSlice =
        stablehlo::SliceOp::create(rewriter, op->getLoc(), superSliceInnerArg,
                                   innerStarts, innerLimits, innerStrides);
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                rightSlice->getResults());
  }

  rewriter.setInsertionPointAfter(ifCondCommSelect);

  auto cperm = stablehlo::CollectivePermuteOp::create(
      rewriter, op->getLoc(), ifCondCommSelect.getResults()[0],
      sourceTargetPairs,
      stablehlo::ChannelHandleAttr::get(op->getContext(), /*handle*/ channel_id,
                                        /*type*/ 0));
  channel_id++;

  Type ifTypes[] = {RankedTensorType::get(
      localRetShape,
      cast<RankedTensorType>(superSliceInnerArg.getType()).getElementType())};
  auto ifCond =
      stablehlo::IfOp::create(rewriter, op->getLoc(), ifTypes, isLeftBlock);
  stablehlo::ReturnOp::create(rewriter, op->getLoc(), ifCond->getResults());

  // we start with               [ superSliceInnerArg ].           : Length SS
  // [superslice] and       we want. [ left  ][                    ][ right ].
  // : Length RS [retshape == superslice + 2 * paddedBounarySize]

  // We assume left and right both fit in < 1 shard for simplicity
  // The result indexes into [i*RS : (i+1):RS], right input indexes [i*SS :
  // (i+1):SS] Consider the case where we are in the middle region, we know i !=
  // 0 and i != N-1, where N is total number of shards in this dimension

  // Let us now ask the question of which data we need for our result (assuming
  // not left/right), and size of pad is nonzero Since we want to store
  // result[i*RS : (i+1):RS], we want to load from input[(i-1)*RS : i:RS] Let's
  // solve for the boundaries. What is the left node L containing the start of
  // our data, and at what offset
  //.   L*SS + offset == (i-1) * RS, where offset < SS
  //   -> L = [ (i-1) * RS ] / SS;  offset = [ (i-1) * RS ] % SS

  // Now consider the fact that we have a symmetric node count. Combined with
  // the fact we have a left/right within a node, means we only need to look to
  // ourselves or our leftmost neighbor (if on left), or rightmost neighbor (if
  // on the right).

  // The leftmost node (not handled in this function), will get data from <left
  // source> and the first (RS / numDevicesAlongDimension - left) elements from
  // itself The second node will get the last [SS / numDevicesAlongDimension -
  // (RS / numDevicesAlongDimension - left)] elems from the leftmost node,
  //.   and keep the first RS / numDevicesAlongDimension - [SS /
  // numDevicesAlongDimension - (RS / numDevicesAlongDimension - left)] elems
  // from itself .                       = left -

  // Case I: for the left part of the comm
  {
    rewriter.createBlock(&ifCond.getTrueBranch(),
                         ifCond.getTrueBranch().begin());

    Value concatArgs[] = {cperm, midOpInnerArg};
    auto innerConcat = stablehlo::ConcatenateOp::create(rewriter, op->getLoc(),
                                                        concatArgs, concatDim);

    SmallVector<Value> dynamicSliceStartSlices(ndims, zero);

    dynamicSliceStartSlices[concatDim] =
        stablehlo::MulOp::create(rewriter, op->getLoc(), pidRem, alpha);

    auto slicedPart = stablehlo::DynamicSliceOp::create(
        rewriter, op->getLoc(), innerConcat, dynamicSliceStartSlices,
        localRetShape);
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                slicedPart->getResults());
  }

  // Case II: for the right part of the comm
  {
    rewriter.createBlock(&ifCond.getFalseBranch(),
                         ifCond.getFalseBranch().begin());

    Value concatArgs[] = {midOpInnerArg, cperm};
    auto innerConcat = stablehlo::ConcatenateOp::create(rewriter, op->getLoc(),
                                                        concatArgs, concatDim);

    SmallVector<Value> dynamicSliceStartSlices(ndims, zero);
    auto limitIndex =
        stablehlo::MulOp::create(rewriter, op->getLoc(), pidRem, alpha);
    auto constVal = stablehlo::ConstantOp::create(
        rewriter, op->getLoc(), partitionIdType,
        cast<ElementsAttr>(makeAttr(
            partitionIdType, cast<RankedTensorType>(innerConcat.getType())
                                     .getShape()[concatDim] -
                                 localRetShape[concatDim])));

    dynamicSliceStartSlices[concatDim] = stablehlo::SubtractOp::create(
        rewriter, op->getLoc(), constVal, limitIndex);

    auto slicedPart = stablehlo::DynamicSliceOp::create(
        rewriter, op->getLoc(), innerConcat, dynamicSliceStartSlices,
        localRetShape);
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                slicedPart->getResults());
  }

  rewriter.setInsertionPointAfter(ifCond);
  return;
}

mlir::ResultRange
wrapCommPatternForEdges(PatternRewriter &rewriter, Operation *op,
                        stablehlo::PartitionIdOp partitionId,
                        stablehlo::ConstantOp zero, Value superSliceInnerArg,
                        Value midOpInnerArg, TensorShardingAttr opSharding,
                        int concatDim, int N, int numDevicesAlongDimension,
                        int ndims, int T, ArrayRef<int64_t> localRetShape,
                        Value isLeftSide, int &channel_id,
                        bool returnResults = true) {
  auto elemType =
      cast<RankedTensorType>(superSliceInnerArg.getType()).getElementType();

  auto commResultShape = llvm::to_vector(
      cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
  commResultShape[concatDim] = N;

  Type ifTypesCommSelect[] = {RankedTensorType::get(commResultShape, elemType)};
  auto ifCondCommSelect = stablehlo::IfOp::create(
      rewriter, op->getLoc(), ifTypesCommSelect, isLeftSide);

  SmallVector<int64_t> innerStrides(ndims, 1);

  {
    rewriter.createBlock(&ifCondCommSelect.getTrueBranch(),
                         ifCondCommSelect.getTrueBranch().begin());

    SmallVector<int64_t> innerStarts1(ndims, 0);
    SmallVector<int64_t> innerLimits1 = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    innerLimits1[concatDim] = N;
    auto endSlice =
        stablehlo::SliceOp::create(rewriter, op->getLoc(), superSliceInnerArg,
                                   innerStarts1, innerLimits1, innerStrides);
    stablehlo::ReturnOp::create(rewriter, op->getLoc(), endSlice->getResults());
  }

  {
    rewriter.createBlock(&ifCondCommSelect.getFalseBranch(),
                         ifCondCommSelect.getFalseBranch().begin());

    SmallVector<int64_t> innerStarts2(ndims, 0);
    SmallVector<int64_t> innerLimits2 = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    innerStarts2[concatDim] = innerLimits2[concatDim] - N;
    auto startSlice =
        stablehlo::SliceOp::create(rewriter, op->getLoc(), superSliceInnerArg,
                                   innerStarts2, innerLimits2, innerStrides);
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                startSlice->getResults());
  }

  rewriter.setInsertionPointAfter(ifCondCommSelect);

  auto sourceTargetIdxs =
      generateShiftPairs(opSharding, concatDim, op, true, true);
  auto sourceTargetIdxsTmp =
      generateShiftPairs(opSharding, concatDim, op, false, true);
  sourceTargetIdxs.append(sourceTargetIdxsTmp.begin(),
                          sourceTargetIdxsTmp.end());

  auto commResult = stablehlo::CollectivePermuteOp::create(
      rewriter, op->getLoc(), ifCondCommSelect.getResults()[0],
      DenseIntElementsAttr::get(
          RankedTensorType::get(
              {(int64_t)(sourceTargetIdxs.size() / 2), (int64_t)2},
              rewriter.getI64Type()),
          sourceTargetIdxs),
      stablehlo::ChannelHandleAttr::get(op->getContext(), /*handle*/ channel_id,
                                        /*type*/ 0));
  channel_id++;

  Type ifTypes[] = {RankedTensorType::get(localRetShape, elemType)};
  auto ifCondInner =
      stablehlo::IfOp::create(rewriter, op->getLoc(), ifTypes, isLeftSide);

  if (returnResults)
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                ifCondInner->getResults());

  {
    rewriter.createBlock(&ifCondInner.getTrueBranch(),
                         ifCondInner.getTrueBranch().begin());

    Value concatArgs[2];
    concatArgs[0] = commResult;

    if ((T / numDevicesAlongDimension) - N ==
        cast<RankedTensorType>(midOpInnerArg.getType()).getShape()[concatDim]) {
      concatArgs[1] = midOpInnerArg;
    } else {
      SmallVector<int64_t> innerStarts3(ndims, 0);
      SmallVector<int64_t> innerLimits3 = llvm::to_vector(
          cast<RankedTensorType>(midOpInnerArg.getType()).getShape());
      innerLimits3[concatDim] = (T / numDevicesAlongDimension) - N;

      auto lhsRightSlice =
          stablehlo::SliceOp::create(rewriter, op->getLoc(), midOpInnerArg,
                                     innerStarts3, innerLimits3, innerStrides);
      concatArgs[1] = lhsRightSlice;
    }

    auto finalResult = stablehlo::ConcatenateOp::create(rewriter, op->getLoc(),
                                                        concatArgs, concatDim);
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                finalResult->getResults());
  }

  {
    rewriter.createBlock(&ifCondInner.getFalseBranch(),
                         ifCondInner.getFalseBranch().begin());

    Value concatArgs[2];
    concatArgs[1] = commResult;

    if (N - 2 * N / numDevicesAlongDimension == 0) {
      concatArgs[0] = midOpInnerArg;
    } else {
      SmallVector<int64_t> innerStarts4(ndims, 0);
      SmallVector<int64_t> innerLimits4 = llvm::to_vector(
          cast<RankedTensorType>(midOpInnerArg.getType()).getShape());
      innerStarts4[concatDim] = N - (2 * N / numDevicesAlongDimension);

      auto rhsLeftSlice =
          stablehlo::SliceOp::create(rewriter, op->getLoc(), midOpInnerArg,
                                     innerStarts4, innerLimits4, innerStrides);
      concatArgs[0] = rhsLeftSlice;
    }

    auto finalResult = stablehlo::ConcatenateOp::create(rewriter, op->getLoc(),
                                                        concatArgs, concatDim);
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                finalResult->getResults());
  }

  rewriter.setInsertionPointAfter(ifCondInner);
  return ifCondInner->getResults();
}

mlir::ResultRange
extendCommPatternForEdges(PatternRewriter &rewriter, Operation *op,
                          stablehlo::PartitionIdOp partitionId,
                          stablehlo::ConstantOp zero, Value innerArg,
                          TensorShardingAttr opSharding, int concatDim, int N,
                          int numDevicesAlongDimension, int ndims, int T,
                          ArrayRef<int64_t> localRetShape, Value isLeftSide,
                          bool returnResults = true) {
  auto elemType = cast<RankedTensorType>(innerArg.getType()).getElementType();

  SmallVector<int64_t> innerStrides(ndims, 1);

  Type ifTypes[] = {RankedTensorType::get(localRetShape, elemType)};
  auto ifCondInner =
      stablehlo::IfOp::create(rewriter, op->getLoc(), ifTypes, isLeftSide);
  if (returnResults)
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                ifCondInner->getResults());

  {
    rewriter.createBlock(&ifCondInner.getTrueBranch(),
                         ifCondInner.getTrueBranch().begin());

    SmallVector<int64_t> innerStarts1(ndims, 0);
    SmallVector<int64_t> innerLimits1 =
        llvm::to_vector(cast<RankedTensorType>(innerArg.getType()).getShape());
    innerLimits1[concatDim] = N;
    auto startSlice =
        stablehlo::SliceOp::create(rewriter, op->getLoc(), innerArg,
                                   innerStarts1, innerLimits1, innerStrides);

    Value concatArgs[2];
    concatArgs[0] = startSlice;
    if ((T / numDevicesAlongDimension) - N ==
        cast<RankedTensorType>(innerArg.getType()).getShape()[concatDim]) {
      concatArgs[1] = innerArg;
    } else {
      SmallVector<int64_t> innerStarts3(ndims, 0);
      SmallVector<int64_t> innerLimits3 = llvm::to_vector(
          cast<RankedTensorType>(innerArg.getType()).getShape());
      innerLimits3[concatDim] = (T / numDevicesAlongDimension) - N;

      auto lhsRightSlice =
          stablehlo::SliceOp::create(rewriter, op->getLoc(), innerArg,
                                     innerStarts3, innerLimits3, innerStrides);
      concatArgs[1] = lhsRightSlice;
    }

    auto finalResult = stablehlo::ConcatenateOp::create(rewriter, op->getLoc(),
                                                        concatArgs, concatDim);
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                finalResult->getResults());
  }

  {
    rewriter.createBlock(&ifCondInner.getFalseBranch(),
                         ifCondInner.getFalseBranch().begin());

    SmallVector<int64_t> innerStarts2(ndims, 0);
    SmallVector<int64_t> innerLimits2 =
        llvm::to_vector(cast<RankedTensorType>(innerArg.getType()).getShape());
    innerStarts2[concatDim] = innerLimits2[concatDim] - N;
    auto endSlice =
        stablehlo::SliceOp::create(rewriter, op->getLoc(), innerArg,
                                   innerStarts2, innerLimits2, innerStrides);

    Value concatArgs[2];
    concatArgs[1] = endSlice;

    if (N - 2 * N / numDevicesAlongDimension == 0) {
      concatArgs[0] = innerArg;
    } else {
      SmallVector<int64_t> innerStarts4(ndims, 0);
      SmallVector<int64_t> innerLimits4 = llvm::to_vector(
          cast<RankedTensorType>(innerArg.getType()).getShape());
      innerStarts4[concatDim] = N - (2 * N / numDevicesAlongDimension);

      auto rhsLeftSlice =
          stablehlo::SliceOp::create(rewriter, op->getLoc(), innerArg,
                                     innerStarts4, innerLimits4, innerStrides);
      concatArgs[0] = rhsLeftSlice;
    }

    auto finalResult = stablehlo::ConcatenateOp::create(rewriter, op->getLoc(),
                                                        concatArgs, concatDim);
    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                finalResult->getResults());
  }

  rewriter.setInsertionPointAfter(ifCondInner);
  return ifCondInner->getResults();
}

bool isZero(ElementsAttr v) {
  if (!v.isSplat())
    return false;

  auto attr = v.getSplatValue<Attribute>();
  if (auto fp = dyn_cast<FloatAttr>(attr)) {
    if (fp.getValue().isZero())
      return true;
  }
  if (auto fp = dyn_cast<IntegerAttr>(attr)) {
    if (fp.getValue().isZero())
      return true;
  }
  return false;
}

bool isZero(Value v) {
  DenseElementsAttr elem;
  if (matchPattern(v, m_Constant(&elem))) {
    return isZero(elem);
  }
  if (auto sdyConstant = v.getDefiningOp<sdy::ConstantOp>()) {
    return isZero(sdyConstant.getValue());
  }
  return false;
}

// TODO: we might need to update this to use the generalized version for the
// generateShiftPairs function
std::tuple<Value, Value, Value, Value, Value, Value>
getChecksForBoundaries(PatternRewriter &rewriter, Operation *op,
                       stablehlo::PartitionIdOp partitionId,
                       int64_t numDevicesAlongDimension,
                       stablehlo::ConstantOp zero) {
  Value leftSide = stablehlo::RemOp::create(
      rewriter, op->getLoc(), partitionId,
      stablehlo::ConstantOp::create(
          rewriter, op->getLoc(), partitionId.getType(),
          cast<ElementsAttr>(
              makeAttr(partitionId.getType(), numDevicesAlongDimension))));
  Value isLeftSide =
      stablehlo::CompareOp::create(rewriter, op->getLoc(), leftSide, zero,
                                   stablehlo::ComparisonDirection::EQ);

  Value rightSide = stablehlo::AddOp::create(
      rewriter, op->getLoc(), partitionId,
      stablehlo::ConstantOp::create(
          rewriter, op->getLoc(),
          cast<ElementsAttr>(makeAttr(partitionId.getType(), 1))));
  rightSide = stablehlo::RemOp::create(
      rewriter, op->getLoc(), rightSide,
      stablehlo::ConstantOp::create(
          rewriter, op->getLoc(), partitionId.getType(),
          cast<ElementsAttr>(
              makeAttr(partitionId.getType(), numDevicesAlongDimension))));
  Value isRightSide =
      stablehlo::CompareOp::create(rewriter, op->getLoc(), rightSide, zero,
                                   stablehlo::ComparisonDirection::EQ);

  Value isNotLeftSide =
      stablehlo::NotOp::create(rewriter, op->getLoc(), isLeftSide);
  Value isNotRightSide =
      stablehlo::NotOp::create(rewriter, op->getLoc(), isRightSide);

  return {isLeftSide,     isRightSide, isNotLeftSide,
          isNotRightSide, leftSide,    rightSide};
}

//   auto [leftPadding, rightPadding, N, T] =
//      getWrapExtendConfiguration(wrapOperandShape[wrapDimension], lhsValue,
//                                 rhsValue, numDevicesAlongDimension);
std::tuple<int, int, int, int>
getWrapExtendConfiguration(int middleSize, int lhsSize, int rhsSize,
                           int numDevicesAlongDimension) {
  int leftPadding = 0;
  int rightPadding = 0;
  int paddedBoundarySize = lhsSize;

  if (lhsSize != rhsSize) {
    if (lhsSize > rhsSize) {
      rightPadding = lhsSize - rhsSize;
      paddedBoundarySize = lhsSize;
    } else {
      leftPadding = rhsSize - lhsSize;
      paddedBoundarySize = rhsSize;
    }
  }

  auto paddedResultSize = middleSize + 2 * paddedBoundarySize;

  if (paddedResultSize % numDevicesAlongDimension != 0) {
    int extra = ((paddedResultSize / numDevicesAlongDimension) + 1) *
                    numDevicesAlongDimension -
                paddedResultSize;

    if (extra % 2 == 0) {
      leftPadding += extra / 2;
      rightPadding += extra / 2;
      paddedBoundarySize += extra / 2;
      paddedResultSize += extra;
    } else {
      // TODO: handle this if we ever need it. basically we find the nearest
      //       multiple of 2 & numDevicesAlongDimension that is larger than T
      return {-1, -1, -1, -1};
    }
  }

  return {leftPadding, rightPadding, paddedBoundarySize, paddedResultSize};
}

// TODO: check mesh attr and ensure only applied to iota tile
// concat(slice2, op, slice1)
struct PeriodicConcatSimplify
    : public OpRewritePattern<stablehlo::ConcatenateOp> {

  int &channel_id;
  PeriodicConcatSimplify(int &channel_id, MLIRContext *context,
                         PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), channel_id(channel_id) {}

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concat,
                                PatternRewriter &rewriter) const override {
    if (concat->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    if (concat.getNumOperands() != 3) {
      return failure();
    }

    auto elemType = concat.getType().getElementType();
    auto ndims = concat.getType().getRank();
    auto concatShape = concat.getType().getShape();
    auto concatDim = concat.getDimension();

    auto concatSharding = mlir::sdy::getSharding(concat);
    if (!concatSharding)
      return failure();
    auto ndevices = getShardingDevices(concatSharding, concatDim, concat);
    int64_t numDevicesAlongDimension = ndevices[concatDim];

    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          concat,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
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

    for (int i = 0; i < ndims; i++) {
      if (leftSliceOp.getStrides()[i] != 1)
        return failure();
      if (rightSliceOp.getStrides()[i] != 1)
        return failure();
      if (i == concatDim) {
        if (leftSliceOp.getStartIndices()[i] <
            rightSliceOp.getStartIndices()[i]) {
          return failure();
        }
      } else {
        if (leftSliceOp.getStartIndices()[i] !=
            rightSliceOp.getStartIndices()[i]) {
          return failure();
        }
      }
    }

    Value superSliceOp;

    auto midSliceOp = midOp.getDefiningOp<stablehlo::SliceOp>();
    if (!midSliceOp) {
      superSliceOp = stablehlo::SliceOp::create(
          rewriter, concat.getLoc(), leftSliceOp.getOperand(),
          rightSliceOp.getStartIndices(), leftSliceOp.getLimitIndices(),
          leftSliceOp.getStrides());
    } else {
      if ((leftSliceOp.getOperand() != midSliceOp.getOperand()) ||
          (rightSliceOp.getStartIndices()[concatDim] !=
           midSliceOp.getStartIndices()[concatDim]) ||
          (leftSliceOp.getLimitIndices()[concatDim] !=
           midSliceOp.getLimitIndices()[concatDim])) {
        // We need to compute the global slice
        superSliceOp = stablehlo::SliceOp::create(
            rewriter, concat.getLoc(), leftSliceOp.getOperand(),
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

    for (int i = 0; i < ndims; i++) {
      if (leftSliceOp.getStrides()[i] != 1 || rightSliceOp.getStrides()[i] != 1)
        return failure();
    }

    for (int i = 0; i < ndims; i++) {
      if (i == concatDim) {
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
    if (!leftSliceSharding)
      return failure();
    auto rightSliceSharding = mlir::sdy::getSharding(rightSliceOp);
    if (!rightSliceSharding)
      return failure();
    if (leftSliceSharding != rightSliceSharding) {
      return failure();
    }

    auto midOpSharding = mlir::sdy::getSharding(midOp);
    if (!midOpSharding)
      return failure();
    if (leftSliceSharding != midOpSharding) {
      return failure();
    }

    TensorShardingAttr op_shardings[] = {concatSharding};
    TensorShardingAttr op_shardings_in[] = {concatSharding, concatSharding};
    TensorShardingPerValueAttr in_shardings =
        TensorShardingPerValueAttr::get(concat.getContext(), op_shardings_in);
    TensorShardingPerValueAttr out_shardings =
        TensorShardingPerValueAttr::get(concat.getContext(), op_shardings);

    SmallVector<StringAttr> manualAxes;
    SmallVector<int64_t> localShape =
        llvm::to_vector(cast<RankedTensorType>(midOp.getType()).getShape());

    updateManualComputationAxesShape(concatSharding, rewriter, concat,
                                     manualAxes, localShape, concatDim);

    if (numDevicesAlongDimension % 2 != 0) {
      return failure();
    }

    auto [leftPadding, rightPadding, N, T] = getWrapExtendConfiguration(
        cast<RankedTensorType>(midOp.getType()).getShape()[concatDim],
        leftSliceOp.getType().getShape()[concatDim],
        rightSliceOp.getType().getShape()[concatDim], numDevicesAlongDimension);

    if (leftPadding == -1 || rightPadding == -1 || N == -1 || T == -1) {
      return failure();
    }

    SmallVector<int64_t> manualOpRetShape = llvm::to_vector(concatShape);

    bool needsSlice = false;
    SmallVector<int64_t> lowPads(ndims, 0);
    SmallVector<int64_t> highPads(ndims, 0);
    SmallVector<int64_t> interior(ndims, 0);
    for (int i = 0; i < ndims; i++) {
      auto numDevicesAlongDimension =
          getNumDevicesAlongDimension(concatSharding, i, concat);
      if (i == concatDim)
        continue;
      auto shape_i = cast<RankedTensorType>(midOp.getType()).getShape()[i];
      if (shape_i % numDevicesAlongDimension == 0)
        continue;
      highPads[i] =
          numDevicesAlongDimension - (shape_i % numDevicesAlongDimension);
      manualOpRetShape[i] += highPads[i];
      needsSlice = true;
    }
    if (needsSlice) {
      auto cst = stablehlo::ConstantOp::create(rewriter, concat.getLoc(),
                                               rewriter.getZeroAttr(elemType));

      superSliceOp =
          stablehlo::PadOp::create(rewriter, concat.getLoc(), superSliceOp, cst,
                                   lowPads, highPads, interior);

      midOp = stablehlo::PadOp::create(rewriter, concat.getLoc(), midOp, cst,
                                       lowPads, highPads, interior);
    }

    manualOpRetShape[concatDim] = T;

    mlir::Type in_tys[2]{
        getLocalType(cast<RankedTensorType>(superSliceOp.getType()),
                     concatSharding, manualAxes, concat),
        getLocalType(cast<RankedTensorType>(midOp.getType()), concatSharding,
                     manualAxes, concat)};
    mlir::Location in_locs[] = {superSliceOp.getLoc(), midOp.getLoc()};

    auto globalResultType = RankedTensorType::get(manualOpRetShape, elemType);
    auto localResultType =
        getLocalType(globalResultType, concatSharding, manualAxes, concat);

    Value manual_ops[] = {superSliceOp, midOp};
    Type manual_types[] = {globalResultType};
    auto manual = sdy::ManualComputationOp::create(
        rewriter, concat.getLoc(), manual_types, manual_ops, in_shardings,
        out_shardings, manualAxes);

    auto blk = rewriter.createBlock(&manual.getBody(), manual.getBody().begin(),
                                    in_tys, in_locs);
    auto superSliceInnerArg = blk->getArgument(0);
    auto midOpInnerArg = blk->getArgument(1);
    auto partitionId =
        stablehlo::PartitionIdOp::create(rewriter, concat.getLoc());

    auto zero = stablehlo::ConstantOp::create(
        rewriter, concat.getLoc(), rewriter.getZeroAttr(partitionId.getType()));

    auto [isLeftSide, isRightSide, isNotLeftSide, isNotRightSide, leftSide,
          rightSide] = getChecksForBoundaries(rewriter, concat, partitionId,
                                              numDevicesAlongDimension, zero);

    SmallVector<int64_t> innerStrides(ndims, 1);

    if (numDevicesAlongDimension != 2) {
      Type ifTypes[] = {localResultType};
      auto if1 = stablehlo::IfOp::create(
          rewriter, concat.getLoc(), ifTypes,
          stablehlo::AndOp::create(rewriter, concat.getLoc(), isNotLeftSide,
                                   isNotRightSide));

      // if ..... !leftSide  && !rightSide
      {
        rewriter.createBlock(&if1.getTrueBranch(), if1.getTrueBranch().begin());

        generateCommPatternForNonEdges(
            rewriter, concat, partitionId, zero, superSliceInnerArg,
            midOpInnerArg, concatSharding, concatDim, N,
            numDevicesAlongDimension, ndims, localResultType.getShape(),
            leftSide, channel_id);
      }

      // else
      {
        rewriter.createBlock(&if1.getFalseBranch(),
                             if1.getFalseBranch().begin());

        wrapCommPatternForEdges(
            rewriter, concat, partitionId, zero, superSliceInnerArg,
            midOpInnerArg, concatSharding, concatDim, N,
            numDevicesAlongDimension, ndims, T, localResultType.getShape(),
            isLeftSide, channel_id);
      }

      rewriter.setInsertionPointAfter(if1);
      sdy::ReturnOp::create(rewriter, concat.getLoc(), if1->getResults());
    } else {
      auto results = wrapCommPatternForEdges(
          rewriter, concat, partitionId, zero, superSliceInnerArg,
          midOpInnerArg, concatSharding, concatDim, N, numDevicesAlongDimension,
          ndims, T, localResultType.getShape(), isLeftSide, channel_id,
          /*returnResults=*/false);
      sdy::ReturnOp::create(rewriter, concat.getLoc(), results);
    }

    if (concat.getType() != manual->getResult(0).getType()) {
      SmallVector<int64_t> sliceStartIndices(ndims, 0);
      SmallVector<int64_t> sliceLimits =
          llvm::to_vector(concat.getType().getShape());
      if (leftPadding > 0) {
        sliceStartIndices[concatDim] += leftPadding;
        sliceLimits[concatDim] += leftPadding;
      }

      rewriter.setInsertionPointAfter(manual);
      rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
          concat, manual->getResults()[0], sliceStartIndices, sliceLimits,
          innerStrides);
    } else {
      rewriter.replaceOp(concat, manual);
    }

    return success();
  }
};

// TODO: check mesh attr and ensure only applied to iota tile
struct WrapCommOptimize : public OpRewritePattern<enzymexla::WrapOp> {

  int &channel_id;
  WrapCommOptimize(int &channel_id, MLIRContext *context,
                   PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), channel_id(channel_id) {}

  LogicalResult matchAndRewrite(enzymexla::WrapOp wrap,
                                PatternRewriter &rewriter) const override {
    if (wrap->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    auto elemType = wrap.getType().getElementType();
    auto ndims = wrap.getType().getRank();
    auto wrapOperandShape = wrap.getOperand().getType().getShape();
    auto wrapShape = wrap.getType().getShape();
    auto wrapDimension = wrap.getDimension();
    auto lhsValue = wrap.getLhs();
    auto rhsValue = wrap.getRhs();

    auto wrapSharding = mlir::sdy::getSharding(wrap);
    if (!wrapSharding)
      return failure();

    TensorShardingAttr opShardings[] = {wrapSharding};
    TensorShardingPerValueAttr inShardings =
        TensorShardingPerValueAttr::get(wrap.getContext(), opShardings);
    TensorShardingPerValueAttr outShardings =
        TensorShardingPerValueAttr::get(wrap.getContext(), opShardings);

    SmallVector<StringAttr> manualAxes;
    SmallVector<int64_t> localShape = llvm::to_vector(wrapOperandShape);

    updateManualComputationAxesShape(wrapSharding, rewriter, wrap, manualAxes,
                                     localShape, wrapDimension);

    auto ndevices = getShardingDevices(wrapSharding, wrapDimension, wrap);
    int64_t numDevicesAlongDimension = ndevices[wrapDimension];

    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          wrap,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    if (numDevicesAlongDimension % 2 != 0) {
      return failure();
    }

    auto [leftPadding, rightPadding, paddedBoundarySize, paddedResultSize] =
        getWrapExtendConfiguration(wrapOperandShape[wrapDimension], lhsValue,
                                   rhsValue, numDevicesAlongDimension);

    if (leftPadding == -1 || rightPadding == -1 || paddedBoundarySize == -1 ||
        paddedResultSize == -1) {
      return failure();
    }

    if (paddedBoundarySize >
        wrapOperandShape[wrapDimension] / numDevicesAlongDimension)
      return rewriter.notifyMatchFailure(
          wrap, "Amount of shift extends past a shard boundary.");

    SmallVector<int64_t> manualOpRetShape = llvm::to_vector(wrapShape);
    Value inputArg = wrap.getOperand();

    bool needsSlice = false;
    SmallVector<int64_t> lowPads(ndims, 0);
    SmallVector<int64_t> highPads(ndims, 0);
    SmallVector<int64_t> interior(ndims, 0);
    for (int i = 0; i < ndims; i++) {
      auto numDevicesAlongDimension =
          getNumDevicesAlongDimension(wrapSharding, i, wrap);
      if (i == wrapDimension)
        continue;
      if (wrap.getType().getShape()[i] % numDevicesAlongDimension == 0)
        continue;
      highPads[i] = numDevicesAlongDimension -
                    (wrap.getType().getShape()[i] % numDevicesAlongDimension);
      manualOpRetShape[i] += highPads[i];
      needsSlice = true;
    }
    if (needsSlice) {
      inputArg = stablehlo::PadOp::create(
          rewriter, wrap.getLoc(), inputArg,
          stablehlo::ConstantOp::create(rewriter, wrap.getLoc(),
                                        rewriter.getZeroAttr(elemType)),
          lowPads, highPads, interior);
    }
    manualOpRetShape[wrapDimension] = paddedResultSize;

    mlir::Type inTys[1]{getLocalType(cast<RankedTensorType>(inputArg.getType()),
                                     wrapSharding, manualAxes, wrap)};
    mlir::Location inLocs[] = {wrap.getLoc()};

    auto globalResultType = RankedTensorType::get(manualOpRetShape, elemType);
    auto localResultType =
        getLocalType(globalResultType, wrapSharding, manualAxes, wrap);

    Value manualOps[] = {inputArg};
    Type manualTypes[] = {globalResultType};
    auto manual = sdy::ManualComputationOp::create(
        rewriter, wrap.getLoc(), manualTypes, manualOps, inShardings,
        outShardings, manualAxes);

    auto blk = rewriter.createBlock(&manual.getBody(), manual.getBody().begin(),
                                    inTys, inLocs);
    auto innerArg = blk->getArgument(0);

    auto partitionId =
        stablehlo::PartitionIdOp::create(rewriter, wrap.getLoc());

    auto zero = stablehlo::ConstantOp::create(
        rewriter, wrap.getLoc(), rewriter.getZeroAttr(partitionId.getType()));

    auto [isLeftSide, isRightSide, isNotLeftSide, isNotRightSide, leftSide,
          rightSide] = getChecksForBoundaries(rewriter, wrap, partitionId,
                                              numDevicesAlongDimension, zero);

    if (numDevicesAlongDimension != 2) {
      Type ifTypes[] = {localResultType};
      auto ifCond = stablehlo::IfOp::create(
          rewriter, wrap.getLoc(), ifTypes,
          stablehlo::AndOp::create(rewriter, wrap.getLoc(), isNotLeftSide,
                                   isNotRightSide));

      {
        rewriter.createBlock(&ifCond.getTrueBranch(),
                             ifCond.getTrueBranch().begin());

        generateCommPatternForNonEdges(
            rewriter, wrap, partitionId, zero, innerArg, innerArg, wrapSharding,
            wrapDimension, paddedBoundarySize, numDevicesAlongDimension, ndims,
            localResultType.getShape(), leftSide, channel_id);
      }

      {
        rewriter.createBlock(&ifCond.getFalseBranch(),
                             ifCond.getFalseBranch().begin());

        wrapCommPatternForEdges(
            rewriter, wrap, partitionId, zero, innerArg, innerArg, wrapSharding,
            wrapDimension, paddedBoundarySize, numDevicesAlongDimension, ndims,
            paddedResultSize, localResultType.getShape(), isLeftSide,
            channel_id);
      }

      rewriter.setInsertionPointAfter(ifCond);
      sdy::ReturnOp::create(rewriter, wrap.getLoc(), ifCond->getResults());
    } else {
      // There are no edges in this case
      auto results = wrapCommPatternForEdges(
          rewriter, wrap, partitionId, zero, innerArg, innerArg, wrapSharding,
          wrapDimension, paddedBoundarySize, numDevicesAlongDimension, ndims,
          paddedResultSize, localResultType.getShape(), isLeftSide, channel_id,
          /*returnResults=*/false);
      sdy::ReturnOp::create(rewriter, wrap.getLoc(), results);
    }

    if (wrap.getType() != manual->getResult(0).getType()) {
      SmallVector<int64_t> sliceStartIndices(ndims, 0);
      SmallVector<int64_t> sliceLimits =
          llvm::to_vector(wrap.getType().getShape());
      SmallVector<int64_t> innerStrides(ndims, 1);

      if (leftPadding > 0) {
        sliceStartIndices[wrapDimension] += leftPadding;
        sliceLimits[wrapDimension] += leftPadding;
      }

      rewriter.setInsertionPointAfter(manual);
      rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
          wrap, manual->getResults()[0], sliceStartIndices, sliceLimits,
          innerStrides);
    } else {
      rewriter.replaceOp(wrap, manual);
    }

    return success();
  }
};

struct WrapToPadCommOptimize : public OpRewritePattern<enzymexla::WrapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::WrapOp wrap,
                                PatternRewriter &rewriter) const override {
    if (wrap->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    auto elemType = wrap.getType().getElementType();
    auto ndims = wrap.getType().getRank();
    auto wrapOperandShape = wrap.getOperand().getType().getShape();
    auto wrapShape = wrap.getType().getShape();
    auto wrapDimension = wrap.getDimension();

    auto wrapSharding = mlir::sdy::getSharding(wrap);
    if (!wrapSharding)
      return failure();

    auto operandSharding = mlir::sdy::getSharding(wrap.getOperand());
    if (!operandSharding)
      return failure();

    if (operandSharding != wrapSharding)
      return failure();

    auto numDevicesAlongDimension =
        getNumDevicesAlongDimension(wrapSharding, wrapDimension, wrap);
    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          wrap,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    SmallVector<int64_t> strides(ndims, 1);

    SmallVector<int64_t> leftStarts(ndims, 0);
    SmallVector<int64_t> leftLimits = llvm::to_vector(wrapOperandShape);
    leftLimits[wrapDimension] = wrap.getRhs();

    auto leftSliceOp =
        stablehlo::SliceOp::create(rewriter, wrap.getLoc(), wrap.getOperand(),
                                   leftStarts, leftLimits, strides);
    sdy::setSharding(leftSliceOp, wrapSharding);

    SmallVector<int64_t> rightStarts(ndims, 0);
    SmallVector<int64_t> rightLimits = llvm::to_vector(wrapOperandShape);
    rightStarts[wrapDimension] = rightLimits[wrapDimension] - wrap.getLhs();

    auto rightSliceOp =
        stablehlo::SliceOp::create(rewriter, wrap.getLoc(), wrap.getOperand(),
                                   rightStarts, rightLimits, strides);
    sdy::setSharding(rightSliceOp, wrapSharding);

    auto zero = stablehlo::ConstantOp::create(rewriter, wrap.getLoc(),
                                              rewriter.getZeroAttr(elemType));

    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);

    padLow[wrapDimension] = wrapShape[wrapDimension] - wrap.getRhs();
    padHigh[wrapDimension] = 0;
    auto paddedLeftSliceOp = stablehlo::PadOp::create(
        rewriter, wrap.getLoc(), leftSliceOp, zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedLeftSliceOp, wrapSharding);

    padLow[wrapDimension] = 0;
    padHigh[wrapDimension] = wrapShape[wrapDimension] - wrap.getLhs();
    auto paddedRightSliceOp = stablehlo::PadOp::create(
        rewriter, wrap.getLoc(), rightSliceOp, zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedRightSliceOp, wrapSharding);

    padLow[wrapDimension] = wrap.getLhs();
    padHigh[wrapDimension] = wrap.getRhs();
    auto paddedWrapOp =
        stablehlo::PadOp::create(rewriter, wrap.getLoc(), wrap.getOperand(),
                                 zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedWrapOp, wrapSharding);

    auto addOp = stablehlo::AddOp::create(rewriter, wrap.getLoc(),
                                          paddedRightSliceOp, paddedWrapOp);
    mlir::sdy::setSharding(addOp, wrapSharding);

    addOp = stablehlo::AddOp::create(rewriter, wrap.getLoc(), addOp,
                                     paddedLeftSliceOp);
    sdy::setSharding(addOp, wrapSharding);

    rewriter.replaceOp(wrap, addOp);
    return success();
  }
};

// Communication optimization pattern to rewrite wrap operations in terms of
// pad and rotate operations. This can enable better optimization of the
// resulting operations in distributed/sharded contexts.
//
// Pattern: wrap(x, lhs=L, rhs=R) =>
//   p0 = pad(x, lhs=L, rhs=R)
//   p1 = rotate(p0, L)
//   p2 = rotate(p0, -R)
//   result = select based on iota to choose between p1 (left), p0 (middle), p2
//   (right)
//
// Note: Only applied when no sharding is present, to be conservative.
// This pattern is intended to work in conjunction with other communication
// optimization patterns for pad and rotate operations.
struct WrapToRotateOptimize : public OpRewritePattern<enzymexla::WrapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::WrapOp wrap,
                                PatternRewriter &rewriter) const override {
    // Check if already inside a manual computation
    if (wrap->getParentOfType<sdy::ManualComputationOp>())
      return failure();

    auto wrapDimension = wrap.getDimension();

    auto wrapSharding = mlir::sdy::getSharding(wrap);
    int64_t numDevicesAlongDimension = -1;
    if (wrapSharding) {
      auto ndevices = getShardingDevices(wrapSharding, wrapDimension, wrap);
      numDevicesAlongDimension = ndevices[wrapDimension];
    }

    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          wrap,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    auto lhs = wrap.getLhs();
    auto rhs = wrap.getRhs();
    auto elemType = wrap.getType().getElementType();
    auto ndims = wrap.getType().getRank();
    auto wrapShape = wrap.getType().getShape();
    auto operandShape = wrap.getOperand().getType().getShape();

    // Create a zero constant for padding
    auto zero = stablehlo::ConstantOp::create(rewriter, wrap.getLoc(),
                                              rewriter.getZeroAttr(elemType));

    // Create pad operation with zeros
    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);
    padLow[wrapDimension] = lhs;
    padHigh[wrapDimension] = rhs;

    auto paddedOp =
        stablehlo::PadOp::create(rewriter, wrap.getLoc(), wrap.getOperand(),
                                 zero, padLow, padHigh, padInner);
    if (wrapSharding) {
      mlir::sdy::setSharding(paddedOp, wrapSharding);
    }

    // Create two rotate operations
    auto rotateRhsPart = enzymexla::RotateOp::create(
        rewriter, wrap.getLoc(), paddedOp.getResult(),
        static_cast<int32_t>(rhs + lhs), static_cast<int32_t>(wrapDimension));
    if (wrapSharding) {
      mlir::sdy::setSharding(rotateRhsPart, wrapSharding);
    }

    auto rotateLhsPart = enzymexla::RotateOp::create(
        rewriter, wrap.getLoc(), paddedOp.getResult(),
        static_cast<int32_t>(wrapShape[wrapDimension] - lhs - rhs),
        static_cast<int32_t>(wrapDimension));
    if (wrapSharding) {
      mlir::sdy::setSharding(rotateLhsPart, wrapSharding);
    }

    // Create iota along the wrap dimension
    auto iota = stablehlo::IotaOp::create(
        rewriter, wrap.getLoc(),
        RankedTensorType::get(wrapShape, rewriter.getI32Type()), wrapDimension);
    if (wrapSharding) {
      mlir::sdy::setSharding(iota, wrapSharding);
    }

    // Use select to choose between the three parts:
    // - left part (iota < lhs): use rotateLhsPart
    // - middle part (lhs <= iota < lhs + operandShape[dim]): use paddedOp
    // - right part (iota >= lhs + operandShape[dim]): use rotateRhsPart
    auto lhsCheckConstOp = stablehlo::ConstantOp::create(
        rewriter, wrap.getLoc(),
        SplatElementsAttr::get(iota.getType(),
                               rewriter.getI32IntegerAttr(lhs)));
    if (wrapSharding) {
      mlir::sdy::setSharding(lhsCheckConstOp, wrapSharding);
    }

    auto rhsCheckConstOp = stablehlo::ConstantOp::create(
        rewriter, wrap.getLoc(),
        SplatElementsAttr::get(
            iota.getType(),
            rewriter.getI32IntegerAttr(lhs + operandShape[wrapDimension])));
    if (wrapSharding) {
      mlir::sdy::setSharding(rhsCheckConstOp, wrapSharding);
    }

    auto lhsCondOp = stablehlo::CompareOp::create(
        rewriter, wrap.getLoc(), iota, lhsCheckConstOp,
        stablehlo::ComparisonDirection::LT);
    if (wrapSharding) {
      mlir::sdy::setSharding(lhsCondOp, wrapSharding);
    }

    auto midAndLhsCondOp = stablehlo::CompareOp::create(
        rewriter, wrap.getLoc(), iota, rhsCheckConstOp,
        stablehlo::ComparisonDirection::LT);
    if (wrapSharding) {
      mlir::sdy::setSharding(midAndLhsCondOp, wrapSharding);
    }

    auto midAndLhs = stablehlo::SelectOp::create(
        rewriter, wrap.getLoc(), lhsCondOp, rotateLhsPart, paddedOp);
    if (wrapSharding) {
      mlir::sdy::setSharding(midAndLhs, wrapSharding);
    }

    auto result = stablehlo::SelectOp::create(
        rewriter, wrap.getLoc(), midAndLhsCondOp, midAndLhs, rotateRhsPart);
    if (wrapSharding) {
      mlir::sdy::setSharding(result, wrapSharding);
    }

    // Replace the wrap with the select
    rewriter.replaceOp(wrap, result);
    return success();
  }
};

// TODO: check mesh attr and ensure only applied to iota tile
struct ExtendCommOptimize : public OpRewritePattern<enzymexla::ExtendOp> {
  int &channel_id;
  ExtendCommOptimize(int &channel_id, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), channel_id(channel_id) {}
  LogicalResult matchAndRewrite(enzymexla::ExtendOp extend,
                                PatternRewriter &rewriter) const override {
    if (extend->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    auto elemType = extend.getType().getElementType();
    auto ndims = extend.getType().getRank();
    auto extendOperandShape = extend.getOperand().getType().getShape();
    auto extendShape = extend.getType().getShape();
    auto extendDimension = extend.getDimension();
    auto lhsValue = extend.getLhs();
    auto rhsValue = extend.getRhs();

    auto extendSharding = mlir::sdy::getSharding(extend);
    if (!extendSharding)
      return failure();

    TensorShardingAttr opShardings[] = {extendSharding};
    TensorShardingPerValueAttr inShardings =
        TensorShardingPerValueAttr::get(extend.getContext(), opShardings);
    TensorShardingPerValueAttr outShardings =
        TensorShardingPerValueAttr::get(extend.getContext(), opShardings);

    SmallVector<StringAttr> manualAxes;
    SmallVector<int64_t> localShape = llvm::to_vector(extendOperandShape);

    updateManualComputationAxesShape(extendSharding, rewriter, extend,
                                     manualAxes, localShape, extendDimension);

    auto ndevices = getShardingDevices(extendSharding, extendDimension, extend);
    int64_t numDevicesAlongDimension = ndevices[extendDimension];

    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          extend,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    if (numDevicesAlongDimension % 2 != 0) {
      return failure();
    }

    auto [leftPadding, rightPadding, paddedBoundarySize, paddedResultSize] =
        getWrapExtendConfiguration(extendOperandShape[extendDimension],
                                   rhsValue, lhsValue,
                                   numDevicesAlongDimension);

    if (leftPadding == -1 || rightPadding == -1 || paddedBoundarySize == -1 ||
        paddedResultSize == -1) {
      return failure();
    }

    if (paddedBoundarySize >
        extendOperandShape[extendDimension] / numDevicesAlongDimension)
      return rewriter.notifyMatchFailure(
          extend, "Amount of shift extends past a shard boundary.");

    SmallVector<int64_t> manualOpRetShape = llvm::to_vector(extendShape);
    Value inputArg = extend.getOperand();

    bool needsSlice = false;
    SmallVector<int64_t> lowPads(ndims, 0);
    SmallVector<int64_t> highPads(ndims, 0);
    SmallVector<int64_t> interior(ndims, 0);
    for (int i = 0; i < ndims; i++) {
      auto numDevicesAlongDimension =
          getNumDevicesAlongDimension(extendSharding, i, extend);
      if (i == extendDimension)
        continue;
      if (extend.getType().getShape()[i] % numDevicesAlongDimension == 0)
        continue;
      highPads[i] = numDevicesAlongDimension -
                    (extend.getType().getShape()[i] % numDevicesAlongDimension);
      manualOpRetShape[i] += highPads[i];
      needsSlice = true;
    }
    if (needsSlice) {
      inputArg = stablehlo::PadOp::create(
          rewriter, extend.getLoc(), inputArg,
          stablehlo::ConstantOp::create(rewriter, extend.getLoc(),
                                        rewriter.getZeroAttr(elemType)),
          lowPads, highPads, interior);
    }
    manualOpRetShape[extendDimension] = paddedResultSize;

    mlir::Type inTys[1]{getLocalType(cast<RankedTensorType>(inputArg.getType()),
                                     extendSharding, manualAxes, extend)};
    mlir::Location inLocs[] = {extend.getLoc()};

    auto globalResultType = RankedTensorType::get(manualOpRetShape, elemType);
    auto localResultType =
        getLocalType(globalResultType, extendSharding, manualAxes, extend);

    Value manualOps[] = {inputArg};
    Type manualTypes[] = {globalResultType};
    auto manual = sdy::ManualComputationOp::create(
        rewriter, extend.getLoc(), manualTypes, manualOps, inShardings,
        outShardings, manualAxes);

    auto blk = rewriter.createBlock(&manual.getBody(), manual.getBody().begin(),
                                    inTys, inLocs);
    auto innerArg = blk->getArgument(0);

    auto partitionId =
        stablehlo::PartitionIdOp::create(rewriter, extend.getLoc());

    auto zero = stablehlo::ConstantOp::create(
        rewriter, extend.getLoc(), rewriter.getZeroAttr(partitionId.getType()));

    auto [isLeftSide, isRightSide, isNotLeftSide, isNotRightSide, leftSide,
          rightSide] = getChecksForBoundaries(rewriter, extend, partitionId,
                                              numDevicesAlongDimension, zero);

    if (numDevicesAlongDimension != 2) {
      Type ifTypes[] = {localResultType};
      auto ifCond = stablehlo::IfOp::create(
          rewriter, extend.getLoc(), ifTypes,
          stablehlo::AndOp::create(rewriter, extend.getLoc(), isNotLeftSide,
                                   isNotRightSide));

      {
        rewriter.createBlock(&ifCond.getTrueBranch(),
                             ifCond.getTrueBranch().begin());

        generateCommPatternForNonEdges(
            rewriter, extend, partitionId, zero, innerArg, innerArg,
            extendSharding, extendDimension, paddedBoundarySize,
            numDevicesAlongDimension, ndims, localResultType.getShape(),
            leftSide, channel_id);
      }

      {
        rewriter.createBlock(&ifCond.getFalseBranch(),
                             ifCond.getFalseBranch().begin());

        extendCommPatternForEdges(
            rewriter, extend, partitionId, zero, innerArg, extendSharding,
            extendDimension, paddedBoundarySize, numDevicesAlongDimension,
            ndims, paddedResultSize, localResultType.getShape(), isLeftSide);
      }

      rewriter.setInsertionPointAfter(ifCond);
      sdy::ReturnOp::create(rewriter, extend.getLoc(), ifCond->getResults());
    } else {
      auto results = extendCommPatternForEdges(
          rewriter, extend, partitionId, zero, innerArg, extendSharding,
          extendDimension, paddedBoundarySize, numDevicesAlongDimension, ndims,
          paddedResultSize, localResultType.getShape(), isLeftSide,
          /*returnResults=*/false);
      sdy::ReturnOp::create(rewriter, extend.getLoc(), results);
    }

    if (extend.getType() != manual->getResult(0).getType()) {
      SmallVector<int64_t> sliceStartIndices(ndims, 0);
      SmallVector<int64_t> sliceLimits =
          llvm::to_vector(extend.getType().getShape());
      SmallVector<int64_t> innerStrides(ndims, 1);

      if (leftPadding > 0) {
        sliceStartIndices[extendDimension] += leftPadding;
        sliceLimits[extendDimension] += leftPadding;
      }

      rewriter.setInsertionPointAfter(manual);
      rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
          extend, manual->getResults()[0], sliceStartIndices, sliceLimits,
          innerStrides);
    } else {
      rewriter.replaceOp(extend, manual);
    }

    return success();
  }
};

struct ExtendToPadCommOptimize : public OpRewritePattern<enzymexla::ExtendOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::ExtendOp extend,
                                PatternRewriter &rewriter) const override {
    if (extend->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    auto elemType = extend.getType().getElementType();
    auto ndims = extend.getType().getRank();
    auto extendOperandShape = extend.getOperand().getType().getShape();
    auto extendShape = extend.getType().getShape();
    auto extendDimension = extend.getDimension();

    auto extendSharding = mlir::sdy::getSharding(extend);
    if (!extendSharding)
      return failure();

    auto operandSharding = mlir::sdy::getSharding(extend.getOperand());
    if (!operandSharding)
      return failure();

    if (operandSharding != extendSharding)
      return failure();

    auto numDevicesAlongDimension =
        getNumDevicesAlongDimension(extendSharding, extendDimension, extend);
    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          extend,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    SmallVector<int64_t> strides(ndims, 1);

    SmallVector<int64_t> leftStarts(ndims, 0);
    SmallVector<int64_t> leftLimits = llvm::to_vector(extendOperandShape);
    leftLimits[extendDimension] = extend.getLhs();

    auto leftSliceOp = stablehlo::SliceOp::create(
        rewriter, extend.getLoc(), extend.getOperand(), leftStarts, leftLimits,
        strides);
    sdy::setSharding(leftSliceOp, extendSharding);

    SmallVector<int64_t> rightStarts(ndims, 0);
    SmallVector<int64_t> rightLimits = llvm::to_vector(extendOperandShape);
    rightStarts[extendDimension] =
        rightLimits[extendDimension] - extend.getRhs();

    auto rightSliceOp = stablehlo::SliceOp::create(
        rewriter, extend.getLoc(), extend.getOperand(), rightStarts,
        rightLimits, strides);
    sdy::setSharding(rightSliceOp, extendSharding);

    auto zero = stablehlo::ConstantOp::create(rewriter, extend.getLoc(),
                                              rewriter.getZeroAttr(elemType));

    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);

    padLow[extendDimension] = 0;
    padHigh[extendDimension] = extendShape[extendDimension] - extend.getLhs();
    auto paddedLeftSliceOp =
        stablehlo::PadOp::create(rewriter, extend.getLoc(), leftSliceOp, zero,
                                 padLow, padHigh, padInner);
    sdy::setSharding(paddedLeftSliceOp, extendSharding);

    padLow[extendDimension] = extendShape[extendDimension] - extend.getRhs();
    padHigh[extendDimension] = 0;
    auto paddedRightSliceOp =
        stablehlo::PadOp::create(rewriter, extend.getLoc(), rightSliceOp, zero,
                                 padLow, padHigh, padInner);
    sdy::setSharding(paddedRightSliceOp, extendSharding);

    padLow[extendDimension] = extend.getLhs();
    padHigh[extendDimension] = extend.getRhs();
    auto paddedExtendOp =
        stablehlo::PadOp::create(rewriter, extend.getLoc(), extend.getOperand(),
                                 zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedExtendOp, extendSharding);

    auto addOp = stablehlo::AddOp::create(rewriter, extend.getLoc(),
                                          paddedLeftSliceOp, paddedExtendOp);
    mlir::sdy::setSharding(addOp, extendSharding);

    addOp = stablehlo::AddOp::create(rewriter, extend.getLoc(), addOp,
                                     paddedRightSliceOp);
    sdy::setSharding(addOp, extendSharding);

    rewriter.replaceOp(extend, addOp);
    return success();
  }
};

struct ExtendToPadCommOptimize2 : public OpRewritePattern<enzymexla::ExtendOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::ExtendOp extend,
                                PatternRewriter &rewriter) const override {
    if (extend->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    auto elemType = extend.getType().getElementType();
    auto ndims = extend.getType().getRank();
    auto extendDimension = extend.getDimension();

    auto extendSharding = mlir::sdy::getSharding(extend);
    if (!extendSharding)
      return failure();

    auto operandSharding = mlir::sdy::getSharding(extend.getOperand());
    if (!operandSharding)
      return failure();

    if (operandSharding != extendSharding)
      return failure();

    auto numDevicesAlongDimension =
        getNumDevicesAlongDimension(extendSharding, extendDimension, extend);
    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          extend,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    SmallVector<int64_t> strides(ndims, 1);

    auto zero = stablehlo::ConstantOp::create(rewriter, extend.getLoc(),
                                              rewriter.getZeroAttr(elemType));

    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);

    padLow[extendDimension] = extend.getRhs() + extend.getLhs();
    padHigh[extendDimension] = 0;
    auto paddedRightSliceOp =
        stablehlo::PadOp::create(rewriter, extend.getLoc(), extend.getOperand(),
                                 zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedRightSliceOp, extendSharding);

    padLow[extendDimension] = extend.getLhs();
    padHigh[extendDimension] = extend.getRhs();
    auto paddedExtendOp =
        stablehlo::PadOp::create(rewriter, extend.getLoc(), extend.getOperand(),
                                 zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedExtendOp, extendSharding);

    Value current = paddedExtendOp;

    auto iota = stablehlo::IotaOp::create(
        rewriter, extend.getLoc(),
        RankedTensorType::get(paddedExtendOp.getType().getShape(),
                              rewriter.getI32Type()),
        extendDimension);
    sdy::setSharding(iota, extendSharding);

    if (extend.getLhs() != 0) {
      padLow[extendDimension] = 0;
      padHigh[extendDimension] = extend.getRhs() + extend.getLhs();
      auto paddedLeftSliceOp = stablehlo::PadOp::create(
          rewriter, extend.getLoc(), extend.getOperand(), zero, padLow, padHigh,
          padInner);
      sdy::setSharding(paddedLeftSliceOp, extendSharding);

      Value lhsValue = stablehlo::ConstantOp::create(
          rewriter, extend.getLoc(),
          SplatElementsAttr::get(iota.getType(),
                                 rewriter.getI32IntegerAttr(extend.getLhs())));

      auto cond = stablehlo::CompareOp::create(
          rewriter, extend.getLoc(), iota, lhsValue,
          stablehlo::ComparisonDirection::LT);
      sdy::setSharding(cond, extendSharding);

      auto selOp = stablehlo::SelectOp::create(rewriter, extend.getLoc(), cond,
                                               paddedLeftSliceOp, current);
      sdy::setSharding(selOp, extendSharding);
      current = selOp;
    }

    if (extend.getRhs() != 0) {
      padLow[extendDimension] = extend.getRhs() + extend.getLhs();
      padHigh[extendDimension] = 0;
      auto paddedRightSliceOp = stablehlo::PadOp::create(
          rewriter, extend.getLoc(), extend.getOperand(), zero, padLow, padHigh,
          padInner);
      sdy::setSharding(paddedRightSliceOp, extendSharding);

      Value rhsValue = stablehlo::ConstantOp::create(
          rewriter, extend.getLoc(),
          SplatElementsAttr::get(
              iota.getType(),
              rewriter.getI32IntegerAttr(
                  extend.getOperand().getType().getShape()[extendDimension] +
                  extend.getLhs())));

      auto cond = stablehlo::CompareOp::create(
          rewriter, extend.getLoc(), iota, rhsValue,
          stablehlo::ComparisonDirection::LT);
      sdy::setSharding(cond, extendSharding);

      auto selOp = stablehlo::SelectOp::create(rewriter, extend.getLoc(), cond,
                                               current, paddedRightSliceOp);
      sdy::setSharding(selOp, extendSharding);
      current = selOp;
    }

    rewriter.replaceOp(extend, current);
    return success();
  }
};

// TODO: check mesh attr and ensure only applied to iota tile
struct RotateCommOptimize : public OpRewritePattern<enzymexla::RotateOp> {

  int &channel_id;
  RotateCommOptimize(int &channel_id, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), channel_id(channel_id) {}
  LogicalResult matchAndRewrite(enzymexla::RotateOp rotate,
                                PatternRewriter &rewriter) const override {
    if (rotate->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    int32_t ndims = rotate.getType().getRank();
    auto elType = rotate.getType().getElementType();
    auto rotateShape = cast<RankedTensorType>(rotate.getType()).getShape();
    auto rotateDimension = rotate.getDimension();

    auto rotateSharding = mlir::sdy::getSharding(rotate);
    if (!rotateSharding)
      return rewriter.notifyMatchFailure(rotate, "No sharding found.");

    TensorShardingAttr opShardings[] = {rotateSharding};
    TensorShardingPerValueAttr inShardings =
        TensorShardingPerValueAttr::get(rotate.getContext(), opShardings);
    TensorShardingPerValueAttr outShardings =
        TensorShardingPerValueAttr::get(rotate.getContext(), opShardings);

    SmallVector<StringAttr> manualAxes;
    SmallVector<int64_t> localShape = llvm::to_vector(rotateShape);

    updateManualComputationAxesShape(rotateSharding, rewriter, rotate,
                                     manualAxes, localShape, rotateDimension);

    int64_t numDevicesAlongDimension =
        getNumDevicesAlongDimension(rotateSharding, rotateDimension, rotate);

    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          rotate,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    SmallVector<int64_t> outputShape = llvm::to_vector(rotateShape);

    int32_t amount = rotate.getAmount();
    bool leftToRight = amount <= outputShape[rotate.getDimension()] / 2;
    if (!leftToRight)
      amount = outputShape[rotate.getDimension()] - amount;
    assert(amount <= outputShape[rotate.getDimension()] / 2);

    bool onlyComm = amount == (outputShape[rotate.getDimension()] /
                               numDevicesAlongDimension);
    if (amount > outputShape[rotate.getDimension()] / numDevicesAlongDimension)
      return rewriter.notifyMatchFailure(
          rotate, "Amount of shift extends past a shard boundary.");

    int32_t rightPadding = 0;
    Value inputArg = rotate.getOperand();
    if (outputShape[rotate.getDimension()] % numDevicesAlongDimension != 0) {
      return rewriter.notifyMatchFailure(
          rotate,
          "Rotation dimension is not divisible by the number of devices");
      // TODO
      int32_t extra =
          ((outputShape[rotateDimension] / numDevicesAlongDimension) + 1) *
          numDevicesAlongDimension;
      rightPadding = extra - outputShape[rotateDimension];
      amount += rightPadding;
      outputShape[rotateDimension] = extra;

      SmallVector<int64_t> padLow(ndims, 0);
      SmallVector<int64_t> padHigh(ndims, 0);
      padHigh[rotateDimension] = rightPadding;
      localShape[rotateDimension] = extra / numDevicesAlongDimension;
      SmallVector<int64_t> padInner(ndims, 0);

      inputArg = stablehlo::PadOp::create(
          rewriter, rotate.getLoc(), rotate.getOperand(),
          stablehlo::ConstantOp::create(rewriter, rotate.getLoc(),
                                        rewriter.getZeroAttr(elType)),
          padLow, padHigh, padInner);
    }
    if (amount > localShape[rotate.getDimension()]) {
      return rewriter.notifyMatchFailure(rotate, "No local tensor remaining!");
    }
    bool needsSlice = false;
    SmallVector<int64_t> lowPads(ndims, 0);
    SmallVector<int64_t> highPads(ndims, 0);
    SmallVector<int64_t> interior(ndims, 0);
    for (int i = 0; i < ndims; i++) {
      auto numDevicesAlongDimension =
          getNumDevicesAlongDimension(rotateSharding, i, rotate);
      if (i == rotateDimension)
        continue;
      if (outputShape[i] % numDevicesAlongDimension == 0)
        continue;
      highPads[i] = numDevicesAlongDimension -
                    (outputShape[i] % numDevicesAlongDimension);
      needsSlice = true;
    }
    if (needsSlice) {
      inputArg = stablehlo::PadOp::create(
          rewriter, rotate.getLoc(), rotate.getOperand(),
          stablehlo::ConstantOp::create(rewriter, rotate.getLoc(),
                                        rewriter.getZeroAttr(elType)),
          lowPads, highPads, interior);
    }

    SmallVector<int64_t> innerStrides(ndims, 1);
    mlir::Type inTyps[1]{
        getLocalType(cast<RankedTensorType>(inputArg.getType()), rotateSharding,
                     manualAxes, rotate)};
    mlir::Location inLocs[] = {rotate.getLoc()};

    Value manualOps[] = {inputArg};
    Type manualTypes[] = {inputArg.getType()};
    auto manual = sdy::ManualComputationOp::create(
        rewriter, rotate.getLoc(), manualTypes, manualOps, inShardings,
        outShardings, manualAxes);

    {
      auto blk = rewriter.createBlock(&manual.getBody(),
                                      manual.getBody().begin(), inTyps, inLocs);
      auto innerArg = blk->getArgument(0);

      SmallVector<int64_t> innerStarts(ndims, 0);
      SmallVector<int64_t> innerLimits = llvm::to_vector(
          cast<RankedTensorType>(innerArg.getType()).getShape());
      if (leftToRight) {
        innerLimits[rotate.getDimension()] = amount;
      } else {
        innerStarts[rotate.getDimension()] =
            innerLimits[rotate.getDimension()] - amount;
      }
      auto commSlice =
          stablehlo::SliceOp::create(rewriter, rotate.getLoc(), innerArg,
                                     innerStarts, innerLimits, innerStrides);

      auto sourceTargetIdxs = generateShiftPairs(
          rotateSharding, rotate.getDimension(), rotate, leftToRight, false);

      auto commResult = stablehlo::CollectivePermuteOp::create(
          rewriter, rotate.getLoc(), commSlice,
          DenseIntElementsAttr::get(
              RankedTensorType::get(
                  {(int64_t)(sourceTargetIdxs.size() / 2), (int64_t)2},
                  rewriter.getI64Type()),
              sourceTargetIdxs),
          stablehlo::ChannelHandleAttr::get(rotate.getContext(),
                                            /*handle*/ channel_id,
                                            /*type*/ 0));
      channel_id++;

      SmallVector<int64_t> innerStartsPresent(ndims, 0);
      SmallVector<int64_t> innerLimitsPresent = llvm::to_vector(
          cast<RankedTensorType>(innerArg.getType()).getShape());
      if (leftToRight) {
        innerStartsPresent[rotate.getDimension()] = amount;
      } else {
        innerLimitsPresent[rotate.getDimension()] =
            innerLimitsPresent[rotate.getDimension()] - amount;
      }

      if (onlyComm) {
        sdy::ReturnOp::create(rewriter, rotate.getLoc(),
                              commResult->getResults());
      } else {
        auto remSlice = stablehlo::SliceOp::create(
            rewriter, rotate.getLoc(), innerArg, innerStartsPresent,
            innerLimitsPresent, innerStrides);

        std::array<Value, 2> concatArgs;
        if (leftToRight)
          concatArgs = {remSlice, commResult};
        else
          concatArgs = {commResult, remSlice};

        auto innerConcat = stablehlo::ConcatenateOp::create(
            rewriter, rotate.getLoc(), concatArgs, rotateDimension);

        sdy::ReturnOp::create(rewriter, rotate.getLoc(),
                              innerConcat->getResults());
      }
    }

    if (manual->getResult(0).getType() != rotate.getType()) {
      rewriter.setInsertionPointAfter(manual);
      SmallVector<int64_t> innerStarts(ndims, 0);
      SmallVector<int64_t> innerLimits =
          llvm::to_vector(rotate.getType().getShape());
      auto sliceRemovePadding = stablehlo::SliceOp::create(
          rewriter, rotate.getLoc(), manual->getResults()[0], innerStarts,
          innerLimits, innerStrides);
      rewriter.replaceOp(rotate, sliceRemovePadding);
    } else {
      rewriter.replaceOp(rotate, manual);
    }

    return success();
  }
};

struct RotateSpmdOptimize : public OpRewritePattern<enzymexla::RotateOp> {

  RotateSpmdOptimize(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}
  LogicalResult matchAndRewrite(enzymexla::RotateOp rotate,
                                PatternRewriter &rewriter) const override {
    if (rotate->getParentOfType<sdy::ManualComputationOp>())
      return failure();

    auto rotateDimension = rotate.getDimension();
    auto rotateSharding = mlir::sdy::getSharding(rotate);
    if (!rotateSharding)
      return rewriter.notifyMatchFailure(rotate, "No sharding found.");

    int64_t numDevicesAlongDimension =
        getNumDevicesAlongDimension(rotateSharding, rotateDimension, rotate);

    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          rotate,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    // Our op is rotate left, the spmd one is rotate right. rotateleft(x) =
    // rotateright(-x), which we add the dim size to make positive.
    std::string opaque =
        "dimension=" + std::to_string(rotateDimension) + ",amount=" +
        std::to_string(rotate.getType().getShape()[rotateDimension] -
                       rotate.getAmount());

    auto fnSym = rewriter.getStringAttr("_SPMDInternalOp_RotateRight");

    // Replace with a custom call
    auto ccall = rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        rotate, rotate->getResultTypes(), rotate->getOperands(), fnSym,
        /*has_side_effect=*/rewriter.getBoolAttr(false),
        /*backend_config=*/rewriter.getStringAttr(opaque),
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr);
    mlir::sdy::setShardings(ccall, rotateSharding);
    return success();
  }
};

struct RotateToPadCommOptimize : public OpRewritePattern<enzymexla::RotateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::RotateOp rotate,
                                PatternRewriter &rewriter) const override {
    if (rotate->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    auto rotateSharding = mlir::sdy::getSharding(rotate);
    if (!rotateSharding)
      return failure();

    auto ndims = rotate.getType().getRank();
    auto elType = rotate.getType().getElementType();
    auto rotateShape = cast<RankedTensorType>(rotate.getType()).getShape();
    auto rotateDimension = rotate.getDimension();

    auto numDevicesAlongDimension =
        getNumDevicesAlongDimension(rotateSharding, rotateDimension, rotate);
    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          rotate,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    // sl0[A:end], sl1[0:A]
    SmallVector<int64_t> strides(ndims, 1);

    SmallVector<int64_t> sl0_starts(ndims, 0);
    SmallVector<int64_t> sl0_ends(rotateShape);
    sl0_starts[rotate.getDimension()] = rotate.getAmount();

    SmallVector<int64_t> sl1_starts(ndims, 0);
    SmallVector<int64_t> sl1_ends(rotateShape);
    sl1_ends[rotate.getDimension()] = rotate.getAmount();

    auto sl0 = stablehlo::SliceOp::create(rewriter, rotate.getLoc(),
                                          rotate.getOperand(), sl0_starts,
                                          sl0_ends, strides);
    sdy::setSharding(sl0, rotateSharding);

    auto sl1 = stablehlo::SliceOp::create(rewriter, rotate.getLoc(),
                                          rotate.getOperand(), sl1_starts,
                                          sl1_ends, strides);
    sdy::setSharding(sl1, rotateSharding);

    auto zero = stablehlo::ConstantOp::create(rewriter, rotate.getLoc(),
                                              rewriter.getZeroAttr(elType));

    SmallVector<int64_t> padInner(ndims, 0);
    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    padHigh[rotate.getDimension()] =
        sl1.getType().getShape()[rotate.getDimension()];
    auto paddedSl0 = stablehlo::PadOp::create(rewriter, rotate.getLoc(), sl0,
                                              zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedSl0, rotateSharding);

    padHigh[rotate.getDimension()] = 0;
    padLow[rotate.getDimension()] =
        sl0.getType().getShape()[rotate.getDimension()];
    auto paddedSl1 = stablehlo::PadOp::create(rewriter, rotate.getLoc(), sl1,
                                              zero, padLow, padHigh, padInner);

    auto addOp = stablehlo::AddOp::create(rewriter, rotate.getLoc(), paddedSl0,
                                          paddedSl1);
    sdy::setSharding(addOp, rotateSharding);

    rewriter.replaceOp(rotate, addOp);
    return success();
  }
};

// TODO: check mesh attr and ensure only applied to iota tile
// we match if exactly one of the operands is small enough that it can be fit
// into a single shard
struct ConcatTwoOperandsCommOptimize
    : public OpRewritePattern<stablehlo::ConcatenateOp> {
  int &channel_id;
  ConcatTwoOperandsCommOptimize(int &channel_id, MLIRContext *context,
                                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), channel_id(channel_id) {}

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concat,
                                PatternRewriter &rewriter) const override {
    if (concat->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    if (concat.getNumOperands() != 2) {
      return failure();
    }

    auto elemType = concat.getType().getElementType();
    auto ndims = concat.getType().getRank();
    auto concatShape = concat.getType().getShape();
    auto concatDimension = concat.getDimension();

    auto concatSharding = mlir::sdy::getSharding(concat);
    if (!concatSharding)
      return failure();

    auto ndevices = getShardingDevices(concatSharding, concatDimension, concat);
    int64_t numDevicesAlongDimension = ndevices[concatDimension];

    auto allOperands = llvm::to_vector(concat.getOperands());

    for (int i = 0; i < 2; i++) {
      auto opSharding = mlir::sdy::getSharding(allOperands[i]);
      if (!opSharding)
        return failure();
      if (opSharding != concatSharding) {
        return failure();
      }
    }

    RankedTensorType originalArgTypes[2] = {
        cast<RankedTensorType>(allOperands[0].getType()),
        cast<RankedTensorType>(allOperands[1].getType()),
    };

    int leftOperandSize = originalArgTypes[0].getShape()[concatDimension];
    int rightOperandSize = originalArgTypes[1].getShape()[concatDimension];

    // Presume XLA does a good job if they are equal
    if (leftOperandSize == rightOperandSize) {
      return failure();
    }

    int64_t fullSize = concatShape[concatDimension];
    int64_t shardSize = fullSize / numDevicesAlongDimension;
    bool commLeft;
    bool concatLeft; // opposite of commLeft but makes code more readable
    if (leftOperandSize < shardSize) {
      if (rightOperandSize < shardSize)
        return failure();
      commLeft = false; // move slices to the right
    } else if (rightOperandSize < shardSize) {
      commLeft = true; // move slices to the left
    } else {
      return failure();
    }
    concatLeft = !commLeft;

    auto meshAxes = concatSharding.getDimShardings()[concatDimension].getAxes();
    if (meshAxes.size() != 1)
      return failure();

    int64_t padding[2] = {0, 0};
    for (int i = 0; i < 2; i++) {
      auto extra = originalArgTypes[i].getShape()[concatDimension] %
                   numDevicesAlongDimension;
      if (extra == 0)
        continue;
      padding[i] = numDevicesAlongDimension - extra;
      SmallVector<int64_t> padLow(ndims, 0);
      SmallVector<int64_t> padHigh(ndims, 0);
      SmallVector<int64_t> padInner(ndims, 0);
      if (i == 0) {
        padLow[concatDimension] += padding[i];
      } else {
        padHigh[concatDimension] += padding[i];
      }
      auto paddedOperand = stablehlo::PadOp::create(
          rewriter, concat.getLoc(), allOperands[i],
          stablehlo::ConstantOp::create(rewriter, concat.getLoc(),
                                        rewriter.getZeroAttr(elemType)),
          padLow, padHigh, padInner);
      sdy::setSharding(paddedOperand, concatSharding);
      allOperands[i] = paddedOperand;
    }

    RankedTensorType paddedArgTypes[2] = {
        cast<RankedTensorType>(allOperands[0].getType()),
        cast<RankedTensorType>(allOperands[1].getType()),
    };

    SmallVector<StringAttr> axis = {
        rewriter.getStringAttr(meshAxes[0].getName())};

    RankedTensorType paddedLocalArgTypes[2] = {
        getLocalType(paddedArgTypes[0], concatSharding, axis, concat),
        getLocalType(paddedArgTypes[1], concatSharding, axis, concat),
    };

    SmallVector<int64_t> globalResultShape = llvm::to_vector(concatShape);
    globalResultShape[concatDimension] =
        paddedArgTypes[0].getShape()[concatDimension] +
        paddedArgTypes[1].getShape()[concatDimension];
    RankedTensorType globalResultType =
        RankedTensorType::get(globalResultShape, elemType);

    auto localResultType =
        getLocalType(globalResultType, concatSharding, axis, concat);

    mlir::Type inTys[2]{paddedLocalArgTypes[0], paddedLocalArgTypes[1]};
    mlir::Location inLocs[] = {concat.getLoc(), concat.getLoc()};

    TensorShardingAttr opShardings[] = {concatSharding};

    TensorShardingAttr opShardingsIn[] = {concatSharding, concatSharding};
    TensorShardingPerValueAttr inShardings =
        TensorShardingPerValueAttr::get(concat.getContext(), opShardingsIn);
    TensorShardingPerValueAttr outShardings =
        TensorShardingPerValueAttr::get(concat.getContext(), opShardings);

    Type manualTypes[] = {globalResultType};

    auto manual = sdy::ManualComputationOp::create(
        rewriter, concat.getLoc(), manualTypes, allOperands, inShardings,
        outShardings, axis);

    SmallVector<int64_t> innerStrides(ndims, 1);

    {
      auto blk = rewriter.createBlock(&manual.getBody(),
                                      manual.getBody().begin(), inTys, inLocs);
      auto extendArg = blk->getArgument(concatLeft ? 0 : 1);
      auto mainArg = blk->getArgument(concatLeft ? 1 : 0);

      int N1 = cast<RankedTensorType>(extendArg.getType())
                   .getShape()[concatDimension];
      int N2 =
          cast<RankedTensorType>(mainArg.getType()).getShape()[concatDimension];

      auto partitionId =
          stablehlo::PartitionIdOp::create(rewriter, concat.getLoc());
      auto partitionIdType = partitionId.getType();
      auto zero = stablehlo::ConstantOp::create(
          rewriter, concat.getLoc(), rewriter.getZeroAttr(partitionIdType));
      auto onePId = stablehlo::ConstantOp::create(
          rewriter, concat.getLoc(), partitionIdType,
          cast<ElementsAttr>(makeAttr(partitionIdType, 1)));

      auto [isLeftSide, isRightSide, isNotLeftSide, isNotRightSide, leftSide,
            rightSide] = getChecksForBoundaries(rewriter, concat, partitionId,
                                                numDevicesAlongDimension, zero);

      stablehlo::SliceOp commSlice;
      if (concatLeft) {
        SmallVector<int64_t> innerStarts(ndims, 0);
        SmallVector<int64_t> innerLimits = llvm::to_vector(
            cast<RankedTensorType>(mainArg.getType()).getShape());
        innerStarts[concatDimension] = N2 - N1;

        commSlice =
            stablehlo::SliceOp::create(rewriter, concat.getLoc(), mainArg,
                                       innerStarts, innerLimits, innerStrides);
      } else {
        SmallVector<int64_t> innerStarts(ndims, 0);
        SmallVector<int64_t> innerLimits = llvm::to_vector(
            cast<RankedTensorType>(mainArg.getType()).getShape());
        innerLimits[concatDimension] = N1;

        commSlice =
            stablehlo::SliceOp::create(rewriter, concat.getLoc(), mainArg,
                                       innerStarts, innerLimits, innerStrides);
      }

      auto shiftPairs = generateShiftPairs(concatSharding, concatDimension,
                                           concat, commLeft, false);

      auto commResult = stablehlo::CollectivePermuteOp::create(
          rewriter, concat.getLoc(), commSlice,
          DenseIntElementsAttr::get(
              RankedTensorType::get(
                  {(int64_t)(shiftPairs.size() / 2), (int64_t)2},
                  rewriter.getI64Type()),
              shiftPairs),
          stablehlo::ChannelHandleAttr::get(concat.getContext(),
                                            /*handle*/ channel_id,
                                            /*type*/ 0));
      channel_id++;

      Type ifTypes[] = {RankedTensorType::get(
          cast<ShapedType>(commResult.getType()).getShape(), elemType)};
      stablehlo::IfOp ifCond =
          stablehlo::IfOp::create(rewriter, concat.getLoc(), ifTypes,
                                  concatLeft ? isLeftSide : isRightSide);

      {
        rewriter.createBlock(&ifCond.getTrueBranch(),
                             ifCond.getTrueBranch().begin());

        stablehlo::ReturnOp::create(rewriter, concat.getLoc(), extendArg);
      }

      {
        rewriter.createBlock(&ifCond.getFalseBranch(),
                             ifCond.getFalseBranch().begin());

        stablehlo::ReturnOp::create(rewriter, concat.getLoc(),
                                    commResult->getResults());
      }

      rewriter.setInsertionPointAfter(ifCond);

      Value concatArgs[2];
      if (concatLeft) {
        concatArgs[0] = ifCond.getResult(0);
        concatArgs[1] = mainArg;
      } else {
        concatArgs[0] = mainArg;
        concatArgs[1] = ifCond.getResult(0);
      }

      auto extendSize =
          cast<ShapedType>(extendArg.getType()).getShape()[concatDimension];

      auto concatResult = stablehlo::ConcatenateOp::create(
          rewriter, concat.getLoc(), concatArgs, concatDimension);

      auto alpha = stablehlo::ConstantOp::create(
          rewriter, concat.getLoc(), partitionIdType,
          cast<ElementsAttr>(makeAttr(partitionIdType,
                                      extendSize / numDevicesAlongDimension)));

      SmallVector<Value> dynamicSliceStartSlices;
      for (int i = 0; i < ndims; i++) {
        if (i == concatDimension) {
          if (concatLeft) {
            dynamicSliceStartSlices.push_back(stablehlo::MulOp::create(
                rewriter, concat.getLoc(),
                stablehlo::SubtractOp::create(rewriter, concat.getLoc(),
                                              partitionId, onePId),
                alpha));
          } else {
            auto diffIdx = stablehlo::MulOp::create(
                rewriter, concat.getLoc(),
                stablehlo::AddOp::create(rewriter, concat.getLoc(), partitionId,
                                         onePId),
                alpha);
            dynamicSliceStartSlices.push_back(stablehlo::SubtractOp::create(
                rewriter, concat.getLoc(),
                stablehlo::ConstantOp::create(
                    rewriter, concat.getLoc(), partitionIdType,
                    cast<ElementsAttr>(makeAttr(partitionIdType, extendSize))),
                diffIdx));
          }
        } else {
          dynamicSliceStartSlices.push_back(zero);
        }
      }

      auto slicedPart = stablehlo::DynamicSliceOp::create(
          rewriter, concat.getLoc(), concatResult, dynamicSliceStartSlices,
          localResultType.getShape());
      sdy::ReturnOp::create(rewriter, concat.getLoc(),
                            slicedPart->getResults());
    }

    if (padding[0] != 0 || padding[1] != 0) {
      SmallVector<int64_t> sliceStartIndices(ndims, 0);
      SmallVector<int64_t> sliceLimits =
          llvm::to_vector(globalResultType.getShape());
      sliceStartIndices[concatDimension] = padding[0];
      sliceLimits[concatDimension] -= padding[1];

      rewriter.setInsertionPointAfter(manual);
      auto sliceRemovePadding = stablehlo::SliceOp::create(
          rewriter, concat.getLoc(), manual->getResults()[0], sliceStartIndices,
          sliceLimits, innerStrides);

      rewriter.replaceOp(concat, sliceRemovePadding);
    } else {
      rewriter.replaceOp(concat, manual);
    }

    return success();
  }
};

struct DUSToPadComm : public OpRewritePattern<stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DynamicUpdateSliceOp dus,
                                PatternRewriter &rewriter) const override {
    auto sharding = mlir::sdy::getSharding(dus);
    if (!sharding)
      return failure();

    auto ndims = dus.getType().getShape().size();
    auto elementType = dus.getType().getElementType();
    auto startIndices = dus.getStartIndices();

    auto operand = dus.getOperand();
    auto operandSharding = mlir::sdy::getSharding(operand);
    if (!operandSharding || (operandSharding != sharding)) {
      // If operand is a constant, then we can construct a new constant
      auto stablehloConstant = operand.getDefiningOp<stablehlo::ConstantOp>();
      auto sdyConstant = operand.getDefiningOp<sdy::ConstantOp>();
      if (stablehloConstant) {
        operand = cast<mlir::TypedValue<mlir::RankedTensorType>>(
            rewriter.clone(*stablehloConstant)->getResult(0));
      } else if (sdyConstant) {
        operand = cast<mlir::TypedValue<mlir::RankedTensorType>>(
            rewriter.clone(*sdyConstant)->getResult(0));
      } else {
        return failure();
      }
      mlir::sdy::setSharding(operand, sharding);
    }

    auto update = dus.getUpdate();
    auto updateSharding = mlir::sdy::getSharding(update);
    if (!updateSharding || (updateSharding != operandSharding)) {
      // If update is a constant, then we can construct a new constant
      auto stablehloConstant = update.getDefiningOp<stablehlo::ConstantOp>();
      auto sdyConstant = update.getDefiningOp<sdy::ConstantOp>();
      if (stablehloConstant) {
        update = cast<mlir::TypedValue<mlir::RankedTensorType>>(
            rewriter.clone(*stablehloConstant)->getResult(0));
      } else if (sdyConstant) {
        update = cast<mlir::TypedValue<mlir::RankedTensorType>>(
            rewriter.clone(*sdyConstant)->getResult(0));
      } else {
        return failure();
      }
      mlir::sdy::setSharding(update, sharding);
    }

    auto operandShape = cast<RankedTensorType>(operand.getType()).getShape();
    auto updateShape = cast<RankedTensorType>(update.getType()).getShape();

    SmallVector<int64_t> constantStartIndices(ndims, 0);
    for (int i = 0; i < ndims; i++) {
      DenseIntElementsAttr cst;
      APInt val;
      if (auto cstAttr = startIndices[i].getDefiningOp<sdy::ConstantOp>()) {
        val = *cast<DenseIntElementsAttr>(cstAttr.getValue()).begin();
      } else if (matchPattern(startIndices[i], m_Constant(&cst))) {
        val = (*cst.begin());
      } else {
        return failure();
      }

      if (val.isNegative())
        return failure();

      constantStartIndices[i] = val.getZExtValue();
    }

    auto zero = stablehlo::ConstantOp::create(
        rewriter, dus.getLoc(), rewriter.getZeroAttr(elementType));
    auto one = stablehlo::ConstantOp::create(rewriter, dus.getLoc(),
                                             rewriter.getOneAttr(elementType));

    SmallVector<int64_t> padInner(ndims, 0);

    SmallVector<int64_t> updatePadLow(ndims, 0);
    SmallVector<int64_t> updatePadHigh(ndims, 0);
    for (int i = 0; i < ndims; i++) {
      updatePadLow[i] = constantStartIndices[i];
      updatePadHigh[i] =
          operandShape[i] - updateShape[i] - constantStartIndices[i];
    }
    Value updatePad = nullptr;
    if (!isZero(update)) {
      auto updatePadOp =
          stablehlo::PadOp::create(rewriter, dus.getLoc(), update, zero,
                                   updatePadLow, updatePadHigh, padInner);
      sdy::setSharding(updatePadOp, sharding);
      updatePad = updatePadOp;
    }

    Value maskedOperand = nullptr;
    if (!isZero(operand)) {
      auto updateType = cast<RankedTensorType>(update.getType());
      auto zeroAttr =
          DenseElementsAttr::get(updateType, rewriter.getZeroAttr(elementType));
      auto zeroUpdateOp = stablehlo::ConstantOp::create(rewriter, dus.getLoc(),
                                                        updateType, zeroAttr);
      sdy::setSharding(zeroUpdateOp, sharding);

      auto maskOp =
          stablehlo::PadOp::create(rewriter, dus.getLoc(), zeroUpdateOp, one,
                                   updatePadLow, updatePadHigh, padInner);
      sdy::setSharding(maskOp, sharding);

      auto maskedOperandOp =
          stablehlo::MulOp::create(rewriter, dus.getLoc(), operand, maskOp);
      sdy::setSharding(maskedOperandOp, sharding);
      maskedOperand = maskedOperandOp;
    }

    Value resultV = nullptr;
    if (maskedOperand && updatePad) {
      auto result = stablehlo::AddOp::create(rewriter, dus.getLoc(),
                                             maskedOperand, updatePad);
      sdy::setSharding(result, sharding);
      resultV = result;
    } else if (maskedOperand) {
      resultV = maskedOperand;
    } else if (updatePad) {
      resultV = updatePad;
    } else {
      auto cst = stablehlo::ConstantOp::create(
          rewriter, dus.getLoc(), dus.getType(),
          cast<ElementsAttr>(rewriter.getZeroAttr(dus.getType())));
      sdy::setSharding(cst, sharding);
      resultV = cst;
    }

    rewriter.replaceOp(dus, resultV);
    return success();
  }
};

// Given two values, an 'operand', and 'update', and a multidimensional index
// set
//.  lowPad[idx], and highPad[idx], index into the data within update if we are
//>= lowPad for all idx and < lowPad + globalUnpaddedUpdateType aka totalShape -
// highPad
void multiDimensionalSelect(Location loc, PatternRewriter &rewriter,
                            RankedTensorType globalResultType,
                            RankedTensorType localResultType,
                            const SmallVectorImpl<int64_t> &lowPads,
                            const SmallVectorImpl<int64_t> &highPads,
                            const SmallVectorImpl<int64_t> &updatedDims,
                            const SmallVectorImpl<int64_t> &updatedShardedDims,
                            Value innerOperand, Value innerUpdate,
                            RankedTensorType globalUnPaddedUpdateType,
                            Operation *op) {
  int ndims = globalResultType.getShape().size();

  auto partitionId = stablehlo::PartitionIdOp::create(rewriter, loc);
  auto partitionType = partitionId.getType();

  DenseMap<std::pair<int64_t, Type>, Value> constantCache;
  auto getOrCreateConstant = [&](int64_t v, Type TT = nullptr) -> Value {
    if (!TT)
      TT = partitionType;
    auto key = std::make_pair(v, TT);
    auto found = constantCache.find(key);
    if (found != constantCache.end())
      return found->second;
    auto cst = stablehlo::ConstantOp::create(
        rewriter, loc, TT, cast<ElementsAttr>(makeAttr(TT, v)));
    constantCache[key] = cst;
    return cst;
  };

  Value innerUpdateVal = innerUpdate;
  if (updatedDims.size() != updatedShardedDims.size()) {
    auto zero = getOrCreateConstant(0);
    SmallVector<Value> newStarts(ndims, zero);
    for (int i = 0; i < ndims; i++) {
      if (llvm::is_contained(updatedShardedDims, i))
        continue;
      newStarts[i] = getOrCreateConstant(lowPads[i]);
    }
    innerUpdateVal = stablehlo::DynamicUpdateSliceOp::create(
        rewriter, loc, innerOperand, innerUpdate, newStarts);
  }

  SmallVector<Value> multiDimIdxs;
  {
    auto sharding = sdy::getSharding(op->getResult(0));
    Value cur = partitionId;
    for (int i = 0; i < localResultType.getShape().size(); i++) {
      auto globalSz = globalResultType.getShape()[i];
      auto localSz = localResultType.getShape()[i];
      auto ndevices = getShardingDevices(sharding, i, op);
      int64_t nDevices = ndevices[i];

      if (globalSz == localSz || nDevices == 1) {
        multiDimIdxs.push_back(getOrCreateConstant(0));
      } else if (i == localResultType.getShape().size() - 1) {
        multiDimIdxs.push_back(cur);
      } else {
        auto cst = getOrCreateConstant(nDevices);
        multiDimIdxs.push_back(
            stablehlo::RemOp::create(rewriter, loc, cur, cst));
      }

      if (i != localResultType.getShape().size() - 1 && nDevices != 1) {
        auto cst = getOrCreateConstant(nDevices);
        cur = stablehlo::DivOp::create(rewriter, loc, cur, cst);
      }
    }
  }

  SmallVector<Value> leftSides;
  SmallVector<Value> rightSides;

  for (auto &&[i, idx] : llvm::enumerate(updatedShardedDims)) {
    Value leftSide;
    if (lowPads[idx] == 0) {
      // No pad, we are never needing to check combining update/operand and
      // can just use update
      leftSide = nullptr;
    } else if (lowPads[idx] % localResultType.getShape()[idx] == 0) {
      // Evenly divisible pad, for example pad 20, inner 50, right 20, local
      // shape is 10 partition 0, and 1 would need the fused update (aka if
      // idx < pad / lowerShape)
      leftSide = stablehlo::CompareOp::create(
          rewriter, loc, multiDimIdxs[idx],
          getOrCreateConstant(lowPads[idx] / localResultType.getShape()[idx]),
          stablehlo::ComparisonDirection::LT);
    } else {
      // Non-evenly divisible pad, for example pad 18, inner 54, right 18,
      // local shape is 10 partition 0, and 1 would need the fused update
      // (aka if idx <= pad / lowerShape)
      leftSide = stablehlo::CompareOp::create(
          rewriter, loc, multiDimIdxs[idx],
          getOrCreateConstant(lowPads[idx] / localResultType.getShape()[idx]),
          stablehlo::ComparisonDirection::LE);
    }

    leftSides.push_back(leftSide);
  }

  for (auto &&[i, idx] : llvm::enumerate(updatedShardedDims)) {
    Value rightSide;
    int64_t startIdx = lowPads[idx] + globalUnPaddedUpdateType.getShape()[idx];
    if (highPads[idx] == 0) {
      // No pad, we are never needing to check combining update/operand and
      // can just use update
      rightSide = nullptr;
    } else if (startIdx % localResultType.getShape()[idx] == 0) {
      // Evenly divisible startIdx, for example pad 20, inner 50, right X,
      // local shape is 10 partition 7, 8, ... would need the fused update
      // (aka if idx >= startIdx / lowerShape)
      rightSide = stablehlo::CompareOp::create(
          rewriter, loc, multiDimIdxs[idx],
          getOrCreateConstant(startIdx / localResultType.getShape()[idx]),
          stablehlo::ComparisonDirection::GE);
    } else {
      // Non-evenly divisible startIdx, for example pad 20, inner 48, right
      // X, local shape is 10 partition 6, 7, 8, ... would need the fused
      // update (aka if idx >= startIdx / lowerShape) partition 6 only needs
      // from 58-60, partitoin 7 uses fully
      rightSide = stablehlo::CompareOp::create(
          rewriter, loc, multiDimIdxs[idx],
          getOrCreateConstant(startIdx / localResultType.getShape()[idx]),
          stablehlo::ComparisonDirection::GE);
    }

    rightSides.push_back(rightSide);
  }

  Value mayContainOperandData = nullptr;
  for (int i = 0; i < updatedShardedDims.size(); i++) {
    if (leftSides[i]) {
      if (mayContainOperandData)
        mayContainOperandData = stablehlo::OrOp::create(
            rewriter, loc, mayContainOperandData, leftSides[i]);
      else
        mayContainOperandData = leftSides[i];
    }
    if (rightSides[i]) {
      if (mayContainOperandData)
        mayContainOperandData = stablehlo::OrOp::create(
            rewriter, loc, mayContainOperandData, rightSides[i]);
      else
        mayContainOperandData = rightSides[i];
    }
  }

  if (updatedShardedDims.size() == 0) {
    sdy::ReturnOp::create(rewriter, loc, innerUpdateVal);
  } else {
    if (updatedShardedDims.size() == 1 && false) {
      // TODO performance optimization, specialize for one dim update, can
      // incorporate from above.
    } else {
      // if (fully in update) {
      assert(mayContainOperandData);
      Type localTypes[] = {localResultType};
      auto if0 = stablehlo::IfOp::create(rewriter, loc, localTypes,
                                         mayContainOperandData);
      sdy::ReturnOp::create(rewriter, loc, if0->getResults());

      {
        rewriter.createBlock(&if0.getTrueBranch(), if0.getTrueBranch().begin());
        Value multiIdx = nullptr;
        for (int i = 0; i < updatedShardedDims.size(); i++) {
          auto TT = RankedTensorType::get(localResultType.getShape(),
                                          rewriter.getIntegerType(32, false));
          auto TTBool = RankedTensorType::get(localResultType.getShape(),
                                              rewriter.getI1Type());
          auto idx = updatedShardedDims[i];

          auto iota = stablehlo::IotaOp::create(rewriter, loc, TT, idx);
          Value lhs = nullptr;

          if (lowPads[idx] == 0) {
            // No pad, we are never needing to check combining
            // update/operand and can just use update
          } else if (lowPads[idx] % localResultType.getShape()[idx] == 0) {
            // Evenly divisible pad, for example pad 20, inner 50, right 20,
            // local shape is 10 partition 0, and 1 would need the fused
            // update (aka if idx < pad / lowerShape) If we are in here, no
            // additional check is needed for within node index. However, we
            // need to check if the given partition itself First, we can
            // special case if the condition to enter here is equivalent to
            // our partition check, do nothing.
            if (mayContainOperandData == leftSides[i]) {
              // No check needed
              lhs = mayContainOperandData;
            } else {
              // Otherwise we could've entered this if statement for other
              // reasons, whether to use is simply the partition check, now
              // broadcasted
              lhs = stablehlo::BroadcastInDimOp::create(
                  rewriter, loc, TTBool, leftSides[i], ArrayRef<int64_t>());
            }
          } else {
            // Non-evenly divisible pad, for example pad 18, inner 54, right
            // 18, local shape is 10 partition 0, and 1 would need the fused
            // update (aka if idx <= pad / lowerShape) The single node test
            // needs to consider both if we're in the right node, and within
            // the transition node, if we're at the point of transition
            // Within the point of transition, we need to consider the
            // offset mod the local result type.
            Value leftSideTransition = stablehlo::CompareOp::create(
                rewriter, loc, iota,
                getOrCreateConstant(
                    lowPads[idx] % localResultType.getShape()[idx], TT),
                stablehlo::ComparisonDirection::LT);
            assert(leftSideTransition.getType() == TTBool);

            // If we know we only have one node (the transition node), we're
            // done if there's only one node in this axis. Otherwise we need
            // the operand if we're in a lower node (independent of
            // transition index), and also need to ensure we don't use the
            // there's a risk that we're in a higher node and accidentally
            // used to enter the if statement. This is because the only
            // entering checks would be the low check (aka if the left
            // partition), or the high check

            // First let's begin by confirming that the leftside transition
            // is only true if within the transition node itself If only one
            // node, that's easy, no additional check required
            if (globalResultType.getShape()[idx] ==
                localResultType.getShape()[idx]) {
              // noop
            } else if (mayContainOperandData == leftSides[i]) {
              // If the only way we could enter here is if we are in a left
              // transition, we won't accidentally update higher nodes so we
              // can also skip the check
            } else {
              // otherwise we fall back and need to check that we're
              // actually at the transition point
              Value atTransition = stablehlo::CompareOp::create(
                  rewriter, loc, multiDimIdxs[idx],
                  getOrCreateConstant(lowPads[idx] /
                                      localResultType.getShape()[idx]),
                  stablehlo::ComparisonDirection::EQ);
              atTransition = stablehlo::BroadcastInDimOp::create(
                  rewriter, loc, TTBool, atTransition, ArrayRef<int64_t>());
              assert(leftSideTransition.getType() == TTBool);
              leftSideTransition = stablehlo::AndOp::create(
                  rewriter, loc, leftSideTransition, atTransition);
            }

            // Now let's check for potential non-transition nodes

            // If the update starts within the first node, no additional
            // check required!
            if (lowPads[idx] < localResultType.getShape()[idx]) {
              // noop
              lhs = leftSideTransition;
            } else {
              // Otherwise we must check both the node idx and the
              // transition idx.
              Value fullyOperandNode = stablehlo::CompareOp::create(
                  rewriter, loc, multiDimIdxs[idx],
                  getOrCreateConstant(lowPads[idx] /
                                      localResultType.getShape()[idx]),
                  stablehlo::ComparisonDirection::LT);
              fullyOperandNode = stablehlo::BroadcastInDimOp::create(
                  rewriter, loc, TTBool, fullyOperandNode, ArrayRef<int64_t>());
              assert(leftSideTransition.getType() == TTBool);
              lhs = stablehlo::OrOp::create(rewriter, loc, leftSideTransition,
                                            fullyOperandNode);
            }
          }

          Value rhs = nullptr;

          int64_t startIdx =
              lowPads[idx] + globalUnPaddedUpdateType.getShape()[idx];
          if (highPads[idx] == 0) {
            // No pad, we are never needing to check combining
            // update/operand and can just use update
          } else if (startIdx % localResultType.getShape()[idx] == 0) {
            // Evenly divisible startIdx, for example pad 20, inner 50,
            // right X, local shape is 10 partition 7, 8, ... would need the
            // fused update (aka if idx >= startIdx / lowerShape) If we are
            // in here, no additional check is needed for within node index.
            // However, we need to check if the given partition itself
            // First, we can special case if the condition to enter here is
            // equivalent to our partition check, do nothing.
            if (mayContainOperandData == rightSides[i]) {
              // No check needed
              rhs = mayContainOperandData;
            } else {
              // Otherwise we could've entered this if statement for other
              // reasons, whether to use is simply the partition check, now
              // broadcasted
              rhs = stablehlo::BroadcastInDimOp::create(
                  rewriter, loc, TTBool, rightSides[i], ArrayRef<int64_t>());
            }
          } else {
            // Non-evenly divisible startIdx, for example pad 20, inner 48,
            // right X, local shape is 10 partition 6, 7, 8, ... would need
            // the fused update (aka if idx >= startIdx / lowerShape)
            // partition 6 only needs from 58-60, partitoin 7 uses fully
            // The single node test needs to consider both if we're in the
            // right node, and within the transition node, if we're at the
            // point of transition Within the point of transition, we need
            // to consider the offset mod the local result type.

            Value rightSideTransition = stablehlo::CompareOp::create(
                rewriter, loc, iota,
                getOrCreateConstant(startIdx % localResultType.getShape()[idx],
                                    TT),
                stablehlo::ComparisonDirection::GE);
            assert(rightSideTransition.getType() == TTBool);

            // If we know we only have one node (the transition node), we're
            // done if there's only one node in this axis. Otherwise we need
            // the operand if we're in a higher node (independent of
            // transition index), and also need to ensure we don't use the
            // there's a risk that we're in a lower node and accidentally
            // used to enter the if statement. This is because the only
            // entering checks would be the high check (aka if the right
            // partition), or the lower check

            // First let's begin by confirming that the leftside transition
            // is only true if within the transition node itself If only one
            // node, that's easy, no additional check required
            if (globalResultType.getShape()[idx] ==
                localResultType.getShape()[idx]) {
              // noop
            } else if (mayContainOperandData == rightSides[i]) {
              // If the only way we could enter here is if we are in a right
              // transition, we won't accidentally lower higher nodes so we
              // can also skip the check
            } else {
              // otherwise we fall back and need to check that we're
              // actually at the transition point
              Value atTransition = stablehlo::CompareOp::create(
                  rewriter, loc, multiDimIdxs[idx],
                  getOrCreateConstant(startIdx /
                                      localResultType.getShape()[idx]),
                  stablehlo::ComparisonDirection::EQ);
              atTransition = stablehlo::BroadcastInDimOp::create(
                  rewriter, loc, TTBool, atTransition, ArrayRef<int64_t>());
              assert(rightSideTransition.getType() == TTBool);
              rightSideTransition = stablehlo::AndOp::create(
                  rewriter, loc, rightSideTransition, atTransition);
            }

            // Now let's check for potential non-transition nodes

            // If the update end within the last node, no additional check
            // required!
            if (startIdx / localResultType.getShape()[idx] ==
                globalResultType.getShape()[idx] /
                        localResultType.getShape()[idx] -
                    1) {
              // noop
              rhs = rightSideTransition;
            } else {
              // Otherwise we must check both the node idx and the
              // transition idx.
              Value fullyOperandNode = stablehlo::CompareOp::create(
                  rewriter, loc, multiDimIdxs[idx],
                  getOrCreateConstant(startIdx /
                                      localResultType.getShape()[idx]),
                  stablehlo::ComparisonDirection::GT);
              fullyOperandNode = stablehlo::BroadcastInDimOp::create(
                  rewriter, loc, TTBool, fullyOperandNode, ArrayRef<int64_t>());
              assert(rightSideTransition.getType() == TTBool);
              rhs = stablehlo::OrOp::create(rewriter, loc, rightSideTransition,
                                            fullyOperandNode);
            }
          }

          // We are in the operand if either lhs or rhs are in operand
          Value inOperand = lhs;
          if (rhs && inOperand != mayContainOperandData) {
            if (inOperand)
              inOperand =
                  stablehlo::OrOp::create(rewriter, loc, inOperand, rhs);
            else
              inOperand = rhs;
          }

          // We are in the operand if either of the indices are in the
          // operand
          if (inOperand == mayContainOperandData ||
              multiIdx == mayContainOperandData) {
            multiIdx = mayContainOperandData;
          } else if (inOperand) {
            if (multiIdx) {
              multiIdx =
                  stablehlo::OrOp::create(rewriter, loc, multiIdx, inOperand);
            } else {
              multiIdx = inOperand;
            }
          }
        }

        auto newV =
            multiIdx == mayContainOperandData
                ? innerOperand
                : stablehlo::SelectOp::create(rewriter, loc, multiIdx,
                                              innerOperand, innerUpdateVal)
                      ->getResult(0);
        stablehlo::ReturnOp::create(rewriter, loc, newV);
      }

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.createBlock(&if0.getFalseBranch(),
                             if0.getFalseBranch().begin());
        stablehlo::ReturnOp::create(rewriter, loc, innerUpdateVal);
      }
    }
  }
}

struct ConcatTwoDUSLike : public OpRewritePattern<stablehlo::ConcatenateOp> {

  int &channel_id;
  ConcatTwoDUSLike(int &channel_id, MLIRContext *context,
                   PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), channel_id(channel_id) {}

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concat,
                                PatternRewriter &rewriter) const override {

    if (concat->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    if (concat.getNumOperands() != 2) {
      return failure();
    }

    auto ndims = concat.getType().getShape().size();
    auto concatDimension = concat.getDimension();
    auto elemType = concat.getType().getElementType();

    auto sharding = mlir::sdy::getSharding(concat);
    if (!sharding)
      return failure();

    auto numDevicesAlongDimension =
        getNumDevicesAlongDimension(sharding, concatDimension, concat);
    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          concat,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    RankedTensorType globalResultType = concat.getType();
    SmallVector<int64_t> shape = llvm::to_vector(globalResultType.getShape());
    bool extraSlice = false;

    SmallVector<StringAttr> manualAxes;
    SmallVector<int64_t> padHigh(ndims, 0);
    for (int i = 0; i < ndims; i++) {
      auto meshAxes = sharding.getDimShardings()[i].getAxes();
      if (meshAxes.size() != 1)
        return failure();

      auto ndevices = getShardingDevices(sharding, i, concat);
      int64_t numDevicesAlongDimension = ndevices[i];

      for (auto axis : meshAxes)
        manualAxes.push_back(rewriter.getStringAttr(axis.getName()));

      if (numDevicesAlongDimension != 1) {
        if (globalResultType.getShape()[i] % numDevicesAlongDimension != 0) {
          int toPad =
              numDevicesAlongDimension -
              (globalResultType.getShape()[i] % numDevicesAlongDimension);
          shape[i] += toPad;
          if (i != concatDimension) {
            padHigh[i] = toPad;
          }
          extraSlice = true;
        }
      }
    }

    SmallVector<int64_t> updatedShardedDims = {(int64_t)concatDimension};
    SmallVector<int64_t> updatedDims = {(int64_t)concatDimension};
    globalResultType = RankedTensorType::get(shape, elemType);
    auto concatDimSize = globalResultType.getShape()[concatDimension];

    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);

    SmallVector<Value> manualOps(concat.getOperands().size());

    for (auto operand : concat.getOperands()) {
      auto operandSharding = mlir::sdy::getSharding(operand);
      if (!operandSharding || (operandSharding != sharding))
        return failure();
    }

    auto zero = stablehlo::ConstantOp::create(rewriter, concat.getLoc(),
                                              rewriter.getZeroAttr(elemType));

    int64_t leftPadding = 0;
    for (auto [i, operand] : llvm::enumerate(concat.getOperands())) {
      auto operandConcatDimSize =
          cast<RankedTensorType>(operand.getType()).getShape()[concatDimension];

      padLow[concatDimension] = leftPadding;
      padHigh[concatDimension] =
          concatDimSize - leftPadding - operandConcatDimSize;

      auto paddedOperand = stablehlo::PadOp::create(
          rewriter, concat.getLoc(), operand, zero, padLow, padHigh, padInner);
      sdy::setSharding(paddedOperand, sharding);
      manualOps[i] = paddedOperand;
      leftPadding += operandConcatDimSize;
    }

    SmallVector<int64_t> lowPads(ndims, 0);
    lowPads[concatDimension] =
        cast<RankedTensorType>(concat.getOperands()[0].getType())
            .getShape()[concatDimension];
    SmallVector<int64_t> highPads(ndims, 0);

    RankedTensorType globalUnPaddedUpdateType =
        cast<RankedTensorType>(concat.getOperands()[1].getType());

    auto localResultType =
        getLocalType(globalResultType, sharding, manualAxes, concat);

    SmallVector<TensorShardingAttr> in_shardings_array = {sharding, sharding};

    TensorShardingAttr out_shardings_array[] = {sharding};

    SmallVector<mlir::Type> inTyps = {localResultType, localResultType};
    SmallVector<mlir::Location> inLocs(inTyps.size(), concat.getLoc());

    Type manualTypes[] = {globalResultType};

    TensorShardingPerValueAttr in_shardings = TensorShardingPerValueAttr::get(
        concat.getContext(), in_shardings_array);
    TensorShardingPerValueAttr out_shardings = TensorShardingPerValueAttr::get(
        concat.getContext(), out_shardings_array);

    auto manual = sdy::ManualComputationOp::create(
        rewriter, concat.getLoc(), manualTypes, manualOps, in_shardings,
        out_shardings, manualAxes);

    {
      auto blk = rewriter.createBlock(&manual.getBody(),
                                      manual.getBody().begin(), inTyps, inLocs);

      auto innerOperand = blk->getArgument(0);
      auto innerUpdate = blk->getArgument(1);
      multiDimensionalSelect(concat.getLoc(), rewriter, globalResultType,
                             localResultType, lowPads, highPads, updatedDims,
                             updatedShardedDims, innerOperand, innerUpdate,
                             globalUnPaddedUpdateType, concat);
    }

    if (!extraSlice) {
      rewriter.replaceOp(concat, manual);
    } else {
      rewriter.setInsertionPointAfter(manual);
      SmallVector<int64_t> starts(ndims, 0);
      SmallVector<int64_t> limits =
          llvm::to_vector(concat.getType().getShape());
      SmallVector<int64_t> interior(ndims, 1);
      auto sl = rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
          concat, manual->getResult(0), starts, limits, interior);
      sdy::setSharding(sl, sharding);
    }
    return success();
  }
};

struct ExtendDUSLike : public OpRewritePattern<enzymexla::ExtendOp> {

  int &channel_id;
  ExtendDUSLike(int &channel_id, MLIRContext *context,
                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), channel_id(channel_id) {}

  LogicalResult matchAndRewrite(enzymexla::ExtendOp concat,
                                PatternRewriter &rewriter) const override {

    if (concat->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    if (concat.getLhs() != 0 && concat.getRhs() != 0) {
      return failure();
    }

    auto ndims = concat.getType().getShape().size();
    auto concatDimension = concat.getDimension();
    auto elemType = concat.getType().getElementType();

    auto sharding = mlir::sdy::getSharding(concat);
    if (!sharding)
      return failure();

    auto numDevicesAlongDimension =
        getNumDevicesAlongDimension(sharding, concatDimension, concat);
    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          concat,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    RankedTensorType globalResultType = concat.getType();
    SmallVector<int64_t> shape = llvm::to_vector(globalResultType.getShape());
    bool extraSlice = false;

    SmallVector<StringAttr> manualAxes;
    SmallVector<int64_t> padHigh(ndims, 0);
    for (int i = 0; i < ndims; i++) {
      auto meshAxes = sharding.getDimShardings()[i].getAxes();
      if (meshAxes.size() != 1)
        return failure();

      auto ndevices = getShardingDevices(sharding, i, concat);
      int64_t numDevicesAlongDimension = ndevices[i];

      for (auto axis : meshAxes)
        manualAxes.push_back(rewriter.getStringAttr(axis.getName()));

      if (numDevicesAlongDimension != 1) {
        if (globalResultType.getShape()[i] % numDevicesAlongDimension != 0) {
          int toPad =
              numDevicesAlongDimension -
              (globalResultType.getShape()[i] % numDevicesAlongDimension);
          shape[i] += toPad;
          padHigh[i] = toPad;
          extraSlice = true;
        }
      }
    }

    SmallVector<int64_t> updatedShardedDims = {(int64_t)concatDimension};
    SmallVector<int64_t> updatedDims = {(int64_t)concatDimension};
    globalResultType = RankedTensorType::get(shape, elemType);

    SmallVector<int64_t> padInner(ndims, 0);

    SmallVector<Value> manualOps(2);

    for (auto operand : concat->getOperands()) {
      auto operandSharding = mlir::sdy::getSharding(operand);
      if (!operandSharding || (operandSharding != sharding))
        return failure();
    }

    auto zero = stablehlo::ConstantOp::create(rewriter, concat.getLoc(),
                                              rewriter.getZeroAttr(elemType));

    for (int i = 0; i < 2; i++) {
      auto operand = concat.getOperand();

      SmallVector<int64_t> padLowLocal(ndims, 0);
      SmallVector<int64_t> padHighLocal = padHigh;
      padLowLocal[concatDimension] +=
          i == 0 ? 0 : (concat.getLhs() + concat.getRhs());
      padHighLocal[concatDimension] +=
          i == 0 ? (concat.getLhs() + concat.getRhs()) : 0;

      auto paddedOperand =
          stablehlo::PadOp::create(rewriter, concat.getLoc(), operand, zero,
                                   padLowLocal, padHighLocal, padInner);
      sdy::setSharding(paddedOperand, sharding);
      manualOps[i] = paddedOperand;
    }

    SmallVector<int64_t> lowPads(ndims, 0);
    lowPads[concatDimension] =
        concat.getLhs() ? concat.getLhs()
                        : cast<RankedTensorType>(concat.getOperand().getType())
                              .getShape()[concat.getDimension()];
    SmallVector<int64_t> highPads(ndims, 0);

    RankedTensorType globalUnPaddedUpdateType =
        cast<RankedTensorType>(concat.getOperand().getType());

    auto localResultType =
        getLocalType(globalResultType, sharding, manualAxes, concat);

    SmallVector<TensorShardingAttr> in_shardings_array = {sharding, sharding};

    TensorShardingAttr out_shardings_array[] = {sharding};

    SmallVector<mlir::Type> inTyps = {localResultType, localResultType};
    SmallVector<mlir::Location> inLocs(inTyps.size(), concat.getLoc());

    Type manualTypes[] = {globalResultType};

    TensorShardingPerValueAttr in_shardings = TensorShardingPerValueAttr::get(
        concat.getContext(), in_shardings_array);
    TensorShardingPerValueAttr out_shardings = TensorShardingPerValueAttr::get(
        concat.getContext(), out_shardings_array);

    auto manual = sdy::ManualComputationOp::create(
        rewriter, concat.getLoc(), manualTypes, manualOps, in_shardings,
        out_shardings, manualAxes);

    {
      auto blk = rewriter.createBlock(&manual.getBody(),
                                      manual.getBody().begin(), inTyps, inLocs);

      auto innerOperand = blk->getArgument(0);
      auto innerUpdate = blk->getArgument(1);
      multiDimensionalSelect(concat.getLoc(), rewriter, globalResultType,
                             localResultType, lowPads, highPads, updatedDims,
                             updatedShardedDims, innerOperand, innerUpdate,
                             globalUnPaddedUpdateType, concat);
    }

    if (!extraSlice) {
      rewriter.replaceOp(concat, manual);
    } else {
      rewriter.setInsertionPointAfter(manual);
      SmallVector<int64_t> starts(ndims, 0);
      SmallVector<int64_t> limits =
          llvm::to_vector(concat.getType().getShape());
      SmallVector<int64_t> interior(ndims, 1);
      auto sl = rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
          concat, manual->getResult(0), starts, limits, interior);
      sdy::setSharding(sl, sharding);
    }
    return success();
  }
};

struct DUSToPadManualCompComm
    : public OpRewritePattern<stablehlo::DynamicUpdateSliceOp> {

  int &channel_id;
  DUSToPadManualCompComm(int &channel_id, MLIRContext *context,
                         PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), channel_id(channel_id) {}

  LogicalResult matchAndRewrite(stablehlo::DynamicUpdateSliceOp dus,
                                PatternRewriter &rewriter) const override {

    // Cannot apply dus rewrite on existing manual computation.
    if (dus->getParentOfType<sdy::ManualComputationOp>())
      return failure();

    auto sharding = mlir::sdy::getSharding(dus);
    if (!sharding)
      return rewriter.notifyMatchFailure(dus, "No sharding found.");

    /*
    {
      mlir::SplatElementsAttr splat;
      matchPattern(dus.getOperand(), m_Constant(&splat));
    if (sharding != mlir::sdy::getSharding(dus.getOperand()) && !splat) {
      return rewriter.notifyMatchFailure(dus, "DUS mismatch sharding");
    }
    }

    {
      mlir::SplatElementsAttr splat;
      matchPattern(dus.getUpdate(), m_Constant(&splat));
      if (sharding != mlir::sdy::getSharding(dus.getUpdate()) && !splat) {

        return rewriter.notifyMatchFailure(dus, "DUS mismatch sharding");
      }
    }
    */

    size_t ndims = dus.getType().getShape().size();
    auto elementType = dus.getType().getElementType();
    auto loc = dus.getLoc();
    SmallVector<int64_t> lowPads;
    SmallVector<int64_t> highPads;
    // Extra padding size we make to ensure things are evenly divisible by the
    // number of devices
    SmallVector<int64_t> extraHighPads;
    // The total padding (highPads + extraHighPads)
    SmallVector<int64_t> totalHighPads;
    SmallVector<int64_t> interior(ndims, 0);
    SmallVector<int64_t> updatedShardedDims;
    SmallVector<int64_t> shardedDims;
    SmallVector<int64_t> updatedDims;

    SmallVector<StringAttr> manualAxes;

    auto UT = cast<RankedTensorType>(dus.getUpdate().getType());
    bool extraSlice = false;

    for (int i = 0; i < ndims; i++) {
      DenseIntElementsAttr curr;
      APInt val;
      if (auto cst =
              dus.getStartIndices()[i].getDefiningOp<sdy::ConstantOp>()) {
        val = *cast<DenseIntElementsAttr>(cst.getValue()).begin();
      } else if (matchPattern(dus.getStartIndices()[i], m_Constant(&curr))) {
        val = (*curr.begin());
      } else {
        return rewriter.notifyMatchFailure(dus, "Non-constant DUS start index");
      }

      if (val.isNegative())
        return rewriter.notifyMatchFailure(dus, "Negative DUS start index");

      auto v2 = val.getZExtValue();
      if (v2 + UT.getShape()[i] > dus.getType().getShape()[i])
        return rewriter.notifyMatchFailure(dus, "Too large DUS start index");

      auto meshAxes = sharding.getDimShardings()[i].getAxes();
      if (meshAxes.size() != 1)
        return failure();
      SmallVector<StringAttr> axis = {
          rewriter.getStringAttr(meshAxes[0].getName())};
      auto localType = getLocalType(dus.getType(), sharding, axis, dus);

      auto rightPad = dus.getType().getShape()[i] - v2 - UT.getShape()[i];

      bool updated = false;
      if (UT.getShape()[i] != dus.getType().getShape()[i]) {
        updated = true;
      }

      for (auto axis : meshAxes)
        manualAxes.push_back(rewriter.getStringAttr(axis.getName()));

      if (localType.getShape()[i] != dus.getType().getShape()[i]) {
        shardedDims.push_back(i);
      }

      lowPads.push_back(v2);
      highPads.push_back(rightPad);
      auto extraPad = 0;

      if (localType.getShape()[i] == 0 ||
          dus.getType().getShape()[i] % localType.getShape()[i] != 0) {
        auto ndevices = getShardingDevices(sharding, i, dus);
        int64_t numDevicesAlongDimension = ndevices[i];
        extraPad = numDevicesAlongDimension -
                   (dus.getType().getShape()[i] % numDevicesAlongDimension);
        extraSlice = true;
      }
      extraHighPads.push_back(extraPad);
      totalHighPads.push_back(extraPad + rightPad);

      if (updated) {
        updatedDims.push_back(i);
        if (localType.getShape()[i] != dus.getType().getShape()[i]) {
          updatedShardedDims.push_back(i);
        }
      }
    }

    DenseMap<std::pair<int64_t, Type>, Value> constantCache;
    auto getOrCreateConstant = [&](int64_t v, Type TT) -> Value {
      auto key = std::make_pair(v, TT);
      auto found = constantCache.find(key);
      if (found != constantCache.end())
        return found->second;
      auto cst = stablehlo::ConstantOp::create(
          rewriter, loc, TT, cast<ElementsAttr>(makeAttr(TT, v)));
      constantCache[key] = cst;
      return cst;
    };

    auto PT = RankedTensorType::get({}, elementType);

    Value globalOperand = dus.getOperand();
    if (extraSlice) {
      SmallVector<int64_t> zeros(ndims, 0);
      auto padOp = stablehlo::PadOp::create(rewriter, loc, dus.getOperand(),
                                            getOrCreateConstant(0, PT), zeros,
                                            extraHighPads, zeros);
      sdy::setShardings(padOp, sdy::getShardingPerValue(dus));
      globalOperand = padOp;
    }
    RankedTensorType globalResultType =
        cast<RankedTensorType>(globalOperand.getType());

    auto localResultType =
        getLocalType(globalResultType, sharding, manualAxes, dus);

    if (updatedDims.size() == 0) {
      rewriter.replaceOp(dus, dus.getUpdate());
      return success();
    }
    if (shardedDims.size() == 0) {
      return failure();
    }

    Value pad2 = nullptr;
    SplatElementsAttr splat = nullptr;

    if (!matchPattern(dus.getUpdate(), m_Constant(&splat))) {
      if (auto cst = dus.getUpdate().getDefiningOp<sdy::ConstantOp>()) {
        if (auto sp = dyn_cast<SplatElementsAttr>(cst.getValue())) {
          splat = sp;
        }
      }
    }
    if (!splat) {
      auto padOp = stablehlo::PadOp::create(rewriter, loc, dus.getUpdate(),
                                            getOrCreateConstant(0, PT), lowPads,
                                            totalHighPads, interior);
      sdy::setShardings(padOp, sdy::getShardingPerValue(dus));
      pad2 = padOp;
    }

    RankedTensorType globalUnPaddedUpdateType =
        cast<RankedTensorType>(dus.getUpdate().getType());
    RankedTensorType globalPaddedUpdateType = nullptr;
    {
      SmallVector<int64_t> padShape =
          llvm::to_vector(globalUnPaddedUpdateType.getShape());
      for (int i = 0; i < padShape.size(); i++) {
        padShape[i] += lowPads[i] + totalHighPads[i];
      }
      globalPaddedUpdateType = RankedTensorType::get(padShape, elementType);
    }

    auto localPaddedUpdateType =
        getLocalType(globalPaddedUpdateType, sharding, manualAxes, dus);

    SmallVector<TensorShardingAttr> in_shardings_array = {
        mlir::sdy::getSharding(dus.getOperand())};
    if (pad2)
      in_shardings_array.push_back(mlir::sdy::getSharding(dus));

    TensorShardingAttr out_shardings_array[] = {mlir::sdy::getSharding(dus)};

    SmallVector<mlir::Type> inTyps = {localResultType};
    if (pad2)
      inTyps.push_back(localPaddedUpdateType);

    SmallVector<mlir::Location> inLocs(inTyps.size(), loc);

    SmallVector<Value> manualOps = {globalOperand};
    if (pad2)
      manualOps.push_back(pad2);
    Type manualTypes[] = {globalResultType};

    TensorShardingPerValueAttr in_shardings =
        TensorShardingPerValueAttr::get(dus.getContext(), in_shardings_array);
    TensorShardingPerValueAttr out_shardings =
        TensorShardingPerValueAttr::get(dus.getContext(), out_shardings_array);

    auto manual = sdy::ManualComputationOp::create(rewriter, loc, manualTypes,
                                                   manualOps, in_shardings,
                                                   out_shardings, manualAxes);

    {
      auto blk = rewriter.createBlock(&manual.getBody(),
                                      manual.getBody().begin(), inTyps, inLocs);

      auto innerOperand = blk->getArgument(0);
      Value innerUpdate;
      if (pad2) {
        innerUpdate = blk->getArgument(1);
      } else {
        innerUpdate = stablehlo::ConstantOp::create(
            rewriter, loc, localPaddedUpdateType,
            splat.resizeSplat(localPaddedUpdateType));
      }

      multiDimensionalSelect(loc, rewriter, globalResultType, localResultType,
                             lowPads, highPads, updatedDims, updatedShardedDims,
                             innerOperand, innerUpdate,
                             globalUnPaddedUpdateType, dus);
    }

    if (!extraSlice) {
      rewriter.replaceOp(dus, manual);
    } else {
      rewriter.setInsertionPointAfter(manual);
      SmallVector<int64_t> starts(ndims, 0);
      auto DT = dus.getType();
      SmallVector<int64_t> limits = llvm::to_vector(DT.getShape());
      SmallVector<int64_t> interior(ndims, 1);
      auto sl = rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
          dus, manual->getResult(0), starts, limits, interior);
      assert(sl.getType() == DT);
      sdy::setSharding(sl, sharding);
    }
    return success();
  }
};

struct ConcatToPadCommOptimize
    : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concat,
                                PatternRewriter &rewriter) const override {
    if (concat->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    auto ndims = concat.getType().getShape().size();
    auto concatShape = concat.getType().getShape();
    auto concatDimension = concat.getDimension();
    auto concatDimSize = concatShape[concatDimension];
    auto elemType = concat.getType().getElementType();

    auto concatSharding = mlir::sdy::getSharding(concat);
    if (!concatSharding)
      return failure();

    auto numDevicesAlongDimension =
        getNumDevicesAlongDimension(concatSharding, concatDimension, concat);
    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          concat,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    if (concat.getNumOperands() == 2 &&
        isRotateLike(concat.getDimension(), concat.getOperands()[0],
                     concat.getOperands()[1])) {
      return rewriter.notifyMatchFailure(concat, "Explicit rotate like comm");
    }

    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);

    SmallVector<Value> addOperands;

    for (auto operand : concat.getOperands()) {
      auto operandSharding = mlir::sdy::getSharding(operand);
      if (!operandSharding || (operandSharding != concatSharding))
        return failure();
    }

    auto zero = stablehlo::ConstantOp::create(rewriter, concat.getLoc(),
                                              rewriter.getZeroAttr(elemType));

    int64_t leftPadding = 0;
    for (auto [i, operand] : llvm::enumerate(concat.getOperands())) {
      auto operandConcatDimSize =
          cast<RankedTensorType>(operand.getType()).getShape()[concatDimension];

      if (isZero(operand)) {
        leftPadding += operandConcatDimSize;
        continue;
      }

      padLow[concatDimension] = leftPadding;
      padHigh[concatDimension] =
          concatDimSize - leftPadding - operandConcatDimSize;

      auto paddedOperand = stablehlo::PadOp::create(
          rewriter, concat.getLoc(), operand, zero, padLow, padHigh, padInner);
      assert(concat.getType() == paddedOperand.getType());
      sdy::setSharding(paddedOperand, concatSharding);
      addOperands.push_back(paddedOperand);
      leftPadding += operandConcatDimSize;
    }

    if (addOperands.size() == 0) {
      auto cst = stablehlo::ConstantOp::create(
          rewriter, concat.getLoc(), concat.getType(),
          cast<ElementsAttr>(rewriter.getZeroAttr(concat.getType())));
      sdy::setSharding(cst, concatSharding);
      rewriter.replaceOp(concat, cst);
    }

    Value sum = addOperands[0];
    for (int i = 1; i < addOperands.size(); i++) {
      auto addOp = stablehlo::AddOp::create(rewriter, concat.getLoc(), sum,
                                            addOperands[i]);
      sdy::setSharding(addOp, concatSharding);
      sum = addOp;
    }

    assert(concat.getType() == sum.getType());
    rewriter.replaceOp(concat, sum);
    return success();
  }
};

struct ConcatToDUSOptimize : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concat,
                                PatternRewriter &rewriter) const override {
    if (concat->getParentOfType<sdy::ManualComputationOp>())
      return failure();
    auto ndims = concat.getType().getShape().size();
    auto concatShape = concat.getType().getShape();
    auto concatDimension = concat.getDimension();
    auto concatDimSize = concatShape[concatDimension];
    auto elemType = concat.getType().getElementType();

    auto concatSharding = mlir::sdy::getSharding(concat);
    if (!concatSharding)
      return failure();

    auto numDevicesAlongDimension =
        getNumDevicesAlongDimension(concatSharding, concatDimension, concat);
    if (numDevicesAlongDimension == 1) {
      return rewriter.notifyMatchFailure(
          concat,
          "numDevicesAlongDimension == 1. Communication is already optimized.");
    }

    if (concat.getNumOperands() == 2 &&
        isRotateLike(concat.getDimension(), concat.getOperands()[0],
                     concat.getOperands()[1])) {
      return rewriter.notifyMatchFailure(concat, "Explicit rotate like comm");
    }

    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);

    SmallVector<Value> addOperands;

    size_t largest_idx = 0;
    for (auto &&[idx, operand] : llvm::enumerate(concat.getOperands())) {
      auto operandSharding = mlir::sdy::getSharding(operand);
      if (!operandSharding || (operandSharding != concatSharding))
        return failure();
      if (cast<RankedTensorType>(operand.getType())
              .getShape()[concatDimension] >
          cast<RankedTensorType>(concat.getOperands()[largest_idx].getType())
              .getShape()[concatDimension]) {
        largest_idx = idx;
      }
    }

    auto zero = stablehlo::ConstantOp::create(rewriter, concat.getLoc(),
                                              rewriter.getZeroAttr(elemType));

    int64_t leftPadding = 0;
    for (auto [i, operand] : llvm::enumerate(concat.getOperands())) {
      auto operandConcatDimSize =
          cast<RankedTensorType>(operand.getType()).getShape()[concatDimension];
      if (i == largest_idx)
        break;
      leftPadding += operandConcatDimSize;
    }

    padLow[concatDimension] = leftPadding;
    padHigh[concatDimension] =
        concatDimSize - leftPadding -
        cast<RankedTensorType>(concat.getOperands()[largest_idx].getType())
            .getShape()[concatDimension];

    auto padStart = stablehlo::PadOp::create(rewriter, concat.getLoc(),
                                             concat.getOperands()[largest_idx],
                                             zero, padLow, padHigh, padInner);
    assert(concat.getType() == padStart.getType());
    sdy::setSharding(padStart, concatSharding);

    Value current = padStart;

    leftPadding = 0;

    auto i32 = RankedTensorType::get({}, concatDimSize < (1ULL << 32)
                                             ? rewriter.getI32Type()
                                             : rewriter.getI64Type());
    auto zeroI32 = stablehlo::ConstantOp::create(rewriter, concat.getLoc(),
                                                 rewriter.getZeroAttr(i32));

    for (auto [i, operand] : llvm::enumerate(concat.getOperands())) {
      auto operandConcatDimSize =
          cast<RankedTensorType>(operand.getType()).getShape()[concatDimension];

      if (isZero(operand) || i == largest_idx) {
        leftPadding += operandConcatDimSize;
        continue;
      }

      SmallVector<Value> idxs(ndims, zeroI32);
      idxs[concatDimension] = stablehlo::ConstantOp::create(
          rewriter, concat.getLoc(), i32,
          cast<ElementsAttr>(makeAttr(i32, leftPadding)));

      auto paddedOperand = stablehlo::DynamicUpdateSliceOp::create(

          rewriter, concat.getLoc(), current, operand, idxs);

      assert(concat.getType() == paddedOperand.getType());
      sdy::setSharding(paddedOperand, concatSharding);
      leftPadding += operandConcatDimSize;
      current = paddedOperand;
    }

    rewriter.replaceOp(concat, current);
    return success();
  }
};

// See https://github.com/EnzymeAD/Enzyme-JAX/issues/854 for the motivation
// TODO: At some point if we can come up with a cost model for this, we can do a
//       greedy search for the best ordering
template <typename opTy>
struct ReorderAssociativeOp : public OpRewritePattern<opTy> {
  using OpRewritePattern<opTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(opTy op,
                                PatternRewriter &rewriter) const override {
    bool isCommutative =
        op->template hasTrait<OpTrait::IsCommutative>() ||
        op->template hasTrait<mlir::hlo::OpTrait::IsCommutative>();

    auto lhsOp = op.getLhs();
    auto rhsOp = op.getRhs();

    auto opSharding = mlir::sdy::getSharding(op);
    if (!opSharding)
      return failure();

    auto lhsSharding = mlir::sdy::getSharding(lhsOp);
    if (!lhsSharding && !lhsOp.hasOneUse())
      return failure();

    auto rhsSharding = mlir::sdy::getSharding(rhsOp);
    if (!rhsSharding && !rhsOp.hasOneUse())
      return failure();

    auto lhsDefOp = lhsOp.template getDefiningOp<opTy>();
    if (!lhsDefOp) {
      auto rhsDefOp = rhsOp.template getDefiningOp<opTy>();
      if (!rhsDefOp)
        return failure();

      // op a (op b c)
      auto aOp = lhsOp;
      auto aOpSharding = lhsSharding;

      auto bOp = rhsDefOp.getLhs();
      auto bOpSharding = mlir::sdy::getSharding(bOp);
      if (!bOpSharding)
        return failure();

      auto cOp = rhsDefOp.getRhs();
      auto cOpSharding = mlir::sdy::getSharding(cOp);
      if (!cOpSharding)
        return failure();

      // good ordering
      if (bOpSharding == cOpSharding)
        return failure();

      if (aOpSharding == cOpSharding && isCommutative) {
        auto newOp = opTy::create(rewriter, op.getLoc(), aOp, cOp);
        sdy::setSharding(newOp, aOpSharding);
        auto newFinalOp = opTy::create(rewriter, op.getLoc(), newOp, bOp);
        sdy::setSharding(newFinalOp, opSharding);
        rewriter.replaceOp(op, newFinalOp); // op (op a c) b
        return success();
      }

      if (aOpSharding == bOpSharding) {
        auto newOp = opTy::create(rewriter, op.getLoc(), aOp, bOp);
        sdy::setSharding(newOp, aOpSharding);
        auto newFinalOp = opTy::create(rewriter, op.getLoc(), newOp, cOp);
        sdy::setSharding(newFinalOp, opSharding);
        rewriter.replaceOp(op, newFinalOp); // op (op a b) c
        return success();
      }
    } else {
      auto rhsDefOp = rhsOp.template getDefiningOp<opTy>();
      if (!rhsDefOp) {
        // op (op a b) c
        auto aOp = lhsDefOp.getLhs();
        auto aOpSharding = mlir::sdy::getSharding(aOp);
        if (!aOpSharding)
          return failure();

        auto bOp = lhsDefOp.getRhs();
        auto bOpSharding = mlir::sdy::getSharding(bOp);
        if (!bOpSharding)
          return failure();

        auto cOp = rhsOp;
        auto cOpSharding = rhsSharding;

        // good ordering
        if (aOpSharding == bOpSharding)
          return failure();

        if (aOpSharding == cOpSharding && isCommutative) {
          auto newOp = opTy::create(rewriter, op.getLoc(), aOp, cOp);
          sdy::setSharding(newOp, aOpSharding);
          auto newFinalOp = opTy::create(rewriter, op.getLoc(), newOp, bOp);
          sdy::setSharding(newFinalOp, opSharding);
          rewriter.replaceOp(op, newFinalOp); // op (op a c) b
          return success();
        }

        if (bOpSharding == cOpSharding) {
          auto newOp = opTy::create(rewriter, op.getLoc(), bOp, cOp);
          sdy::setSharding(newOp, bOpSharding);
          auto newFinalOp = opTy::create(rewriter, op.getLoc(), aOp, newOp);
          sdy::setSharding(newFinalOp, opSharding);
          rewriter.replaceOp(op, newFinalOp); // op a (op b c)
          return success();
        }
      } else {
        // op (op a b) (op c d)
        auto aOp = lhsDefOp.getLhs();
        auto aOpSharding = sdy::getSharding(aOp);
        if (!aOpSharding)
          return failure();

        auto bOp = lhsDefOp.getRhs();
        auto bOpSharding = sdy::getSharding(bOp);
        if (!bOpSharding)
          return failure();

        auto cOp = rhsDefOp.getLhs();
        auto cOpSharding = sdy::getSharding(cOp);
        if (!cOpSharding)
          return failure();

        auto dOp = rhsDefOp.getRhs();
        auto dOpSharding = sdy::getSharding(dOp);
        if (!dOpSharding)
          return failure();

        // good ordering
        if ((aOpSharding == bOpSharding) || (cOpSharding == dOpSharding))
          return failure();

        if (!isCommutative) // all nicer variants need to be commutative
          return failure();

        if (aOpSharding == cOpSharding && bOpSharding == dOpSharding) {
          auto newLhsOp = opTy::create(rewriter, op.getLoc(), aOp, cOp);
          sdy::setSharding(newLhsOp, aOpSharding);
          auto newRhsOp = opTy::create(rewriter, op.getLoc(), bOp, dOp);
          sdy::setSharding(newRhsOp, bOpSharding);
          auto newOp = opTy::create(rewriter, op.getLoc(), newLhsOp, newRhsOp);
          sdy::setSharding(newOp, opSharding);
          rewriter.replaceOp(op, newOp); // op (op a c) (op b d)
          return success();
        }

        if (aOpSharding == dOpSharding && bOpSharding == cOpSharding) {
          auto newLhsOp = opTy::create(rewriter, op.getLoc(), aOp, dOp);
          sdy::setSharding(newLhsOp, aOpSharding);
          auto newRhsOp = opTy::create(rewriter, op.getLoc(), bOp, cOp);
          sdy::setSharding(newRhsOp, bOpSharding);
          auto newOp = opTy::create(rewriter, op.getLoc(), newLhsOp, newRhsOp);
          sdy::setSharding(newOp, opSharding);
          rewriter.replaceOp(op, newOp); // op (op a d) (op b c)
          return success();
        }
      }
    }

    return failure();
  }
};

struct OptimizeCommunicationPass
    : public enzyme::impl::OptimizeCommunicationBase<
          OptimizeCommunicationPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    int channel_id = 1;

    getOperation()->walk([&](stablehlo::CollectivePermuteOp perm) {
      if (auto attr = perm.getChannelHandle())
        channel_id = std::max(channel_id, (int)attr->getHandle() + 1);
    });

    if (periodic_concat > 0)
      patterns.add<PeriodicConcatSimplify>(channel_id, context,
                                           PatternBenefit(periodic_concat));

    if (concat_to_pad_comm > 0)
      patterns.add<ConcatToPadCommOptimize>(context,
                                            PatternBenefit(concat_to_pad_comm));

    if (concat_to_dus > 0)
      patterns.add<ConcatToDUSOptimize>(context, PatternBenefit(concat_to_dus));

    if (concat_two_operands_comm > 0)
      patterns.add<ConcatTwoOperandsCommOptimize>(
          channel_id, context, PatternBenefit(concat_two_operands_comm));

    if (rotate_comm > 0)
      patterns.add<RotateCommOptimize>(channel_id, context,
                                       PatternBenefit(rotate_comm));

    if (rotate_spmd > 0)
      patterns.add<RotateSpmdOptimize>(context, PatternBenefit(rotate_comm));

    if (rotate_to_pad_comm > 0)
      patterns.add<RotateToPadCommOptimize>(context,
                                            PatternBenefit(rotate_to_pad_comm));

    if (wrap_comm > 0)
      patterns.add<WrapCommOptimize>(channel_id, context,
                                     PatternBenefit(wrap_comm));

    if (wrap_to_pad_comm > 0)
      patterns.add<WrapToPadCommOptimize>(context,
                                          PatternBenefit(wrap_to_pad_comm));

    if (wrap_to_rotate > 0)
      patterns.add<WrapToRotateOptimize>(context,
                                         PatternBenefit(wrap_to_rotate));

    if (extend_comm > 0)
      patterns.add<ExtendCommOptimize>(channel_id, context,
                                       PatternBenefit(extend_comm));

    if (extend_to_pad_comm > 0)
      patterns.add<ExtendToPadCommOptimize>(context,
                                            PatternBenefit(extend_to_pad_comm));

    if (extend_to_pad_comm2 > 0)
      patterns.add<ExtendToPadCommOptimize2>(
          context, PatternBenefit(extend_to_pad_comm2));

    if (dus_to_pad_manual_comp_comm > 0)
      patterns.add<DUSToPadManualCompComm>(
          channel_id, context, PatternBenefit(dus_to_pad_manual_comp_comm));

    if (concat_two_dus_like > 0)
      patterns.add<ConcatTwoDUSLike>(channel_id, context,
                                     PatternBenefit(concat_two_dus_like));

    if (extend_dus_like > 0)
      patterns.add<ExtendDUSLike>(channel_id, context,
                                  PatternBenefit(extend_dus_like));

    if (dus_to_pad_comm > 0)
      patterns.add<DUSToPadComm>(context, PatternBenefit(dus_to_pad_comm));

    if (reorder_associative > 0) {
      patterns.add<ReorderAssociativeOp<stablehlo::AddOp>,
                   ReorderAssociativeOp<stablehlo::MulOp>,
                   ReorderAssociativeOp<stablehlo::MinOp>,
                   ReorderAssociativeOp<stablehlo::MaxOp>,
                   ReorderAssociativeOp<stablehlo::AndOp>,
                   ReorderAssociativeOp<stablehlo::OrOp>,
                   ReorderAssociativeOp<stablehlo::XorOp>>(
          context, PatternBenefit(reorder_associative));
    }

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
