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
    int numDevicesAlongDimension, int ndims, SmallVector<int64_t> localRetShape,
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
  auto alpha = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), partitionIdType,
      makeAttr(partitionIdType,
               2 * paddedBoundarySize / numDevicesAlongDimension)
          .cast<ElementsAttr>());

  // pId % numDevicesAlongDimension
  auto pidRem = rewriter.create<stablehlo::RemOp>(
      op->getLoc(), partitionId,
      rewriter.create<stablehlo::ConstantOp>(
          op->getLoc(), partitionIdType,
          makeAttr(partitionIdType, numDevicesAlongDimension)
              .cast<ElementsAttr>()));

  auto numDevicesAlongDimension_2 = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), partitionIdType,
      makeAttr(partitionIdType, numDevicesAlongDimension / 2)
          .cast<ElementsAttr>());
  auto isLeftBlock = rewriter.create<stablehlo::CompareOp>(
      op->getLoc(), leftSide, numDevicesAlongDimension_2,
      stablehlo::ComparisonDirection::LT);

  auto commShape = llvm::to_vector(
      cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
  commShape[concatDim] = paddedBoundarySize;
  Type ifTypesCommSelect[] = {RankedTensorType::get(
      commShape,
      superSliceInnerArg.getType().cast<RankedTensorType>().getElementType())};
  auto ifCondCommSelect = rewriter.create<stablehlo::IfOp>(
      op->getLoc(), ifTypesCommSelect, isLeftBlock);

  {
    rewriter.createBlock(&ifCondCommSelect.getTrueBranch(),
                         ifCondCommSelect.getTrueBranch().begin());

    SmallVector<int64_t> innerStarts(ndims, 0);
    SmallVector<int64_t> innerLimits = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    innerStarts[concatDim] = innerLimits[concatDim] - paddedBoundarySize;

    auto leftSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), superSliceInnerArg, innerStarts, innerLimits,
        innerStrides);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), leftSlice->getResults());
  }

  {
    rewriter.createBlock(&ifCondCommSelect.getFalseBranch(),
                         ifCondCommSelect.getFalseBranch().begin());

    SmallVector<int64_t> innerStarts(ndims, 0);
    SmallVector<int64_t> innerLimits = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    innerLimits[concatDim] = paddedBoundarySize;

    auto rightSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), superSliceInnerArg, innerStarts, innerLimits,
        innerStrides);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         rightSlice->getResults());
  }

  rewriter.setInsertionPointAfter(ifCondCommSelect);

  auto cperm = rewriter.create<stablehlo::CollectivePermuteOp>(
      op->getLoc(), ifCondCommSelect.getResults()[0], sourceTargetPairs,
      stablehlo::ChannelHandleAttr::get(op->getContext(), /*handle*/ channel_id,
                                        /*type*/ 0));
  channel_id++;

  Type ifTypes[] = {RankedTensorType::get(
      localRetShape,
      superSliceInnerArg.getType().cast<RankedTensorType>().getElementType())};
  auto ifCond =
      rewriter.create<stablehlo::IfOp>(op->getLoc(), ifTypes, isLeftBlock);
  rewriter.create<stablehlo::ReturnOp>(op->getLoc(), ifCond->getResults());

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
    auto innerConcat = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);

    SmallVector<Value> dynamicSliceStartSlices(ndims, zero);

    dynamicSliceStartSlices[concatDim] =
        rewriter.create<stablehlo::MulOp>(op->getLoc(), pidRem, alpha);

    auto slicedPart = rewriter.create<stablehlo::DynamicSliceOp>(
        op->getLoc(), innerConcat, dynamicSliceStartSlices, localRetShape);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         slicedPart->getResults());
  }

  // Case II: for the right part of the comm
  {
    rewriter.createBlock(&ifCond.getFalseBranch(),
                         ifCond.getFalseBranch().begin());

    Value concatArgs[] = {midOpInnerArg, cperm};
    auto innerConcat = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);

    SmallVector<Value> dynamicSliceStartSlices(ndims, zero);
    auto limitIndex =
        rewriter.create<stablehlo::MulOp>(op->getLoc(), pidRem, alpha);
    auto constVal = rewriter.create<stablehlo::ConstantOp>(
        op->getLoc(), partitionIdType,
        makeAttr(partitionIdType, innerConcat.getType()
                                          .cast<RankedTensorType>()
                                          .getShape()[concatDim] -
                                      localRetShape[concatDim])
            .cast<ElementsAttr>());

    dynamicSliceStartSlices[concatDim] = rewriter.create<stablehlo::SubtractOp>(
        op->getLoc(), constVal, limitIndex);

    auto slicedPart = rewriter.create<stablehlo::DynamicSliceOp>(
        op->getLoc(), innerConcat, dynamicSliceStartSlices, localRetShape);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
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
                        int ndims, int T, SmallVector<int64_t> localRetShape,
                        Value isLeftSide, int &channel_id,
                        bool returnResults = true) {
  auto elemType =
      superSliceInnerArg.getType().cast<RankedTensorType>().getElementType();

  auto commResultShape = llvm::to_vector(
      cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
  commResultShape[concatDim] = N;

  Type ifTypesCommSelect[] = {RankedTensorType::get(commResultShape, elemType)};
  auto ifCondCommSelect = rewriter.create<stablehlo::IfOp>(
      op->getLoc(), ifTypesCommSelect, isLeftSide);

  SmallVector<int64_t> innerStrides(ndims, 1);

  {
    rewriter.createBlock(&ifCondCommSelect.getTrueBranch(),
                         ifCondCommSelect.getTrueBranch().begin());

    SmallVector<int64_t> innerStarts1(ndims, 0);
    SmallVector<int64_t> innerLimits1 = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    innerLimits1[concatDim] = N;
    auto endSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), superSliceInnerArg, innerStarts1, innerLimits1,
        innerStrides);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), endSlice->getResults());
  }

  {
    rewriter.createBlock(&ifCondCommSelect.getFalseBranch(),
                         ifCondCommSelect.getFalseBranch().begin());

    SmallVector<int64_t> innerStarts2(ndims, 0);
    SmallVector<int64_t> innerLimits2 = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    innerStarts2[concatDim] = innerLimits2[concatDim] - N;
    auto startSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), superSliceInnerArg, innerStarts2, innerLimits2,
        innerStrides);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         startSlice->getResults());
  }

  rewriter.setInsertionPointAfter(ifCondCommSelect);

  auto sourceTargetIdxs =
      generateShiftPairs(opSharding, concatDim, op, true, true);
  auto sourceTargetIdxsTmp =
      generateShiftPairs(opSharding, concatDim, op, false, true);
  sourceTargetIdxs.append(sourceTargetIdxsTmp.begin(),
                          sourceTargetIdxsTmp.end());

  auto commResult = rewriter.create<stablehlo::CollectivePermuteOp>(
      op->getLoc(), ifCondCommSelect.getResults()[0],
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
      rewriter.create<stablehlo::IfOp>(op->getLoc(), ifTypes, isLeftSide);

  if (returnResults)
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
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

      auto lhsRightSlice = rewriter.create<stablehlo::SliceOp>(
          op->getLoc(), midOpInnerArg, innerStarts3, innerLimits3,
          innerStrides);
      concatArgs[1] = lhsRightSlice;
    }

    auto finalResult = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
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

      auto rhsLeftSlice = rewriter.create<stablehlo::SliceOp>(
          op->getLoc(), midOpInnerArg, innerStarts4, innerLimits4,
          innerStrides);
      concatArgs[0] = rhsLeftSlice;
    }

    auto finalResult = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
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
                          SmallVector<int64_t> localRetShape, Value isLeftSide,
                          bool returnResults = true) {
  auto elemType = innerArg.getType().cast<RankedTensorType>().getElementType();

  SmallVector<int64_t> innerStrides(ndims, 1);

  Type ifTypes[] = {RankedTensorType::get(localRetShape, elemType)};
  auto ifCondInner =
      rewriter.create<stablehlo::IfOp>(op->getLoc(), ifTypes, isLeftSide);
  if (returnResults)
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         ifCondInner->getResults());

  {
    rewriter.createBlock(&ifCondInner.getTrueBranch(),
                         ifCondInner.getTrueBranch().begin());

    SmallVector<int64_t> innerStarts1(ndims, 0);
    SmallVector<int64_t> innerLimits1 =
        llvm::to_vector(cast<RankedTensorType>(innerArg.getType()).getShape());
    innerLimits1[concatDim] = N;
    auto startSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), innerArg, innerStarts1, innerLimits1, innerStrides);

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

      auto lhsRightSlice = rewriter.create<stablehlo::SliceOp>(
          op->getLoc(), innerArg, innerStarts3, innerLimits3, innerStrides);
      concatArgs[1] = lhsRightSlice;
    }

    auto finalResult = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         finalResult->getResults());
  }

  {
    rewriter.createBlock(&ifCondInner.getFalseBranch(),
                         ifCondInner.getFalseBranch().begin());

    SmallVector<int64_t> innerStarts2(ndims, 0);
    SmallVector<int64_t> innerLimits2 =
        llvm::to_vector(cast<RankedTensorType>(innerArg.getType()).getShape());
    innerStarts2[concatDim] = innerLimits2[concatDim] - N;
    auto endSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), innerArg, innerStarts2, innerLimits2, innerStrides);

    Value concatArgs[2];
    concatArgs[1] = endSlice;

    if (N - 2 * N / numDevicesAlongDimension == 0) {
      concatArgs[0] = innerArg;
    } else {
      SmallVector<int64_t> innerStarts4(ndims, 0);
      SmallVector<int64_t> innerLimits4 = llvm::to_vector(
          cast<RankedTensorType>(innerArg.getType()).getShape());
      innerStarts4[concatDim] = N - (2 * N / numDevicesAlongDimension);

      auto rhsLeftSlice = rewriter.create<stablehlo::SliceOp>(
          op->getLoc(), innerArg, innerStarts4, innerLimits4, innerStrides);
      concatArgs[0] = rhsLeftSlice;
    }

    auto finalResult = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         finalResult->getResults());
  }

  rewriter.setInsertionPointAfter(ifCondInner);
  return ifCondInner->getResults();
}

// TODO: we might need to update this to use the generalized version for the
// generateShiftPairs function
std::tuple<Value, Value, Value, Value, Value, Value>
getChecksForBoundaries(PatternRewriter &rewriter, Operation *op,
                       stablehlo::PartitionIdOp partitionId,
                       int64_t numDevicesAlongDimension,
                       stablehlo::ConstantOp zero) {
  Value leftSide = rewriter.create<stablehlo::RemOp>(
      op->getLoc(), partitionId,
      rewriter.create<stablehlo::ConstantOp>(
          op->getLoc(), partitionId.getType(),
          makeAttr(partitionId.getType(), numDevicesAlongDimension)
              .cast<ElementsAttr>()));
  Value isLeftSide = rewriter.create<stablehlo::CompareOp>(
      op->getLoc(), leftSide, zero, stablehlo::ComparisonDirection::EQ);

  Value rightSide = rewriter.create<stablehlo::AddOp>(
      op->getLoc(), partitionId,
      rewriter.create<stablehlo::ConstantOp>(
          op->getLoc(),
          makeAttr(partitionId.getType(), 1).cast<ElementsAttr>()));
  rightSide = rewriter.create<stablehlo::RemOp>(
      op->getLoc(), rightSide,
      rewriter.create<stablehlo::ConstantOp>(
          op->getLoc(), partitionId.getType(),
          makeAttr(partitionId.getType(), numDevicesAlongDimension)
              .cast<ElementsAttr>()));
  Value isRightSide = rewriter.create<stablehlo::CompareOp>(
      op->getLoc(), rightSide, zero, stablehlo::ComparisonDirection::EQ);

  Value isNotLeftSide =
      rewriter.create<stablehlo::NotOp>(op->getLoc(), isLeftSide);
  Value isNotRightSide =
      rewriter.create<stablehlo::NotOp>(op->getLoc(), isRightSide);

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
      superSliceOp = rewriter.create<stablehlo::SliceOp>(
          concat.getLoc(), leftSliceOp.getOperand(),
          rightSliceOp.getStartIndices(), leftSliceOp.getLimitIndices(),
          leftSliceOp.getStrides());
    } else {
      if ((leftSliceOp.getOperand() != midSliceOp.getOperand()) ||
          (rightSliceOp.getStartIndices()[concatDim] !=
           midSliceOp.getStartIndices()[concatDim]) ||
          (leftSliceOp.getLimitIndices()[concatDim] !=
           midSliceOp.getLimitIndices()[concatDim])) {
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

    SmallVector<StringAttr> manual_axes;
    SmallVector<int64_t> localShape =
        llvm::to_vector(cast<RankedTensorType>(midOp.getType()).getShape());

    updateManualComputationAxesShape(concatSharding, rewriter, concat,
                                     manual_axes, localShape, concatDim);

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

    SmallVector<int64_t> localRetShape = llvm::to_vector(concatShape);
    SmallVector<int64_t> manualOpRetShape = llvm::to_vector(concatShape);
    for (int i = 0; i < localRetShape.size(); i++) {
      if (i == concatDim) {
        localRetShape[i] = T / ndevices[i];
        continue;
      }
      localRetShape[i] /= ndevices[i];
    }
    manualOpRetShape[concatDim] = T;

    mlir::Type in_tys[2]{RankedTensorType::get(localShape, elemType),
                         RankedTensorType::get(localShape, elemType)};
    mlir::Location in_locs[] = {superSliceOp.getLoc(), midOp.getLoc()};

    Value manual_ops[] = {superSliceOp, midOp};
    Type manual_types[] = {RankedTensorType::get(manualOpRetShape, elemType)};
    auto manual = rewriter.create<sdy::ManualComputationOp>(
        concat.getLoc(), manual_types, manual_ops, in_shardings, out_shardings,
        manual_axes);

    auto blk = rewriter.createBlock(&manual.getBody(), manual.getBody().begin(),
                                    in_tys, in_locs);
    auto superSliceInnerArg = blk->getArgument(0);
    auto midOpInnerArg = blk->getArgument(1);
    auto partitionId =
        rewriter.create<stablehlo::PartitionIdOp>(concat.getLoc());

    auto zero = rewriter.create<stablehlo::ConstantOp>(
        concat.getLoc(), rewriter.getZeroAttr(partitionId.getType()));

    auto [isLeftSide, isRightSide, isNotLeftSide, isNotRightSide, leftSide,
          rightSide] = getChecksForBoundaries(rewriter, concat, partitionId,
                                              numDevicesAlongDimension, zero);

    SmallVector<int64_t> innerStrides(ndims, 1);

    if (numDevicesAlongDimension != 2) {
      Type ifTypes[] = {RankedTensorType::get(localRetShape, elemType)};
      auto if1 = rewriter.create<stablehlo::IfOp>(
          concat.getLoc(), ifTypes,
          rewriter.create<stablehlo::AndOp>(concat.getLoc(), isNotLeftSide,
                                            isNotRightSide));

      // if ..... !leftSide  && !rightSide
      {
        rewriter.createBlock(&if1.getTrueBranch(), if1.getTrueBranch().begin());

        generateCommPatternForNonEdges(rewriter, concat, partitionId, zero,
                                       superSliceInnerArg, midOpInnerArg,
                                       concatSharding, concatDim, N,
                                       numDevicesAlongDimension, ndims,
                                       localRetShape, leftSide, channel_id);
      }

      // else
      {
        rewriter.createBlock(&if1.getFalseBranch(),
                             if1.getFalseBranch().begin());

        wrapCommPatternForEdges(rewriter, concat, partitionId, zero,
                                superSliceInnerArg, midOpInnerArg,
                                concatSharding, concatDim, N,
                                numDevicesAlongDimension, ndims, T,
                                localRetShape, isLeftSide, channel_id);
      }

      rewriter.setInsertionPointAfter(if1);
      rewriter.create<sdy::ReturnOp>(concat.getLoc(), if1->getResults());
    } else {
      auto results = wrapCommPatternForEdges(
          rewriter, concat, partitionId, zero, superSliceInnerArg,
          midOpInnerArg, concatSharding, concatDim, N, numDevicesAlongDimension,
          ndims, T, localRetShape, isLeftSide, channel_id,
          /*returnResults=*/false);
      rewriter.create<sdy::ReturnOp>(concat.getLoc(), results);
    }

    if (leftPadding != 0 || rightPadding != 0) {
      SmallVector<int64_t> sliceStartIndices(ndims, 0);
      SmallVector<int64_t> sliceLimits = llvm::to_vector(
          cast<RankedTensorType>(manual->getResults()[0].getType()).getShape());

      if (leftPadding > 0) {
        sliceStartIndices[concatDim] = leftPadding;
      }

      if (rightPadding > 0) {
        sliceLimits[concatDim] -= rightPadding;
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

    SmallVector<int64_t> localRetShape = llvm::to_vector(wrapShape);
    SmallVector<int64_t> manualOpRetShape = llvm::to_vector(wrapShape);
    for (int i = 0; i < localRetShape.size(); i++) {
      if (i == wrapDimension) {
        localRetShape[i] = paddedResultSize / ndevices[i];
        continue;
      }
      localRetShape[i] /= ndevices[i];
    }
    manualOpRetShape[wrapDimension] = paddedResultSize;

    mlir::Type inTys[1]{RankedTensorType::get(localShape, elemType)};
    mlir::Location inLocs[] = {wrap.getLoc()};

    Value manualOps[] = {wrap.getOperand()};
    Type manualTypes[] = {RankedTensorType::get(manualOpRetShape, elemType)};
    auto manual = rewriter.create<sdy::ManualComputationOp>(
        wrap.getLoc(), manualTypes, manualOps, inShardings, outShardings,
        manualAxes);

    auto blk = rewriter.createBlock(&manual.getBody(), manual.getBody().begin(),
                                    inTys, inLocs);
    auto innerArg = blk->getArgument(0);

    auto partitionId = rewriter.create<stablehlo::PartitionIdOp>(wrap.getLoc());

    auto zero = rewriter.create<stablehlo::ConstantOp>(
        wrap.getLoc(), rewriter.getZeroAttr(partitionId.getType()));

    auto [isLeftSide, isRightSide, isNotLeftSide, isNotRightSide, leftSide,
          rightSide] = getChecksForBoundaries(rewriter, wrap, partitionId,
                                              numDevicesAlongDimension, zero);

    if (numDevicesAlongDimension != 2) {
      Type ifTypes[] = {RankedTensorType::get(localRetShape, elemType)};
      auto ifCond = rewriter.create<stablehlo::IfOp>(
          wrap.getLoc(), ifTypes,
          rewriter.create<stablehlo::AndOp>(wrap.getLoc(), isNotLeftSide,
                                            isNotRightSide));

      {
        rewriter.createBlock(&ifCond.getTrueBranch(),
                             ifCond.getTrueBranch().begin());

        generateCommPatternForNonEdges(
            rewriter, wrap, partitionId, zero, innerArg, innerArg, wrapSharding,
            wrapDimension, paddedBoundarySize, numDevicesAlongDimension, ndims,
            localRetShape, leftSide, channel_id);
      }

      {
        rewriter.createBlock(&ifCond.getFalseBranch(),
                             ifCond.getFalseBranch().begin());

        wrapCommPatternForEdges(
            rewriter, wrap, partitionId, zero, innerArg, innerArg, wrapSharding,
            wrapDimension, paddedBoundarySize, numDevicesAlongDimension, ndims,
            paddedResultSize, localRetShape, isLeftSide, channel_id);
      }

      rewriter.setInsertionPointAfter(ifCond);
      rewriter.create<sdy::ReturnOp>(wrap.getLoc(), ifCond->getResults());
    } else {
      // There are no edges in this case
      auto results = wrapCommPatternForEdges(
          rewriter, wrap, partitionId, zero, innerArg, innerArg, wrapSharding,
          wrapDimension, paddedBoundarySize, numDevicesAlongDimension, ndims,
          paddedResultSize, localRetShape, isLeftSide, channel_id,
          /*returnResults=*/false);
      rewriter.create<sdy::ReturnOp>(wrap.getLoc(), results);
    }

    if (leftPadding != 0 || rightPadding != 0) {
      SmallVector<int64_t> sliceStartIndices(ndims, 0);
      SmallVector<int64_t> sliceLimits = llvm::to_vector(
          cast<RankedTensorType>(manual->getResults()[0].getType()).getShape());
      SmallVector<int64_t> innerStrides(ndims, 1);

      if (leftPadding > 0) {
        sliceStartIndices[wrapDimension] = leftPadding;
      }

      if (rightPadding > 0) {
        sliceLimits[wrapDimension] -= rightPadding;
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
    leftLimits[wrapDimension] = wrap.getLhs();

    auto leftSliceOp = rewriter.create<stablehlo::SliceOp>(
        wrap.getLoc(), wrap.getOperand(), leftStarts, leftLimits, strides);
    sdy::setSharding(leftSliceOp, wrapSharding);

    SmallVector<int64_t> rightStarts(ndims, 0);
    SmallVector<int64_t> rightLimits = llvm::to_vector(wrapOperandShape);
    rightStarts[wrapDimension] = rightLimits[wrapDimension] - wrap.getRhs();

    auto rightSliceOp = rewriter.create<stablehlo::SliceOp>(
        wrap.getLoc(), wrap.getOperand(), rightStarts, rightLimits, strides);
    sdy::setSharding(rightSliceOp, wrapSharding);

    auto zero = rewriter.create<stablehlo::ConstantOp>(
        wrap.getLoc(), rewriter.getZeroAttr(elemType));

    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);

    padLow[wrapDimension] = wrapShape[wrapDimension] - wrap.getLhs();
    padHigh[wrapDimension] = 0;
    auto paddedLeftSliceOp = rewriter.create<stablehlo::PadOp>(
        wrap.getLoc(), leftSliceOp, zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedLeftSliceOp, wrapSharding);

    padLow[wrapDimension] = 0;
    padHigh[wrapDimension] = wrapShape[wrapDimension] - wrap.getRhs();
    auto paddedRightSliceOp = rewriter.create<stablehlo::PadOp>(
        wrap.getLoc(), rightSliceOp, zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedRightSliceOp, wrapSharding);

    padLow[wrapDimension] = wrap.getLhs();
    padHigh[wrapDimension] = wrap.getRhs();
    auto paddedWrapOp = rewriter.create<stablehlo::PadOp>(
        wrap.getLoc(), wrap.getOperand(), zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedWrapOp, wrapSharding);

    auto addOp = rewriter.create<stablehlo::AddOp>(
        wrap.getLoc(), paddedLeftSliceOp, paddedRightSliceOp);
    mlir::sdy::setSharding(addOp, wrapSharding);

    addOp =
        rewriter.create<stablehlo::AddOp>(wrap.getLoc(), addOp, paddedWrapOp);
    sdy::setSharding(addOp, wrapSharding);

    rewriter.replaceOp(wrap, addOp);
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

    SmallVector<int64_t> localRetShape = llvm::to_vector(extendShape);
    SmallVector<int64_t> manualOpRetShape = llvm::to_vector(extendShape);
    for (int i = 0; i < localRetShape.size(); i++) {
      if (i == extendDimension) {
        localRetShape[i] = paddedResultSize / ndevices[i];
        continue;
      }
      localRetShape[i] /= ndevices[i];
    }
    manualOpRetShape[extendDimension] = paddedResultSize;

    mlir::Type inTys[1]{RankedTensorType::get(localShape, elemType)};
    mlir::Location inLocs[] = {extend.getLoc()};

    Value manualOps[] = {extend.getOperand()};
    Type manualTypes[] = {RankedTensorType::get(manualOpRetShape, elemType)};
    auto manual = rewriter.create<sdy::ManualComputationOp>(
        extend.getLoc(), manualTypes, manualOps, inShardings, outShardings,
        manualAxes);

    auto blk = rewriter.createBlock(&manual.getBody(), manual.getBody().begin(),
                                    inTys, inLocs);
    auto innerArg = blk->getArgument(0);

    auto partitionId =
        rewriter.create<stablehlo::PartitionIdOp>(extend.getLoc());

    auto zero = rewriter.create<stablehlo::ConstantOp>(
        extend.getLoc(), rewriter.getZeroAttr(partitionId.getType()));

    auto [isLeftSide, isRightSide, isNotLeftSide, isNotRightSide, leftSide,
          rightSide] = getChecksForBoundaries(rewriter, extend, partitionId,
                                              numDevicesAlongDimension, zero);

    if (numDevicesAlongDimension != 2) {
      Type ifTypes[] = {RankedTensorType::get(localRetShape, elemType)};
      auto ifCond = rewriter.create<stablehlo::IfOp>(
          extend.getLoc(), ifTypes,
          rewriter.create<stablehlo::AndOp>(extend.getLoc(), isNotLeftSide,
                                            isNotRightSide));

      {
        rewriter.createBlock(&ifCond.getTrueBranch(),
                             ifCond.getTrueBranch().begin());

        generateCommPatternForNonEdges(rewriter, extend, partitionId, zero,
                                       innerArg, innerArg, extendSharding,
                                       extendDimension, paddedBoundarySize,
                                       numDevicesAlongDimension, ndims,
                                       localRetShape, leftSide, channel_id);
      }

      {
        rewriter.createBlock(&ifCond.getFalseBranch(),
                             ifCond.getFalseBranch().begin());

        extendCommPatternForEdges(
            rewriter, extend, partitionId, zero, innerArg, extendSharding,
            extendDimension, paddedBoundarySize, numDevicesAlongDimension,
            ndims, paddedResultSize, localRetShape, isLeftSide);
      }

      rewriter.setInsertionPointAfter(ifCond);
      rewriter.create<sdy::ReturnOp>(extend.getLoc(), ifCond->getResults());
    } else {
      auto results = extendCommPatternForEdges(
          rewriter, extend, partitionId, zero, innerArg, extendSharding,
          extendDimension, paddedBoundarySize, numDevicesAlongDimension, ndims,
          paddedResultSize, localRetShape, isLeftSide, /*returnResults=*/false);
      rewriter.create<sdy::ReturnOp>(extend.getLoc(), results);
    }

    if (leftPadding != 0 || rightPadding != 0) {
      SmallVector<int64_t> sliceStartIndices(ndims, 0);
      SmallVector<int64_t> sliceLimits = llvm::to_vector(
          cast<RankedTensorType>(manual->getResults()[0].getType()).getShape());
      SmallVector<int64_t> innerStrides(ndims, 1);

      if (leftPadding > 0) {
        sliceStartIndices[extendDimension] = leftPadding;
      }

      if (rightPadding > 0) {
        sliceLimits[extendDimension] -= rightPadding;
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

    auto leftSliceOp = rewriter.create<stablehlo::SliceOp>(
        extend.getLoc(), extend.getOperand(), leftStarts, leftLimits, strides);
    sdy::setSharding(leftSliceOp, extendSharding);

    SmallVector<int64_t> rightStarts(ndims, 0);
    SmallVector<int64_t> rightLimits = llvm::to_vector(extendOperandShape);
    rightStarts[extendDimension] =
        rightLimits[extendDimension] - extend.getRhs();

    auto rightSliceOp = rewriter.create<stablehlo::SliceOp>(
        extend.getLoc(), extend.getOperand(), rightStarts, rightLimits,
        strides);
    sdy::setSharding(rightSliceOp, extendSharding);

    auto zero = rewriter.create<stablehlo::ConstantOp>(
        extend.getLoc(), rewriter.getZeroAttr(elemType));

    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);

    padLow[extendDimension] = 0;
    padHigh[extendDimension] = extendShape[extendDimension] - extend.getLhs();
    auto paddedLeftSliceOp = rewriter.create<stablehlo::PadOp>(
        extend.getLoc(), leftSliceOp, zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedLeftSliceOp, extendSharding);

    padLow[extendDimension] = extendShape[extendDimension] - extend.getRhs();
    padHigh[extendDimension] = 0;
    auto paddedRightSliceOp = rewriter.create<stablehlo::PadOp>(
        extend.getLoc(), rightSliceOp, zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedRightSliceOp, extendSharding);

    padLow[extendDimension] = extend.getLhs();
    padHigh[extendDimension] = extend.getRhs();
    auto paddedExtendOp = rewriter.create<stablehlo::PadOp>(
        extend.getLoc(), extend.getOperand(), zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedExtendOp, extendSharding);

    auto addOp = rewriter.create<stablehlo::AddOp>(
        extend.getLoc(), paddedLeftSliceOp, paddedRightSliceOp);
    mlir::sdy::setSharding(addOp, extendSharding);

    addOp = rewriter.create<stablehlo::AddOp>(extend.getLoc(), addOp,
                                              paddedExtendOp);
    sdy::setSharding(addOp, extendSharding);

    rewriter.replaceOp(extend, addOp);
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

      inputArg = rewriter.create<stablehlo::PadOp>(
          rotate.getLoc(), rotate.getOperand(),
          rewriter.create<stablehlo::ConstantOp>(rotate.getLoc(),
                                                 rewriter.getZeroAttr(elType)),
          padLow, padHigh, padInner);
    }
    if (amount > localShape[rotate.getDimension()]) {
      return rewriter.notifyMatchFailure(rotate, "No local tensor remaining!");
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
      if (leftToRight) {
        innerLimits[rotate.getDimension()] = amount;
      } else {
        innerStarts[rotate.getDimension()] =
            innerLimits[rotate.getDimension()] - amount;
      }
      auto commSlice = rewriter.create<stablehlo::SliceOp>(
          rotate.getLoc(), innerArg, innerStarts, innerLimits, innerStrides);

      auto sourceTargetIdxs = generateShiftPairs(
          rotateSharding, rotate.getDimension(), rotate, leftToRight, false);

      auto commResult = rewriter.create<stablehlo::CollectivePermuteOp>(
          rotate.getLoc(), commSlice,
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
        rewriter.create<sdy::ReturnOp>(rotate.getLoc(),
                                       commResult->getResults());
      } else {
        auto remSlice = rewriter.create<stablehlo::SliceOp>(
            rotate.getLoc(), innerArg, innerStartsPresent, innerLimitsPresent,
            innerStrides);

        std::array<Value, 2> concatArgs;
        if (leftToRight)
          concatArgs = {remSlice, commResult};
        else
          concatArgs = {commResult, remSlice};

        auto innerConcat = rewriter.create<stablehlo::ConcatenateOp>(
            rotate.getLoc(), concatArgs, rotateDimension);

        rewriter.create<sdy::ReturnOp>(rotate.getLoc(),
                                       innerConcat->getResults());
      }
    }

    if (rightPadding != 0) {
      rewriter.setInsertionPointAfter(manual);

      SmallVector<int64_t> innerStarts(ndims, 0);
      SmallVector<int64_t> innerLimits = llvm::to_vector(outputShape);
      innerLimits[rotateDimension] -= rightPadding;

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

struct RotateToPadCommOptimize : public OpRewritePattern<enzymexla::RotateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::RotateOp rotate,
                                PatternRewriter &rewriter) const override {
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

    auto sl0 = rewriter.create<stablehlo::SliceOp>(
        rotate.getLoc(), rotate.getOperand(), sl0_starts, sl0_ends, strides);
    sdy::setSharding(sl0, rotateSharding);

    auto sl1 = rewriter.create<stablehlo::SliceOp>(
        rotate.getLoc(), rotate.getOperand(), sl1_starts, sl1_ends, strides);
    sdy::setSharding(sl1, rotateSharding);

    auto zero = rewriter.create<stablehlo::ConstantOp>(
        rotate.getLoc(), rewriter.getZeroAttr(elType));

    SmallVector<int64_t> padInner(ndims, 0);
    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    padHigh[rotate.getDimension()] =
        sl1.getType().getShape()[rotate.getDimension()];
    auto paddedSl0 = rewriter.create<stablehlo::PadOp>(
        rotate.getLoc(), sl0, zero, padLow, padHigh, padInner);
    sdy::setSharding(paddedSl0, rotateSharding);

    padHigh[rotate.getDimension()] = 0;
    padLow[rotate.getDimension()] =
        sl0.getType().getShape()[rotate.getDimension()];
    auto paddedSl1 = rewriter.create<stablehlo::PadOp>(
        rotate.getLoc(), sl1, zero, padLow, padHigh, padInner);

    auto addOp = rewriter.create<stablehlo::AddOp>(rotate.getLoc(), paddedSl0,
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
    auto leftOperand = allOperands[0];
    auto rightOperand = allOperands[1];

    auto leftOperandSharding = mlir::sdy::getSharding(leftOperand);
    if (!leftOperandSharding)
      return failure();
    auto rightOperandSharding = mlir::sdy::getSharding(rightOperand);
    if (!rightOperandSharding)
      return failure();

    if (leftOperandSharding != rightOperandSharding ||
        leftOperandSharding != concatSharding) {
      return failure();
    }

    auto leftOperandSize = cast<RankedTensorType>(leftOperand.getType())
                               .getShape()[concatDimension];
    auto rightOperandSize = cast<RankedTensorType>(rightOperand.getType())
                                .getShape()[concatDimension];

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

    TensorShardingAttr opShardings[] = {concatSharding};

    auto meshAttr = mlir::sdy::getCommonMesh(opShardings, opShardings, concat);
    if (!meshAttr)
      return failure();

    TensorShardingAttr opShardingsIn[] = {concatSharding, concatSharding};
    TensorShardingPerValueAttr inShardings =
        TensorShardingPerValueAttr::get(concat.getContext(), opShardingsIn);
    TensorShardingPerValueAttr outShardings =
        TensorShardingPerValueAttr::get(concat.getContext(), opShardings);

    SmallVector<StringAttr> manualAxes;
    SmallVector<int64_t> tmpConcatShape = llvm::to_vector(concatShape);

    updateManualComputationAxesShape(concatSharding, rewriter, concat,
                                     manualAxes, tmpConcatShape,
                                     concatDimension);

    Value extendOperand;
    Value mainOperand;
    if (concatLeft) {
      extendOperand = leftOperand;
      mainOperand = rightOperand;
    } else {
      extendOperand = rightOperand;
      mainOperand = leftOperand;
    }

    auto extra = concatShape[concatDimension] % numDevicesAlongDimension;
    int64_t rightPadding = 0;
    int64_t leftPadding = 0;
    if (extra != 0) {
      if (concatLeft) {
        leftPadding = numDevicesAlongDimension - extra;
      } else {
        rightPadding = numDevicesAlongDimension - extra;
      }
    }

    int64_t diff = (cast<RankedTensorType>(extendOperand.getType())
                        .getShape()[concatDimension] +
                    leftPadding + rightPadding) *
                   (numDevicesAlongDimension - 1);

    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);
    if (concatLeft) {
      padHigh[concatDimension] = diff;
      padLow[concatDimension] = leftPadding;
    } else {
      padLow[concatDimension] = diff;
      padHigh[concatDimension] = rightPadding;
    }

    auto reshardedExtendOp = rewriter.create<stablehlo::PadOp>(
        concat.getLoc(), extendOperand,
        rewriter.create<stablehlo::ConstantOp>(concat.getLoc(),
                                               rewriter.getZeroAttr(elemType)),
        padLow, padHigh, padInner);
    sdy::setSharding(reshardedExtendOp, concatSharding);

    SmallVector<int64_t> localExtendShape = llvm::to_vector(
        cast<RankedTensorType>(reshardedExtendOp.getType()).getShape());
    SmallVector<int64_t> mainOperandShape = llvm::to_vector(
        cast<RankedTensorType>(mainOperand.getType()).getShape());
    SmallVector<int64_t> localRetShape(ndims, 0);
    SmallVector<int64_t> manualOpRetShape = llvm::to_vector(concatShape);
    manualOpRetShape[concatDimension] += leftPadding + rightPadding;

    for (int i = 0; i < ndims; i++) {
      localExtendShape[i] = localExtendShape[i] / ndevices[i];
      mainOperandShape[i] = mainOperandShape[i] / ndevices[i];
      localRetShape[i] = manualOpRetShape[i] / ndevices[i];
    }

    mlir::Type inTys[2]{RankedTensorType::get(localExtendShape, elemType),
                        RankedTensorType::get(mainOperandShape, elemType)};
    mlir::Location inLocs[] = {reshardedExtendOp.getLoc(),
                               mainOperand.getLoc()};

    Value manualOps[] = {reshardedExtendOp, mainOperand};
    Type manualTypes[] = {RankedTensorType::get(manualOpRetShape, elemType)};
    auto manual = rewriter.create<sdy::ManualComputationOp>(
        concat.getLoc(), manualTypes, manualOps, inShardings, outShardings,
        manualAxes);

    SmallVector<int64_t> innerStrides(ndims, 1);

    {
      auto blk = rewriter.createBlock(&manual.getBody(),
                                      manual.getBody().begin(), inTys, inLocs);
      auto extendArg = blk->getArgument(0);
      auto mainArg = blk->getArgument(1);

      int N1 = cast<RankedTensorType>(extendArg.getType())
                   .getShape()[concatDimension];
      int N2 =
          cast<RankedTensorType>(mainArg.getType()).getShape()[concatDimension];

      auto partitionId =
          rewriter.create<stablehlo::PartitionIdOp>(concat.getLoc());
      auto partitionIdType = partitionId.getType();
      auto zero = rewriter.create<stablehlo::ConstantOp>(
          concat.getLoc(), rewriter.getZeroAttr(partitionIdType));
      auto onePId = rewriter.create<stablehlo::ConstantOp>(
          concat.getLoc(), partitionIdType,
          makeAttr(partitionIdType, 1).cast<ElementsAttr>());

      auto [isLeftSide, isRightSide, isNotLeftSide, isNotRightSide, leftSide,
            rightSide] = getChecksForBoundaries(rewriter, concat, partitionId,
                                                numDevicesAlongDimension, zero);

      stablehlo::SliceOp commSlice;
      if (concatLeft) {
        SmallVector<int64_t> innerStarts(ndims, 0);
        SmallVector<int64_t> innerLimits = llvm::to_vector(
            cast<RankedTensorType>(mainArg.getType()).getShape());
        innerStarts[concatDimension] = N2 - N1;

        commSlice = rewriter.create<stablehlo::SliceOp>(
            concat.getLoc(), mainArg, innerStarts, innerLimits, innerStrides);
      } else {
        SmallVector<int64_t> innerStarts(ndims, 0);
        SmallVector<int64_t> innerLimits = llvm::to_vector(
            cast<RankedTensorType>(mainArg.getType()).getShape());
        innerLimits[concatDimension] = N1;

        commSlice = rewriter.create<stablehlo::SliceOp>(
            concat.getLoc(), mainArg, innerStarts, innerLimits, innerStrides);
      }

      auto shiftPairs = generateShiftPairs(concatSharding, concatDimension,
                                           concat, commLeft, false);

      auto commResult = rewriter.create<stablehlo::CollectivePermuteOp>(
          concat.getLoc(), commSlice,
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
          commResult.getType().cast<ShapedType>().getShape(), elemType)};
      stablehlo::IfOp ifCond = rewriter.create<stablehlo::IfOp>(
          concat.getLoc(), ifTypes, concatLeft ? isLeftSide : isRightSide);

      {
        rewriter.createBlock(&ifCond.getTrueBranch(),
                             ifCond.getTrueBranch().begin());

        rewriter.create<stablehlo::ReturnOp>(concat.getLoc(), extendArg);
      }

      {
        rewriter.createBlock(&ifCond.getFalseBranch(),
                             ifCond.getFalseBranch().begin());

        rewriter.create<stablehlo::ReturnOp>(concat.getLoc(),
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
          extendArg.getType().cast<ShapedType>().getShape()[concatDimension];

      auto concatResult = rewriter.create<stablehlo::ConcatenateOp>(
          concat.getLoc(), concatArgs, concatDimension);

      auto alpha = rewriter.create<stablehlo::ConstantOp>(
          concat.getLoc(), partitionIdType,
          makeAttr(partitionIdType, extendSize / numDevicesAlongDimension)
              .cast<ElementsAttr>());

      SmallVector<Value> dynamicSliceStartSlices;
      for (int i = 0; i < ndims; i++) {
        if (i == concatDimension) {
          if (concatLeft) {
            dynamicSliceStartSlices.push_back(rewriter.create<stablehlo::MulOp>(
                concat.getLoc(),
                rewriter.create<stablehlo::SubtractOp>(concat.getLoc(),
                                                       partitionId, onePId),
                alpha));
          } else {
            auto diffIdx = rewriter.create<stablehlo::MulOp>(
                concat.getLoc(),
                rewriter.create<stablehlo::AddOp>(concat.getLoc(), partitionId,
                                                  onePId),
                alpha);
            dynamicSliceStartSlices.push_back(
                rewriter.create<stablehlo::SubtractOp>(
                    concat.getLoc(),
                    rewriter.create<stablehlo::ConstantOp>(
                        concat.getLoc(), partitionIdType,
                        makeAttr(partitionIdType, extendSize)
                            .cast<ElementsAttr>()),
                    diffIdx));
          }
        } else {
          dynamicSliceStartSlices.push_back(zero);
        }
      }

      auto slicedPart = rewriter.create<stablehlo::DynamicSliceOp>(
          concat.getLoc(), concatResult, dynamicSliceStartSlices,
          localRetShape);
      rewriter.create<sdy::ReturnOp>(concat.getLoc(), slicedPart->getResults());
    }

    if (leftPadding != 0 || rightPadding != 0) {
      SmallVector<int64_t> sliceStartIndices(ndims, 0);
      SmallVector<int64_t> sliceLimits = llvm::to_vector(
          cast<RankedTensorType>(manual->getResults()[0].getType()).getShape());
      SmallVector<int64_t> innerStrides(ndims, 1);
      if (leftPadding > 0) {
        sliceStartIndices[concatDimension] = leftPadding;
      }
      if (rightPadding > 0) {
        sliceLimits[concatDimension] -= rightPadding;
      }

      rewriter.setInsertionPointAfter(manual);
      auto sliceRemovePadding = rewriter.create<stablehlo::SliceOp>(
          concat.getLoc(), manual->getResults()[0], sliceStartIndices,
          sliceLimits, innerStrides);

      rewriter.replaceOp(concat, sliceRemovePadding);
    } else {
      rewriter.replaceOp(concat, manual);
    }

    return success();
  }
};

struct DUSToPadComm : public OpRewritePattern<stablehlo::DynamicUpdateSliceOp> {

  int &channel_id;
  DUSToPadComm(int &channel_id, MLIRContext *context,
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

      if (localType.getShape()[i] != dus.getType().getShape()[i]) {
        shardedDims.push_back(i);
        manualAxes.push_back(axis[0]);
      }

      lowPads.push_back(v2);
      highPads.push_back(rightPad);
      auto extraPad = 0;
      ;
      if (dus.getType().getShape()[i] % localType.getShape()[i] != 0) {
        auto numDevicesAlongDimension =
            dus.getType().getShape()[i] / localType.getShape()[i];
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
    Type partitionType = nullptr;
    auto getOrCreateConstant = [&](int64_t v, Type TT = nullptr) -> Value {
      if (!TT)
        TT = partitionType;
      auto key = std::make_pair(v, TT);
      auto found = constantCache.find(key);
      if (found != constantCache.end())
        return found->second;
      auto cst = rewriter.create<stablehlo::ConstantOp>(
          loc, TT, makeAttr(TT, v).cast<ElementsAttr>());
      constantCache[key] = cst;
      return cst;
    };

    auto PT = RankedTensorType::get({}, elementType);

    Value globalOperand = dus.getOperand();
    if (extraSlice) {
      SmallVector<int64_t> zeros(ndims, 0);
      auto padOp = rewriter.create<stablehlo::PadOp>(
          loc, dus.getOperand(), getOrCreateConstant(0, PT), zeros,
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
      auto padOp = rewriter.create<stablehlo::PadOp>(
          loc, dus.getUpdate(), getOrCreateConstant(0, PT), lowPads,
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

    auto manual = rewriter.create<sdy::ManualComputationOp>(
        loc, manualTypes, manualOps, in_shardings, out_shardings, manualAxes);

    {
      auto blk = rewriter.createBlock(&manual.getBody(),
                                      manual.getBody().begin(), inTyps, inLocs);

      auto partitionId = rewriter.create<stablehlo::PartitionIdOp>(loc);
      partitionType = partitionId.getType();

      constantCache.clear();

      auto innerOperand = blk->getArgument(0);
      Value innerUpdate;
      if (pad2) {
        innerUpdate = blk->getArgument(1);
      } else {
        innerUpdate = rewriter.create<stablehlo::ConstantOp>(
            loc, localPaddedUpdateType,
            splat.resizeSplat(localPaddedUpdateType));
      }

      Value innerUpdateVal = innerUpdate;
      if (updatedDims.size() != updatedShardedDims.size()) {
        auto zero = getOrCreateConstant(0);
        SmallVector<Value> newStarts(ndims, zero);
        for (int i = 0; i < ndims; i++) {
          if (llvm::is_contained(updatedShardedDims, i))
            continue;
          newStarts[i] = getOrCreateConstant(lowPads[i]);
        }
        innerUpdateVal = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
            loc, innerOperand, innerUpdate, newStarts);
      }

      SmallVector<Value> multiDimIdxs;
      {
        Value cur = partitionId;
        for (int i = 0; i < localResultType.getShape().size(); i++) {
          auto globalSz = globalResultType.getShape()[i];
          auto localSz = localResultType.getShape()[i];
          if (globalSz == localSz) {
            multiDimIdxs.push_back(getOrCreateConstant(0));
            continue;
          }
          if (i == localResultType.getShape().size() - 1) {
            multiDimIdxs.push_back(cur);
          } else {
            auto nDevices = globalSz / localSz;
            auto cst = getOrCreateConstant(nDevices);
            multiDimIdxs.push_back(
                rewriter.create<stablehlo::RemOp>(loc, cur, cst));
            cur = rewriter.create<stablehlo::DivOp>(loc, cur, cst);
          }
        }
      }

      SmallVector<Value> leftSides;
      SmallVector<Value> rightSides;

      auto i1VTy = RankedTensorType::get({}, rewriter.getI1Type());

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
          leftSide = rewriter.create<stablehlo::CompareOp>(
              loc, multiDimIdxs[idx],
              getOrCreateConstant(lowPads[idx] /
                                  localResultType.getShape()[idx]),
              stablehlo::ComparisonDirection::LT);
        } else {
          // Non-evenly divisible pad, for example pad 18, inner 54, right 18,
          // local shape is 10 partition 0, and 1 would need the fused update
          // (aka if idx <= pad / lowerShape)
          leftSide = rewriter.create<stablehlo::CompareOp>(
              loc, multiDimIdxs[idx],
              getOrCreateConstant(lowPads[idx] /
                                  localResultType.getShape()[idx]),
              stablehlo::ComparisonDirection::LE);
        }

        leftSides.push_back(leftSide);
      }

      for (auto &&[i, idx] : llvm::enumerate(updatedShardedDims)) {
        Value rightSide;
        int64_t startIdx =
            lowPads[idx] + globalUnPaddedUpdateType.getShape()[idx];
        if (highPads[idx] == 0) {
          // No pad, we are never needing to check combining update/operand and
          // can just use update
          rightSide = nullptr;
        } else if (startIdx % localResultType.getShape()[idx] == 0) {
          // Evenly divisible startIdx, for example pad 20, inner 50, right X,
          // local shape is 10 partition 7, 8, ... would need the fused update
          // (aka if idx >= startIdx / lowerShape)
          rightSide = rewriter.create<stablehlo::CompareOp>(
              loc, multiDimIdxs[idx],
              getOrCreateConstant(startIdx / localResultType.getShape()[idx]),
              stablehlo::ComparisonDirection::GE);
        } else {
          // Non-evenly divisible startIdx, for example pad 20, inner 48, right
          // X, local shape is 10 partition 6, 7, 8, ... would need the fused
          // update (aka if idx >= startIdx / lowerShape) partition 6 only needs
          // from 58-60, partitoin 7 uses fully
          rightSide = rewriter.create<stablehlo::CompareOp>(
              loc, multiDimIdxs[idx],
              getOrCreateConstant(startIdx / localResultType.getShape()[idx]),
              stablehlo::ComparisonDirection::GE);
        }

        rightSides.push_back(rightSide);
      }

      Value mayContainOperandData = nullptr;
      for (int i = 0; i < updatedShardedDims.size(); i++) {
        if (leftSides[i]) {
          if (mayContainOperandData)
            mayContainOperandData = rewriter.create<stablehlo::OrOp>(
                loc, mayContainOperandData, leftSides[i]);
          else
            mayContainOperandData = leftSides[i];
        }
        if (rightSides[i]) {
          if (mayContainOperandData)
            mayContainOperandData = rewriter.create<stablehlo::OrOp>(
                loc, mayContainOperandData, rightSides[i]);
          else
            mayContainOperandData = rightSides[i];
        }
      }

      if (updatedShardedDims.size() == 0) {
        rewriter.create<sdy::ReturnOp>(loc, innerUpdateVal);
      } else {
        if (updatedShardedDims.size() == 1 && false) {
          // TODO performance optimization, specialize for one dim update, can
          // incorporate from above.
        } else {
          // if (fully in update) {
          assert(mayContainOperandData);
          Type localTypes[] = {localResultType};
          auto if0 = rewriter.create<stablehlo::IfOp>(loc, localTypes,
                                                      mayContainOperandData);
          rewriter.create<sdy::ReturnOp>(loc, if0->getResults());

          {
            rewriter.createBlock(&if0.getTrueBranch(),
                                 if0.getTrueBranch().begin());
            Value multiIdx = nullptr;
            for (int i = 0; i < updatedShardedDims.size(); i++) {
              auto TT =
                  RankedTensorType::get(localResultType.getShape(),
                                        rewriter.getIntegerType(32, false));
              auto TTBool = RankedTensorType::get(localResultType.getShape(),
                                                  rewriter.getI1Type());
              auto idx = updatedShardedDims[i];

              auto iota = rewriter.create<stablehlo::IotaOp>(loc, TT, idx);
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
                } else {
                  // Otherwise we could've entered this if statement for other
                  // reasons, whether to use is simply the partition check, now
                  // broadcasted
                  lhs = rewriter.create<stablehlo::BroadcastInDimOp>(
                      loc, TTBool, leftSides[i], ArrayRef<int64_t>());
                }
              } else {
                // Non-evenly divisible pad, for example pad 18, inner 54, right
                // 18, local shape is 10 partition 0, and 1 would need the fused
                // update (aka if idx <= pad / lowerShape) The single node test
                // needs to consider both if we're in the right node, and within
                // the transition node, if we're at the point of transition
                // Within the point of transition, we need to consider the
                // offset mod the local result type.
                Value leftSideTransition =
                    rewriter.create<stablehlo::CompareOp>(
                        loc, iota,
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
                  Value atTransition = rewriter.create<stablehlo::CompareOp>(
                      loc, multiDimIdxs[idx],
                      getOrCreateConstant(lowPads[idx] /
                                          localResultType.getShape()[idx]),
                      stablehlo::ComparisonDirection::EQ);
                  atTransition = rewriter.create<stablehlo::BroadcastInDimOp>(
                      loc, TTBool, atTransition, ArrayRef<int64_t>());
                  assert(leftSideTransition.getType() == TTBool);
                  leftSideTransition = rewriter.create<stablehlo::AndOp>(
                      loc, leftSideTransition, atTransition);
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
                  Value fullyOperandNode =
                      rewriter.create<stablehlo::CompareOp>(
                          loc, multiDimIdxs[idx],
                          getOrCreateConstant(lowPads[idx] /
                                              localResultType.getShape()[idx]),
                          stablehlo::ComparisonDirection::LT);
                  fullyOperandNode =
                      rewriter.create<stablehlo::BroadcastInDimOp>(
                          loc, TTBool, fullyOperandNode, ArrayRef<int64_t>());
                  assert(leftSideTransition.getType() == TTBool);
                  lhs = rewriter.create<stablehlo::OrOp>(
                      loc, leftSideTransition, fullyOperandNode);
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
                } else {
                  // Otherwise we could've entered this if statement for other
                  // reasons, whether to use is simply the partition check, now
                  // broadcasted
                  rhs = rewriter.create<stablehlo::BroadcastInDimOp>(
                      loc, TTBool, rightSides[i], ArrayRef<int64_t>());
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

                Value rightSideTransition =
                    rewriter.create<stablehlo::CompareOp>(
                        loc, iota,
                        getOrCreateConstant(
                            startIdx % localResultType.getShape()[idx], TT),
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
                  Value atTransition = rewriter.create<stablehlo::CompareOp>(
                      loc, multiDimIdxs[idx],
                      getOrCreateConstant(startIdx /
                                          localResultType.getShape()[idx]),
                      stablehlo::ComparisonDirection::EQ);
                  atTransition = rewriter.create<stablehlo::BroadcastInDimOp>(
                      loc, TTBool, atTransition, ArrayRef<int64_t>());
                  assert(rightSideTransition.getType() == TTBool);
                  rightSideTransition = rewriter.create<stablehlo::AndOp>(
                      loc, rightSideTransition, atTransition);
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
                  Value fullyOperandNode =
                      rewriter.create<stablehlo::CompareOp>(
                          loc, multiDimIdxs[idx],
                          getOrCreateConstant(startIdx /
                                              localResultType.getShape()[idx]),
                          stablehlo::ComparisonDirection::GT);
                  fullyOperandNode =
                      rewriter.create<stablehlo::BroadcastInDimOp>(
                          loc, TTBool, fullyOperandNode, ArrayRef<int64_t>());
                  assert(rightSideTransition.getType() == TTBool);
                  rhs = rewriter.create<stablehlo::OrOp>(
                      loc, rightSideTransition, fullyOperandNode);
                }
              }

              // We are in the operand if either lhs or rhs are in operand
              Value inOperand = lhs;
              if (rhs) {
                if (inOperand)
                  inOperand =
                      rewriter.create<stablehlo::OrOp>(loc, inOperand, rhs);
                else
                  inOperand = rhs;
              }
              assert(inOperand);

              // We are in the operand if either of the indices are in the
              // operand
              if (multiIdx) {
                multiIdx =
                    rewriter.create<stablehlo::OrOp>(loc, multiIdx, inOperand);
              } else {
                multiIdx = inOperand;
              }
            }

            auto newV = rewriter.create<stablehlo::SelectOp>(
                loc, multiIdx, innerOperand, innerUpdateVal);
            rewriter.create<stablehlo::ReturnOp>(loc, newV->getResults());
          }

          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.createBlock(&if0.getFalseBranch(),
                                 if0.getFalseBranch().begin());
            rewriter.create<stablehlo::ReturnOp>(loc, innerUpdateVal);
          }
        }
      }
    }

    if (!extraSlice) {
      rewriter.replaceOp(dus, manual);
    } else {
      rewriter.setInsertionPointAfter(manual);
      SmallVector<int64_t> starts(ndims, 0);
      SmallVector<int64_t> limits = llvm::to_vector(dus.getType().getShape());
      SmallVector<int64_t> interior(ndims, 1);
      auto sl = rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
          dus, manual->getResult(0), starts, limits, interior);
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

    SmallVector<int64_t> padLow(ndims, 0);
    SmallVector<int64_t> padHigh(ndims, 0);
    SmallVector<int64_t> padInner(ndims, 0);

    SmallVector<Value> addOperands(concat.getOperands().size());

    auto zero = rewriter.create<stablehlo::ConstantOp>(
        concat.getLoc(), rewriter.getZeroAttr(elemType));

    int64_t leftPadding = 0;
    for (auto [i, operand] : llvm::enumerate(concat.getOperands())) {
      auto operandSharding = mlir::sdy::getSharding(operand);
      if (!operandSharding || (operandSharding != concatSharding))
        return failure();

      auto operandConcatDimSize =
          cast<RankedTensorType>(operand.getType()).getShape()[concatDimension];

      padLow[concatDimension] = leftPadding;
      padHigh[concatDimension] =
          concatDimSize - leftPadding - operandConcatDimSize;

      auto paddedOperand = rewriter.create<stablehlo::PadOp>(
          concat.getLoc(), operand, zero, padLow, padHigh, padInner);
      sdy::setSharding(paddedOperand, concatSharding);
      addOperands[i] = paddedOperand;
      leftPadding += operandConcatDimSize;
    }

    if (addOperands.size() == 1) {
      rewriter.replaceOp(concat, addOperands[0]);
      return success();
    }

    stablehlo::AddOp addOp = rewriter.create<stablehlo::AddOp>(
        concat.getLoc(), addOperands[0], addOperands[1]);
    sdy::setSharding(addOp, concatSharding);
    for (int i = 2; i < addOperands.size(); i++) {
      addOp = rewriter.create<stablehlo::AddOp>(concat.getLoc(), addOp,
                                                addOperands[i]);
      sdy::setSharding(addOp, concatSharding);
    }
    rewriter.replaceOp(concat, addOp);
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

    if (concat_two_operands_comm > 0)
      patterns.add<ConcatTwoOperandsCommOptimize>(
          channel_id, context, PatternBenefit(concat_two_operands_comm));

    if (rotate_comm > 0)
      patterns.add<RotateCommOptimize>(channel_id, context,
                                       PatternBenefit(rotate_comm));

    if (rotate_to_pad_comm > 0)
      patterns.add<RotateToPadCommOptimize>(context,
                                            PatternBenefit(rotate_to_pad_comm));

    if (wrap_comm > 0)
      patterns.add<WrapCommOptimize>(channel_id, context,
                                     PatternBenefit(wrap_comm));

    if (wrap_to_pad_comm > 0)
      patterns.add<WrapToPadCommOptimize>(context,
                                          PatternBenefit(wrap_to_pad_comm));

    if (extend_comm > 0)
      patterns.add<ExtendCommOptimize>(channel_id, context,
                                       PatternBenefit(extend_comm));

    if (extend_to_pad_comm > 0)
      patterns.add<ExtendToPadCommOptimize>(context,
                                            PatternBenefit(extend_to_pad_comm));

    if (dus_to_pad_comm > 0)
      patterns.add<DUSToPadComm>(channel_id, context,
                                 PatternBenefit(dus_to_pad_comm));

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
