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

// From https://github.com/openxla/shardy/blob/0d88b5d25971bd66272195ceeb2288cde72997d0/shardy/dialect/sdy/ir/verifiers.cc#L765C1-L782C2
// Returns the accumulated axes size of a tensor sharding with respect to manual
// axes.
// If an axis in dimShardingAxes belongs to manualAxes, it's an axis
// the user is doing a manual computation on, thus the ManualComputationOp's
// body will have tensors smaller wrt this manual axis.
template<typename T>
int64_t accumulatedManualAxesSize(
    ArrayRef<AxisRefAttr> dimShardingAxes,
    const T& manualAxes, MeshAttr mesh) {
  int64_t axesFactor = 1;
  for (AxisRefAttr axisRef : dimShardingAxes) {
    if (llvm::is_contained(manualAxes, axisRef.getName())) {
      axesFactor *= axisRef.getSize(mesh);
    }
  }

  return axesFactor;
}

// From https://github.com/openxla/shardy/blob/0d88b5d25971bd66272195ceeb2288cde72997d0/shardy/dialect/sdy/ir/verifiers.cc#L869
template<typename T>
RankedTensorType getLocalType(RankedTensorType globalType, TensorShardingAttr sharding, const T& manualAxesSet, Operation *op) {
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

  auto dimShardings = shardingAttr.getDimShardings();
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
    int numDevicesAlongDimension, int ndims, int commSize,
    SmallVector<int64_t> localRetShape, Value leftSide) {
  auto sourceTargetPairsVec =
      generateShiftPairs(opSharding, concatDim, op, /*leftToRight*/true, /*onlyEdges*/false, /*splitHalfComm*/true);
  auto sourceTargetPairs = DenseIntElementsAttr::get(
      RankedTensorType::get(
          {(int64_t)(sourceTargetPairsVec.size() / 2), (int64_t)2},
          rewriter.getI64Type()),
      sourceTargetPairsVec);

  SmallVector<int64_t> innerStrides(ndims, 1);

  auto partitionIdType = partitionId.getType();
  auto alpha = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), partitionIdType,
      makeAttr(partitionIdType, 2 * paddedBoundarySize / numDevicesAlongDimension)
          .cast<ElementsAttr>());
  auto onePId = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), partitionIdType,
      makeAttr(partitionIdType, 1).cast<ElementsAttr>());

  auto numDevicesAlongDimension_2 = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), partitionIdType,
      makeAttr(partitionIdType, numDevicesAlongDimension / 2)
          .cast<ElementsAttr>());
  auto isLeftBlock = rewriter.create<stablehlo::CompareOp>(
      op->getLoc(), leftSide, numDevicesAlongDimension_2,
      stablehlo::ComparisonDirection::LT);

  Type ifTypes[] = {RankedTensorType::get(
      localRetShape,
      superSliceInnerArg.getType().cast<RankedTensorType>().getElementType())};
  auto ifCond =
      rewriter.create<stablehlo::IfOp>(op->getLoc(), ifTypes, isLeftBlock);
  rewriter.create<stablehlo::ReturnOp>(op->getLoc(), ifCond->getResults());

  // we start with               [ superSliceInnerArg ].           : Length SS [superslice]
  // and       we want. [ left  ][                    ][ right ].  : Length RS [retshape == superslice + 2 * paddedBounarySize]

  // We assume left and right both fit in < 1 shard for simplicity
  // The result indexes into [i*RS : (i+1):RS], right input indexes [i*SS : (i+1):SS]
  // Consider the case where we are in the middle region, we know i != 0 and i != N-1, where N is total number of shards in this dimension

  // Let us now ask the question of which data we need for our result (assuming not left/right), and size of pad is nonzero
  // Since we want to store result[i*RS : (i+1):RS], we want to load from input[(i-1)*RS : i:RS]
  // Let's solve for the boundaries. What is the left node L containing the start of our data, and at what offset
  //.   L*SS + offset == (i-1) * RS, where offset < SS
  //   -> L = [ (i-1) * RS ] / SS;  offset = [ (i-1) * RS ] % SS 

  // Now consider the fact that we have a symmetric node count. Combined with the fact we have a left/right within a node, means we only need to look to ourselves
  // or our leftmost neighbor (if on left), or rightmost neighbor (if on the right).

  // The leftmost node (not handled in this function), will get data from <left source> and the first (RS / numDevicesAlongDimension - left) elements from itself
  // The second node will get the last [SS / numDevicesAlongDimension - (RS / numDevicesAlongDimension - left)] elems from the leftmost node,
  //.   and keep the first RS / numDevicesAlongDimension - [SS / numDevicesAlongDimension - (RS / numDevicesAlongDimension - left)] elems from itself
  //.                       = left - 


  // Case I: for the left part of the comm
  {
    rewriter.createBlock(&ifCond.getTrueBranch(),
                         ifCond.getTrueBranch().begin());

    SmallVector<int64_t> innerStarts(ndims, 0);
    SmallVector<int64_t> innerLimits = llvm::to_vector(
        cast<RankedTensorType>(midOpInnerArg.getType()).getShape());
    innerStarts[concatDim] = innerLimits[concatDim] - commSize;

    auto leftSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), midOpInnerArg, innerStarts, innerLimits, innerStrides);

    auto cperm = rewriter.create<stablehlo::CollectivePermuteOp>(
        op->getLoc(), leftSlice, sourceTargetPairs,
        stablehlo::ChannelHandleAttr::get(op->getContext(), /*handle*/ 1,
                                          /*type*/ 0));

    Value concatArgs[] = {leftSlice, midOpInnerArg};
    auto innerConcat = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);

    SmallVector<Value> dynamicSliceStartSlices(ndims, zero);
    dynamicSliceStartSlices[concatDim] = rewriter.create<stablehlo::MulOp>(
            op->getLoc(),
            rewriter.create<stablehlo::SubtractOp>(op->getLoc(), partitionId,
                                                   onePId),
            alpha);

    auto slicedPart = rewriter.create<stablehlo::DynamicSliceOp>(
        op->getLoc(), innerConcat, dynamicSliceStartSlices, localRetShape);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         slicedPart->getResults());
  }

  // Case II: for the right part of the comm
  {
    rewriter.createBlock(&ifCond.getFalseBranch(),
                         ifCond.getFalseBranch().begin());

    SmallVector<int64_t> innerStarts(ndims, 0);
    SmallVector<int64_t> innerLimits = llvm::to_vector(
        cast<RankedTensorType>(midOpInnerArg.getType()).getShape());
    innerLimits[concatDim] = commSize;

    auto rightSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), midOpInnerArg, innerStarts, innerLimits, innerStrides);

    auto cperm = rewriter.create<stablehlo::CollectivePermuteOp>(
        op->getLoc(), rightSlice, sourceTargetPairs,
        stablehlo::ChannelHandleAttr::get(op->getContext(), /*handle*/ 1,
                                          /*type*/ 0));

    Value concatArgs[] = {midOpInnerArg, rightSlice};
    auto innerConcat = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);

    SmallVector<Value> dynamicSliceStartSlices(ndims, zero);
        auto diffIdx = rewriter.create<stablehlo::MulOp>(
            op->getLoc(),
            rewriter.create<stablehlo::AddOp>(op->getLoc(), partitionId,
                                              onePId),
            alpha);
     dynamicSliceStartSlices[concatDim] = rewriter.create<stablehlo::SubtractOp>(
                op->getLoc(),
                rewriter.create<stablehlo::ConstantOp>(
                    op->getLoc(), partitionIdType,
                    makeAttr(partitionIdType, paddedBoundarySize).cast<ElementsAttr>()),
                diffIdx);

    auto slicedPart = rewriter.create<stablehlo::DynamicSliceOp>(
        op->getLoc(), innerConcat, dynamicSliceStartSlices, localRetShape);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         slicedPart->getResults());
  }

  return;
}

void wrapCommPatternForEdges(PatternRewriter &rewriter, Operation *op,
                             stablehlo::PartitionIdOp partitionId,
                             stablehlo::ConstantOp zero,
                             Value superSliceInnerArg, Value midOpInnerArg,
                             TensorShardingAttr opSharding, int concatDim,
                             int N, int numDevicesAlongDimension, int ndims,
                             int T, SmallVector<int64_t> localRetShape,
                             Value isLeftSide) {
  auto elemType =
      superSliceInnerArg.getType().cast<RankedTensorType>().getElementType();

  SmallVector<int64_t> innerStrides(ndims, 1);

  SmallVector<int64_t> innerStarts1(ndims, 0);
  SmallVector<int64_t> innerLimits1 = llvm::to_vector(
      cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
  innerLimits1[concatDim] = N;
  auto endSlice = rewriter.create<stablehlo::SliceOp>(
      op->getLoc(), superSliceInnerArg, innerStarts1, innerLimits1,
      innerStrides);

  auto sourceTargetIdxsLeftEdges =
      generateShiftPairs(opSharding, concatDim, op, true, true);

  auto commResult1 = rewriter.create<stablehlo::CollectivePermuteOp>(
      op->getLoc(), endSlice,
      DenseIntElementsAttr::get(
          RankedTensorType::get(
              {(int64_t)(sourceTargetIdxsLeftEdges.size() / 2), (int64_t)2},
              rewriter.getI64Type()),
          sourceTargetIdxsLeftEdges),
      stablehlo::ChannelHandleAttr::get(op->getContext(), /*handle*/ 1,
                                        /*type*/ 0));

  SmallVector<int64_t> innerStarts2(ndims, 0);
  SmallVector<int64_t> innerLimits2 = llvm::to_vector(
      cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
  innerStarts2[concatDim] = innerLimits2[concatDim] - N;
  auto startSlice = rewriter.create<stablehlo::SliceOp>(
      op->getLoc(), superSliceInnerArg, innerStarts2, innerLimits2,
      innerStrides);

  auto sourceTargetIdxsRightEdges =
      generateShiftPairs(opSharding, concatDim, op, false, true);

  auto commResult2 = rewriter.create<stablehlo::CollectivePermuteOp>(
      op->getLoc(), startSlice,
      DenseIntElementsAttr::get(
          RankedTensorType::get(
              {(int64_t)(sourceTargetIdxsRightEdges.size() / 2), (int64_t)2},
              rewriter.getI64Type()),
          sourceTargetIdxsRightEdges),
      stablehlo::ChannelHandleAttr::get(op->getContext(), /*handle*/ 1,
                                        /*type*/ 0));

  Type ifTypes[] = {RankedTensorType::get(localRetShape, elemType)};
  auto ifCondInner =
      rewriter.create<stablehlo::IfOp>(op->getLoc(), ifTypes, isLeftSide);
  rewriter.create<stablehlo::ReturnOp>(op->getLoc(), ifCondInner->getResults());

  {
    rewriter.createBlock(&ifCondInner.getTrueBranch(),
                         ifCondInner.getTrueBranch().begin());

    SmallVector<int64_t> innerStarts3(ndims, 0);
    SmallVector<int64_t> innerLimits3 = llvm::to_vector(
        cast<RankedTensorType>(midOpInnerArg.getType()).getShape());
    innerLimits3[concatDim] = (T / numDevicesAlongDimension) - N;

    auto lhsRightSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), midOpInnerArg, innerStarts3, innerLimits3, innerStrides);

    Value concatArgs[] = {commResult1, lhsRightSlice};
    auto finalResult = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         finalResult->getResults());
  }

  {
    rewriter.createBlock(&ifCondInner.getFalseBranch(),
                         ifCondInner.getFalseBranch().begin());

    SmallVector<int64_t> innerStarts4(ndims, 0);
    SmallVector<int64_t> innerLimits4 = llvm::to_vector(
        cast<RankedTensorType>(midOpInnerArg.getType()).getShape());
    innerStarts4[concatDim] = N - (2 * N / numDevicesAlongDimension);

    auto rhsLeftSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), midOpInnerArg, innerStarts4, innerLimits4, innerStrides);

    Value concatArgs[] = {rhsLeftSlice, commResult2};
    auto finalResult = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         finalResult->getResults());
  }

  return;
}

void extendCommPatternForEdges(PatternRewriter &rewriter, Operation *op,
                               stablehlo::PartitionIdOp partitionId,
                               stablehlo::ConstantOp zero, Value innerArg,
                               TensorShardingAttr opSharding, int concatDim,
                               int N, int numDevicesAlongDimension, int ndims,
                               int T, SmallVector<int64_t> localRetShape,
                               Value isLeftSide) {
  auto elemType = innerArg.getType().cast<RankedTensorType>().getElementType();

  SmallVector<int64_t> innerStrides(ndims, 1);

  Type ifTypes[] = {RankedTensorType::get(localRetShape, elemType)};
  auto ifCondInner =
      rewriter.create<stablehlo::IfOp>(op->getLoc(), ifTypes, isLeftSide);
  rewriter.create<stablehlo::ReturnOp>(op->getLoc(), ifCondInner->getResults());

  {
    rewriter.createBlock(&ifCondInner.getTrueBranch(),
                         ifCondInner.getTrueBranch().begin());

    SmallVector<int64_t> innerStarts1(ndims, 0);
    SmallVector<int64_t> innerLimits1 =
        llvm::to_vector(cast<RankedTensorType>(innerArg.getType()).getShape());
    innerLimits1[concatDim] = N;
    auto startSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), innerArg, innerStarts1, innerLimits1, innerStrides);

    SmallVector<int64_t> innerStarts3(ndims, 0);
    SmallVector<int64_t> innerLimits3 =
        llvm::to_vector(cast<RankedTensorType>(innerArg.getType()).getShape());
    innerLimits3[concatDim] = (T / numDevicesAlongDimension) - N;

    auto lhsRightSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), innerArg, innerStarts3, innerLimits3, innerStrides);

    Value concatArgs[] = {startSlice, lhsRightSlice};
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

    SmallVector<int64_t> innerStarts4(ndims, 0);
    SmallVector<int64_t> innerLimits4 =
        llvm::to_vector(cast<RankedTensorType>(innerArg.getType()).getShape());
    innerStarts4[concatDim] = N - (2 * N / numDevicesAlongDimension);

    auto rhsLeftSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), innerArg, innerStarts4, innerLimits4, innerStrides);

    Value concatArgs[] = {rhsLeftSlice, endSlice};
    auto finalResult = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatArgs, concatDim);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(),
                                         finalResult->getResults());
  }

  return;
}

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
    int extra =
        ((paddedResultSize / numDevicesAlongDimension) + 1) * numDevicesAlongDimension - paddedResultSize;

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
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concat,
                                PatternRewriter &rewriter) const override {
    if (concat.getNumOperands() != 3) {
      return failure();
    }

    auto elemType = concat.getType().getElementType();
    auto ndims = concat.getType().getRank();
    auto concatShape = concat.getType().getShape();
    auto concatDim = concat.getDimension();

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
    if (leftSliceSharding != rightSliceSharding) {
      return failure();
    }

    auto midOpSharding = mlir::sdy::getSharding(midOp);
    if (leftSliceSharding != midOpSharding) {
      return failure();
    }

    auto concatSharding = mlir::sdy::getSharding(concat);
    if (!concatSharding)
      return failure();

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

    auto ndevices = getShardingDevices(concatSharding, concatDim, concat);
    int64_t numDevicesAlongDimension = ndevices[concatDim];

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

    Type ifTypes[] = {RankedTensorType::get(localRetShape, elemType)};
    auto if1 = rewriter.create<stablehlo::IfOp>(
        concat.getLoc(), ifTypes,
        rewriter.create<stablehlo::AndOp>(concat.getLoc(), isNotLeftSide,
                                          isNotRightSide));

    SmallVector<int64_t> innerStrides(ndims, 1);

    // if ..... !leftSide  && !rightSide
    {
      rewriter.createBlock(&if1.getTrueBranch(), if1.getTrueBranch().begin());

      generateCommPatternForNonEdges(
          rewriter, concat, partitionId, zero, superSliceInnerArg,
          midOpInnerArg, concatSharding, concatDim, N, numDevicesAlongDimension,
          ndims, N - 2 * N / numDevicesAlongDimension, localRetShape, leftSide);
    }

    // else
    {
      rewriter.createBlock(&if1.getFalseBranch(), if1.getFalseBranch().begin());

      wrapCommPatternForEdges(rewriter, concat, partitionId, zero,
                              superSliceInnerArg, midOpInnerArg, concatSharding,
                              concatDim, N, numDevicesAlongDimension, ndims, T,
                              localRetShape, isLeftSide);
    }

    rewriter.setInsertionPointAfter(if1);
    rewriter.create<sdy::ReturnOp>(concat.getLoc(), if1->getResults());

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
  using OpRewritePattern::OpRewritePattern;

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

    if (numDevicesAlongDimension % 2 != 0) {
      return failure();
    }

    auto [leftPadding, rightPadding, paddedBoundarySize, paddedResultSize] =
        getWrapExtendConfiguration(wrapOperandShape[wrapDimension], lhsValue,
                                   rhsValue, numDevicesAlongDimension);

    if (leftPadding == -1 || rightPadding == -1 || paddedBoundarySize == -1 || paddedResultSize == -1) {
      return failure();
    }

    if (paddedBoundarySize >= numDevicesAlongDimension) {
      return failure();
    }

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
          paddedBoundarySize - 2 * paddedBoundarySize / numDevicesAlongDimension, localRetShape, leftSide);
    }

    {
      rewriter.createBlock(&ifCond.getFalseBranch(),
                           ifCond.getFalseBranch().begin());

      wrapCommPatternForEdges(rewriter, wrap, partitionId, zero, innerArg,
                              innerArg, wrapSharding, wrapDimension, paddedBoundarySize,
                              numDevicesAlongDimension, ndims, paddedResultSize, localRetShape,
                              isLeftSide);
    }

    rewriter.setInsertionPointAfter(ifCond);
    rewriter.create<sdy::ReturnOp>(wrap.getLoc(), ifCond->getResults());

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

// TODO: check mesh attr and ensure only applied to iota tile
struct ExtendCommOptimize : public OpRewritePattern<enzymexla::ExtendOp> {
  using OpRewritePattern::OpRewritePattern;

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

    if (numDevicesAlongDimension % 2 != 0) {
      return failure();
    }

    auto [leftPadding, rightPadding, paddedBoundarySize, paddedResultSize] = getWrapExtendConfiguration(
        extendOperandShape[extendDimension], rhsValue, lhsValue,
        numDevicesAlongDimension);

    if (leftPadding == -1 || rightPadding == -1 || paddedBoundarySize == -1 || paddedResultSize == -1) {
      return failure();
    }

    if (paddedBoundarySize >= numDevicesAlongDimension) {
      return failure();
    }

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

    Type ifTypes[] = {RankedTensorType::get(localRetShape, elemType)};
    auto ifCond = rewriter.create<stablehlo::IfOp>(
        extend.getLoc(), ifTypes,
        rewriter.create<stablehlo::AndOp>(extend.getLoc(), isNotLeftSide,
                                          isNotRightSide));

    {
      rewriter.createBlock(&ifCond.getTrueBranch(),
                           ifCond.getTrueBranch().begin());

      generateCommPatternForNonEdges(
          rewriter, extend, partitionId, zero, innerArg, innerArg,
          extendSharding, extendDimension, paddedBoundarySize, numDevicesAlongDimension, ndims,
          paddedBoundarySize - 2 * paddedBoundarySize / numDevicesAlongDimension, localRetShape, leftSide);
    }

    {
      rewriter.createBlock(&ifCond.getFalseBranch(),
                           ifCond.getFalseBranch().begin());

      extendCommPatternForEdges(rewriter, extend, partitionId, zero, innerArg,
                                extendSharding, extendDimension, paddedBoundarySize,
                                numDevicesAlongDimension, ndims, paddedResultSize,
                                localRetShape, isLeftSide);
    }

    rewriter.setInsertionPointAfter(ifCond);
    rewriter.create<sdy::ReturnOp>(extend.getLoc(), ifCond->getResults());

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

// TODO: check mesh attr and ensure only applied to iota tile
struct RotateCommOptimize : public OpRewritePattern<enzymexla::RotateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::RotateOp rotate,
                                PatternRewriter &rewriter) const override {
    int32_t ndims = rotate.getType().getRank();
    auto elType = rotate.getType().getElementType();
    auto rotateShape = cast<RankedTensorType>(rotate.getType()).getShape();

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
                                     manualAxes, localShape,
                                     rotate.getDimension());

    int64_t numDevicesAlongDimension = getNumDevicesAlongDimension(
        rotateSharding, rotate.getDimension(), rotate);

    SmallVector<int64_t> outputShape = llvm::to_vector(rotateShape);

    int32_t amount = rotate.getAmount();
    bool leftToRight = amount <= outputShape[rotate.getDimension()] / 2;
    if (!leftToRight)
      amount = outputShape[rotate.getDimension()] - amount;
    assert(amount <= outputShape[rotate.getDimension()] / 2);

    if (amount >= outputShape[rotate.getDimension()] / numDevicesAlongDimension)
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
          stablehlo::ChannelHandleAttr::get(rotate.getContext(), /*handle*/ 1,
                                            /*type*/ 0));

      SmallVector<int64_t> innerStartsPresent(ndims, 0);
      SmallVector<int64_t> innerLimitsPresent = llvm::to_vector(
          cast<RankedTensorType>(innerArg.getType()).getShape());
      if (leftToRight) {
        innerStartsPresent[rotate.getDimension()] = amount;
      } else {
        innerLimitsPresent[rotate.getDimension()] =
            innerLimitsPresent[rotate.getDimension()] - amount;
      }
      auto remSlice = rewriter.create<stablehlo::SliceOp>(
          rotate.getLoc(), innerArg, innerStartsPresent, innerLimitsPresent,
          innerStrides);

      std::array<Value, 2> concatArgs;
      if (leftToRight)
        concatArgs = {remSlice, commResult};
      else
        concatArgs = {commResult, remSlice};

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


struct DUSToPadComm : public OpRewritePattern<stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DynamicUpdateSliceOp dus,
                                PatternRewriter &rewriter) const override {

    auto sharding = mlir::sdy::getSharding(dus);
    if (!sharding)
      return rewriter.notifyMatchFailure(dus, "No sharding found.");

    if (sharding != mlir::sdy::getSharding(dus.getOperand())) {
      return rewriter.notifyMatchFailure(dus, "DUS mismatch sharding"); 
    }

    if (sharding != mlir::sdy::getSharding(dus.getUpdate())) {
      return rewriter.notifyMatchFailure(dus, "DUS mismatch sharding"); 
    }

    SmallVector<int64_t> lowPads;
    SmallVector<int64_t> highPads;
    SmallVector<int64_t> interior(dus.getType().getShape().size(), 0);
    SmallVector<int64_t> updatedDims;
    SmallVector<StringAttr> manualAxes;


    for (int i=0; i<dus.getType().getShape().size(); i++) {
      llvm::APInt val;
      if (!matchPattern(dus.getStartIndices()[i], m_ConstantInt(&val)))
        return failure();

      if (val.isNegative())
        return failure();

      auto v2 = val.getZExtValue();
      if (v2 + cast<RankedTensorType>(dus.getOperand().getType()).getShape()[i] > dus.getType().getShape()[i]);
        return failure();

      lowPads.push_back(v2);
      highPads.push_back(dus.getType().getShape()[i] - v2 - cast<RankedTensorType>(dus.getOperand().getType()).getShape()[i]);


      auto meshAxes = sharding.getDimShardings()[i].getAxes();
      if (meshAxes.size() != 1) return failure();
      SmallVector<StringAttr> axis = {rewriter.getStringAttr(meshAxes[0].getName())};
      auto localType = getLocalType(dus.getType(), sharding, axis, dus);

      // Make sure that the operand can only update the outer edge (aka the update doesn't spill into two shards)
      if (localType.getShape()[i] < lowPads.back()) {
        return failure();
      }
      if (localType.getShape()[i] < highPads.back()) {
        return failure();
      }

      bool updated = false;
      if (val != 0 || val + cast<RankedTensorType>(dus.getOperand().getType()).getShape()[i] != dus.getType().getShape()[i]) {
        updated = true;
      }
      if (updated) {
        // Don't support an update on a non-sharded axis.
        if (localType.getShape()[i] == dus.getType().getShape()[i]) {
          return failure();
        }
        updatedDims.push_back(i);
        manualAxes.push_back(axis[0]);
      }
    }

    auto localType = getLocalType(dus.getType(), sharding, manualAxes, dus);

    if (updatedDims.size() == 0) {
      rewriter.replaceOp(dus, dus.getUpdate());
      return success();
    }

    if (updatedDims.size() > 3) {
      return failure();
    }

    auto PT = RankedTensorType::get({}, dus.getType().getElementType());
    auto pad2 = rewriter.create<stablehlo::PadOp>(dus.getLoc(), dus.getUpdate(), rewriter.create<stablehlo::ConstantOp>(dus.getLoc(), PT, rewriter.getZeroAttr(PT).cast<ElementsAttr>()), lowPads, highPads, interior);
    sdy::setShardings(pad2, sdy::getShardingPerValue(dus));

    TensorShardingAttr shardings[] = { sharding };

    mlir::Type inTyps[] = { pad2.getType(), dus.getType() };
    mlir::Location inLocs[] = {dus.getLoc(), dus.getLoc()};

    Value manualOps[] = {dus.getOperand(), pad2};
    Type manualTypes[] = {dus.getType()};

    TensorShardingPerValueAttr in_shardings =
        TensorShardingPerValueAttr::get(dus.getContext(), shardings);

    auto manual = rewriter.create<sdy::ManualComputationOp>(
        dus.getLoc(), manualTypes, manualOps, in_shardings, in_shardings,
        manualAxes);

    {
      auto blk = rewriter.createBlock(&manual.getBody(),
                                      manual.getBody().begin(), inTyps, inLocs);

      auto innerUpdate = blk->getArgument(0);
      auto innerOperand = blk->getArgument(1);

      auto partitionId = rewriter.create<stablehlo::PartitionIdOp>(dus.getLoc());

      SmallVector<Value> multiDimIdxs;
      if (updatedDims.size() != 1) {
        llvm::errs() << " TODO, figure out row vs col vs etc order of parition id, picked one for now\n";
      }
      {
        Value cur = partitionId;
        for (int i=0; i<updatedDims.size()-1; i++) {
          auto cst = rewriter.create<stablehlo::ConstantOp>(dus.getLoc(), partitionId.getType(), makeAttr(partitionId.getType(), localType.getShape()[updatedDims[i]] ).cast<ElementsAttr>() );
          multiDimIdxs.push_back(rewriter.create<stablehlo::RemOp>(dus.getLoc(), cur, cst));
          cur = rewriter.create<stablehlo::DivOp>(dus.getLoc(), cur, cst);
        }
        multiDimIdxs.push_back(cur);
      }

      SmallVector<Value> leftSides;
      SmallVector<Value> rightSides;
      for (auto &&[i, idx] : llvm::enumerate(updatedDims)) {
        leftSides.push_back(
          rewriter.create<stablehlo::CompareOp>(dus.getLoc(), multiDimIdxs[i],

            rewriter.create<stablehlo::ConstantOp>(dus.getLoc(), partitionId.getType(), makeAttr(partitionId.getType(), lowPads[i]).cast<ElementsAttr>()
            ),
            stablehlo::ComparisonDirection::EQ
        ) );
      }
      for (auto &&[i, idx] : llvm::enumerate(updatedDims)) {
        rightSides.push_back(
          rewriter.create<stablehlo::CompareOp>(dus.getLoc(),
            multiDimIdxs[i],
              rewriter.create<stablehlo::ConstantOp>(dus.getLoc(), partitionId.getType(), makeAttr(partitionId.getType(), dus.getType().getShape()[i]/ localType.getShape()[idx] - 1).cast<ElementsAttr>()

              ),             stablehlo::ComparisonDirection::EQ

              ) );
      }

      auto i1VTy = leftSides[0].getType();
      auto falseV = rewriter.create<stablehlo::ConstantOp>(dus.getLoc(), i1VTy, makeAttr(i1VTy, false).cast<ElementsAttr>());

      Value onAnyBoundary = falseV;

      for (int i=0; i<updatedDims.size(); i++) {
        onAnyBoundary = rewriter.create<stablehlo::OrOp>(dus.getLoc(), onAnyBoundary, leftSides[i]);
        onAnyBoundary = rewriter.create<stablehlo::OrOp>(dus.getLoc(), onAnyBoundary, rightSides[i]);
      }


      if (updatedDims.size() == 1 && false) {
        // TODO performance optimization, specialize for one dim update, can incorporate from above.
      } else {
        // if (fully in update) {
        auto if0 = rewriter.create<stablehlo::IfOp>(dus.getLoc(), manualTypes, onAnyBoundary);
        rewriter.create<stablehlo::ReturnOp>(dus.getLoc(), if0->getResults());

        {
          rewriter.createBlock(&if0.getTrueBranch(), if0.getTrueBranch().begin());
          Value idx = nullptr;
          SmallVector<int64_t> toBC;
          SmallVector<int64_t> dimSizes;
          auto R1 = RankedTensorType::get(localType.getShape(), rewriter.getI1Type());
          for (int i=0; i<updatedDims.size(); i++) {
            auto TT = RankedTensorType::get(localType.getShape(), rewriter.getI32Type());
            auto iota = rewriter.create<stablehlo::IotaOp>(dus.getLoc(), R1, updatedDims[i]);
            auto lhs = rewriter.create<stablehlo::CompareOp>(dus.getLoc(), iota, 
                rewriter.create<stablehlo::ConstantOp>(dus.getLoc(), TT, makeAttr(TT, lowPads[updatedDims[i]]).cast<ElementsAttr>()),
                stablehlo::ComparisonDirection::LT
                );
            auto rhs = rewriter.create<stablehlo::CompareOp>(dus.getLoc(), iota, 
                rewriter.create<stablehlo::ConstantOp>(dus.getLoc(), TT, makeAttr(TT, localType.getShape()[updatedDims[i]] - highPads[updatedDims[i]]).cast<ElementsAttr>()),
                stablehlo::ComparisonDirection::GE
                );
            auto inUpdate = rewriter.create<stablehlo::SelectOp>(
              dus.getLoc(),
              leftSides[i],
              lhs,
              rhs
            );
            toBC.push_back(i);
            dimSizes.push_back(localType.getShape()[updatedDims[i]]);
            if (idx) {
              idx = rewriter.create<stablehlo::OrOp>(dus.getLoc(), idx, inUpdate);
            } else {
              idx = inUpdate;
            }
          }

          auto newV = rewriter.create<stablehlo::SelectOp>(dus.getLoc(), idx, innerOperand, innerUpdate);
          rewriter.create<stablehlo::ReturnOp>(dus.getLoc(), newV->getResults());
        }

        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.createBlock(&if0.getFalseBranch(), if0.getFalseBranch().begin());
          rewriter.create<stablehlo::ReturnOp>(dus.getLoc(), innerUpdate);
        }
      }
    }

    rewriter.replaceOp(dus, manual);
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

    if (permute_like_raise > 0)
    patterns.add<PeriodicConcatSimplify, RotateCommOptimize, WrapCommOptimize,
                 ExtendCommOptimize>(context, PatternBenefit(permute_like_raise));

    if (pad_like_raise > 0)
    patterns.add<DUSToPadComm>(context, PatternBenefit(pad_like_raise));

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
