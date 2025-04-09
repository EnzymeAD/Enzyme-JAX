#include "mhlo/IR/hlo_ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
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

template <typename T> Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));
  if (isa<FloatType>(elemType))
    return FloatAttr::get(elemType, val);
  else
    return IntegerAttr::get(elemType, val);
}

using namespace mlir::sdy;

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

    // TODO: relax the slice op condition
    auto midOp = allOperands[1]; // .getDefiningOp<stablehlo::SliceOp>();
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

    auto meshAttr =
        mlir::sdy::getCommonMesh(op_shardings, op_shardings, concat);

    for (auto meshAxis : meshAttr.getAxes()) {
      manual_axes.push_back(rewriter.getStringAttr(meshAxis.getName()));
    }

    auto dim_shardings = op_shardings[0].getDimShardings();
    SmallVector<int32_t> ndevices;
    for (auto dim_sharding : dim_shardings) {
      int32_t total_size = 1;
      for (auto axis : dim_sharding.getAxes()) {
        total_size *= meshAttr.getAxisSize(axis.getName());
      }
      ndevices.push_back(total_size);
    }

    // TODO: easy to lift the != 1 condition, but we don't need it rn, and any
    //       operation along that dim is super cheap
    if (ndevices.size() != 3 || ndevices[0] != 1) {
      return failure();
    }

    int64_t nx, ny;
    if (concat.getDimension() == 1) {
      ny = ndevices[1];
      nx = ndevices[2];
    } else {
      ny = ndevices[2];
      nx = ndevices[1];
    }

    if (ny % 2 != 0) {
      // TODO: Lift this condition. For the center most blocks, we will have 2
      //       comms from the left and right
      return failure();
    }

    SmallVector<int64_t> localShape =
        llvm::to_vector(cast<RankedTensorType>(midOp.getType()).getShape());
    assert(ndevices.size() == localShape.size());
    assert(localShape.size() == ndevices.size());
    for (int i = 0; i < localShape.size(); i++) {
      localShape[i] /= ndevices[i];
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

    if (T % ny != 0) {
      int extra = ((T / ny) + 1) * ny - T;

      if (extra % 2 == 0) {
        left_padding += extra / 2;
        right_padding += extra / 2;
        N += extra / 2;
        T += extra;
      } else {
        // TODO: handle this if we ever need it. basically we find the nearest
        //       multiple of 2 & ny that is larger than T
        return failure();
      }
    }

    SmallVector<int64_t> localRetShape =
        llvm::to_vector(concat.getType().getShape());
    SmallVector<int64_t> manualOpRetShape =
        llvm::to_vector(concat.getType().getShape());
    for (int i = 0; i < localRetShape.size(); i++) {
      if (i == concat.getDimension()) {
        localRetShape[i] = T / ny;
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

    auto devices_along_dim = ndevices[concat.getDimension()];

    auto zero = rewriter.create<stablehlo::ConstantOp>(
        concat.getLoc(), rewriter.getZeroAttr(partition_id.getType()));

    Value leftSide = rewriter.create<stablehlo::RemOp>(
        concat.getLoc(), partition_id,
        rewriter.create<stablehlo::ConstantOp>(
            concat.getLoc(), partition_id.getType(),
            makeAttr(partition_id.getType(), ny).cast<ElementsAttr>()));
    Value isLeftSide = rewriter.create<stablehlo::CompareOp>(
        concat.getLoc(), leftSide, zero, stablehlo::ComparisonDirection::EQ);

    // partition_id == (ny -1) (2ny - 1) ...
    Value rightSide = rewriter.create<stablehlo::AddOp>(
        concat.getLoc(), partition_id,
        rewriter.create<stablehlo::ConstantOp>(
            concat.getLoc(),
            makeAttr(partition_id.getType(), 1).cast<ElementsAttr>()));
    rightSide = rewriter.create<stablehlo::RemOp>(
        concat.getLoc(), rightSide,
        rewriter.create<stablehlo::ConstantOp>(
            concat.getLoc(), partition_id.getType(),
            makeAttr(partition_id.getType(), ny).cast<ElementsAttr>()));
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
        makeAttr(partition_id.getType(), ny / 2).cast<ElementsAttr>());

    Value isLeftBlock = rewriter.create<stablehlo::CompareOp>(
        concat.getLoc(), leftSide, ny_2, stablehlo::ComparisonDirection::LT);

    int64_t commSize = N - 2 * N / ny;
    auto if3 =
        rewriter.create<stablehlo::IfOp>(concat.getLoc(), ifTypes, isLeftBlock);

    SmallVector<int64_t> source_target_ids(2 * (ny - 2) * nx);
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < (ny - 2); j++) {
        int idx = i * (ny - 2) + j;
        int partition_idx = i * ny + j;
        if (j + 1 < (ny / 2)) {
          source_target_ids[2 * idx] = partition_idx;
          source_target_ids[2 * idx + 1] = partition_idx + 1;
        } else {
          source_target_ids[2 * idx] = partition_idx + 2;
          source_target_ids[2 * idx + 1] = partition_idx + 1;
        }
      }
    }
    auto source_target_pairs_ty = RankedTensorType::get(
        {(int64_t)((ny - 2) * nx), (int64_t)2}, rewriter.getI64Type());

    Value alpha = rewriter.create<stablehlo::ConstantOp>(
        concat.getLoc(), partition_id.getType(),
        makeAttr(partition_id.getType(), 2 * N / ny).cast<ElementsAttr>());
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
          concat.getLoc(), leftSlice,
          DenseIntElementsAttr::get(source_target_pairs_ty, source_target_ids),
          stablehlo::ChannelHandleAttr::get(concat.getContext(), /*handle*/ 0,
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
          concat.getLoc(), rightSlice,
          DenseIntElementsAttr::get(source_target_pairs_ty, source_target_ids),
          stablehlo::ChannelHandleAttr::get(concat.getContext(), /*handle*/ 0,
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

    SmallVector<int64_t> source_target_ids2(2 * nx);
    for (int i = 0; i < 2 * nx; i += 2) {
      int idx = i / 2;
      source_target_ids2[i] = idx * ny;
      source_target_ids2[i + 1] = (idx + 1) * ny - 1;
    }

    auto source_target_pairs_ty2 = RankedTensorType::get(
        {(int64_t)(nx), (int64_t)2}, rewriter.getI64Type());
    auto result_1 = rewriter.create<stablehlo::CollectivePermuteOp>(
        concat.getLoc(), end_slice,
        DenseIntElementsAttr::get(source_target_pairs_ty2, source_target_ids2),
        stablehlo::ChannelHandleAttr::get(concat.getContext(), /*handle*/ 0,
                                          /*type*/ 0));

    SmallVector<int64_t> inner_starts4(concat.getType().getShape().size(), 0);
    SmallVector<int64_t> inner_limits4 = llvm::to_vector(
        cast<RankedTensorType>(superSliceInnerArg.getType()).getShape());
    inner_starts4[concat.getDimension()] =
        inner_limits4[concat.getDimension()] - N;
    auto start_slice = rewriter.create<stablehlo::SliceOp>(
        concat.getLoc(), superSliceInnerArg, inner_starts4, inner_limits4,
        inner_strides);

    auto source_target_pairs_ty3 = RankedTensorType::get(
        {(int64_t)(nx), (int64_t)2}, rewriter.getI64Type());
    SmallVector<int64_t> source_target_ids3(2 * nx);
    for (int i = 0; i < 2 * nx; i += 2) {
      int idx = i / 2;
      source_target_ids3[i] = (idx + 1) * ny - 1;
      source_target_ids3[i + 1] = idx * ny;
    }

    auto result_2 = rewriter.create<stablehlo::CollectivePermuteOp>(
        concat.getLoc(), start_slice,
        DenseIntElementsAttr::get(source_target_pairs_ty3, source_target_ids3),
        stablehlo::ChannelHandleAttr::get(concat.getContext(), /*handle*/ 0,
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
      inner_limits5[concat.getDimension()] = (T / ny) - N;

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
      inner_starts6[concat.getDimension()] = N - (2 * N / ny);

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

struct OptimizeCommunicationPass
    : public enzyme::impl::OptimizeCommunicationBase<
          OptimizeCommunicationPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<PeriodicConcatSimplify>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
