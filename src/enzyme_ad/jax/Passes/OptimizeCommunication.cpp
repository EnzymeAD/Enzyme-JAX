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

struct SliceConcatSimplify : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concat,
                                PatternRewriter &rewriter) const override {
    if (concat.getNumOperands() != 3) {
      return failure();
    }

    SmallVector<stablehlo::SliceOp> ops;
    for (Value operand : concat.getOperands()) {
      auto sliceOp = operand.getDefiningOp<stablehlo::SliceOp>();
      if (!sliceOp)
        return failure();

      if (!sliceOp->hasOneUse())
        return failure();

      ops.push_back(sliceOp);
    }

    for (auto sl : ops) {
      for (int i = 0; i < concat.getType().getShape().size(); i++) {
        if (sl.getStrides()[i] != 1) {
          return failure();
        }
        if (sl.getOperand() != ops[0].getOperand()) {
          return failure();
        }
        if (i == concat.getDimension()) {
          continue;
        }
        if (sl.getStartIndices()[i] != ops[0].getStartIndices()[i]) {
          return failure();
        }
        if (sl.getLimitIndices()[i] != ops[0].getLimitIndices()[i]) {
          return failure();
        }
      }
    }

    /*
    %26 = stablehlo.slice %arg11 [8:520, 1:1023, 2034:2040] {mhlo.sharding =
    "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) ->
    tensor<512x1022x6xf64> loc(#loc1191) %27 = stablehlo.slice %arg11 [8:520,
    1:1023, 8:2040] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} :
    (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64> loc(#loc1191) %23 =
    stablehlo.slice %arg11 [8:520, 1:1023, 8:16] {mhlo.sharding =
    "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) ->
    tensor<512x1022x8xf64> loc(#loc1190) %36 = stablehlo.concatenate %26, %27,
    %23, dim = 2 {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} :
    (tensor<512x1022x6xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x8xf64>)
    -> tensor<512x1022x2046xf64> loc(#loc1262
    */

    // LHS = 6
    // RHS = 8 in above
    int lhsSize = ops[0].getType().getShape()[concat.getDimension()];
    int rhsSize = ops[2].getType().getShape()[concat.getDimension()];
    if (ops[2].getStartIndices()[concat.getDimension()] !=
        ops[1].getStartIndices()[concat.getDimension()])
      return failure();

    if (ops[0].getLimitIndices()[concat.getDimension()] !=
        ops[1].getLimitIndices()[concat.getDimension()])
      return failure();

    TensorShardingAttr op_shardings[] = {mlir::sdy::getSharding(ops[1])};
    TensorShardingPerValueAttr in_shardings =
        TensorShardingPerValueAttr::get(concat.getContext(), op_shardings);
    TensorShardingPerValueAttr out_shardings = in_shardings;
    SmallVector<StringAttr> manual_axes;

    Value manual_ops[] = {ops[1]};
    Type manual_types[] = {concat.getType()};

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

    SmallVector<int64_t> localShape =
        llvm::to_vector(ops[1].getType().getShape());
    assert(ndevices.size() == localShape.size());
    assert(localShape.size() == ndevices.size());
    for (int i = 0; i < localShape.size(); i++) {
      localShape[i] /= ndevices[i];
    }

    SmallVector<int64_t> localRetShape =
        llvm::to_vector(concat.getType().getShape());
    for (int i = 0; i < localRetShape.size(); i++) {
      localRetShape[i] /= ndevices[i];
    }

    mlir::Type in_tys[1]{
        RankedTensorType::get(localShape, ops[1].getType().getElementType())};
    mlir::Location in_locs[] = {ops[1].getLoc()};

    auto manual = rewriter.create<sdy::ManualComputationOp>(
        concat.getLoc(), manual_types, manual_ops, in_shardings, out_shardings,
        manual_axes);

    auto blk = rewriter.createBlock(&manual.getBody(), manual.getBody().begin(),
                                    in_tys, in_locs);
    auto arg = blk->getArgument(0);
    auto partition_id =
        rewriter.create<stablehlo::PartitionIdOp>(concat.getLoc());

    auto T1 = ops[1].getType().getShape()[concat.getDimension()];
    auto T2 = concat.getType().getShape()[concat.getDimension()];

    auto N1 = T1;
    auto N2 = ops[0].getType().getShape()[concat.getDimension()];
    auto N3 = ops[2].getType().getShape()[concat.getDimension()];
    auto T = N1 + N2 + N3;

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

    auto devices_along_dim = ndevices[concat.getDimension()];

    auto zero = rewriter.create<stablehlo::ConstantOp>(
        concat.getLoc(), rewriter.getZeroAttr(partition_id.getType()));

    Value leftSide = rewriter.create<stablehlo::RemOp>(
        concat.getLoc(), partition_id,
        rewriter.create<stablehlo::ConstantOp>(
            concat.getLoc(), partition_id.getType(),
            makeAttr(partition_id.getType(), ny).cast<ElementsAttr>()));
    leftSide = rewriter.create<stablehlo::CompareOp>(
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
    rightSide = rewriter.create<stablehlo::CompareOp>(
        concat.getLoc(), rightSide, zero, stablehlo::ComparisonDirection::EQ);

    auto notLeft = rewriter.create<stablehlo::NotOp>(concat.getLoc(), leftSide);
    auto notRight =
        rewriter.create<stablehlo::NotOp>(concat.getLoc(), rightSide);
    Type ifTypes[] = {RankedTensorType::get(localRetShape,
                                            concat.getType().getElementType())};
    auto if1 = rewriter.create<stablehlo::IfOp>(
        concat.getLoc(), ifTypes,
        rewriter.create<stablehlo::AndOp>(concat.getLoc(), notLeft, notRight));
    rewriter.create<sdy::ReturnOp>(concat.getLoc(), if1->getResults());

    rewriter.createBlock(&if1.getTrueBranch(), if1.getTrueBranch().begin());

    // if ..... !leftSide  && !rightSide
    SmallVector<int64_t> inner_starts(concat.getType().getShape().size(), 0);
    SmallVector<int64_t> inner_limits =
        llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
    SmallVector<int64_t> inner_strides(concat.getType().getShape().size(), 1);
    inner_starts[concat.getDimension()] = T1 / ny; // (T2 - T1) / ny;
    auto local_data_for_comm = rewriter.create<stablehlo::SliceOp>(
        concat.getLoc(), arg, inner_starts, inner_limits, inner_strides);

    // SmallVector<int64_t> source_ids((ny - 1) * nx);
    // SmallVector<int64_t> target_ids((ny - 1) * nx);
    // send_idx = 0;
    // for (int i = 0; i < (ny - 1) * nx; i++) {
    //   source_ids[i] = send_idx;
    //   target_ids[i] = send_idx + 1;
    // }

    // for simplicity send for all pairs
    SmallVector<int64_t> source_target_ids(2 * ny * nx);
    for (int i = 0; i < ny * nx; i++) {
      source_target_ids[i] = i;
      source_target_ids[i + nx * ny] = i + 1;
    }

    auto source_target_pairs_ty = RankedTensorType::get(
        {(int64_t)(nx * ny), (int64_t)2}, rewriter.getI64Type());

    auto cperm = rewriter.create<stablehlo::CollectivePermuteOp>(
        concat.getLoc(), local_data_for_comm->getResult(0),
        DenseIntElementsAttr::get(source_target_pairs_ty, source_target_ids),
        stablehlo::ChannelHandleAttr::get(concat.getContext(), /*handle*/ 0,
                                          /*type*/ 0));

    SmallVector<int64_t> inner_starts2(concat.getType().getShape().size(), 0);
    SmallVector<int64_t> inner_limits2 =
        llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
    inner_limits2[concat.getDimension()] = (T2 - T1) / ny;
    auto already_available_slice = rewriter.create<stablehlo::SliceOp>(
        concat.getLoc(), arg, inner_starts2, inner_limits2, inner_strides);

    Value concat_args[] = {cperm, already_available_slice}; // fix api
    auto ifneither = rewriter.create<stablehlo::ConcatenateOp>(
        concat.getLoc(), concat_args, concat.getDimension());
    rewriter.create<stablehlo::ReturnOp>(concat.getLoc(),
                                         ifneither->getResults());

    // else

    rewriter.createBlock(&if1.getFalseBranch(), if1.getFalseBranch().begin());

    SmallVector<int64_t> inner_starts3(concat.getType().getShape().size(), 0);
    SmallVector<int64_t> inner_limits3 =
        llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
    inner_limits3[concat.getDimension()] = rhsSize;
    auto end_slice = rewriter.create<stablehlo::SliceOp>(
        concat.getLoc(), arg, inner_starts3, inner_limits3, inner_strides);

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
    SmallVector<int64_t> inner_limits4 =
        llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
    inner_starts4[concat.getDimension()] =
        inner_limits4[concat.getDimension()] -
        ops[0].getType().getShape()[concat.getDimension()];
    auto start_slice = rewriter.create<stablehlo::SliceOp>(
        concat.getLoc(), arg, inner_starts4, inner_limits4, inner_strides);

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
        rewriter.create<stablehlo::IfOp>(concat.getLoc(), ifTypes, leftSide);
    rewriter.create<stablehlo::ReturnOp>(concat.getLoc(), if2->getResults());

    // if lhsSide
    {
      rewriter.createBlock(&if2.getTrueBranch(), if2.getTrueBranch().begin());

      SmallVector<int64_t> inner_starts5(concat.getType().getShape().size(), 0);
      SmallVector<int64_t> inner_limits5 =
          llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
      inner_limits5[concat.getDimension()] = (T / ny) - N2;

      auto lhsRightSlice = rewriter.create<stablehlo::SliceOp>(
          concat.getLoc(), arg, inner_starts5, inner_limits5, inner_strides);

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
      SmallVector<int64_t> inner_limits6 =
          llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
      inner_starts6[concat.getDimension()] = N3 - ((N2 + N3) / ny);

      auto rhsLeftSlice = rewriter.create<stablehlo::SliceOp>(
          concat.getLoc(), arg, inner_starts6, inner_limits6, inner_strides);

      Value concat_args2[] = {rhsLeftSlice, result_2};
      auto final_result = rewriter.create<stablehlo::ConcatenateOp>(
          concat.getLoc(), concat_args2, concat.getDimension());
      rewriter.create<stablehlo::ReturnOp>(concat.getLoc(),
                                           final_result->getResults());
    }

    llvm::errs() << "manual: " << manual << "\n";
    rewriter.replaceOp(concat, manual);

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

    patterns.add<SliceConcatSimplify>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
