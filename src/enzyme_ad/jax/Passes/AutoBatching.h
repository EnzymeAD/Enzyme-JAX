#pragma once

#include "src/enzyme_ad/jax/CheckedRewrite.h"
#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

// Loading the header causes a bunch of ambiguous errors
// #include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
namespace mlir {
namespace enzyme {
struct WhileLoopInfo;
}; // namespace enzyme
}; // namespace mlir

enum class BatchLiftingMode {
  DYNAMIC_SLICE,
  DEFINED_OUTSIDE_WHILE,
  CONSTANT,
  NEEDS_HOISTING_OUTSIDE_WHILE,
  AFFINE_INDEX,
};

template <typename OpTy> struct SliceInfo {
  OpTy sliceOp;
  llvm::SmallVector<int64_t> dimensions;
  bool intermediateReshape;
  std::optional<llvm::SmallVector<int64_t>> explicitReshapeShape;
};

template <typename OpTy> struct BatchOperandConstructionInfo {
  llvm::SmallVector<SliceInfo<OpTy>> slices;
  int32_t sliceOperandIndex;
  bool intermediateReshape;
};

SliceInfo<mlir::stablehlo::SliceOp>
constructSliceInfo(mlir::stablehlo::SliceOp sliceOp);

void ComputeSliceDimension(
    llvm::ArrayRef<SliceInfo<mlir::stablehlo::SliceOp>> slices,
    int64_t *sliceDim);

inline mlir::Operation *CheckElementwise(mlir::Operation *op) {
  if (op && mlir::stablehlo::hasTraitElementwise(op)) {
    return op;
  }
  return nullptr;
}

struct ConcatInsertDimToBatchBase
    : public mlir::enzyme::CheckedOpRewritePattern<
          mlir::stablehlo::ConcatenateOp, ConcatInsertDimToBatchBase> {
  using Base =
      mlir::enzyme::CheckedOpRewritePattern<mlir::stablehlo::ConcatenateOp,
                                            ConcatInsertDimToBatchBase>;
  using Base::Base;

  ConcatInsertDimToBatchBase(
      std::function<mlir::Operation *(mlir::Operation *)> isValidTargetOp,
      mlir::MLIRContext *ctx, mlir::PatternBenefit benefit = 1)
      : Base(ctx, benefit), isValidTargetOp(isValidTargetOp) {}

  llvm::LogicalResult
  matchAndRewriteImpl(mlir::stablehlo::ConcatenateOp concatOp,
                      mlir::PatternRewriter &rewriter) const;

protected:
  std::function<mlir::Operation *(mlir::Operation *)> isValidTargetOp;
};

template <typename OpTy>
struct ConcatInsertDimToBatch : public ConcatInsertDimToBatchBase {
  ConcatInsertDimToBatch(mlir::MLIRContext *ctx,
                         mlir::PatternBenefit benefit = 1)
      : ConcatInsertDimToBatchBase(
            [](mlir::Operation *op) -> mlir::Operation * {
              return llvm::dyn_cast_or_null<OpTy>(op);
            },
            ctx, benefit) {}
};

struct ConcatInsertDimElementwiseToBatch : public ConcatInsertDimToBatchBase {
  ConcatInsertDimElementwiseToBatch(mlir::MLIRContext *ctx,
                                    mlir::PatternBenefit benefit = 1)
      : ConcatInsertDimToBatchBase(CheckElementwise, ctx, benefit) {}
};

struct SliceToBatchBase
    : public mlir::enzyme::CheckedOpRewritePattern<mlir::stablehlo::SliceOp,
                                                   SliceToBatchBase> {
  using Base = mlir::enzyme::CheckedOpRewritePattern<mlir::stablehlo::SliceOp,
                                                     SliceToBatchBase>;
  using Base::Base;

  SliceToBatchBase(
      std::function<mlir::Operation *(mlir::Operation *)> isValidTargetOp,
      mlir::MLIRContext *ctx, mlir::PatternBenefit benefit = 1)
      : Base(ctx, benefit), isValidTargetOp(isValidTargetOp) {}

  llvm::LogicalResult
  matchAndRewriteImpl(mlir::stablehlo::SliceOp sliceOp,
                      mlir::PatternRewriter &rewriter) const;

protected:
  std::function<mlir::Operation *(mlir::Operation *)> isValidTargetOp;
};

template <typename OpTy> struct SliceToBatch : public SliceToBatchBase {
  SliceToBatch(mlir::MLIRContext *ctx, mlir::PatternBenefit benefit = 1)
      : SliceToBatchBase(
            [](mlir::Operation *op) -> mlir::Operation * {
              return llvm::dyn_cast_or_null<OpTy>(op);
            },
            ctx, benefit) {}
};

template <typename OpTy>
struct SliceToBatchWithReshapeLikeCheck : public SliceToBatchBase {
  SliceToBatchWithReshapeLikeCheck(mlir::MLIRContext *ctx,
                                   mlir::PatternBenefit benefit = 1)
      : SliceToBatchBase(
            [](mlir::Operation *op) -> mlir::Operation * {
              if (!op) {
                return nullptr;
              }
              if (auto reshapeLike = llvm::dyn_cast<OpTy>(op)) {
                if (!mlir::stablehlo::OpIsReshapeLike(reshapeLike)) {
                  return reshapeLike;
                }
              }
              return nullptr;
            },
            ctx, benefit) {}
};

struct SliceToBatchElementwise : public SliceToBatchBase {
  SliceToBatchElementwise(mlir::MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : SliceToBatchBase(CheckElementwise, ctx, benefit) {}
};

bool liftOperationByBatching(
    mlir::PatternRewriter &rewriter, mlir::stablehlo::WhileOp whileOp,
    llvm::ArrayRef<SliceInfo<mlir::stablehlo::DynamicSliceOp>> slices,
    mlir::Operation *op, mlir::enzyme::WhileLoopInfo info);

bool liftReduceLikeOperation(
    mlir::PatternRewriter &rewriter, mlir::stablehlo::WhileOp whileOp,
    llvm::ArrayRef<SliceInfo<mlir::stablehlo::DynamicSliceOp>> slices,
    mlir::Operation *op, mlir::enzyme::WhileLoopInfo info);

bool liftChainToReduceWindow(
    mlir::PatternRewriter &rewriter, mlir::stablehlo::WhileOp whileOp,
    llvm::ArrayRef<SliceInfo<mlir::stablehlo::DynamicSliceOp>> slices,
    mlir::Operation *op, mlir::enzyme::WhileLoopInfo info);

struct GreedyWhileLoopBatchFission
    : public mlir::enzyme::CheckedOpRewritePattern<
          mlir::stablehlo::WhileOp, GreedyWhileLoopBatchFission> {
  using Base =
      mlir::enzyme::CheckedOpRewritePattern<mlir::stablehlo::WhileOp,
                                            GreedyWhileLoopBatchFission>;
  using Base::Base;

  mlir::LogicalResult
  matchAndRewriteImpl(mlir::stablehlo::WhileOp whileOp,
                      mlir::PatternRewriter &rewriter) const;

private:
  enum class IsValidForBatchingResult {
    VALID,
    OPERAND_NOT_ACCESSIBLE_FROM_PARENT,
    DYNAMIC_START_INDEX,
    NO_INDUCTION_VARIABLE_DETECTED,
  };

  static bool isValidForBatchingResult(IsValidForBatchingResult result) {
    return result == IsValidForBatchingResult::VALID;
  }

  struct ValidBatchingInfo {
    IsValidForBatchingResult result;
    llvm::SmallVector<int64_t> dimensions;
  };

  ValidBatchingInfo
  isDynamicSliceValidForBatching(mlir::stablehlo::DynamicSliceOp sliceOp,
                                 mlir::enzyme::WhileLoopInfo &loopInfo,
                                 mlir::Block &whileBody,
                                 mlir::stablehlo::WhileOp whileOp) const;
};

struct WhileElementwiseReductionToReduce
    : public mlir::enzyme::CheckedOpRewritePattern<
          mlir::stablehlo::WhileOp, WhileElementwiseReductionToReduce> {
  using Base =
      mlir::enzyme::CheckedOpRewritePattern<mlir::stablehlo::WhileOp,
                                            WhileElementwiseReductionToReduce>;
  using Base::Base;

  mlir::LogicalResult
  matchAndRewriteImpl(mlir::stablehlo::WhileOp whileOp,
                      mlir::PatternRewriter &rewriter) const;
};

/*
Analyzes loop arguments and checks if a

while (%iterarg = %x) {
  dynamic_slice %iterarg %idxs...
  ...
  %iterarg = dynamic_update_slice %iterarg %idxs2...
}

If dynamic_slice only reads indices of %iterarg only before they are updated,
we can transform the code into the following:

while (%iterarg = %x) {
  dynamic_slice %x %idxs...
  ...
  %iterarg = dynamic_update_slice %x %idxs2...
}
*/
struct RemoveLoopCarriedDependenciesFromWhileLoadOperations
    : public mlir::enzyme::CheckedOpRewritePattern<
          mlir::stablehlo::WhileOp,
          RemoveLoopCarriedDependenciesFromWhileLoadOperations> {
  using Base = mlir::enzyme::CheckedOpRewritePattern<
      mlir::stablehlo::WhileOp,
      RemoveLoopCarriedDependenciesFromWhileLoadOperations>;
  using Base::Base;

  mlir::LogicalResult
  matchAndRewriteImpl(mlir::stablehlo::WhileOp whileOp,
                      mlir::PatternRewriter &rewriter) const;

private:
  bool extractDynamicUpdateSliceUpdate(
      mlir::Operation *op, mlir::BlockArgument blockArg,
      llvm::SmallVectorImpl<mlir::Value> &startIndices,
      llvm::SmallVectorImpl<int64_t> &sliceSizes,
      mlir::enzyme::WhileLoopInfo &info) const;
};

struct WhileIsCopySimplify
    : public mlir::enzyme::CheckedOpRewritePattern<mlir::stablehlo::WhileOp,
                                                   WhileIsCopySimplify> {
  using Base = mlir::enzyme::CheckedOpRewritePattern<mlir::stablehlo::WhileOp,
                                                     WhileIsCopySimplify>;

  using Base::Base;

  mlir::LogicalResult
  matchAndRewriteImpl(mlir::stablehlo::WhileOp whileOp,
                      mlir::PatternRewriter &rewriter) const;

private:
  std::optional<llvm::SmallVector<mlir::Operation *>> extractValidUpdateChain(
      mlir::PatternRewriter &rewriter,
      mlir::stablehlo::DynamicUpdateSliceOp dusOp,
      mlir::stablehlo::WhileOp whileOp,
      llvm::MapVector<mlir::Value, mlir::enzyme::WhileLoopInfo::AffineIndexInfo>
          &affineIndexInfo,
      mlir::enzyme::WhileLoopInfo &info) const;

  bool extractValidUpdateChainInner(
      mlir::PatternRewriter &rewriter, mlir::Operation *op,
      mlir::stablehlo::WhileOp whileOp,
      llvm::SmallVectorImpl<mlir::Operation *> &updateChain) const;

  template <typename OpTy>
  llvm::SmallVector<int64_t> getInductionVariableDimension(
      OpTy op,
      llvm::MapVector<mlir::Value, mlir::enzyme::WhileLoopInfo::AffineIndexInfo>
          &affineIndexInfo,
      mlir::stablehlo::WhileOp whileOp,
      mlir::enzyme::WhileLoopInfo &info) const;

  llvm::SmallVector<int64_t> getInductionVariableDimension(
      mlir::OperandRange startIndices,
      llvm::MapVector<mlir::Value, mlir::enzyme::WhileLoopInfo::AffineIndexInfo>
          &affineIndexInfo,
      mlir::stablehlo::WhileOp whileOp,
      mlir::enzyme::WhileLoopInfo &info) const;
};

namespace mlir {
namespace enzyme {

struct AutoBatchingPassPipelineOptions {
  bool enableSliceToBatch;
  bool enableConcatInsertDimToBatch;
  std::string whileLoopBatchingMode;
  bool enableWhileElementwiseReductionToReduce;
  bool enableWhileIsCopySimplify;
  bool enableRemoveLoopCarriedDependenciesFromWhileLoadOperations;
};

void populateAutoBatchingPassPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx,
                                      AutoBatchingPassPipelineOptions options);

} // namespace enzyme
} // namespace mlir
