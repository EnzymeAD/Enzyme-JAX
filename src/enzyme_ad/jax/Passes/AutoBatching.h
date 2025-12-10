#pragma once

#include "src/enzyme_ad/jax/CheckedRewrite.h"
#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include <tuple>

// Loading the header causes a bunch of ambiguous errors
// #include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
namespace mlir {
namespace enzyme {
struct WhileLoopInfo;
}; // namespace enzyme
}; // namespace mlir

std::tuple<bool, bool> allSameBool(const llvm::SmallVector<bool> &bools);
bool allOpsAreUnique(const llvm::SmallVector<mlir::Operation *> &ops);

enum class BatchLiftingMode {
  DYNAMIC_SLICE,
  DEFINED_OUTSIDE_WHILE,
  CONSTANT,
  NEEDS_HOISTING_OUTSIDE_WHILE,
  AFFINE_INDEX,
};

struct BatchOperandConstructionInfo {
  mlir::stablehlo::SliceOp sliceOp;
  int32_t sliceOperandIndex;
  int32_t sliceDim;
  int32_t nbatches;
  bool intermediateReshape;
};

template <typename OpTy> struct SliceInfo {
  OpTy sliceOp;
  llvm::SmallVector<mlir::Value> dynamicStartIndices;
  llvm::SmallVector<int64_t> startIndices;
  llvm::SmallVector<int64_t> sliceSizes;
  int64_t sliceDim;
  int64_t sliceStart;
  bool supported;
};

SliceInfo<mlir::stablehlo::SliceOp>
constructSliceInfo(mlir::stablehlo::SliceOp sliceOp);

bool areSlicesContiguous(
    llvm::SmallVector<SliceInfo<mlir::stablehlo::SliceOp>> &slices);

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

private:
  bool validReshapeOpInsertDimForBatching(
      mlir::stablehlo::ReshapeOp op, int64_t dim,
      llvm::SmallVectorImpl<int64_t> &intermediateInsertions) const;
  bool validBroadcastInDimOpInsertDimForBatching(
      mlir::stablehlo::BroadcastInDimOp op, int64_t dim,
      llvm::SmallVectorImpl<int64_t> &intermediateInsertions) const;
};

template <typename OpTy>
struct ConcatInsertDimToBatch : public ConcatInsertDimToBatchBase {
  ConcatInsertDimToBatch(mlir::MLIRContext *ctx,
                         mlir::PatternBenefit benefit = 1)
      : ConcatInsertDimToBatchBase(
            [](mlir::Operation *op) -> mlir::Operation * {
              if (!op)
                return nullptr;
              return llvm::dyn_cast<OpTy>(op);
            },
            ctx, benefit) {}
};

struct ConcatInsertDimElementwiseToBatch : public ConcatInsertDimToBatchBase {
  ConcatInsertDimElementwiseToBatch(mlir::MLIRContext *ctx,
                                    mlir::PatternBenefit benefit = 1)
      : ConcatInsertDimToBatchBase(
            [](mlir::Operation *op) -> mlir::Operation * {
              if (!op)
                return nullptr;
              if (op->hasTrait<mlir::OpTrait::Elementwise>())
                return op;
              return nullptr;
            },
            ctx, benefit) {}
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
              if (!op)
                return nullptr;
              return llvm::dyn_cast<OpTy>(op);
            },
            ctx, benefit) {}
};

struct SliceToBatchElementwise : public SliceToBatchBase {
  SliceToBatchElementwise(mlir::MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : SliceToBatchBase(
            [](mlir::Operation *op) -> mlir::Operation * {
              if (!op)
                return nullptr;
              if (op->hasTrait<mlir::OpTrait::Elementwise>())
                return op;
              return nullptr;
            },
            ctx, benefit) {}
};

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

  struct DynamicSliceInfo {
    mlir::stablehlo::DynamicSliceOp sliceOp;
    llvm::SmallVector<int64_t> dimensions;
    bool intermediateReshape;
    bool needsManualReshape;
    llvm::SmallVector<int64_t> reshapeShape;
  };

  struct ValidBatchingInfo {
    IsValidForBatchingResult result;
    llvm::SmallVector<int64_t> dimensions;
  };

  ValidBatchingInfo isDynamicSliceValidForBatching(
      mlir::stablehlo::DynamicSliceOp sliceOp,
      llvm::MapVector<mlir::Value, mlir::enzyme::WhileLoopInfo::AffineIndexInfo>
          &affineIndexInfoMap,
      mlir::Block &whileBody, mlir::stablehlo::WhileOp whileOp) const;

  bool liftOperationByBatching(mlir::PatternRewriter &rewriter,
                               mlir::stablehlo::WhileOp whileOp,
                               llvm::ArrayRef<DynamicSliceInfo> sliceOps,
                               mlir::Operation *op,
                               mlir::enzyme::WhileLoopInfo info) const;
};
