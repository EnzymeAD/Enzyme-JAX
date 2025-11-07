#pragma once

#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "src/enzyme_ad/jax/CheckedRewrite.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include <tuple>
#include <vector>

// Loading the header causes a bunch of ambiguous errors
// #include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
namespace mlir {
namespace enzyme {
struct WhileLoopInfo;
}; // namespace enzyme
}; // namespace mlir

std::tuple<bool, bool> allSameBool(const llvm::SmallVector<bool> &bools);
bool allOpsAreUnique(const llvm::SmallVector<mlir::Operation *> &ops);

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

  SliceInfo(OpTy sliceOp, llvm::SmallVector<mlir::Value> dynamicStartIndices,
            llvm::SmallVector<int64_t> startIndices,
            llvm::SmallVector<int64_t> sliceSizes, int64_t sliceDim,
            int64_t sliceStart, bool supported)
      : sliceOp(sliceOp), dynamicStartIndices(std::move(dynamicStartIndices)),
        startIndices(std::move(startIndices)),
        sliceSizes(std::move(sliceSizes)), sliceDim(sliceDim),
        sliceStart(sliceStart), supported(supported) {}
};

SliceInfo<mlir::stablehlo::SliceOp>
constructSliceInfo(mlir::stablehlo::SliceOp sliceOp);
SliceInfo<mlir::stablehlo::DynamicSliceOp>
constructSliceInfo(mlir::stablehlo::DynamicSliceOp sliceOp);

bool areSlicesContiguous(
    llvm::SmallVector<SliceInfo<mlir::stablehlo::SliceOp>> &slices);
bool areSlicesContiguous(
    llvm::SmallVector<SliceInfo<mlir::stablehlo::DynamicSliceOp>> &slices);

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
  bool validReshapeOpInsertDimForBatching(mlir::Operation *op,
                                          int64_t dim) const;
  bool validBroadcastInDimOpInsertDimForBatching(mlir::Operation *op,
                                                 int64_t dim) const;
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

// TODO: we need to be a bit more smart about this pattern. Everytime this
// pattern is applied we increase the dimension of the tensor by 1. struct
// SliceToBatchReshape : public SliceToBatchBase {
//   SliceToBatchReshape(mlir::MLIRContext *ctx, mlir::PatternBenefit benefit =
//   1)
//       : SliceToBatchBase(
//             [](mlir::Operation *op) -> mlir::Operation * {
//               if (!op)
//                 return nullptr;
//               if (auto reshapeOp = dyn_cast<mlir::stablehlo::ReshapeOp>(op))
//               {
//                 if (reshapeIsTranspose(reshapeOp))
//                   return op;

//                 auto inputType = cast<mlir::RankedTensorType>(
//                     reshapeOp.getOperand().getType());
//                 auto outputType = cast<mlir::RankedTensorType>(
//                     reshapeOp.getResult().getType());

//                 auto insertionDims = mlir::enzyme::findReshapeInsertionDims(
//                     inputType, outputType);
//                 if (!insertionDims.empty())
//                   return op;

//                 // We need to be a bit careful about deletions to prevent an
//                 infinite
//                 // loop of insertions and deletions.
//                 // auto deletionDims =
//                 mlir::enzyme::findReshapeInsertionDims(
//                 //     outputType, inputType);
//                 // if (!deletionDims.empty())
//                 //   return op;
//               }
//               return nullptr;
//             },
//             ctx, benefit) {}
// };

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
  struct DynamicSliceInfo {
    mlir::stablehlo::DynamicSliceOp sliceOp;
    int64_t inductionVarDimension;
    bool intermediateReshape;
    llvm::SmallVector<int64_t> reshapeShape;
    int64_t offset;
  };

  enum class BatchLiftingMode {
    DYNAMIC_SLICE,
    DEFINED_OUTSIDE_WHILE,
    CONSTANT,
  };

  enum class IsValidForBatchingResult {
    VALID,
    OPERAND_NOT_ACCESSIBLE_FROM_PARENT,
    NOT_FULL_SLICE,
    MULTIPLE_INDUCTION_VARIABLE_SLICE_DIMS,
    MULTIPLE_INDICES_FROM_BODY,
  };

  struct ValidBatchingInfo {
    IsValidForBatchingResult result;
    int64_t sliceDim;
  };

  ValidBatchingInfo isDynamicSliceValidForBatching(
      mlir::stablehlo::DynamicSliceOp sliceOp, mlir::Value iterVar,
      int64_t limit, mlir::Block &whileBody, mlir::Block *parentBlock) const;

  bool liftOperationByBatching(mlir::PatternRewriter &rewriter,
                               mlir::stablehlo::WhileOp whileOp,
                               llvm::ArrayRef<DynamicSliceInfo> sliceOps,
                               mlir::Operation *op,
                               mlir::enzyme::WhileLoopInfo info) const;
};
