#pragma once

#include "mlir/IR/PatternMatch.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include <tuple>
#include <vector>

struct BatchOperandConstructionInfo {
  mlir::stablehlo::SliceOp sliceOp;
  int32_t sliceOperandIndex;
  int32_t sliceDim;
  int32_t nbatches;
  bool intermediateReshape;
};

struct ConcatInsertDimToBatchBase
    : public mlir::OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using Base = mlir::OpRewritePattern<mlir::stablehlo::ConcatenateOp>;
  using Base::Base;

  ConcatInsertDimToBatchBase(
      std::function<mlir::Operation *(mlir::Operation *)> isValidTargetOp,
      mlir::MLIRContext *ctx, mlir::PatternBenefit benefit = 1)
      : Base(ctx, benefit), isValidTargetOp(isValidTargetOp) {}

  llvm::LogicalResult
  matchAndRewrite(mlir::stablehlo::ConcatenateOp concatOp,
                  mlir::PatternRewriter &rewriter) const override;

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

struct SliceToBatchBase
    : public mlir::OpRewritePattern<mlir::stablehlo::SliceOp> {
  using Base = mlir::OpRewritePattern<mlir::stablehlo::SliceOp>;
  using Base::Base;

  SliceToBatchBase(
      std::function<mlir::Operation *(mlir::Operation *)> isValidTargetOp,
      mlir::MLIRContext *ctx, mlir::PatternBenefit benefit = 1)
      : Base(ctx, benefit), isValidTargetOp(isValidTargetOp) {}

  llvm::LogicalResult
  matchAndRewrite(mlir::stablehlo::SliceOp sliceOp,
                  mlir::PatternRewriter &rewriter) const override;

private:
  struct SliceInfo {
    mlir::stablehlo::SliceOp sliceOp;
    llvm::SmallVector<int64_t> startIndices;
    llvm::SmallVector<int64_t> endIndices;
    llvm::SmallVector<int64_t> strides;
    int64_t sliceDim;
    int64_t sliceStart;
    bool supported;
  };

  SliceInfo extractSliceInfo(mlir::stablehlo::SliceOp slice) const;
  bool areSlicesContiguous(llvm::SmallVector<SliceInfo> &slices) const;
  std::tuple<bool, bool>
  allSameBool(const llvm::SmallVector<bool> &bools) const;
  std::tuple<bool, bool>
  allSameBool(const llvm::SmallVector<mlir::Operation *> &ops) const;
  bool allOpsAreUnique(const llvm::SmallVector<mlir::Operation *> &ops) const;

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
