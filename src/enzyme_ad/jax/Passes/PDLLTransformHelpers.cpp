//===- PDLLTransformHelpers.cpp - Configurable PDLL helpers -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/PDLLTransformHelpers.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace mlir {
namespace enzyme {

std::optional<RewriteExtent> parseRewriteExtent(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<RewriteExtent>>(name)
      .Case("pure-local", RewriteExtent::PureLocal)
      .Case("checked-local", RewriteExtent::CheckedLocal)
      .Case("checked-global", RewriteExtent::CheckedGlobal)
      .Default(std::nullopt);
}

//===----------------------------------------------------------------------===//
// SliceExtend commute helpers
//===----------------------------------------------------------------------===//

LogicalResult
hasMultipleMatchingExtendUsers(::mlir::stablehlo::SliceOp sliceOp,
                               ::mlir::enzymexla::ExtendOp extendOp) {
  Value baseOperand = sliceOp.getOperand();
  int64_t targetDim = extendOp.getDimension();
  int64_t targetLhs = extendOp.getLhs();
  int64_t targetRhs = extendOp.getRhs();

  // Look for at least one OTHER user that matches the extend characteristics.
  for (Operation *user : baseOperand.getUsers()) {
    if (user == sliceOp.getOperation())
      continue; // Skip the trigger slice.

    // Case 1: Direct Extend.
    if (auto directExtend = mlir::dyn_cast<enzymexla::ExtendOp>(user)) {
      if (directExtend.getDimension() == targetDim &&
          directExtend.getLhs() == targetLhs &&
          directExtend.getRhs() == targetRhs) {
        return success();
      }
      continue;
    }

    // Case 2: Extend of Slice.
    if (auto sliceUser = mlir::dyn_cast<stablehlo::SliceOp>(user)) {
      if (!sliceUser->hasOneUse())
        continue;

      auto extendOfSlice =
          mlir::dyn_cast<enzymexla::ExtendOp>(*sliceUser->user_begin());
      if (extendOfSlice && extendOfSlice.getDimension() == targetDim &&
          extendOfSlice.getLhs() == targetLhs &&
          extendOfSlice.getRhs() == targetRhs) {
        return success();
      }
    }
  }

  return failure();
}

Operation *commuteExtendAndSlice(PatternRewriter &rewriter,
                                 ::mlir::enzymexla::ExtendOp extendOp,
                                 ::mlir::stablehlo::SliceOp sliceOp,
                                 bool doGlobalDeduplication) {
  Value baseOperand = sliceOp.getOperand();
  int64_t dim = extendOp.getDimension();
  int64_t lhs = extendOp.getLhs();
  int64_t rhs = extendOp.getRhs();
  Location loc = extendOp.getLoc();

  // 1. Create the single shared base ExtendOp.
  auto baseType = mlir::cast<RankedTensorType>(baseOperand.getType());
  llvm::SmallVector<int64_t> newExtendShape =
      llvm::to_vector(baseType.getShape());
  newExtendShape[dim] += (lhs + rhs);
  auto newExtendType =
      RankedTensorType::get(newExtendShape, baseType.getElementType());

  // Set insertion point to ensure dominance.
  if (auto subOp = baseOperand.getDefiningOp()) {
    rewriter.setInsertionPointAfter(subOp);
  } else {
    rewriter.setInsertionPointToStart(
        mlir::cast<BlockArgument>(baseOperand).getOwner());
  }

  auto newBaseExtendOp = enzymexla::ExtendOp::create(
      rewriter, loc, newExtendType, baseOperand, lhs, rhs, dim);

  // 2. Optionally, eagerly replace all OTHER matching sibling users.
  if (doGlobalDeduplication) {
    // Snapshot users because we are about to mutate the use-list.
    llvm::SmallVector<Operation *> users(baseOperand.getUsers().begin(),
                                         baseOperand.getUsers().end());

    for (Operation *user : users) {
      if (user == sliceOp.getOperation())
        continue; // Skip the trigger slice; handled at the end.
      if (user == newBaseExtendOp.getOperation())
        continue;

      // Case A: Direct Extend.
      if (auto directExtend = mlir::dyn_cast<enzymexla::ExtendOp>(user)) {
        if (directExtend.getDimension() != dim ||
            directExtend.getLhs() != lhs || directExtend.getRhs() != rhs)
          continue;

        if (directExtend.getResult().getType() == newBaseExtendOp.getType()) {
          rewriter.replaceOp(directExtend, newBaseExtendOp.getResult());
        } else {
          rewriter.setInsertionPoint(directExtend);
          auto castOp = rewriter.create<mlir::tensor::CastOp>(
              loc, directExtend.getResult().getType(),
              newBaseExtendOp.getResult());
          rewriter.replaceOp(directExtend, castOp.getResult());
        }
        continue;
      }

      // Case B: Extend of Slice.
      if (auto sliceUser = mlir::dyn_cast<stablehlo::SliceOp>(user)) {
        if (!sliceUser->hasOneUse())
          continue;

        auto extendOfSlice =
            mlir::dyn_cast<enzymexla::ExtendOp>(*sliceUser->user_begin());
        if (!extendOfSlice || extendOfSlice.getDimension() != dim ||
            extendOfSlice.getLhs() != lhs || extendOfSlice.getRhs() != rhs)
          continue;

        if (sliceUser.getStartIndices()[dim] != 0 ||
            sliceUser.getLimitIndices()[dim] != baseType.getShape()[dim] ||
            sliceUser.getStrides()[dim] != 1)
          continue;

        rewriter.setInsertionPoint(extendOfSlice);
        llvm::SmallVector<int64_t> newSliceStarts =
            llvm::to_vector(sliceUser.getStartIndices());
        llvm::SmallVector<int64_t> newSliceLimits =
            llvm::to_vector(sliceUser.getLimitIndices());
        llvm::SmallVector<int64_t> newSliceStrides =
            llvm::to_vector(sliceUser.getStrides());

        newSliceStarts[dim] = 0;
        newSliceLimits[dim] = newExtendShape[dim];
        newSliceStrides[dim] = 1;

        auto newSliceOp = rewriter.create<stablehlo::SliceOp>(
            loc, extendOfSlice.getResult().getType(),
            newBaseExtendOp.getResult(), newSliceStarts, newSliceLimits,
            newSliceStrides);
        rewriter.replaceOp(extendOfSlice, newSliceOp.getResult());
      }
    }
  }

  // 3. Create the new commuted SliceOp to replace the trigger Extend.
  llvm::SmallVector<int64_t> newStarts =
      llvm::to_vector(sliceOp.getStartIndices());
  llvm::SmallVector<int64_t> newLimits =
      llvm::to_vector(sliceOp.getLimitIndices());
  llvm::SmallVector<int64_t> newStrides = llvm::to_vector(sliceOp.getStrides());

  newStarts[dim] = 0;
  newLimits[dim] = newExtendShape[dim];
  newStrides[dim] = 1;

  rewriter.setInsertionPoint(extendOp);
  return rewriter.create<stablehlo::SliceOp>(
      loc, extendOp.getResult().getType(), newBaseExtendOp.getResult(),
      newStarts, newLimits, newStrides);
}

void registerSliceExtendDynamicPDLLBindings(RewritePatternSet &patterns,
                                            RewriteExtent extent) {
  PDLPatternModule &pdl = patterns.getPDLPatterns();

  // Bind the configurable profitability check. The high-level overload of
  // `registerConstraintFunction` lets us receive typed Op arguments
  // directly; the framework unwraps the underlying PDLValues.
  pdl.registerConstraintFunction(
      "ConfigurableProfitabilityCheck",
      [extent](PatternRewriter &, stablehlo::SliceOp sliceOp,
               enzymexla::ExtendOp extendOp) -> LogicalResult {
        if (extent == RewriteExtent::PureLocal)
          return success();
        return hasMultipleMatchingExtendUsers(sliceOp, extendOp);
      });

  // Bind the configurable rewrite. The high-level overload of
  // `registerRewriteFunction` lets us return the replacement Op directly;
  // the framework packages it into the PDL result list.
  pdl.registerRewriteFunction(
      "ConfigurableCommute",
      [extent](PatternRewriter &rewriter, enzymexla::ExtendOp extendOp,
               stablehlo::SliceOp sliceOp) -> Operation * {
        bool doGlobal = (extent == RewriteExtent::CheckedGlobal);
        return commuteExtendAndSlice(rewriter, extendOp, sliceOp, doGlobal);
      });
}

} // namespace enzyme
} // namespace mlir
