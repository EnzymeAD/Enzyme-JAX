//===- StableHLODUSToConcat.cpp - Fuse StableHLO dus(dus) to dus(concat) --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to fuse StableHLO dus(dus) to dus(concat)
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"

// #include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
// #include "mlir/Dialect/Affine/Analysis/Utils.h"
// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Affine/IR/AffineValueMap.h"
// #include "mlir/Dialect/Affine/LoopUtils.h"
// #include "mlir/Dialect/Affine/Passes.h"
// #include "mlir/Dialect/Affine/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_STABLEHLODUSTOCONCAT
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

static int
getConcatenateDimIfConcatenable(stablehlo::DynamicUpdateSliceOp op1,
                                stablehlo::DynamicUpdateSliceOp op2) {
  auto getShape = [](Value value) -> SmallVector<int64_t> {
    if (auto shapedType = value.getType().dyn_cast<ShapedType>()) {
      return llvm::to_vector(shapedType.getShape());
    }
    return {};
  };

  SmallVector<int64_t> shape1 = getShape(op1.getUpdate());
  SmallVector<int64_t> shape2 = getShape(op2.getUpdate());

  // Check if ranks match
  if (shape1.size() != shape2.size())
    return -1;

  if (shape1.empty())
    return -1;

  int mismatchedDims = 0;
  int concatenateDim = -1;
  for (const auto &[i, dims] : llvm::enumerate(llvm::zip(shape1, shape2))) {
    auto [dim1, dim2] = dims;
    if (dim1 != dim2) {
      mismatchedDims++;
      concatenateDim = i;
      if (mismatchedDims > 1)
        return -1;
    }
  }

  auto startIndices1 = op1.getStartIndices();
  auto startIndices2 = op2.getStartIndices();

  int mismatchedIndices = 0;
  for (const auto &[i, startIndices] :
       llvm::enumerate(llvm::zip(startIndices1, startIndices2))) {
    auto [idx1, idx2] = startIndices;
    auto const1 = idx1.getDefiningOp<stablehlo::ConstantOp>();
    auto const2 = idx2.getDefiningOp<stablehlo::ConstantOp>();

    // For constants, compare actual values
    // If one is constant and other isn't, they don't match
    // Otherwise compare SSA values directly
    auto getConstantInt = [](Value val) -> std::optional<int64_t> {
      if (auto constOp = val.getDefiningOp<stablehlo::ConstantOp>()) {
        //      constOp->dump();
        //      constOp.getValue().dump();
        if (auto intAttr =
                constOp.getValue().dyn_cast<DenseIntElementsAttr>()) {
          APInt apIntValue = intAttr.getValues<APInt>()[0];
          return apIntValue.getSExtValue();
        }
      }
      return std::nullopt;
    };

    auto val1 = getConstantInt(idx1);
    auto val2 = getConstantInt(idx2);

    if (!val1 && !val2)
      if (startIndices1 != startIndices2)
        return -1;

    if (!val1 || !val2)
      return -1;

    if (*val1 != *val2) {
      mismatchedIndices++;
      if (concatenateDim != -1 && concatenateDim != i)
        return -1;
      if (mismatchedIndices > 1)
        return -1;
      if (shape1[i] + *val1 != *val2) // && shape2[i] + *val2 != *val1)
        return -1;
      // op1->dump();
      // op2->dump();
      // llvm::errs() << shape1[i] << ' ' << *val1 << ' ' << shape2[i] << ' '
      //              << *val2 << '\n';
      concatenateDim = i;
    }
  }
  return concatenateDim;
}
/*
TH1 -> 1x20x30 0, 1, 2
       1x20x30 1, 1, 2
-> concat
TH2 -> 1x20x30 0, 1, 2
       1x20x30 1, 2, 2
-> false
TH3 -> 1x20x30 0, 1, 2
       1x20x30 0, 1, 2
-> overwrite
TH4 -> 1x20x30 0, 1, 2
       2x20x30 0, 1, 2
-> overwrite
TH5 -> 2x20x30 0, 1, 2
       1x20x30 0, 1, 2
-> concat + overwrite
TH6 -> 2x20x30 0, 1, 2
       1x20x30 2, 1, 2
-> concat
TH7 -> 1x20x30 2, 1, 2
       2x20x30 0, 1, 2
-> concat
*/

static bool isConcatenable(stablehlo::DynamicUpdateSliceOp op1,
                           stablehlo::DynamicUpdateSliceOp op2) {
  return getConcatenateDimIfConcatenable(op1, op2) != -1;
}

namespace {
struct StableHLODUSToConcatPass
    : public enzyme::impl::StableHLODUSToConcatBase<StableHLODUSToConcatPass> {
  using StableHLODUSToConcatBase::StableHLODUSToConcatBase;

  void runOnOperation() override {
    std::vector<stablehlo::DynamicUpdateSliceOp> worklist;

    auto op = getOperation();
    auto *ctx = op->getContext();
    OpBuilder builder(ctx);

    op->walk<WalkOrder::PostOrder>([&](stablehlo::DynamicUpdateSliceOp dus) {
      if (auto definingDUS =
              dus.getOperand()
                  .getDefiningOp<stablehlo::DynamicUpdateSliceOp>()) {
        if (!dus.getOperand().hasOneUse())
          return;
        if (isConcatenable(definingDUS, dus))
          worklist.push_back(dus);
      }
    });

    for (auto dus : worklist) {
      //    dus->dump();
      auto definingDUS =
          dus.getOperand().getDefiningOp<stablehlo::DynamicUpdateSliceOp>();
      int concatenateDim = getConcatenateDimIfConcatenable(definingDUS, dus);

      if (concatenateDim == -1)
        continue;
      auto getConstantInt = [](Value val) -> int64_t {
        auto constOp = val.getDefiningOp<stablehlo::ConstantOp>();
        auto intAttr = constOp.getValue().cast<DenseIntElementsAttr>();
        APInt apIntValue = intAttr.getValues<APInt>()[0];
        return apIntValue.getSExtValue();
      };

      auto val1 = getConstantInt(definingDUS.getStartIndices()[concatenateDim]);
      auto val2 = getConstantInt(dus.getStartIndices()[concatenateDim]);

      // auto dus1 = (val1 < val2) ? definingDUS : dus;
      // auto dus2 = (val1 < val2) ? dus : definingDUS;
      auto dus1 = definingDUS;
      auto dus2 = dus;
      Value update1 = dus1.getUpdate();
      Value update2 = dus2.getUpdate();
      builder.setInsertionPoint(dus);
      auto concat = builder.create<stablehlo::ConcatenateOp>(
          dus.getLoc(), mlir::ValueRange{update1, update2},
          builder.getI64IntegerAttr(concatenateDim));

      SmallVector<Value> newIndices(dus1.getStartIndices());
      auto newUpdate = builder.create<stablehlo::DynamicUpdateSliceOp>(
          dus.getLoc(), dus1.getOperand(), concat.getResult(), newIndices);

      // 6. Replace all uses and clean up
      dus2.getResult().replaceAllUsesWith(newUpdate.getResult());
      dus->erase();
      definingDUS->erase();
    }
  }
};
} // namespace
