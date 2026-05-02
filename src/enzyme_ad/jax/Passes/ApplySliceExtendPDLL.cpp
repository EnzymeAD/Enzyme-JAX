//===- ApplySliceExtendPDLL.cpp - Apply the SliceExtendCommute PDLL ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a simple pass that runs the PDLL pattern defined in
// `Passes/PDLL/sliceextend.pdll`, which commutes a `stablehlo.slice` followed
// by an `enzymexla.extend` into an `enzymexla.extend` followed by a
// `stablehlo.slice`.
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_APPLYSLICEEXTENDPDLLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "src/enzyme_ad/jax/Passes/PDLL/SliceExtendPDLLPatterns.h.inc"

struct ApplySliceExtendPDLLPass
    : public enzyme::impl::ApplySliceExtendPDLLPassBase<
          ApplySliceExtendPDLLPass> {
  using ApplySliceExtendPDLLPassBase::ApplySliceExtendPDLLPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList(ctx);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() override {
    if (failed(applyPatternsGreedily(getOperation(), patterns)))
      signalPassFailure();
  }

  FrozenRewritePatternSet patterns;
};
} // end anonymous namespace
