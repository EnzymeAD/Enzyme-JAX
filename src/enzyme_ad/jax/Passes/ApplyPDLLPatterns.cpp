//===- ApplyPDLLPatterns.cpp - Apply all PDLL patterns -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a generic pass that runs every PDLL pattern set
// defined under `Passes/PDLL/`.
//
// Adding a new `.pdll` file is a three-step operation:
//   1. Drop the new `Foo.pdll` file in `src/enzyme_ad/jax/Passes/PDLL/`.
//   2. Add a corresponding `gentbl_cc_library` entry in the BUILD file
//      (model it after `MorePatternsPDLLPatternsIncGen`) and add the new
//      `*IncGen` target to the main library `deps`.
//   3. Add a `PDLL_PATTERN_SET(...)` line below pointing at the generated
//      header. The C++ side will then automatically include the patterns in
//      the generic application pass.
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_APPLYPDLLPATTERNSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

// Each generated PDLL header defines `populateGeneratedPDLLPatterns` at file
// scope, so we wrap each include in its own anonymous nested namespace to
// avoid symbol clashes. To register an additional `.pdll` pattern set, just
// add a new `namespace { #include ... }` block and a matching call in
// `populateAllPDLLPatterns` below.

namespace slice_extend_pdll {
#include "src/enzyme_ad/jax/Passes/PDLL/SliceExtendPDLLPatterns.h.inc"
} // namespace slice_extend_pdll

namespace more_patterns_pdll {
#include "src/enzyme_ad/jax/Passes/PDLL/MorePatternsPDLLPatterns.h.inc"
} // namespace more_patterns_pdll

static void populateAllPDLLPatterns(RewritePatternSet &patterns) {
  slice_extend_pdll::populateGeneratedPDLLPatterns(patterns);
  more_patterns_pdll::populateGeneratedPDLLPatterns(patterns);
}

struct ApplyPDLLPatternsPass
    : public enzyme::impl::ApplyPDLLPatternsPassBase<ApplyPDLLPatternsPass> {
  using ApplyPDLLPatternsPassBase::ApplyPDLLPatternsPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList(ctx);
    populateAllPDLLPatterns(patternList);
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
