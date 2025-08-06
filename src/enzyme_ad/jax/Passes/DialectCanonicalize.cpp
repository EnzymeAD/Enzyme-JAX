//===- DialectCanonicalize.cpp - Dialect-restricted canonicalizer pass ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_DIALECTCANONICALIZE
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct DialectCanonicalizePass
    : mlir::enzyme::impl::DialectCanonicalizeBase<DialectCanonicalizePass> {
  using DialectCanonicalizeBase::DialectCanonicalizeBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    for (const std::string &dialectName : dialects) {
      if (Dialect *dialect = getContext().getLoadedDialect(dialectName)) {
        dialect->getCanonicalizationPatterns(patterns);
        continue;
      }

      // If the dialect is not loaded, ignore it because it means no ops from
      // this dialect could have been constructed.
      getOperation()->emitWarning()
          << "requested canonicalization for an unloaded dialetct '"
          << dialectName << "'";
    }
    LogicalResult result =
        applyPatternsGreedily(getOperation(), std::move(patterns),
                              GreedyRewriteConfig()
                                  .setMaxNumRewrites(maxNumRewrites)
                                  .setMaxIterations(maxIterations)
                                  .enableFolding(enableFolding)
                                  .enableConstantCSE(enableConstantCSE));
    if (failed(result)) {
      getOperation()->emitError()
          << "greedy pattern application failed in this scope";
      return signalPassFailure();
    }
  }
};
} // namespace
