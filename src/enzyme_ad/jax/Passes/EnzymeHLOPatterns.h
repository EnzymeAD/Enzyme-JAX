//===- EnzymeHLOPatterns.h - functions to register patterns -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

namespace mlir {
class MLIRContext;
class PatternBenefit;
class RewritePatternSet;
} // namespace mlir

#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h.inc"

namespace mlir::transform {
void addPadDotGeneral(RewritePatternSet &patterns, bool postPad,
                      MLIRContext &context, PatternBenefit benefit);
void addNoNanAddSubSimplify(RewritePatternSet &patterns,
                            bool allowOnFloatingPointMath, MLIRContext &context,
                            PatternBenefit benefit);
void addIotaSimplify(RewritePatternSet &patterns, int64_t maxConstantExpansion,
                     MLIRContext &context, PatternBenefit benefit);
void addBroadcastInDimSimplify(RewritePatternSet &patterns,
                               int64_t maxConstantExpansion,
                               MLIRContext &context, PatternBenefit benefit);
void addSelectOpCanon(RewritePatternSet &patterns, int64_t maxConstantExpansion,
                      MLIRContext &context, PatternBenefit benefit);
void addConcatenateOpCanon(RewritePatternSet &patterns,
                           int64_t maxConstantExpansion, MLIRContext &context,
                           PatternBenefit benefit);
} // namespace mlir::transform
