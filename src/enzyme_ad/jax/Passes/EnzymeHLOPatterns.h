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
void addWhileSimplify(RewritePatternSet &patterns, bool hoist_all,
                      MLIRContext &context, PatternBenefit benefit);
void addSliceLICM(RewritePatternSet &patterns, bool single_user,
                  MLIRContext &context, PatternBenefit benefit);
void addDUSLICM(RewritePatternSet &patterns, bool single_user,
                  MLIRContext &context, PatternBenefit benefit);
void addPadLICM(RewritePatternSet &patterns, bool single_user,
                  MLIRContext &context, PatternBenefit benefit);
void addElementwiseLICM(RewritePatternSet &patterns, bool single_user,
                  MLIRContext &context, PatternBenefit benefit);
void addConcatenateLICM(RewritePatternSet &patterns, bool single_user,
                  MLIRContext &context, PatternBenefit benefit);
void addBroadcastInDimLICM(RewritePatternSet &patterns, bool single_user,
                  MLIRContext &context, PatternBenefit benefit);
void addReshapeLICM(RewritePatternSet &patterns, bool single_user,
                  MLIRContext &context, PatternBenefit benefit);
void addTransposeLICM(RewritePatternSet &patterns, bool single_user,
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
