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
void addSumToConv(RewritePatternSet &patterns, bool collapseDims,
                  MLIRContext &context, PatternBenefit benefit);
void addPadDotGeneral(RewritePatternSet &patterns, bool postPad,
                      MLIRContext &context, PatternBenefit benefit);
void addNoNanCompareSimplify(RewritePatternSet &patterns,
                             bool allowOnFloatingPointMath,
                             MLIRContext &context, PatternBenefit benefit);
void addNoNanSelfSubSimplify(RewritePatternSet &patterns,
                             bool allowOnFloatingPointMath,
                             MLIRContext &context, PatternBenefit benefit);
void addNoNanAddSubSimplify(RewritePatternSet &patterns,
                            bool allowOnFloatingPointMath, MLIRContext &context,
                            PatternBenefit benefit);
void addNoNanMulSimplify(RewritePatternSet &patterns,
                         bool allowOnFloatingPointMath, MLIRContext &context,
                         PatternBenefit benefit);
void addNoNanDivSimplify(RewritePatternSet &patterns,
                         bool allowOnFloatingPointMath, MLIRContext &context,
                         PatternBenefit benefit);
void addNoNanZeroBasePowSimplify(RewritePatternSet &patterns,
                                 bool allowOnFloatingPointMath,
                                 MLIRContext &context, PatternBenefit benefit);
void addIotaSimplify(RewritePatternSet &patterns, int64_t maxConstantExpansion,
                     MLIRContext &context, PatternBenefit benefit);
void addRecognizeFromConstant(RewritePatternSet &patterns, int64_t minFoldSize,
                              MLIRContext &context, PatternBenefit benefit);
void addConcatConstProp(RewritePatternSet &patterns,
                        int64_t maxConstantExpansion, MLIRContext &context,
                        PatternBenefit benefit);
void addScatterConstFold(RewritePatternSet &patterns,
                         int64_t maxConstantExpansion, MLIRContext &context,
                         PatternBenefit benefit);
void addPadSimplify(RewritePatternSet &patterns, int64_t maxConstantExpansion,
                    MLIRContext &context, PatternBenefit benefit);
void addDynamicUpdateSliceConstProp(RewritePatternSet &patterns,
                                    int64_t maxConstantExpansion,
                                    MLIRContext &context,
                                    PatternBenefit benefit);
void addWhileSimplify(RewritePatternSet &patterns, bool hoist_all,
                      MLIRContext &context, PatternBenefit benefit);
void addWhileLICM(RewritePatternSet &patterns, bool hoist_all,
                  MLIRContext &context, PatternBenefit benefit);
void addSliceLICM(RewritePatternSet &patterns, bool single_user,
                  MLIRContext &context, PatternBenefit benefit);
void addDotGeneralLICM(RewritePatternSet &patterns, bool single_user,
                       MLIRContext &context, PatternBenefit benefit);
void addReverseLICM(RewritePatternSet &patterns, bool single_user,
                    MLIRContext &context, PatternBenefit benefit);
void addReduceLICM(RewritePatternSet &patterns, bool single_user,
                   MLIRContext &context, PatternBenefit benefit);
void addReduceWindowLICM(RewritePatternSet &patterns, bool single_user,
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
void addConvolutionLICM(RewritePatternSet &patterns, bool single_user,
                        MLIRContext &context, PatternBenefit benefit);
void addDynamicSliceLICM(RewritePatternSet &patterns, bool single_user,
                         MLIRContext &context, PatternBenefit benefit);
void addScatterLICM(RewritePatternSet &patterns, bool single_user,
                    MLIRContext &context, PatternBenefit benefit);
void addGatherLICM(RewritePatternSet &patterns, bool single_user,
                   MLIRContext &context, PatternBenefit benefit);
void addIotaLICM(RewritePatternSet &patterns, bool single_user,
                 MLIRContext &context, PatternBenefit benefit);
void addBroadcastInDimSimplify(RewritePatternSet &patterns,
                               int64_t maxConstantExpansion,
                               MLIRContext &context, PatternBenefit benefit);
void addSelectOpCanon(RewritePatternSet &patterns, int64_t maxConstantExpansion,
                      MLIRContext &context, PatternBenefit benefit);
void addConcatenateOpCanon(RewritePatternSet &patterns,
                           int64_t maxConstantExpansion, MLIRContext &context,
                           PatternBenefit benefit);
void addTransposeElementwise(RewritePatternSet &patterns, bool onlySingleUser,
                             MLIRContext &context, PatternBenefit benefit);
void addReshapeElementwise(RewritePatternSet &patterns, bool onlySingleUser,
                           MLIRContext &context, PatternBenefit benefit);
void addReshapeElementwiseOnlyFusible(RewritePatternSet &patterns,
                                      bool onlySingleUser, MLIRContext &context,
                                      PatternBenefit benefit);
void addReshapeSlice(RewritePatternSet &patterns, bool onlySingleUser,
                     MLIRContext &context, PatternBenefit benefit);
void addReshapeDynamicSlice(RewritePatternSet &patterns, bool onlySingleUser,
                            MLIRContext &context, PatternBenefit benefit);
void addExtendUnaryElementwise(RewritePatternSet &patterns, bool onlySingleUser,
                               MLIRContext &context, PatternBenefit benefit);
void addWrapUnaryElementwise(RewritePatternSet &patterns, bool onlySingleUser,
                             MLIRContext &context, PatternBenefit benefit);
void addSelfAddToConvolutionLike(RewritePatternSet &patterns,
                                 bool allowEmitConvolution,
                                 MLIRContext &context, PatternBenefit benefit);
void addSelfSubtractToConvolutionLike(RewritePatternSet &patterns,
                                      bool allowEmitConvolution,
                                      MLIRContext &context,
                                      PatternBenefit benefit);
void addSelfMulToConvolutionLike(RewritePatternSet &patterns,
                                 bool allowEmitConvolution,
                                 MLIRContext &context, PatternBenefit benefit);
void addEnzymeHLOUnroll(RewritePatternSet &patterns, int64_t maxNumIterations,
                        MLIRContext &context, PatternBenefit benefit);

} // namespace mlir::transform
