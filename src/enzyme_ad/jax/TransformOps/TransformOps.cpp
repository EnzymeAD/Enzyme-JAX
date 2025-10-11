//===- TransformOps.cpp - Definition of transform extension ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/TransformOps/TransformOps.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
#include "src/enzyme_ad/jax/TransformOps/OpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/TransformOps.cpp.inc"
#include "src/enzyme_ad/jax/TransformOps/TransformOpsImpl.cpp.inc"

using namespace mlir;
using namespace mlir::enzyme;

namespace mlir {
namespace transform {

void ApplyPadDotGeneralPatterns::populatePatterns(RewritePatternSet &patterns) {
  addPadDotGeneral(patterns, getParameter(), *getContext(),
                   PatternBenefit(getBenefit().value_or(1)));
}
void ApplyNoNanCompareSimplify::populatePatterns(RewritePatternSet &patterns) {
  addNoNanCompareSimplify(patterns, getParameter(), *getContext(),
                          PatternBenefit(getBenefit().value_or(1)));
}
void ApplyNoNanSelfSubSimplify::populatePatterns(RewritePatternSet &patterns) {
  addNoNanSelfSubSimplify(patterns, getParameter(), *getContext(),
                          PatternBenefit(getBenefit().value_or(1)));
}
void ApplyNoNanAddSubSimplify::populatePatterns(RewritePatternSet &patterns) {
  addNoNanAddSubSimplify(patterns, getParameter(), *getContext(),
                         PatternBenefit(getBenefit().value_or(1)));
}
void ApplyNoNanMulSimplify::populatePatterns(RewritePatternSet &patterns) {
  addNoNanMulSimplify(patterns, getParameter(), *getContext(),
                      PatternBenefit(getBenefit().value_or(1)));
}
void ApplyNoNanDivSimplify::populatePatterns(RewritePatternSet &patterns) {
  addNoNanDivSimplify(patterns, getParameter(), *getContext(),
                      PatternBenefit(getBenefit().value_or(1)));
}
void ApplyNoNanZeroBasePowSimplify::populatePatterns(
    RewritePatternSet &patterns) {
  addNoNanZeroBasePowSimplify(patterns, getParameter(), *getContext(),
                              PatternBenefit(getBenefit().value_or(1)));
}
void ApplySelfSubtractToConvolutionLikePatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addSelfSubtractToConvolutionLike(patterns, getParameter(), *getContext(),
                                   PatternBenefit(getBenefit().value_or(1)));
}
void ApplySelfAddToConvolutionLikePatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addSelfAddToConvolutionLike(patterns, getParameter(), *getContext(),
                              PatternBenefit(getBenefit().value_or(1)));
}
void ApplySelfMulToConvolutionLikePatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addSelfMulToConvolutionLike(patterns, getParameter(), *getContext(),
                              PatternBenefit(getBenefit().value_or(1)));
}
void ApplyWhileSimplifyPatterns::populatePatterns(RewritePatternSet &patterns) {
  addWhileSimplify(patterns, getParameter(), *getContext(),
                   PatternBenefit(getBenefit().value_or(0)));
}
void ApplyWhileLICMPatterns::populatePatterns(RewritePatternSet &patterns) {
  addWhileLICM(patterns, getParameter(), *getContext(),
               PatternBenefit(getBenefit().value_or(0)));
}
void ApplySliceLICMPatterns::populatePatterns(RewritePatternSet &patterns) {
  addSliceLICM(patterns, getParameter(), *getContext(),
               PatternBenefit(getBenefit().value_or(1)));
}
void ApplyDUSLICMPatterns::populatePatterns(RewritePatternSet &patterns) {
  addDUSLICM(patterns, getParameter(), *getContext(),
             PatternBenefit(getBenefit().value_or(1)));
}
void ApplyPadLICMPatterns::populatePatterns(RewritePatternSet &patterns) {
  addPadLICM(patterns, getParameter(), *getContext(),
             PatternBenefit(getBenefit().value_or(1)));
}
void ApplyElementwiseLICMPatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addElementwiseLICM(patterns, getParameter(), *getContext(),
                     PatternBenefit(getBenefit().value_or(1)));
}
void ApplyConcatenateLICMPatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addConcatenateLICM(patterns, getParameter(), *getContext(),
                     PatternBenefit(getBenefit().value_or(1)));
}
void ApplyBroadcastInDimLICMPatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addBroadcastInDimLICM(patterns, getParameter(), *getContext(),
                        PatternBenefit(getBenefit().value_or(1)));
}
void ApplyReshapeLICMPatterns::populatePatterns(RewritePatternSet &patterns) {
  addReshapeLICM(patterns, getParameter(), *getContext(),
                 PatternBenefit(getBenefit().value_or(1)));
}
void ApplyTransposeLICMPatterns::populatePatterns(RewritePatternSet &patterns) {
  addTransposeLICM(patterns, getParameter(), *getContext(),
                   PatternBenefit(getBenefit().value_or(1)));
}
void ApplyIotaSimplifyPatterns::populatePatterns(RewritePatternSet &patterns) {
  addIotaSimplify(patterns, getParameter(), *getContext(),
                  PatternBenefit(getBenefit().value_or(1)));
}
void ApplyConcatConstPropPatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addConcatConstProp(patterns, getParameter(), *getContext(),
                     PatternBenefit(getBenefit().value_or(1)));
}
void ApplyScatterConstFoldPatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addScatterConstFold(patterns, getParameter(), *getContext(),
                      PatternBenefit(getBenefit().value_or(1)));
}
void ApplyPadSimplifyPatterns::populatePatterns(RewritePatternSet &patterns) {
  addPadSimplify(patterns, getParameter(), *getContext(),
                 PatternBenefit(getBenefit().value_or(1)));
}
void ApplyDynamicUpdateSliceConstPropPatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addDynamicUpdateSliceConstProp(patterns, getParameter(), *getContext(),
                                 PatternBenefit(getBenefit().value_or(1)));
}
void ApplyBroadcastInDimSimplifyPatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addBroadcastInDimSimplify(patterns, getParameter(), *getContext(),
                            PatternBenefit(getBenefit().value_or(1)));
}
void ConcatenateOpCanonPatterns::populatePatterns(RewritePatternSet &patterns) {
  addConcatenateOpCanon(patterns, getParameter(), *getContext(),
                        PatternBenefit(getBenefit().value_or(1)));
}
void SelectOpCanonPatterns::populatePatterns(RewritePatternSet &patterns) {
  addSelectOpCanon(patterns, getParameter(), *getContext(),
                   PatternBenefit(getBenefit().value_or(1)));
}
void ApplyTransposeElementwisePatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addTransposeElementwise(patterns, getParameter(), *getContext(),
                          PatternBenefit(getBenefit().value_or(1)));
}
void ApplyReshapeElementwisePatterns::populatePatterns(
    RewritePatternSet &patterns) {
  addReshapeElementwise(patterns, getParameter(), *getContext(),
                        PatternBenefit(getBenefit().value_or(1)));
}
void ApplyReshapeSlicePatterns::populatePatterns(RewritePatternSet &patterns) {
  addReshapeSlice(patterns, getParameter(), *getContext(),
                  PatternBenefit(getBenefit().value_or(1)));
}
void ApplySumToConvPatterns::populatePatterns(RewritePatternSet &patterns) {
  addSumToConv(patterns, getParameter(), *getContext(),
               PatternBenefit(getBenefit().value_or(0)));
}
void ExtendUnaryElementwise::populatePatterns(RewritePatternSet &patterns) {
  addExtendUnaryElementwise(patterns, getParameter(), *getContext(),
                            PatternBenefit(getBenefit().value_or(0)));
}
void WrapUnaryElementwise::populatePatterns(RewritePatternSet &patterns) {
  addWrapUnaryElementwise(patterns, getParameter(), *getContext(),
                          PatternBenefit(getBenefit().value_or(0)));
}

} // namespace transform
} // namespace mlir

namespace {
class EnzymeJaxTransformExtension
    : public transform::TransformDialectExtension<EnzymeJaxTransformExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnzymeJaxTransformExtension)
  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/TransformOps/TransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::enzyme::registerEnzymeJaxTransformExtension(
    DialectRegistry &registry) {
  registry.addExtensions<EnzymeJaxTransformExtension>();
}

template <typename... OpType> static SmallVector<StringRef> extractNames() {
  return {OpType::getOperationName()...};
}

SmallVector<StringRef> mlir::enzyme::getTransformOperationNames() {
  return extractNames<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/TransformOps/TransformOps.cpp.inc"
      >();
}
