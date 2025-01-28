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
void ApplyNoNanAddSubSimplify::populatePatterns(RewritePatternSet &patterns) {
  addNoNanAddSubSimplify(patterns, getParameter(), *getContext(),
                         PatternBenefit(getBenefit().value_or(1)));
}
void ApplyIotaSimplifyPatterns::populatePatterns(RewritePatternSet &patterns) {
  addIotaSimplify(patterns, getParameter(), *getContext(),
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
