//===- TransformOps.cpp - Definition of transform extension ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/TransformOps/StablehloTransformOps.h"
#include "src/enzyme_ad/jax/Passes/StablehloOptPatterns.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/StablehloTransformOps.cpp.inc"

using namespace mlir;

namespace {
class StablehloTransformExtension
    : public transform::TransformDialectExtension<StablehloTransformExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StablehloTransformExtension)
  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/TransformOps/StablehloTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::enzyme::registerStablehloTransformExtension(
    DialectRegistry &registry) {
  registry.addExtensions<StablehloTransformExtension>();
}

template <typename... OpType> static SmallVector<StringRef> extractNames() {
  return {OpType::getOperationName()...};
}

SmallVector<StringRef> mlir::enzyme::getTransformOperationNames() {
  return extractNames<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/TransformOps/StablehloTransformOps.cpp.inc"
      >();
}
