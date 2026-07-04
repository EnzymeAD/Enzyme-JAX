//===- ImpulseTransformOps.cpp - Definition of impulse transform extension ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformOps.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformOps.cpp.inc"
#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformOpsImpl.cpp.inc"

using namespace mlir;

namespace {
class ImpulseTransformExtension
    : public transform::TransformDialectExtension<ImpulseTransformExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImpulseTransformExtension)
  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::enzyme::registerImpulseTransformExtension(
    DialectRegistry &registry) {
  registry.addExtensions<ImpulseTransformExtension>();
}
