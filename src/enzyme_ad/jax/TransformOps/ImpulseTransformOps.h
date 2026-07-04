//===- ImpulseTransformOps.h - Declarations of Transform extension --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformOps.h.inc"
#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformPatterns.h.inc"

namespace mlir {
namespace enzyme {
void registerImpulseTransformExtension(mlir::DialectRegistry &registry);

} // namespace enzyme
} // namespace mlir
