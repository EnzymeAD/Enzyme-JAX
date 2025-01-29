//===---- RaisingTransformOps.h - Declarations of Transform extension  ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "src/enzyme_ad/jax/TransformOps/OpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/RaisingTransformOps.h.inc"
#include "src/enzyme_ad/jax/TransformOps/RaisingTransformPatterns.h.inc"

namespace mlir {
namespace enzyme {
void registerRaisingTransformExtension(mlir::DialectRegistry &registry);

} // namespace enzyme
} // namespace mlir
