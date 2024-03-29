//===- TransformOps.h - Declarations of Transform extension -----*- C++ -*-===//
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
#include "src/enzyme_ad/jax/TransformOps/TransformOps.h.inc"

namespace mlir {
namespace enzyme {
void registerEnzymeJaxTransformExtension(mlir::DialectRegistry &registry);

SmallVector<StringRef> getTransformOperationNames();

void registerGenerateApplyPatternsPass();
void registerRemoveTransformPass();
} // namespace enzyme
} // namespace mlir
