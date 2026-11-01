//===- Passes.h - SQL passes --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_SQL_PASSES_H
#define ENZYME_AD_JAX_SQL_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/SQL/Dialect.h"

namespace mlir::sql {
#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/SQL/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/SQL/Passes.h.inc"
} // namespace mlir::sql

#endif // ENZYME_AD_JAX_SQL_PASSES_H
