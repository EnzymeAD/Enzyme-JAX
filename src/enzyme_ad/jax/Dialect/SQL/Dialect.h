//===- Dialect.h - SQL dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_DIALECT_SQL_DIALECT_H
#define ENZYME_AD_JAX_DIALECT_SQL_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/enzyme_ad/jax/Dialect/SQL/SQLDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/SQL/SQLTypes.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/SQL/SQLOps.h.inc"

#endif // ENZYME_AD_JAX_DIALECT_SQL_DIALECT_H
