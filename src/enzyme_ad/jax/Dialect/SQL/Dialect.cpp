//===- Dialect.cpp - SQL dialect -----------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Dialect/SQL/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/enzyme_ad/jax/Dialect/SQL/SQLDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/SQL/SQLTypes.cpp.inc"

void mlir::sql::SQLDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/enzyme_ad/jax/Dialect/SQL/SQLTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/SQL/SQLOps.cpp.inc"
      >();
}
