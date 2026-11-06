//===- Parser.h - SQL text to SQL dialect --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_DIALECT_SQL_PARSER_H
#define ENZYME_AD_JAX_DIALECT_SQL_PARSER_H

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::sql {
// Parse one supported SELECT statement. Diagnose malformed/unsupported input
// at loc. On failure, leave the builder and its insertion block unchanged.
FailureOr<Value> parseSQL(Location loc, OpBuilder &builder, StringRef query);
} // namespace mlir::sql

#endif // ENZYME_AD_JAX_DIALECT_SQL_PARSER_H
