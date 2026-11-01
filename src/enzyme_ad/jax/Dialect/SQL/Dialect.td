//===- Dialect.td - SQL dialect ----------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_DIALECT_SQL_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_SQL_DIALECT_TD

include "mlir/IR/DialectBase.td"

def SQLDialect : Dialect {
  let name = "sql";
  let summary = "SQL expressions and database operations";
  let cppNamespace = "::mlir::sql";
  let useDefaultTypePrinterParser = 1;
}

#endif // ENZYME_AD_JAX_DIALECT_SQL_DIALECT_TD
