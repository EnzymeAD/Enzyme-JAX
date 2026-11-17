//===- Types.td - SQL dialect types ------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_DIALECT_SQL_TYPES_TD
#define ENZYME_AD_JAX_DIALECT_SQL_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"

class SQLType<string name, string typeMnemonic> : TypeDef<SQLDialect, name> {
  let mnemonic = typeMnemonic;
}

def SQLExprType : SQLType<"Expr", "expr"> {
  let summary = "SQL expression";
}

def SQLBoolType : SQLType<"Bool", "bool"> {
  let summary = "SQL predicate";
}

#endif // ENZYME_AD_JAX_DIALECT_SQL_TYPES_TD
