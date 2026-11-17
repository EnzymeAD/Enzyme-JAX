//===- Passes.td - SQL passes ------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_SQL_PASSES_TD
#define ENZYME_AD_JAX_SQL_PASSES_TD

include "mlir/Pass/PassBase.td"

def SQLRaisingPass : Pass<"sql-raising", "ModuleOp"> {
  let summary = "Raise PostgreSQL calls and parse constant SQL queries";
  let dependentDialects = ["sql::SQLDialect"];
}

def SQLLowerPass : Pass<"sql-lower", "ModuleOp"> {
  let summary = "Serialize SQL and reconstruct PostgreSQL calls";
  let dependentDialects = [
    "sql::SQLDialect", "LLVM::LLVMDialect", "func::FuncDialect",
    "memref::MemRefDialect"
  ];
}

#endif // ENZYME_AD_JAX_SQL_PASSES_TD
