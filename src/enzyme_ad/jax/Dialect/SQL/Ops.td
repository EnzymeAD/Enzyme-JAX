//===- Ops.td - SQL dialect operations ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_DIALECT_SQL_OPS_TD
#define ENZYME_AD_JAX_DIALECT_SQL_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "Types.td"

class SQLOp<string mnemonic, list<Trait> traits = []>
    : Op<SQLDialect, mnemonic, traits>;

// Keep imported handles in their original representation. In particular, an
// LLVM pointer must not travel through index or a hard-coded integer width.
def SQLHandle : AnyTypeOf<[Index, LLVM_AnyPointer, AnyMemRef]>;
def SQLCount : AnyTypeOf<[Index, AnySignlessInteger]>;

def IntOp : SQLOp<"int", [Pure]> {
  let summary = "Integer literal represented as SQL text";
  let arguments = (ins StrAttr:$expr);
  let results = (outs SQLExprType:$result);
}

def ColumnOp : SQLOp<"column", [Pure]> {
  let summary = "Column reference";
  let arguments = (ins StrAttr:$expr);
  let results = (outs SQLExprType:$result);
}

def WhereOp : SQLOp<"where", [Pure]> {
  let summary = "WHERE clause wrapping a predicate";
  let arguments = (ins SQLBoolType:$expr);
  let results = (outs SQLExprType:$result);
}

def BoolArithOp : SQLOp<"bool_arith", [Pure]> {
  let summary = "Comparison of SQL expressions";
  let arguments = (ins SQLExprType:$left, SQLExprType:$right, StrAttr:$op);
  let results = (outs SQLBoolType:$result);
}

def CalcBoolOp : SQLOp<"calc_bool", [Pure]> {
  let summary = "Predicate represented as SQL text";
  let arguments = (ins StrAttr:$expr);
  let results = (outs SQLBoolType:$result);
}

def ArithOp : SQLOp<"arith", [Pure]> {
  let summary = "Arithmetic on SQL expressions";
  let arguments = (ins SQLExprType:$left, SQLExprType:$right, StrAttr:$op);
  let results = (outs SQLExprType:$result);
}

def AndOp : SQLOp<"and", [Pure]> {
  let summary = "Conjunction of SQL predicates";
  let arguments = (ins SQLBoolType:$left, SQLBoolType:$right);
  let results = (outs SQLBoolType:$result);
}

def OrOp : SQLOp<"or", [Pure]> {
  let summary = "Disjunction of SQL predicates";
  let arguments = (ins SQLBoolType:$left, SQLBoolType:$right);
  let results = (outs SQLBoolType:$result);
}

def TableOp : SQLOp<"table", [Pure]> {
  let summary = "Table reference";
  let arguments = (ins StrAttr:$expr);
  let results = (outs SQLExprType:$result);
}

def SQLConstantStringOp : SQLOp<"str_constant", [Pure]> {
  let summary = "Constant query string";
  let arguments = (ins StrAttr:$input);
  let results = (outs AnyType:$result);
}

def SQLToStringOp : SQLOp<"to_string", [Pure]> {
  let summary = "Serialize an SQL expression";
  let arguments = (ins SQLExprType:$expr);
  let results = (outs AnyType:$result);
}

def SQLBoolToStringOp : SQLOp<"bool_to_string", [Pure]> {
  let summary = "Serialize an SQL predicate";
  let arguments = (ins SQLBoolType:$expr);
  let results = (outs AnyType:$result);
}

def SQLStringConcatOp : SQLOp<"string_concat", [Pure]> {
  let summary = "Concatenate query strings";
  let arguments = (ins Variadic<AnyType>:$expr);
  let results = (outs AnyType:$result);
}

def ConstantBoolOp : SQLOp<"constant_bool", [Pure]> {
  let summary = "True SQL predicate (also used for an absent WHERE clause)";
  let results = (outs SQLBoolType:$result);
}

def SelectOp : SQLOp<"select", [Pure]> {
  let summary = "SELECT query";
  // A negative limit represents an unbounded query.
  let arguments = (ins Variadic<SQLExprType>:$columns,
                       SQLExprType:$table, SQLExprType:$where, SI64Attr:$limit);
  let results = (outs SQLExprType:$result);
}

def AllColumnsOp : SQLOp<"all_columns", [Pure]> {
  let summary = "All columns in a projection";
  let results = (outs SQLExprType:$result);
}

def UnparsedOp : SQLOp<"unparsed", [Pure]> {
  let summary = "Query string awaiting SQL parsing";
  let arguments = (ins AnyType:$input);
  let results = (outs SQLExprType:$result);
}

def ExecuteOp : SQLOp<"execute"> {
  let summary = "Execute a query on a connection";
  let arguments = (ins SQLHandle:$conn, SQLExprType:$command);
  let results = (outs SQLHandle:$result);
}

def NumResultsOp : SQLOp<"num_results"> {
  let summary = "Number of rows in a query result";
  let arguments = (ins SQLHandle:$handle);
  let results = (outs SQLCount:$result);
}

def GetValueOp : SQLOp<"get_value"> {
  let summary = "Read a query result value";
  // Preserve the reference dialect's operand order.
  let arguments = (ins SQLHandle:$handle, SQLCount:$column, SQLCount:$row);
  let results = (outs AnyType:$result);
}

#endif // ENZYME_AD_JAX_DIALECT_SQL_OPS_TD
