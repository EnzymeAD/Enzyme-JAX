// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s --sql-raising | FileCheck %s
// RUN: enzymexlamlir-opt %s --sql-raising --sql-raising | FileCheck %s

// CHECK-LABEL: func.func @precedence()
// CHECK: %[[A:.*]] = "sql.column"() <{expr = "a"}>
// CHECK: %[[TWO:.*]] = "sql.int"() <{expr = "2"}>
// CHECK: %[[THREE:.*]] = "sql.int"() <{expr = "3"}>
// CHECK: %[[MUL:.*]] = "sql.arith"(%[[TWO]], %[[THREE]]) <{op = "*"}>
// CHECK: "sql.arith"(%[[A]], %[[MUL]]) <{op = "+"}>
// CHECK: "sql.table"() <{expr = "schema.t"}>
// CHECK: %[[X:.*]] = "sql.bool_arith"({{.*}}) <{op = "="}>
// CHECK: "sql.int"() <{expr = "-2"}>
// CHECK: %[[Y:.*]] = "sql.bool_arith"({{.*}}) <{op = ">="}>
// CHECK: %[[Z:.*]] = "sql.bool_arith"({{.*}}) <{op = "<>"}>
// CHECK: %[[AND:.*]] = "sql.and"(%[[Y]], %[[Z]])
// CHECK: %[[OR:.*]] = "sql.or"(%[[X]], %[[AND]])
// CHECK: "sql.where"(%[[OR]])
// CHECK: %[[QUERY:.*]] = "sql.select"({{.*}}) <{limit = 9 : si64}>
// CHECK: return %[[QUERY]] : !sql.expr
func.func @precedence() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "  seLect a + 2 * 3 FROM schema.t WHERE x = 1 OR y >= -2 AND z <> 4 LIMIT 9; -- done"}> : () -> memref<?xi8>
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// CHECK-LABEL: func.func @no_clauses()
// CHECK: %[[VALUE:.*]] = "sql.int"() <{expr = "42"}>
// CHECK: %[[TABLE:.*]] = "sql.table"() <{expr = ""}>
// CHECK: %[[TRUE:.*]] = "sql.constant_bool"()
// CHECK: %[[WHERE:.*]] = "sql.where"(%[[TRUE]])
// CHECK: "sql.select"(%[[VALUE]], %[[TABLE]], %[[WHERE]]) <{limit = -1 : si64}>
func.func @no_clauses() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT 42"}> : () -> memref<?xi8>
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// CHECK-LABEL: func.func @parentheses_and_identifiers()
// CHECK: "sql.column"() <{expr = "selectivity"}>
// CHECK: "sql.column"() <{expr = "b"}>
// CHECK: "sql.table"() <{expr = "orders"}>
// CHECK: %[[OR:.*]] = "sql.or"({{.*}})
// CHECK: "sql.and"(%[[OR]], %{{.*}})
// CHECK: "sql.select"({{.*}}) <{limit = -1 : si64}>
func.func @parentheses_and_identifiers() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT selectivity, b FROM orders WHERE (x=1 OR y=2) AND z=3 LIMIT ALL"}> : () -> memref<?xi8>
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// CHECK-LABEL: func.func @quoted_and_comments()
// CHECK: "sql.column"() <{expr = "\22a\22\22b\22"}>
// CHECK: "sql.table"() <{expr = "\22S\22.\22T\22"}>
// CHECK: "sql.calc_bool"() <{expr = "FALSE"}>
func.func @quoted_and_comments() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "/* comment */ SELECT \22a\22\22b\22 FROM \22S\22.\22T\22 -- line\0A WHERE FALSE"}> : () -> memref<?xi8>
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// CHECK-LABEL: func.func @star_and_integer_bounds()
// CHECK: "sql.all_columns"()
// CHECK: "sql.int"() <{expr = "-9223372036854775808"}>
// CHECK: "sql.select"({{.*}}) <{limit = 9223372036854775807 : si64}>
func.func @star_and_integer_bounds() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT *, -9223372036854775808 FROM t LIMIT 9223372036854775807"}> : () -> memref<?xi8>
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}
