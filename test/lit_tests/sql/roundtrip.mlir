// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s | enzymexlamlir-opt | FileCheck %s

// Adapted from Polygeist's SQL dialect tests. Reparse the printed IR to check
// both directions of the registered dialect's assembly support.

// CHECK-LABEL: func.func @test_int()
// CHECK: %[[INT:.*]] = "sql.int"() <{expr = "42"}> : () -> !sql.expr
// CHECK: return %[[INT]] : !sql.expr
func.func @test_int() -> !sql.expr {
  %0 = "sql.int"() <{expr = "42"}> : () -> !sql.expr
  return %0 : !sql.expr
}

// CHECK-LABEL: func.func @test_arith()
// CHECK: %[[A:.*]] = "sql.column"() <{expr = "a"}> : () -> !sql.expr
// CHECK: %[[B:.*]] = "sql.column"() <{expr = "b"}> : () -> !sql.expr
// CHECK: %[[SUM:.*]] = "sql.arith"(%[[A]], %[[B]]) <{op = "+"}> : (!sql.expr, !sql.expr) -> !sql.expr
// CHECK: return %[[SUM]] : !sql.expr
func.func @test_arith() -> !sql.expr {
  %a = "sql.column"() <{expr = "a"}> : () -> !sql.expr
  %b = "sql.column"() <{expr = "b"}> : () -> !sql.expr
  %sum = "sql.arith"(%a, %b) <{op = "+"}> : (!sql.expr, !sql.expr) -> !sql.expr
  return %sum : !sql.expr
}

// CHECK-LABEL: func.func @test_boolean()
// CHECK: %[[A:.*]] = "sql.calc_bool"() <{expr = "x > 0"}> : () -> !sql.bool
// CHECK: %[[B:.*]] = "sql.calc_bool"() <{expr = "y < 5"}> : () -> !sql.bool
// CHECK: %[[AND:.*]] = "sql.and"(%[[A]], %[[B]]) : (!sql.bool, !sql.bool) -> !sql.bool
// CHECK: %[[CONSTANT:.*]] = "sql.constant_bool"() : () -> !sql.bool
// CHECK: %[[OR:.*]] = "sql.or"(%[[AND]], %[[CONSTANT]]) : (!sql.bool, !sql.bool) -> !sql.bool
// CHECK: return %[[OR]] : !sql.bool
func.func @test_boolean() -> !sql.bool {
  %a = "sql.calc_bool"() <{expr = "x > 0"}> : () -> !sql.bool
  %b = "sql.calc_bool"() <{expr = "y < 5"}> : () -> !sql.bool
  %and = "sql.and"(%a, %b) : (!sql.bool, !sql.bool) -> !sql.bool
  %constant = "sql.constant_bool"() : () -> !sql.bool
  %or = "sql.or"(%and, %constant) : (!sql.bool, !sql.bool) -> !sql.bool
  return %or : !sql.bool
}

// CHECK-LABEL: func.func @test_select()
// CHECK: %[[COL:.*]] = "sql.column"() <{expr = "data"}> : () -> !sql.expr
// CHECK: %[[TABLE:.*]] = "sql.table"() <{expr = "mytable"}> : () -> !sql.expr
// CHECK: %[[VALUE:.*]] = "sql.int"() <{expr = "0"}> : () -> !sql.expr
// CHECK: %[[PRED:.*]] = "sql.bool_arith"(%[[COL]], %[[VALUE]]) <{op = ">"}> : (!sql.expr, !sql.expr) -> !sql.bool
// CHECK: %[[WHERE:.*]] = "sql.where"(%[[PRED]]) : (!sql.bool) -> !sql.expr
// CHECK: %[[QUERY:.*]] = "sql.select"(%[[COL]], %[[TABLE]], %[[WHERE]]) <{limit = 1 : si64}> : (!sql.expr, !sql.expr, !sql.expr) -> !sql.expr
// CHECK: return %[[QUERY]] : !sql.expr
func.func @test_select() -> !sql.expr {
  %col = "sql.column"() <{expr = "data"}> : () -> !sql.expr
  %table = "sql.table"() <{expr = "mytable"}> : () -> !sql.expr
  %value = "sql.int"() <{expr = "0"}> : () -> !sql.expr
  %pred = "sql.bool_arith"(%col, %value) <{op = ">"}> : (!sql.expr, !sql.expr) -> !sql.bool
  %where = "sql.where"(%pred) : (!sql.bool) -> !sql.expr
  %query = "sql.select"(%col, %table, %where) <{limit = 1 : si64}> : (!sql.expr, !sql.expr, !sql.expr) -> !sql.expr
  return %query : !sql.expr
}

// Check that the variadic projection leaves the final table/WHERE operands
// intact, and that the unbounded limit representation round trips.
// CHECK-LABEL: func.func @test_multiple_columns()
// CHECK: %[[ALL:.*]] = "sql.all_columns"() : () -> !sql.expr
// CHECK: %[[COL:.*]] = "sql.column"() <{expr = "data"}> : () -> !sql.expr
// CHECK: %[[TABLE:.*]] = "sql.table"() <{expr = "mytable"}> : () -> !sql.expr
// CHECK: %[[PRED:.*]] = "sql.calc_bool"() <{expr = "1 = 1"}> : () -> !sql.bool
// CHECK: %[[WHERE:.*]] = "sql.where"(%[[PRED]]) : (!sql.bool) -> !sql.expr
// CHECK: %[[QUERY:.*]] = "sql.select"(%[[ALL]], %[[COL]], %[[TABLE]], %[[WHERE]]) <{limit = -1 : si64}> : (!sql.expr, !sql.expr, !sql.expr, !sql.expr) -> !sql.expr
// CHECK: return %[[QUERY]] : !sql.expr
func.func @test_multiple_columns() -> !sql.expr {
  %all = "sql.all_columns"() : () -> !sql.expr
  %col = "sql.column"() <{expr = "data"}> : () -> !sql.expr
  %table = "sql.table"() <{expr = "mytable"}> : () -> !sql.expr
  %pred = "sql.calc_bool"() <{expr = "1 = 1"}> : () -> !sql.bool
  %where = "sql.where"(%pred) : (!sql.bool) -> !sql.expr
  %query = "sql.select"(%all, %col, %table, %where) <{limit = -1 : si64}> : (!sql.expr, !sql.expr, !sql.expr, !sql.expr) -> !sql.expr
  return %query : !sql.expr
}
