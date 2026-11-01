// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s --sql-lower --split-input-file | FileCheck %s --implicit-check-not='"sql.' --implicit-check-not='!sql.' --implicit-check-not=sql.source_call
// RUN: enzymexlamlir-opt %s --sql-lower --split-input-file | enzymexlamlir-opt | FileCheck %s

llvm.func fastcc @PQexec(!llvm.ptr<3>, !llvm.ptr<5>) -> !llvm.ptr<3>
llvm.func @PQntuples(!llvm.ptr<3>) -> i64
llvm.func @PQgetvalue(!llvm.ptr<3>, i16, i64) -> !llvm.ptr<7>

// Parentheses must preserve both right-associated arithmetic trees and an OR
// nested below AND. The query pointer's space differs from the connection's.
// CHECK: llvm.mlir.global private constant @[[$GLOBAL:[^ (]+]]("SELECT (a - (b - c)), (a / (b / c)) FROM schema.t WHERE (((a > 1) OR (b < 2)) AND (c = 3)) LIMIT 7\00")
// CHECK-SAME: addr_space = 5 : i32
// CHECK-LABEL: llvm.func @structured(
// CHECK-SAME: %[[CONN:[^: ,()]+]]: !llvm.ptr<3>
// CHECK: %[[ROW:[^: ,()]+]] = llvm.mlir.constant(2 : i16)
// CHECK: %[[COL:[^: ,()]+]] = llvm.mlir.constant(7 : i64)
// CHECK: %[[TEXT:[^: ,()]+]] = llvm.mlir.addressof @[[$GLOBAL]] : !llvm.ptr<5>
// CHECK: %[[HANDLE:[^: ,()]+]] = llvm.call fastcc @PQexec(%[[CONN]], %[[TEXT]]) : (!llvm.ptr<3>, !llvm.ptr<5>) -> !llvm.ptr<3>
// CHECK: llvm.call @PQntuples(%[[HANDLE]]) : (!llvm.ptr<3>) -> i64
// CHECK: %[[VALUE:[^: ,()]+]] = llvm.call @PQgetvalue(%[[HANDLE]], %[[ROW]], %[[COL]]) : (!llvm.ptr<3>, i16, i64) -> !llvm.ptr<7>
// CHECK: llvm.return %[[VALUE]] : !llvm.ptr<7>
llvm.func @structured(%conn: !llvm.ptr<3>) -> !llvm.ptr<7> {
  %row = llvm.mlir.constant(2 : i16) : i16
  %col = llvm.mlir.constant(7 : i64) : i64
  %a = "sql.column"() <{expr = "a"}> : () -> !sql.expr
  %b = "sql.column"() <{expr = "b"}> : () -> !sql.expr
  %c = "sql.column"() <{expr = "c"}> : () -> !sql.expr
  %bc = "sql.arith"(%b, %c) <{op = "-"}> : (!sql.expr, !sql.expr) -> !sql.expr
  %sub = "sql.arith"(%a, %bc) <{op = "-"}> : (!sql.expr, !sql.expr) -> !sql.expr
  %quot = "sql.arith"(%b, %c) <{op = "/"}> : (!sql.expr, !sql.expr) -> !sql.expr
  %div = "sql.arith"(%a, %quot) <{op = "/"}> : (!sql.expr, !sql.expr) -> !sql.expr
  %one = "sql.int"() <{expr = "1"}> : () -> !sql.expr
  %two = "sql.int"() <{expr = "2"}> : () -> !sql.expr
  %three = "sql.int"() <{expr = "3"}> : () -> !sql.expr
  %p = "sql.bool_arith"(%a, %one) <{op = ">"}> : (!sql.expr, !sql.expr) -> !sql.bool
  %q = "sql.bool_arith"(%b, %two) <{op = "<"}> : (!sql.expr, !sql.expr) -> !sql.bool
  %r = "sql.bool_arith"(%c, %three) <{op = "="}> : (!sql.expr, !sql.expr) -> !sql.bool
  %or = "sql.or"(%p, %q) : (!sql.bool, !sql.bool) -> !sql.bool
  %and = "sql.and"(%or, %r) : (!sql.bool, !sql.bool) -> !sql.bool
  %where = "sql.where"(%and) : (!sql.bool) -> !sql.expr
  %table = "sql.table"() <{expr = "schema.t"}> : () -> !sql.expr
  %query = "sql.select"(%sub, %div, %table, %where) <{limit = 7 : si64}> : (!sql.expr, !sql.expr, !sql.expr, !sql.expr) -> !sql.expr
  %handle = "sql.execute"(%conn, %query) : (!llvm.ptr<3>, !sql.expr) -> !llvm.ptr<3>
  %count = "sql.num_results"(%handle) : (!llvm.ptr<3>) -> i64
  %value = "sql.get_value"(%handle, %col, %row) : (!llvm.ptr<3>, i64, i16) -> !llvm.ptr<7>
  llvm.return %value : !llvm.ptr<7>
}

// -----

// No spurious FROM, WHERE, or LIMIT is emitted for absent clauses.
// CHECK: llvm.mlir.global private constant @[[$GLOBAL:[^ (]+]]("SELECT 42\00")
// CHECK-LABEL: func.func @no_clauses()
// CHECK: %[[TEXT:[^: ,()]+]] = llvm.mlir.addressof @[[$GLOBAL]] : !llvm.ptr
// CHECK: return %[[TEXT]] : !llvm.ptr
func.func @no_clauses() -> !llvm.ptr {
  %value = "sql.int"() <{expr = "42"}> : () -> !sql.expr
  %table = "sql.table"() <{expr = ""}> : () -> !sql.expr
  %true = "sql.constant_bool"() : () -> !sql.bool
  %where = "sql.where"(%true) : (!sql.bool) -> !sql.expr
  %query = "sql.select"(%value, %table, %where) <{limit = -1 : si64}> : (!sql.expr, !sql.expr, !sql.expr) -> !sql.expr
  %text = "sql.to_string"(%query) : (!sql.expr) -> !llvm.ptr
  return %text : !llvm.ptr
}

// -----

// Constant string support also covers Boolean serialization and concatenation.
// CHECK: memref.global "private" constant @[[$GLOBAL:[^ :]+]] : memref<13xi8>
// CHECK-LABEL: func.func @string_support()
// CHECK: %[[BUFFER:[^: ,()]+]] = memref.get_global @[[$GLOBAL]] : memref<13xi8>
// CHECK: %[[TEXT:[^: ,()]+]] = memref.cast %[[BUFFER]] : memref<13xi8> to memref<?xi8>
// CHECK: return %[[TEXT]] : memref<?xi8>
func.func @string_support() -> memref<?xi8> {
  %prefix = "sql.str_constant"() <{input = "WHERE "}> : () -> memref<?xi8>
  %true = "sql.constant_bool"() : () -> !sql.bool
  %predicate = "sql.calc_bool"() <{expr = "TRUE"}> : () -> !sql.bool
  %bool = "sql.bool_to_string"(%predicate) : (!sql.bool) -> memref<?xi8>
  %result = "sql.string_concat"(%prefix, %bool) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
  return %result : memref<?xi8>
}

// -----

// Existing declarations provide the ABI when there is no source_call metadata.
func.func private @PQexec(memref<?xi8>, memref<?xi8>) -> memref<?xi8>
// CHECK: memref.global "private" constant @[[$GLOBAL:[^ :]+]] : memref<9xi8>
// CHECK-LABEL: func.func @func_constant(
// CHECK-SAME: %[[CONN:[^: ,()]+]]: memref<?xi8>
// CHECK: %[[BUFFER:[^: ,()]+]] = memref.get_global @[[$GLOBAL]] : memref<9xi8>
// CHECK: %[[TEXT:[^: ,()]+]] = memref.cast %[[BUFFER]] : memref<9xi8> to memref<?xi8>
// CHECK: %[[HANDLE:[^: ,()]+]] = {{(func\.)?}}call @PQexec(%[[CONN]], %[[TEXT]]) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
// CHECK: return %[[HANDLE]] : memref<?xi8>
func.func @func_constant(%conn: memref<?xi8>) -> memref<?xi8> {
  %value = "sql.int"() <{expr = "1"}> : () -> !sql.expr
  %table = "sql.table"() <{expr = ""}> : () -> !sql.expr
  %true = "sql.constant_bool"() : () -> !sql.bool
  %where = "sql.where"(%true) : (!sql.bool) -> !sql.expr
  %query = "sql.select"(%value, %table, %where) <{limit = -1 : si64}> : (!sql.expr, !sql.expr, !sql.expr) -> !sql.expr
  %handle = "sql.execute"(%conn, %query) : (memref<?xi8>, !sql.expr) -> memref<?xi8>
  return %handle : memref<?xi8>
}

// -----

// CHECK: llvm.mlir.global private constant @[[$GLOBAL:[^ (]+]]("((a + (b * c)) % 2)\00")
// CHECK-LABEL: func.func @arithmetic()
// CHECK: %[[TEXT:[^: ,()]+]] = llvm.mlir.addressof @[[$GLOBAL]] : !llvm.ptr
// CHECK: return %[[TEXT]] : !llvm.ptr
func.func @arithmetic() -> !llvm.ptr {
  %a = "sql.column"() <{expr = "a"}> : () -> !sql.expr
  %b = "sql.column"() <{expr = "b"}> : () -> !sql.expr
  %c = "sql.column"() <{expr = "c"}> : () -> !sql.expr
  %two = "sql.int"() <{expr = "2"}> : () -> !sql.expr
  %mul = "sql.arith"(%b, %c) <{op = "*"}> : (!sql.expr, !sql.expr) -> !sql.expr
  %add = "sql.arith"(%a, %mul) <{op = "+"}> : (!sql.expr, !sql.expr) -> !sql.expr
  %mod = "sql.arith"(%add, %two) <{op = "%"}> : (!sql.expr, !sql.expr) -> !sql.expr
  %text = "sql.to_string"(%mod) : (!sql.expr) -> !llvm.ptr
  return %text : !llvm.ptr
}

// -----

llvm.func @PQexec(!llvm.ptr<1>, !llvm.ptr<2>) -> !llvm.ptr<1>
// CHECK-LABEL: llvm.func @raw_query(
// CHECK-SAME: %[[CONN:[^: ,()]+]]: !llvm.ptr<1>, %[[TEXT:[^: ,()]+]]: !llvm.ptr<2>
// CHECK: %[[RESULT:[^: ,()]+]] = llvm.call @PQexec(%[[CONN]], %[[TEXT]]) : (!llvm.ptr<1>, !llvm.ptr<2>) -> !llvm.ptr<1>
// CHECK: llvm.return %[[RESULT]] : !llvm.ptr<1>
llvm.func @raw_query(%conn: !llvm.ptr<1>, %text: !llvm.ptr<2>) -> !llvm.ptr<1> {
  %query = "sql.unparsed"(%text) : (!llvm.ptr<2>) -> !sql.expr
  %result = "sql.execute"(%conn, %query) : (!llvm.ptr<1>, !sql.expr) -> !llvm.ptr<1>
  llvm.return %result : !llvm.ptr<1>
}

// -----

// CHECK-LABEL: func.func @singleton_concat(
// CHECK-SAME: %[[TEXT:[^: ,()]+]]: !llvm.ptr<5>
// CHECK-NEXT: return %[[TEXT]] : !llvm.ptr<5>
func.func @singleton_concat(%text: !llvm.ptr<5>) -> !llvm.ptr<5> {
  %result = "sql.string_concat"(%text) : (!llvm.ptr<5>) -> !llvm.ptr<5>
  return %result : !llvm.ptr<5>
}
