// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s | enzymexlamlir-opt | FileCheck %s

// These tests exercise operation assembly only; no query parsing, execution,
// serialization, or lowering pass is run.

// CHECK-LABEL: func.func @strings(
// CHECK: %[[STRING:.*]] = "sql.str_constant"() <{input = "SELECT "}> : () -> memref<?xi8>
// CHECK: %[[EXPR:.*]] = "sql.to_string"(%{{.*}}) : (!sql.expr) -> memref<?xi8>
// CHECK: %[[PRED:.*]] = "sql.bool_to_string"(%{{.*}}) : (!sql.bool) -> memref<?xi8>
// CHECK: %[[CONCAT:.*]] = "sql.string_concat"(%[[STRING]], %[[EXPR]], %[[PRED]]) : (memref<?xi8>, memref<?xi8>, memref<?xi8>) -> memref<?xi8>
// CHECK: %[[QUERY:.*]] = "sql.unparsed"(%[[CONCAT]]) : (memref<?xi8>) -> !sql.expr
// CHECK: return %[[QUERY]] : !sql.expr
func.func @strings(%expr: !sql.expr, %pred: !sql.bool) -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT "}> : () -> memref<?xi8>
  %text = "sql.to_string"(%expr) : (!sql.expr) -> memref<?xi8>
  %bool = "sql.bool_to_string"(%pred) : (!sql.bool) -> memref<?xi8>
  %concat = "sql.string_concat"(%str, %text, %bool) : (memref<?xi8>, memref<?xi8>, memref<?xi8>) -> memref<?xi8>
  %query = "sql.unparsed"(%concat) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// CHECK-LABEL: func.func @database_ops(
// CHECK: %[[HANDLE:.*]] = "sql.execute"(%{{.*}}, %{{.*}}) : (index, !sql.expr) -> index
// CHECK: %[[COUNT:.*]] = "sql.num_results"(%[[HANDLE]]) : (index) -> index
// CHECK: %[[VALUE:.*]] = "sql.get_value"(%[[HANDLE]], %{{.*}}, %{{.*}}) : (index, index, index) -> i32
// CHECK: return %[[COUNT]], %[[VALUE]] : index, i32
func.func @database_ops(%conn: index, %query: !sql.expr, %column: index, %row: index) -> (index, i32) {
  %handle = "sql.execute"(%conn, %query) : (index, !sql.expr) -> index
  %count = "sql.num_results"(%handle) : (index) -> index
  %value = "sql.get_value"(%handle, %column, %row) : (index, index, index) -> i32
  return %count, %value : index, i32
}
