// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s --sql-raising --split-input-file | FileCheck %s

llvm.mlir.global private @mutable_query("not a SELECT\00")
llvm.mlir.global private constant @unterminated("SELECT 1")
llvm.mlir.global private constant @embedded_nul("SELECT 1\00unsupported suffix")
llvm.mlir.global external constant @interposable("not a SELECT\00")
llvm.func @PQexec(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
llvm.func @unrelated(!llvm.ptr) -> !llvm.ptr

// CHECK-LABEL: llvm.func @mutable_string(
// CHECK: %[[TEXT:.*]] = llvm.mlir.addressof @mutable_query
// CHECK: %[[QUERY:.*]] = "sql.unparsed"(%[[TEXT]])
// CHECK: "sql.execute"(%{{.*}}, %[[QUERY]])
llvm.func @mutable_string(%conn: !llvm.ptr) -> !llvm.ptr {
  %text = llvm.mlir.addressof @mutable_query : !llvm.ptr
  %result = llvm.call @PQexec(%conn, %text) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.return %result : !llvm.ptr
}

// CHECK-LABEL: llvm.func @no_terminator(
// CHECK: "sql.unparsed"
// CHECK: "sql.execute"
llvm.func @no_terminator(%conn: !llvm.ptr) -> !llvm.ptr {
  %text = llvm.mlir.addressof @unterminated : !llvm.ptr
  %result = llvm.call @PQexec(%conn, %text) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.return %result : !llvm.ptr
}

// An externally replaceable global is not a known query even if constant.
// CHECK-LABEL: llvm.func @interposable_string(
// CHECK: "sql.unparsed"
// CHECK: "sql.execute"
llvm.func @interposable_string(%conn: !llvm.ptr) -> !llvm.ptr {
  %text = llvm.mlir.addressof @interposable : !llvm.ptr
  %result = llvm.call @PQexec(%conn, %text) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.return %result : !llvm.ptr
}

// CHECK-LABEL: llvm.func @nonzero_gep(
// CHECK: %[[OFFSET:.*]] = llvm.getelementptr
// CHECK: "sql.unparsed"(%[[OFFSET]])
llvm.func @nonzero_gep(%conn: !llvm.ptr) -> !llvm.ptr {
  %base = llvm.mlir.addressof @embedded_nul : !llvm.ptr
  %text = llvm.getelementptr %base[1] : (!llvm.ptr) -> !llvm.ptr, i8
  %result = llvm.call @PQexec(%conn, %text) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.return %result : !llvm.ptr
}

// C-string length, rather than the full global's length, determines the query.
// CHECK-LABEL: llvm.func @embedded_terminator(
// CHECK: "sql.int"() <{expr = "1"}>
// CHECK: "sql.select"
// CHECK: "sql.execute"
llvm.func @embedded_terminator(%conn: !llvm.ptr) -> !llvm.ptr {
  %text = llvm.mlir.addressof @embedded_nul : !llvm.ptr
  %result = llvm.call @PQexec(%conn, %text) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.return %result : !llvm.ptr
}

// CHECK-LABEL: llvm.func @indirect_and_unrelated(
// CHECK: llvm.call %{{.*}}
// CHECK: llvm.call @unrelated
llvm.func @indirect_and_unrelated(%fn: !llvm.ptr, %conn: !llvm.ptr, %text: !llvm.ptr) -> !llvm.ptr {
  %result = llvm.call %fn(%conn, %text) : !llvm.ptr, (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %other = llvm.call @unrelated(%result) : (!llvm.ptr) -> !llvm.ptr
  llvm.return %other : !llvm.ptr
}

// -----

// A same-named local implementation is not assumed to be libpq.
llvm.func @PQexec(%conn: !llvm.ptr, %text: !llvm.ptr) -> !llvm.ptr {
  llvm.return %conn : !llvm.ptr
}
// CHECK-LABEL: llvm.func @local_definition(
// CHECK: llvm.call @PQexec
llvm.func @local_definition(%conn: !llvm.ptr, %text: !llvm.ptr) -> !llvm.ptr {
  %result = llvm.call @PQexec(%conn, %text) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.return %result : !llvm.ptr
}

// -----

llvm.func @PQexec(...) -> !llvm.ptr
// CHECK-LABEL: llvm.func @variadic_declaration(
// CHECK: llvm.call @PQexec
llvm.func @variadic_declaration(%conn: !llvm.ptr, %text: !llvm.ptr) -> !llvm.ptr {
  %result = llvm.call @PQexec(%conn, %text) vararg(!llvm.func<ptr (...)>) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.return %result : !llvm.ptr
}

// -----

func.func private @PQgetvalue(!llvm.ptr, i32, i32) -> !llvm.ptr
// CHECK-LABEL: func.func @attributes(
// CHECK: "sql.get_value"
// CHECK-SAME: sql.source_call = {attributes = {
// CHECK-SAME: callee = @PQgetvalue
// CHECK-SAME: test.annotation = "retain me"
// CHECK-SAME: kind = "func"
func.func @attributes(%handle: !llvm.ptr, %row: i32, %col: i32) -> !llvm.ptr {
  %result = func.call @PQgetvalue(%handle, %row, %col) {test.annotation = "retain me"} : (!llvm.ptr, i32, i32) -> !llvm.ptr
  return %result : !llvm.ptr
}
