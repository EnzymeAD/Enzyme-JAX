// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s --sql-raising --split-input-file --verify-diagnostics

llvm.func @PQexec(!llvm.ptr, !llvm.ptr) -> i64
llvm.func @wrong_exec_result(%conn: !llvm.ptr, %text: !llvm.ptr) -> i64 {
  // expected-error @+1 {{unsupported PQexec signature}}
  %result = llvm.call @PQexec(%conn, %text) : (!llvm.ptr, !llvm.ptr) -> i64
  llvm.return %result : i64
}

// -----

llvm.func @PQntuples(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @wrong_argument_count(%handle: !llvm.ptr) -> i32 {
  // expected-error @+1 {{unsupported PQntuples signature}}
  %result = llvm.call @PQntuples(%handle, %handle) : (!llvm.ptr, !llvm.ptr) -> i32
  llvm.return %result : i32
}

// -----

llvm.func @PQgetvalue(!llvm.ptr, i32, i32) -> i32
llvm.func @numeric_getvalue(%handle: !llvm.ptr, %row: i32, %col: i32) -> i32 {
  // expected-error @+1 {{unsupported PQgetvalue signature}}
  %result = llvm.call @PQgetvalue(%handle, %row, %col) : (!llvm.ptr, i32, i32) -> i32
  llvm.return %result : i32
}

// -----

func.func private @PQgetvalue(memref<?xi8>, f32, i32) -> memref<?xi8>
func.func @floating_row(%handle: memref<?xi8>, %row: f32, %col: i32) -> memref<?xi8> {
  // expected-error @+1 {{unsupported PQgetvalue signature}}
  %result = func.call @PQgetvalue(%handle, %row, %col) : (memref<?xi8>, f32, i32) -> memref<?xi8>
  return %result : memref<?xi8>
}

// -----

llvm.mlir.global private constant @bad_query("SELECT FROM t\00")
llvm.func @PQexec(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
llvm.func @malformed_call_query(%conn: !llvm.ptr) -> !llvm.ptr {
  %text = llvm.mlir.addressof @bad_query : !llvm.ptr
  // expected-error @+1 {{expected an identifier}}
  %result = llvm.call @PQexec(%conn, %text) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.return %result : !llvm.ptr
}
