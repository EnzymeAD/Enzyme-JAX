// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s --sql-raising --split-input-file | FileCheck %s --implicit-check-not=llvm.ptrtoint --implicit-check-not=llvm.inttoptr --implicit-check-not=arith.index_cast --implicit-check-not=itoa --implicit-check-not=atoi
// RUN: enzymexlamlir-opt %s --sql-raising --split-input-file | enzymexlamlir-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --sql-raising --canonicalize --split-input-file | FileCheck %s --check-prefix=EFFECT

// Bound SSA/operand matches so CHECK-SAME can inspect both the source-call
// metadata signature and the SQL operation's final type signature.

module attributes {llvm.data_layout = "e-p:32:32"} {
  llvm.mlir.global private constant @query("SELECT data FROM t\00")
  llvm.func fastcc @PQexec(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.func @PQntuples(!llvm.ptr) -> i32
  llvm.func @PQgetvalue(!llvm.ptr, i32, i32) -> !llvm.ptr

  // CHECK: llvm.func fastcc @PQexec(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK-LABEL: llvm.func @query32(
  // CHECK-SAME: %[[CONN:[^: ,()]+]]: !llvm.ptr
  // CHECK: %[[ROW:[^: ,()]+]] = llvm.mlir.constant(2 : i32)
  // CHECK: %[[COL:[^: ,()]+]] = llvm.mlir.constant(7 : i32)
  // CHECK: %[[QUERY:[^: ,()]+]] = "sql.select"({{[^)]*}})
  // CHECK: %[[HANDLE:[^: ,()]+]] = "sql.execute"(%[[CONN]], %[[QUERY]])
  // CHECK-SAME: sql.source_call =
  // CHECK-SAME: arg_attrs =
  // CHECK-SAME: llvm.noundef
  // CHECK-SAME: llvm.nonnull
  // CHECK-SAME: callee = @PQexec
  // CHECK-SAME: calling_convention = #llvm.cconv<fastcc>
  // CHECK-SAME: kind = "llvm"
  // CHECK-SAME: type = (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK-SAME: : (!llvm.ptr, !sql.expr) -> !llvm.ptr
  // CHECK: %[[COUNT:[^: ,()]+]] = "sql.num_results"(%[[HANDLE]])
  // CHECK-SAME: sql.source_call =
  // CHECK-SAME: callee = @PQntuples
  // CHECK-SAME: kind = "llvm"
  // CHECK-SAME: type = (!llvm.ptr) -> i32
  // CHECK-SAME: : (!llvm.ptr) -> i32
  // CHECK: %[[VALUE:[^: ,()]+]] = "sql.get_value"(%[[HANDLE]], %[[COL]], %[[ROW]])
  // CHECK-SAME: sql.source_call =
  // CHECK-SAME: callee = @PQgetvalue
  // CHECK-SAME: kind = "llvm"
  // CHECK-SAME: type = (!llvm.ptr, i32, i32) -> !llvm.ptr
  // CHECK-SAME: : (!llvm.ptr, i32, i32) -> !llvm.ptr
  // CHECK: llvm.return %[[VALUE]] : !llvm.ptr
  // EFFECT-LABEL: llvm.func @query32(
  // EFFECT: "sql.execute"
  // EFFECT: "sql.num_results"
  // EFFECT: "sql.get_value"
  llvm.func @query32(%conn: !llvm.ptr) -> !llvm.ptr {
    %row = llvm.mlir.constant(2 : i32) : i32
    %col = llvm.mlir.constant(7 : i32) : i32
    %base = llvm.mlir.addressof @query : !llvm.ptr
    %query = llvm.getelementptr %base[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    %handle = llvm.call fastcc @PQexec(%conn, %query) : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull}) -> !llvm.ptr
    %count = llvm.call @PQntuples(%handle) : (!llvm.ptr) -> i32
    %value = llvm.call @PQgetvalue(%handle, %row, %col) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    llvm.return %value : !llvm.ptr
  }
}

// -----

module attributes {llvm.data_layout = "e-p:64:64-p1:32:32"} {
  llvm.func @PQexec(!llvm.ptr<1>, !llvm.ptr) -> !llvm.ptr<1>
  llvm.func @PQntuples(!llvm.ptr<1>) -> i64
  llvm.func @PQgetvalue(!llvm.ptr<1>, i16, i64) -> !llvm.ptr<1>

  // CHECK-LABEL: llvm.func @dynamic_and_address_spaces(
  // CHECK-SAME: %[[CONN:[^: ,()]+]]: !llvm.ptr<1>, %[[TEXT:[^: ,()]+]]: !llvm.ptr, %[[ROW:[^: ,()]+]]: i16, %[[COL:[^: ,()]+]]: i64
  // CHECK: %[[QUERY:[^: ,()]+]] = "sql.unparsed"(%[[TEXT]]) : (!llvm.ptr) -> !sql.expr
  // CHECK: %[[HANDLE:[^: ,()]+]] = "sql.execute"(%[[CONN]], %[[QUERY]])
  // CHECK-SAME: sql.source_call =
  // CHECK-SAME: callee = @PQexec
  // CHECK-SAME: kind = "llvm"
  // CHECK-SAME: type = (!llvm.ptr<1>, !llvm.ptr) -> !llvm.ptr<1>
  // CHECK-SAME: : (!llvm.ptr<1>, !sql.expr) -> !llvm.ptr<1>
  // CHECK: %[[COUNT:[^: ,()]+]] = "sql.num_results"(%[[HANDLE]])
  // CHECK-SAME: sql.source_call =
  // CHECK-SAME: callee = @PQntuples
  // CHECK-SAME: kind = "llvm"
  // CHECK-SAME: type = (!llvm.ptr<1>) -> i64
  // CHECK-SAME: : (!llvm.ptr<1>) -> i64
  // CHECK: %[[VALUE:[^: ,()]+]] = "sql.get_value"(%[[HANDLE]], %[[COL]], %[[ROW]])
  // CHECK-SAME: sql.source_call =
  // CHECK-SAME: callee = @PQgetvalue
  // CHECK-SAME: kind = "llvm"
  // CHECK-SAME: type = (!llvm.ptr<1>, i16, i64) -> !llvm.ptr<1>
  // CHECK-SAME: : (!llvm.ptr<1>, i64, i16) -> !llvm.ptr<1>
  // CHECK: llvm.return %[[VALUE]] : !llvm.ptr<1>
  llvm.func @dynamic_and_address_spaces(%conn: !llvm.ptr<1>, %query: !llvm.ptr, %row: i16, %col: i64) -> !llvm.ptr<1> {
    %handle = llvm.call @PQexec(%conn, %query) : (!llvm.ptr<1>, !llvm.ptr) -> !llvm.ptr<1>
    %count = llvm.call @PQntuples(%handle) : (!llvm.ptr<1>) -> i64
    %value = llvm.call @PQgetvalue(%handle, %row, %col) : (!llvm.ptr<1>, i16, i64) -> !llvm.ptr<1>
    llvm.return %value : !llvm.ptr<1>
  }
}

// -----

func.func private @PQexec(memref<?xi8>, memref<?xi8>) -> memref<?xi8>
func.func private @PQntuples(memref<?xi8>) -> i64
func.func private @PQgetvalue(memref<?xi8>, i64, i32) -> memref<?xi8>

// CHECK-LABEL: func.func @frontend_memrefs(
// CHECK-SAME: %[[CONN:[^: ,()]+]]: memref<?xi8>, %[[TEXT:[^: ,()]+]]: memref<?xi8>, %[[ROW:[^: ,()]+]]: i64, %[[COL:[^: ,()]+]]: i32
// CHECK: %[[QUERY:[^: ,()]+]] = "sql.unparsed"(%[[TEXT]])
// CHECK-SAME: : (memref<?xi8>) -> !sql.expr
// CHECK: %[[HANDLE:[^: ,()]+]] = "sql.execute"(%[[CONN]], %[[QUERY]])
// CHECK-SAME: sql.source_call =
// CHECK-SAME: callee = @PQexec
// CHECK-SAME: kind = "func"
// CHECK-SAME: type = (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
// CHECK-SAME: : (memref<?xi8>, !sql.expr) -> memref<?xi8>
// CHECK: %[[COUNT:[^: ,()]+]] = "sql.num_results"(%[[HANDLE]])
// CHECK-SAME: sql.source_call =
// CHECK-SAME: callee = @PQntuples
// CHECK-SAME: kind = "func"
// CHECK-SAME: type = (memref<?xi8>) -> i64
// CHECK-SAME: : (memref<?xi8>) -> i64
// CHECK: %[[VALUE:[^: ,()]+]] = "sql.get_value"(%[[HANDLE]], %[[COL]], %[[ROW]])
// CHECK-SAME: sql.source_call =
// CHECK-SAME: callee = @PQgetvalue
// CHECK-SAME: kind = "func"
// CHECK-SAME: type = (memref<?xi8>, i64, i32) -> memref<?xi8>
// CHECK-SAME: : (memref<?xi8>, i32, i64) -> memref<?xi8>
// CHECK: return %[[COUNT]], %[[VALUE]] : i64, memref<?xi8>
func.func @frontend_memrefs(%conn: memref<?xi8>, %query: memref<?xi8>, %row: i64, %col: i32) -> (i64, memref<?xi8>) {
  %handle = func.call @PQexec(%conn, %query) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
  %count = func.call @PQntuples(%handle) : (memref<?xi8>) -> i64
  %value = func.call @PQgetvalue(%handle, %row, %col) : (memref<?xi8>, i64, i32) -> memref<?xi8>
  return %count, %value : i64, memref<?xi8>
}
