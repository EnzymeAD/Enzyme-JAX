// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s --sql-raising --sql-lower --split-input-file | FileCheck %s --implicit-check-not='"sql.' --implicit-check-not='!sql.' --implicit-check-not=sql.source_call --implicit-check-not=llvm.ptrtoint --implicit-check-not=llvm.inttoptr --implicit-check-not=arith.index_cast --implicit-check-not=atoi --implicit-check-not=itoa
// RUN: enzymexlamlir-opt %s --sql-raising --sql-lower --split-input-file | enzymexlamlir-opt | FileCheck %s

module attributes {llvm.data_layout = "e-p:64:64-p1:32:32-p2:32:32-p3:32:32"} {
  llvm.func fastcc @PQexec(!llvm.ptr<1>, !llvm.ptr<2>) -> !llvm.ptr<1>
  llvm.func @PQntuples(!llvm.ptr<1>) -> i64
  llvm.func @PQgetvalue(!llvm.ptr<1>, i16, i64) -> !llvm.ptr<3>

  // Exact SSA operands, not just their types, must survive the round trip.
  // CHECK: llvm.func fastcc @PQexec(!llvm.ptr<1>, !llvm.ptr<2>) -> !llvm.ptr<1>
  // CHECK-LABEL: llvm.func @dynamic(
  // CHECK-SAME: %[[CONN:[^: ,()]+]]: !llvm.ptr<1>, %[[QUERY:[^: ,()]+]]: !llvm.ptr<2>, %[[ROW:[^: ,()]+]]: i16, %[[COL:[^: ,()]+]]: i64
  // CHECK: %[[HANDLE:[^: ,()]+]] = llvm.call fastcc @PQexec(%[[CONN]], %[[QUERY]])
  // CHECK-SAME: test.saved = "exec"
  // CHECK-SAME: : (!llvm.ptr<1>, !llvm.ptr<2> {llvm.nonnull, llvm.noundef}) -> !llvm.ptr<1>
  // CHECK: llvm.call @PQntuples(%[[HANDLE]])
  // CHECK-SAME: test.saved = "count"
  // CHECK-SAME: : (!llvm.ptr<1>) -> i64
  // CHECK: %[[VALUE:[^: ,()]+]] = llvm.call @PQgetvalue(%[[HANDLE]], %[[ROW]], %[[COL]])
  // CHECK-SAME: test.saved = "value"
  // CHECK-SAME: : (!llvm.ptr<1>, i16, i64) ->
  // CHECK-SAME: !llvm.ptr<3>
  // CHECK-SAME: llvm.nonnull
  // CHECK: llvm.return %[[VALUE]] : !llvm.ptr<3>
  llvm.func @dynamic(%conn: !llvm.ptr<1>, %query: !llvm.ptr<2>, %row: i16, %col: i64) -> !llvm.ptr<3> {
    %handle = llvm.call fastcc @PQexec(%conn, %query) {test.saved = "exec"} : (!llvm.ptr<1>, !llvm.ptr<2> {llvm.nonnull, llvm.noundef}) -> !llvm.ptr<1>
    %count = llvm.call @PQntuples(%handle) {test.saved = "count"} : (!llvm.ptr<1>) -> i64
    %value = llvm.call @PQgetvalue(%handle, %row, %col) {test.saved = "value"} : (!llvm.ptr<1>, i16, i64) -> (!llvm.ptr<3> {llvm.nonnull})
    llvm.return %value : !llvm.ptr<3>
  }
}

// -----

module attributes {llvm.data_layout = "e-p:32:32-p1:32:32-p5:32:32"} {
  llvm.mlir.global private constant @input("SELECT a - (b - c) FROM t WHERE (x=1 OR y=2) AND z=3 LIMIT 4\00") {addr_space = 5 : i32}
  llvm.func fastcc @PQexec(!llvm.ptr<1>, !llvm.ptr<5>) -> !llvm.ptr<1>

  // CHECK: llvm.mlir.global private constant @[[$GLOBAL:[^ (]+]]("SELECT (a - (b - c)) FROM t WHERE (((x = 1) OR (y = 2)) AND (z = 3)) LIMIT 4\00")
  // CHECK-SAME: addr_space = 5 : i32
  // CHECK-LABEL: llvm.func @constant_query(
  // CHECK-SAME: %[[CONN:[^: ,()]+]]: !llvm.ptr<1>
  // CHECK: %[[QUERY:[^: ,()]+]] = llvm.mlir.addressof @[[$GLOBAL]] : !llvm.ptr<5>
  // CHECK: %[[HANDLE:[^: ,()]+]] = llvm.call fastcc @PQexec(%[[CONN]], %[[QUERY]])
  // CHECK-SAME: : (!llvm.ptr<1>, !llvm.ptr<5> {llvm.nonnull}) -> !llvm.ptr<1>
  // CHECK: llvm.return %[[HANDLE]] : !llvm.ptr<1>
  llvm.func @constant_query(%conn: !llvm.ptr<1>) -> !llvm.ptr<1> {
    %query = llvm.mlir.addressof @input : !llvm.ptr<5>
    %handle = llvm.call fastcc @PQexec(%conn, %query) : (!llvm.ptr<1>, !llvm.ptr<5> {llvm.nonnull}) -> !llvm.ptr<1>
    llvm.return %handle : !llvm.ptr<1>
  }
}

// -----

func.func private @PQexec(memref<?xi8>, memref<?xi8>) -> memref<?xi8>
func.func private @PQntuples(memref<?xi8>) -> i64
func.func private @PQgetvalue(memref<?xi8>, i64, i32) -> memref<?xi8>

// CHECK-LABEL: func.func @frontend(
// CHECK-SAME: %[[CONN:[^: ,()]+]]: memref<?xi8>, %[[QUERY:[^: ,()]+]]: memref<?xi8>, %[[ROW:[^: ,()]+]]: i64, %[[COL:[^: ,()]+]]: i32
// CHECK: %[[HANDLE:[^: ,()]+]] = {{(func\.)?}}call @PQexec(%[[CONN]], %[[QUERY]])
// CHECK-SAME: test.saved = "exec"
// CHECK-SAME: : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
// CHECK: %[[COUNT:[^: ,()]+]] = {{(func\.)?}}call @PQntuples(%[[HANDLE]]) : (memref<?xi8>) -> i64
// CHECK: %[[VALUE:[^: ,()]+]] = {{(func\.)?}}call @PQgetvalue(%[[HANDLE]], %[[ROW]], %[[COL]])
// CHECK-SAME: test.saved = "value"
// CHECK-SAME: : (memref<?xi8>, i64, i32) -> memref<?xi8>
// CHECK: return %[[COUNT]], %[[VALUE]] : i64, memref<?xi8>
func.func @frontend(%conn: memref<?xi8>, %query: memref<?xi8>, %row: i64, %col: i32) -> (i64, memref<?xi8>) {
  %handle = func.call @PQexec(%conn, %query) {test.saved = "exec"} : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
  %count = func.call @PQntuples(%handle) : (memref<?xi8>) -> i64
  %value = func.call @PQgetvalue(%handle, %row, %col) {test.saved = "value"} : (memref<?xi8>, i64, i32) -> memref<?xi8>
  return %count, %value : i64, memref<?xi8>
}
