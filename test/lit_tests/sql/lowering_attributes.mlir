// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s --sql-raising --sql-lower | FileCheck %s --implicit-check-not='"sql.' --implicit-check-not='!sql.' --implicit-check-not=sql.source_call
// RUN: enzymexlamlir-opt %s --sql-raising --sql-lower | enzymexlamlir-opt | FileCheck %s

// The input allocation is 64 bytes and aligned to 32. Serialization shortens
// its SQL text, but the call's dereferenceability/alignment promises must remain
// valid. Match all 64 output bytes, not just a prefix of the query string.
// CHECK: llvm.mlir.global private constant @[[$GLOBAL:[^ (]+]]("SELECT 1\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")
// CHECK-SAME: addr_space = 5 : i32
// CHECK-SAME: alignment = 32 : i64
// CHECK-LABEL: llvm.func @buffer_attributes(
// CHECK-SAME: %[[CONN:[^: ,()]+]]: !llvm.ptr<1>
// CHECK: %[[QUERY:[^: ,()]+]] = llvm.mlir.addressof @[[$GLOBAL]] : !llvm.ptr<5>
// CHECK: %[[RESULT:[^: ,()]+]] = llvm.call fastcc @PQexec(%[[CONN]], %[[QUERY]])
// CHECK-SAME: test.saved = "retain"
// CHECK-SAME: : (!llvm.ptr<1>, !llvm.ptr<5> {llvm.align = 32 : i64, llvm.dereferenceable = 64 : i64, llvm.nonnull}) -> !llvm.ptr<1>
// CHECK: llvm.return %[[RESULT]] : !llvm.ptr<1>
llvm.mlir.global private constant @input("SELECT 1                                                       \00") {addr_space = 5 : i32, alignment = 32 : i64}
llvm.func fastcc @PQexec(!llvm.ptr<1>, !llvm.ptr<5> {llvm.align = 16 : i64}) -> !llvm.ptr<1>
llvm.func @buffer_attributes(%conn: !llvm.ptr<1>) -> !llvm.ptr<1> {
  %query = llvm.mlir.addressof @input : !llvm.ptr<5>
  %result = llvm.call fastcc @PQexec(%conn, %query) {test.saved = "retain"} : (!llvm.ptr<1>, !llvm.ptr<5> {llvm.align = 32 : i64, llvm.dereferenceable = 64 : i64, llvm.nonnull}) -> !llvm.ptr<1>
  llvm.return %result : !llvm.ptr<1>
}
