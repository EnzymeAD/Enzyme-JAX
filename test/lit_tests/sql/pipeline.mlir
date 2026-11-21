// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Run Reactant's production LLVM importer, default pipeline, and exporter.
// No independent SQL pass flags or replacement pipelines are used here.
// RUN: %S/../reactant-pipeline-driver cpu %S/Inputs/pipeline.ll > %t.cpu.ll
// RUN: FileCheck %s --input-file=%t.cpu.ll --implicit-check-not=sql. --implicit-check-not=ptrtoint --implicit-check-not=inttoptr
// RUN: FileCheck %s --check-prefix=DECL --input-file=%t.cpu.ll
// RUN: %S/../reactant-pipeline-driver xla %S/Inputs/pipeline.ll > %t.xla.ll
// RUN: FileCheck %s --input-file=%t.xla.ll --implicit-check-not=sql. --implicit-check-not=ptrtoint --implicit-check-not=inttoptr
// RUN: FileCheck %s --check-prefix=DECL --input-file=%t.xla.ll
// RUN: %S/../reactant-pipeline-driver cpu %S/Inputs/pipeline.ll --mlir | FileCheck %s --check-prefix=MLIR --implicit-check-not='"sql.' --implicit-check-not='!sql.' --implicit-check-not=sql.source_call

// Reformatting and full parentheses prove that the constant query passed
// through structured SQL IR. The original source global cannot satisfy this.
// Declaration alignment and the stronger call-site alignment/dereferenceable
// attributes must both survive rematerialization in address space 5.
// CHECK: @[[$GLOBAL:__sql_query[^ ]*]] = private addrspace(5) constant [128 x i8] c"SELECT (a - (b - c)) FROM t WHERE ((x = 1) OR ((y = 2) AND (z = 3))) LIMIT 4\00
// CHECK-SAME: , align 32
// CHECK-LABEL: define ptr addrspace(1) @constant_query(
// CHECK-SAME: ptr addrspace(1) %[[CONN:[^ ,)]+]]
// CHECK: %[[HANDLE:[^ ,=]+]] = call fastcc nonnull ptr addrspace(1) @PQexec(ptr addrspace(1) %[[CONN]], ptr addrspace(5) nonnull align 32 dereferenceable(128) @[[$GLOBAL]])
// CHECK: ret ptr addrspace(1) %[[HANDLE]]

// CHECK-LABEL: define ptr addrspace(3) @dynamic_query(
// CHECK-SAME: ptr addrspace(1) %[[CONN:[^ ,)]+]], ptr addrspace(5) %[[TEXT:[^ ,)]+]], i16 %[[ROW:[^ ,)]+]], i64 %[[COLUMN:[^ ,)]+]]
// CHECK: %[[HANDLE:[^ ,=]+]] = call fastcc nonnull ptr addrspace(1) @PQexec(ptr addrspace(1) %[[CONN]], ptr addrspace(5) nonnull %[[TEXT]])
// CHECK: call i64 @PQntuples(ptr addrspace(1) %[[HANDLE]])
// CHECK: %[[VALUE:[^ ,=]+]] = call nonnull ptr addrspace(3) @PQgetvalue(ptr addrspace(1) %[[HANDLE]], i16 %[[ROW]], i64 %[[COLUMN]])
// CHECK: ret ptr addrspace(3) %[[VALUE]]

// Check declarations separately, independently of their order among definitions.
// DECL-DAG: declare fastcc nonnull ptr addrspace(1) @PQexec(ptr addrspace(1), ptr addrspace(5) align 16)
// DECL-DAG: declare i64 @PQntuples(ptr addrspace(1))
// DECL-DAG: declare nonnull ptr addrspace(3) @PQgetvalue(ptr addrspace(1), i16, i64)

// MLIR: llvm.mlir.global private constant @__sql_query
// MLIR-SAME: SELECT (a - (b - c)) FROM t WHERE ((x = 1) OR ((y = 2) AND (z = 3))) LIMIT 4
// MLIR-LABEL: llvm.func @constant_query(
// MLIR: llvm.call fastcc @PQexec
// MLIR-LABEL: llvm.func @dynamic_query(
// MLIR: llvm.call fastcc @PQexec
// MLIR: llvm.call @PQntuples
// MLIR: llvm.call @PQgetvalue
