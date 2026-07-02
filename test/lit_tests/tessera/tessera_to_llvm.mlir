// RUN: enzymexlamlir-opt %s -tessera-to-llvm -split-input-file | FileCheck %s

tessera.define @tessera_simple_func() attributes {byRefArgs = array<i1>, byRefTypes = [], pure = false, tessera.original_name = "simple_func"} {
  tessera.return
}

// CHECK: llvm.func @simple_func()
// CHECK-NEXT: llvm.return

// -----

tessera.define @tessera_func_with_args(%arg0: i32, %arg1: f32) -> i32 attributes {byRefArgs = array<i1: false, false>, byRefTypes = [], pure = false, tessera.original_name = "func_with_args"} {
  tessera.return %arg0 : i32
}

// CHECK: llvm.func @func_with_args(%arg0: i32, %arg1: f32) -> i32
// CHECK-NEXT: llvm.return %arg0 : i32

// -----

tessera.define @tessera_helper() attributes {byRefArgs = array<i1>, byRefTypes = [], pure = false, tessera.original_name = "helper"} {
  tessera.return
}

tessera.define @tessera_func_with_call() attributes {byRefArgs = array<i1>, byRefTypes = [], pure = false, tessera.original_name = "func_with_call"} {
  tessera.call @tessera_helper() {op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 0, 0>} : () -> ()
  tessera.return
}

// CHECK: llvm.func @helper
// CHECK-NEXT: llvm.return

// CHECK: llvm.func @func_with_call
// CHECK-NEXT: llvm.call @helper() : () -> ()
// CHECK-NEXT: llvm.return

// -----

tessera.define @tessera_sret_func(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}, %arg1: !llvm.ptr {llvm.noundef, llvm.readonly}) 
attributes {byRefArgs = array<i1: true>, byRefTypes = [!llvm.struct<(f32, f32)>], linkage = #llvm.linkage<external>, pure = true, tessera.original_name = "sret_func"} {
  %0 = llvm.load %arg1 {alignment = 8 : i64} : !llvm.ptr -> f32
  llvm.store %0, %arg0 {alignment = 8 : i64} : f32, !llvm.ptr
  tessera.return
}

llvm.func @caller() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(f32, f32)>
  %4 = tessera.call @tessera_sret_func(%3) {arg_attrs = [{llvm.nonnull, llvm.noundef}], op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>, tessera.loaded_operands = array<i32: 0>} : (!llvm.struct<(f32, f32)>) -> !llvm.struct<(f32, f32)>
  llvm.store %4, %1 : !llvm.struct<(f32, f32)>, !llvm.ptr
  llvm.return
}

// CHECK: llvm.func @sret_func(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}, %arg1: !llvm.ptr {llvm.noundef, llvm.readonly})
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %arg1 {alignment = 8 : i64} : !llvm.ptr -> f32
// CHECK-NEXT: llvm.store %[[LOAD]], %arg0 {alignment = 8 : i64} : f32, !llvm.ptr
// CHECK-NEXT: llvm.return

// CHECK: llvm.func @caller()
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A1:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[A2:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[A2]] : !llvm.ptr -> !llvm.struct<(f32, f32)>
// CHECK-NEXT: %[[SRET:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[A3:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> : (i32) -> !llvm.ptr
// CHECK-NEXT: llvm.store %[[LOADED]], %[[A3]] : !llvm.struct<(f32, f32)>, !llvm.ptr
// CHECK-NEXT: llvm.call @sret_func(%[[SRET]], %[[A3]]) : (!llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}, !llvm.ptr {llvm.nonnull, llvm.noundef}) -> ()
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[SRET]] : !llvm.ptr -> !llvm.struct<(f32, f32)>
// CHECK-NEXT: llvm.store %[[LOADED]], %[[A1]] : !llvm.struct<(f32, f32)>, !llvm.ptr
// CHECK-NEXT: llvm.return
