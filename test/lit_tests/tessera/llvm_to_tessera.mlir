// RUN: enzymexlamlir-opt %s -llvm-to-tessera -split-input-file | FileCheck %s

llvm.func @simple_func() attributes {tessera_op = "tessera_simple_func()"} {
  llvm.return
}

// CHECK: tessera.define @tessera_simple_func()
// CHECK-SAME: byRefTypes = []
// CHECK-SAME: pure = false
// CHECK-SAME: tessera.original_name = "simple_func"
// CHECK-NEXT: tessera.return

// -----


llvm.func @func_with_args(%arg0: i32, %arg1: f32) -> i32 attributes {tessera_op = "tessera_func_with_args(x, y)"} {
  llvm.return %arg0 : i32
}

// CHECK: tessera.define @tessera_func_with_args(%arg0: i32, %arg1: f32)
// CHECK-SAME: byRefTypes = [unit, unit]
// CHECK-SAME: pure = false
// CHECK-SAME: tessera.original_name = "func_with_args"
// CHECK-NEXT: tessera.return %arg0 : i32

// -----

llvm.mlir.global internal constant @_ZL26__tessera_byref_arg_type_0() : !llvm.struct<(f32)>

llvm.func @pure_func(%arg0: i32, %arg1: !llvm.ptr) -> i32 attributes {pure_tessera_op = "tessera_pure_func(x, y:byref):globals=0"} {
  llvm.return %arg0 : i32
}

// CHECK: tessera.define @tessera_pure_func(%arg0: i32, %arg1: !llvm.ptr)
// CHECK-SAME: byRefTypes = [unit, !llvm.struct<(f32)>]
// CHECK-SAME: pure = true
// CHECK-SAME: tessera.original_name = "pure_func"
// CHECK-NEXT: tessera.return %arg0 : i32

// -----

llvm.func @helper() attributes {tessera_op = "tessera_helper()"} {
  llvm.return
}

llvm.func @func_with_call() attributes {tessera_op = "tessera_func_with_call()"} {
  llvm.call @helper() : () -> ()
  llvm.return
}

// CHECK: tessera.define @tessera_helper()
// CHECK-SAME: byRefTypes = []
// CHECK-SAME: pure = false
// CHECK-SAME: tessera.original_name = "helper"
// CHECK-NEXT: tessera.return

// CHECK: tessera.define @tessera_func_with_call()
// CHECK-SAME: byRefTypes = []
// CHECK-SAME: pure = false
// CHECK-SAME: tessera.original_name = "func_with_call"
// CHECK-NEXT: tessera.call @tessera_helper()
// CHECK-NEXT: tessera.return

// -----

llvm.func @regular_function() {
  llvm.return
}

// CHECK: llvm.func @regular_function()
// CHECK-NEXT: llvm.return

// -----

llvm.func @func_with_indirect_call(%arg0: !llvm.ptr) {
  llvm.call %arg0() : !llvm.ptr, () -> ()
  llvm.return
}

// CHECK: llvm.func @func_with_indirect_call(%arg0: !llvm.ptr)
// CHECK-NEXT: llvm.call %arg0() : !llvm.ptr, () -> ()
// CHECK-NEXT: llvm.return

// -----

llvm.mlir.global internal constant @_ZL26__tessera_byref_arg_type_1() : !llvm.struct<(f32, f32)>

llvm.func @sret_func(%arg0: !llvm.ptr {llvm.sret = !llvm.struct<(f32, f32)>, llvm.align = 8 : i64, llvm.nonnull}, %arg1: !llvm.ptr {llvm.noundef, llvm.readonly}) attributes {pure_tessera_op = "tessera_sret_func(x:byref):globals=1"} {
  %0 = llvm.load %arg1 {alignment = 8 : i64} : !llvm.ptr -> f32
  llvm.store %0, %arg0 {alignment = 8 : i64} : f32, !llvm.ptr
  llvm.return
}

llvm.func @caller() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.call @sret_func(%1, %2) : (!llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}, !llvm.ptr {llvm.nonnull, llvm.noundef}) -> ()
  llvm.return
}

// CHECK: tessera.define @tessera_sret_func(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}, %arg1: !llvm.ptr {llvm.noundef, llvm.readonly})
// CHECK-SAME: byRefTypes = [!llvm.struct<(f32, f32)>]
// CHECK-SAME: pure = true
// CHECK-SAME: tessera.original_name = "sret_func"
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %arg1 {alignment = 8 : i64} : !llvm.ptr -> f32
// CHECK-NEXT: llvm.store %[[LOAD]], %arg0 {alignment = 8 : i64} : f32, !llvm.ptr
// CHECK-NEXT: tessera.return

// CHECK: llvm.func @caller()
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A1:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[A2:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[A2]] : !llvm.ptr -> !llvm.struct<(f32, f32)>
// CHECK-NEXT: %[[RES:.*]] = tessera.call @tessera_sret_func(%[[LOADED]])
// CHECK-SAME: arg_attrs = [{llvm.nonnull, llvm.noundef}]
// CHECK-SAME: tessera.loaded_operands = array<i32: 0>
// CHECK-SAME: (!llvm.struct<(f32, f32)>) -> !llvm.struct<(f32, f32)>
// CHECK-NEXT: llvm.store %[[RES]], %[[A1]] : !llvm.struct<(f32, f32)>, !llvm.ptr
// CHECK-NEXT: llvm.return