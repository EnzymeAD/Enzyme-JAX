// RUN: enzymexlamlir-opt %s -llvm-to-tessera | FileCheck %s

llvm.func @simple_func() attributes {tessera.convert = #tessera<convert "tessera_simple_func" byref = [] sizes = [] pure = false>} {
  llvm.return
}

// CHECK: tessera.define @tessera_simple_func()
// CHECK-SAME: tessera.convert = #tessera<convert tessera_simple_func byref = [] sizes = [] pure = false>
// CHECK-SAME: tessera.original_name = "simple_func"
// CHECK-NEXT: tessera.return

// -----


llvm.func @func_with_args(%arg0: i32, %arg1: f32) -> i32 attributes {tessera.convert = #tessera<convert "tessera_func_with_args" byref = [false, false] sizes = [4, 4] pure = false>} {
  llvm.return %arg0 : i32
}

// CHECK: tessera.define @tessera_func_with_args(%arg0: i32, %arg1: f32)
// CHECK-SAME: tessera.convert = #tessera<convert tessera_func_with_args byref = [false, false] sizes = [4, 4] pure = false>
// CHECK-SAME: tessera.original_name = "func_with_args"
// CHECK-NEXT: tessera.return %arg0 : i32

// -----

llvm.func @pure_func(%arg0: i32, %arg1: f32) -> i32 attributes {tessera.convert = #tessera<convert "tessera_pure_func" byref = [false, false] sizes = [4, 4] pure = true>} {
  llvm.return %arg0 : i32
}

// CHECK: tessera.define @tessera_pure_func(%arg0: i32, %arg1: f32)
// CHECK-SAME: tessera.convert = #tessera<convert tessera_pure_func byref = [false, false] sizes = [4, 4] pure = true>
// CHECK-SAME: tessera.original_name = "pure_func"
// CHECK-SAME: tessera.side_effect_free
// CHECK-NEXT: tessera.return %arg0 : i32

// -----

llvm.func @helper() attributes {tessera.convert = #tessera<convert "tessera_helper" byref = [] sizes = [] pure = false>} {
  llvm.return
}

llvm.func @func_with_call() attributes {tessera.convert = #tessera<convert "tessera_func_with_call" byref = [] sizes = [] pure = false>} {
  llvm.call @helper() : () -> ()
  llvm.return
}

// CHECK: tessera.define @tessera_helper()
// CHECK-SAME: tessera.convert = #tessera<convert tessera_helper byref = [] sizes = [] pure = false>
// CHECK-SAME: tessera.original_name = "helper"
// CHECK-NEXT: tessera.return

// CHECK: tessera.define @tessera_func_with_call()
// CHECK-SAME: tessera.convert = #tessera<convert tessera_func_with_call byref = [] sizes = [] pure = false>
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

llvm.func @sret_func(%arg0: !llvm.ptr {llvm.sret = !llvm.struct<(f32, f32)>, llvm.align = 8 : i64, llvm.nonnull}, %arg1: !llvm.ptr {llvm.noundef, llvm.readonly}) attributes {tessera.convert = #tessera<convert "tessera_sret_func" byref = [true] sizes = [64] pure = true>} {
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

// CHECK-LABEL: tessera.define @tessera_sret_func
// CHECK-SAME: tessera.convert = #tessera<convert tessera_sret_func byref = [true] sizes = [64] pure = true>
// CHECK-SAME: tessera.original_name = "sret_func"
// CHECK-SAME: tessera.side_effect_free
// CHECK-SAME: tessera.sret_attrs = {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %arg1
// CHECK-SAME: !llvm.ptr -> f32
// CHECK-NEXT: llvm.store %[[LOAD]], %arg0 {alignment = 8 : i64} : f32, !llvm.ptr
// CHECK-NEXT: tessera.return

// CHECK-LABEL: llvm.func @caller
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A1:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[A2:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[A2]] : !llvm.ptr -> i512
// CHECK-NEXT: %[[RES:.*]] = tessera.call @tessera_sret_func(%[[LOADED]])
// CHECK-SAME: tessera.loaded_operands = array<i32: 0>
// CHECK-SAME: (i512) -> !llvm.struct<(f32, f32)>
// CHECK-NEXT: llvm.store %[[RES]], %[[A1]] : !llvm.struct<(f32, f32)>, !llvm.ptr
// CHECK-NEXT: llvm.return
