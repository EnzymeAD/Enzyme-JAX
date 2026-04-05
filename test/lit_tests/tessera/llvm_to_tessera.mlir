// RUN: enzymexlamlir-opt %s -llvm-to-tessera | FileCheck %s

llvm.func @simple_func() attributes {tessera.convert = #tessera<convert "tessera_simple_func">} {
  llvm.return
}

// CHECK-LABEL: tessera.define @tessera_simple_func
// CHECK-SAME: tessera.original_name = "simple_func"
// CHECK: tessera.return

// -----


llvm.func @func_with_args(%arg0: i32, %arg1: f32) -> i32 attributes {tessera.convert = #tessera<convert "tessera_func_with_args">} {
  llvm.return %arg0 : i32
}

// CHECK-LABEL: tessera.define @tessera_func_with_args
// CHECK: tessera.return %arg0 : i32

// -----

llvm.func @helper() attributes {tessera.convert = #tessera<convert "tessera_helper">} {
  llvm.return
}

llvm.func @func_with_call() attributes {tessera.convert = #tessera<convert "tessera_func_with_call">} {
  llvm.call @helper() : () -> ()
  llvm.return
}

// CHECK-LABEL: tessera.define @tessera_helper
// CHECK: tessera.return

// CHECK-LABEL: tessera.define @tessera_func_with_call
// CHECK: tessera.call @tessera_helper()
// CHECK: tessera.return

// -----

llvm.func @regular_function() {
  llvm.return
}

// CHECK-LABEL: llvm.func @regular_function
// CHECK: llvm.return

// -----

llvm.func @func_with_indirect_call(%arg0: !llvm.ptr) {
  llvm.call %arg0() : !llvm.ptr, () -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @func_with_indirect_call
// CHECK: llvm.call %arg0() : !llvm.ptr, () -> ()

// -----

llvm.func @sret_func(%arg0: !llvm.ptr {llvm.sret = !llvm.struct<(f32, f32)>, llvm.align = 8 : i64, llvm.nonnull}, %arg1: !llvm.ptr {llvm.noundef, llvm.readonly}) attributes {tessera.convert = #tessera<convert "tessera_sret_func">} {
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
// CHECK-SAME: tessera.sret_attrs = {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}
// CHECK: tessera.return

// CHECK-LABEL: llvm.func @caller
// CHECK: %[[RES:.*]] = tessera.call @tessera_sret_func
// CHECK-SAME: -> !llvm.struct<(f32, f32)>
// CHECK: llvm.store %[[RES]], %{{.*}} : !llvm.struct<(f32, f32)>, !llvm.ptr
