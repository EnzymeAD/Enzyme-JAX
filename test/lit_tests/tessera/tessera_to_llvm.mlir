// RUN: enzymexlamlir-opt %s -tessera-to-llvm | FileCheck %s

tessera.define @tessera_simple_func() attributes {tessera.original_name = "simple_func", CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, unnamed_addr = 0 : i64, visibility_ = 0 : i64} {
  tessera.return
}

// CHECK-LABEL: llvm.func @simple_func
// CHECK: llvm.return

// -----

tessera.define @tessera_func_with_args(%arg0: i32, %arg1: f32) -> i32 attributes {tessera.original_name = "func_with_args", CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, unnamed_addr = 0 : i64, visibility_ = 0 : i64} {
  tessera.return %arg0 : i32
}

// CHECK-LABEL: llvm.func @func_with_args
// CHECK: llvm.return %arg0 : i32

// -----

tessera.define @tessera_helper() attributes {tessera.original_name = "helper", CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, unnamed_addr = 0 : i64, visibility_ = 0 : i64} {
  tessera.return
}

tessera.define @tessera_func_with_call() attributes {tessera.original_name = "func_with_call", CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, unnamed_addr = 0 : i64, visibility_ = 0 : i64} {
  tessera.call @tessera_helper() {CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 0, 0>} : () -> ()
  tessera.return
}

// CHECK-LABEL: llvm.func @helper
// CHECK: llvm.return

// CHECK-LABEL: llvm.func @func_with_call
// CHECK: llvm.call @helper() : () -> ()
// CHECK: llvm.return

// -----

tessera.define @tessera_sret_func(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}, %arg1: !llvm.ptr {llvm.noundef, llvm.readonly}) attributes {CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, tessera.original_name = "sret_func", tessera.sret_attrs = {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}, unnamed_addr = 0 : i64, visibility_ = 0 : i64} {
  %0 = llvm.load %arg1 {alignment = 8 : i64} : !llvm.ptr -> f32
  llvm.store %0, %arg0 {alignment = 8 : i64} : f32, !llvm.ptr
  tessera.return
}

llvm.func @caller() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = tessera.call @tessera_sret_func(%2) {CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, arg_attrs = [{llvm.nonnull, llvm.noundef}], fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>} : (!llvm.ptr) -> !llvm.struct<(f32, f32)>
  llvm.store %3, %1 : !llvm.struct<(f32, f32)>, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: llvm.func @sret_func
// CHECK-SAME: !llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}
// CHECK-LABEL: llvm.func @caller
// CHECK: %[[A1:.*]] = llvm.alloca
// CHECK: %[[A2:.*]] = llvm.alloca
// CHECK: %[[SRET:.*]] = llvm.alloca
// CHECK: llvm.call @sret_func(%[[SRET]], %[[A2]])
// CHECK-SAME: !llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}
// CHECK: %[[LOADED:.*]] = llvm.load %[[SRET]]
// CHECK: llvm.store %[[LOADED]], {{.*}}
// CHECK-NOT: tessera.call
