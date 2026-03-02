// RUN: enzymexlamlir-opt %s -tessera-to-llvm | FileCheck %s

tessera.define @tessera_simple_func() -> !llvm.void attributes {tessera.original_name = "simple_func", CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, unnamed_addr = 0 : i64, visibility_ = 0 : i64} {
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

tessera.define @tessera_helper() -> !llvm.void attributes {tessera.original_name = "helper", CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, unnamed_addr = 0 : i64, visibility_ = 0 : i64} {
  tessera.return
}

tessera.define @tessera_func_with_call() -> !llvm.void attributes {tessera.original_name = "func_with_call", CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, unnamed_addr = 0 : i64, visibility_ = 0 : i64} {
  tessera.call @tessera_helper() {CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 0, 0>} : () -> ()
  tessera.return
}

// CHECK-LABEL: llvm.func @helper
// CHECK-NEXT: llvm.return

// CHECK-LABEL: llvm.func @func_with_call
// CHECK: llvm.call @helper() : () -> ()
// CHECK: llvm.return
