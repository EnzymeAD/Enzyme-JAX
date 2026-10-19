// RUN: enzymexlamlir-opt --libdevice-funcs-raise --split-input-file %s | FileCheck %s

// arith.bitcast holds its shape, so only a cast whose two ends agree on one can
// be said with it. An integer reinterpreted as a vector of floats is a cast
// LLVM allows and arith has no spelling for, and it stays as it was.

llvm.func @scalar_to_vector(%a: i64) -> vector<2xf32> {
  %r = llvm.bitcast %a : i64 to vector<2xf32>
  llvm.return %r : vector<2xf32>
}

// CHECK-LABEL: llvm.func @scalar_to_vector
// CHECK:         llvm.bitcast
// CHECK-NOT:     arith.bitcast

// -----

// The other way round is no better.

llvm.func @vector_to_scalar(%a: vector<2xf32>) -> i64 {
  %r = llvm.bitcast %a : vector<2xf32> to i64
  llvm.return %r : i64
}

// CHECK-LABEL: llvm.func @vector_to_scalar
// CHECK:         llvm.bitcast
// CHECK-NOT:     arith.bitcast

// -----

// Two ends of one shape still raise.

llvm.func @scalar_to_scalar(%a: i32) -> f32 {
  %r = llvm.bitcast %a : i32 to f32
  llvm.return %r : f32
}

// CHECK-LABEL: llvm.func @scalar_to_scalar
// CHECK:         arith.bitcast
