// RUN: enzymexlamlir-opt %s --canonicalize-parallel | FileCheck %s

// trunci of a wider index_cast of an index is the narrower index_cast: both
// are the low bits of the index.
module {
  func.func @narrow(%i: index) -> i32 {
    %0 = arith.index_cast %i : index to i64
    %1 = arith.trunci %0 : i64 to i32
    return %1 : i32
  }

  // a truncation of something else is left alone
  func.func @other(%x: i64) -> i32 {
    %0 = arith.trunci %x : i64 to i32
    return %0 : i32
  }
}

// CHECK:  func.func @narrow(%arg0: index) -> i32 {
// CHECK-NEXT:    %0 = arith.index_cast %arg0 : index to i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }

// CHECK:  func.func @other(%arg0: i64) -> i32 {
// CHECK-NEXT:    %0 = arith.trunci %arg0 : i64 to i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
