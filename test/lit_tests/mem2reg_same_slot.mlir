// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// A single dominating store lets a load forward its value only when both
// address the same slot; a store to a[i] says nothing about a load of a[j].
// Identical maps over identical operands is the one case where sameness is
// a fact rather than an analysis. MFEM's batched LOR assembly filled a
// 16-int scratch array at one index and read it back at another -- the
// forwarded value made every LOR H1/ND/RT system matrix wrong.

module {
  func.func @no_forward_different_index(%v: i32, %i: index, %j: index) -> i32 {
    %a = memref.alloca() : memref<16xi32>
    affine.store %v, %a[symbol(%i)] : memref<16xi32>
    %r = affine.load %a[symbol(%j)] : memref<16xi32>
    return %r : i32
  }

  func.func @forward_same_index(%v: i32, %i: index) -> i32 {
    %a = memref.alloca() : memref<16xi32>
    affine.store %v, %a[symbol(%i)] : memref<16xi32>
    %r = affine.load %a[symbol(%i)] : memref<16xi32>
    return %r : i32
  }
}

// CHECK-LABEL: func.func @no_forward_different_index
// CHECK:         affine.store
// CHECK:         %[[R:.+]] = affine.load
// CHECK:         return %[[R]]

// CHECK-LABEL: func.func @forward_same_index
// CHECK-NOT:     affine.load
// CHECK:         return %arg0 : i32
