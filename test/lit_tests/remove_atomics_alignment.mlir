// RUN: enzymexlamlir-opt --remove-atomics %s | FileCheck %s

// The load and store that replace an atomic are of its address, and keep the
// alignment it named.

#map = affine_map<(d0) -> (d0)>
module {
  func.func @rmw(%v: f32, %a: memref<?xf32> {llvm.noalias}) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %r = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] monotonic <{alignment = 2 : i64}> : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @rmw(%arg0: f32, %arg1: memref<?xf32> {llvm.noalias}) {
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:     affine.parallel (%arg2) = (0) to (4) {
// CHECK-NEXT:       %1 = affine.load %arg1[%arg2] {alignment = 2 : i64} : memref<?xf32>
// CHECK-NEXT:       %2 = arith.addf %1, %arg0 : f32
// CHECK-NEXT:       affine.store %2, %arg1[%arg2] {alignment = 2 : i64} : memref<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:   }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:   return
// CHECK-NEXT: }
