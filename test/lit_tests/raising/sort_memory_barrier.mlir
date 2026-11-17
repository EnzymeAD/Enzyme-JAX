// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(sort-memory,raise-affine-to-stablehlo,enzyme-hlo-opt)" --split-input-file | FileCheck %s

// sort-memory must not hoist cross-thread loads above stores across an
// enzymexla.barrier inside affine.parallel.
func.func @barrier_sort_memory(%out: memref<32xf64, 1>, %in: memref<32xf64, 1>) {
  %scr = memref.alloca() : memref<32xf64>
  affine.parallel (%t) = (0) to (32) {
    %v = affine.load %in[%t] : memref<32xf64, 1>
    affine.store %v, %scr[%t] : memref<32xf64>
    "enzymexla.barrier"(%t) : (index) -> ()
    %x = affine.load %scr[31 - %t] : memref<32xf64>
    affine.store %x, %out[%t] : memref<32xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @barrier_sort_memory_raised(
// CHECK-SAME: %[[OUT:.*]]: tensor<32xf64>, %[[IN:.*]]: tensor<32xf64>) -> (tensor<32xf64>, tensor<32xf64>)
// CHECK-NOT: enzymexla.barrier
// CHECK: %[[REV:.*]] = stablehlo.reverse %[[IN]], dims = [0] : tensor<32xf64>
// CHECK: return %[[REV]], %[[IN]] : tensor<32xf64>, tensor<32xf64>
