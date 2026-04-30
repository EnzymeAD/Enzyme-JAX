// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

func.func @gpu_wrapper_symbol(%arg0: memref<?xf64>, %arg1: index) {
  %c601 = arith.constant 601 : index
  %sym = arith.subi %c601, %arg1 : index
  
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  
  // CHECK: enzymexla.gpu_wrapper
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c32, %c1, %c1) ({
    affine.parallel (%arg2) = (0) to (32) {
      %1 = affine.load %arg0[symbol(%sym)] : memref<?xf64>
      affine.store %1, %arg0[%arg2] : memref<?xf64>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}
