// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// CHECK-LABEL: func.func @gpu_wrapper_symbol
// CHECK: enzymexla.xla_wrapper @[[RAISED:[a-zA-Z0-9_$]+]]

// CHECK: func.func private @[[RAISED]]
// CHECK: %[[C608:.*]] = stablehlo.constant dense<608> : tensor<i64>
// CHECK: %[[MUL:.*]] = arith.muli {{.*}}, %[[C608]]
// CHECK: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[MUL]], dims = []
// CHECK: %[[ADD:.*]] = arith.addi {{.*}}, %[[BROADCAST]]
// CHECK: stablehlo.gather
// CHECK: stablehlo.dynamic_update_slice
// CHECK: return
func.func @gpu_wrapper_symbol(%arg0: memref<?xf64>, %sym: index) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c32, %c1, %c1) ({
    affine.parallel (%arg2) = (0) to (32) {
      // Use a symbol passed as argument to force gather!
      %1 = affine.load %arg0[%arg2 + symbol(%sym) * 608] : memref<?xf64>
      affine.store %1, %arg0[%arg2] : memref<?xf64>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}
