// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

func.func @index_cast_add(%arg0: memref<?xi32>, %arg1: memref<?xi32>) {
  affine.parallel (%arg2) = (0) to (10) {
    // CHECK: %[[IV:.*]] = stablehlo.iota dim = 0 : tensor<10xi64>
    // CHECK: %[[CONV:.*]] = stablehlo.convert {{.*}} : (tensor<10xi64>) -> tensor<10xi32>
    %0 = arith.index_castui %arg2 : index to i32
    
    // CHECK: %[[ADD:.*]] = arith.addi %[[CONV]], %[[CONV]] : tensor<10xi32>
    %1 = arith.addi %0, %0 : i32
    
    affine.store %1, %arg1[%arg2] : memref<?xi32>
  }
  return
}
