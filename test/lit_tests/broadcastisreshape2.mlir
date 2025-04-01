// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module @reactant_repeat attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
   func.func @main(%arg0: tensor<2xf64>) -> tensor<1x4xf64> {
     %0 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<2xf64>) -> tensor<1x1x2x2xf64>
     %1 = stablehlo.reshape %0 : (tensor<1x1x2x2xf64>) -> tensor<1x4xf64>
     return %1 : tensor<1x4xf64>
   }
 }

// CHECK:   func.func @main(%arg0: tensor<2xf64>) -> tensor<1x4xf64> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<2xf64>) -> tensor<1x1x2x2xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x1x2x2xf64>) -> tensor<1x4xf64>
// CHECK-NEXT:    return %1 : tensor<1x4xf64>
// CHECK-NEXT:  }