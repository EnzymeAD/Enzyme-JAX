// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

#map_lb = affine_map<(d0) -> (d0, 0)>
#map_ub = affine_map<(d0) -> (100, d0)>

module {
  func.func private @dynamic_for_min_max(%bounds: memref<2xi64>, %tensor: memref<100xf32>) {
    %lb_i64 = affine.load %bounds[0] : memref<2xi64>
    %ub_i64 = affine.load %bounds[1] : memref<2xi64>
    %lb = arith.index_cast %lb_i64 : i64 to index
    %ub = arith.index_cast %ub_i64 : i64 to index
    affine.for %i = max #map_lb(%lb) to min #map_ub(%ub) {
      %val = affine.load %tensor[0] : memref<100xf32>
      %res = arith.addf %val, %val : f32
      affine.store %res, %tensor[0] : memref<100xf32>
    }
    return
  }

  func.func @main(%arg0: tensor<2xi64>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
    %0 = enzymexla.jit_call @dynamic_for_min_max(%arg0, %arg1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]} : (tensor<2xi64>, tensor<100xf32>) -> tensor<100xf32>
    return %0 : tensor<100xf32>
  }
}

// CHECK-LABEL: func.func private @dynamic_for_min_max_raised
// CHECK:         %[[LB:.+]] = stablehlo.slice %{{.*}} [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:         %[[LB_R:.+]] = stablehlo.reshape %[[LB]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:         %[[UB:.+]] = stablehlo.slice %{{.*}} [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:         %[[UB_R:.+]] = stablehlo.reshape %[[UB]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:         %[[C0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:         %[[MAX:.+]] = stablehlo.maximum %[[LB_R]], %[[C0]] : tensor<i64>
// CHECK:         %[[C100:.+]] = stablehlo.constant dense<100> : tensor<i64>
// CHECK:         %[[MIN:.+]] = stablehlo.minimum %[[C100]], %[[UB_R]] : tensor<i64>
// CHECK:         stablehlo.while(%[[ITER:.+]] = %[[MAX]], {{.*}})
// CHECK:         cond {
// CHECK:           %[[CMP:.+]] = stablehlo.compare  LT, %[[ITER]], %[[MIN]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK:           stablehlo.return %[[CMP]] : tensor<i1>
// CHECK:         } do {
