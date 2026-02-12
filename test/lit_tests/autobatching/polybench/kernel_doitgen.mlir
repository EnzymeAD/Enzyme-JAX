// RUN: enzymexlamlir-opt --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s
// RUN: enzymexlamlir-opt --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt %s | FileCheck %s --check-prefix=FULL

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.dot_general_broadcast_in_dim_sort_dims
    } : !transform.any_op
    transform.yield 
  }
  func.func @kernel_doitgen(%arg0: tensor<18x15x12xf32>, %arg1: tensor<18x18xf32>) -> (tensor<18x15x12xf32>, tensor<18x18xf32>) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.slice %arg1 [0:15, 0:15] : (tensor<18x18xf32>) -> tensor<15x15xf32>
    %1 = stablehlo.slice %arg0 [0:15, 0:15, 0:12] : (tensor<18x15x12xf32>) -> tensor<15x15x12xf32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<15x15xf32>) -> tensor<15x15xf32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [2, 1, 0] : (tensor<15x15x12xf32>) -> tensor<12x15x15xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<15x15xf32>) -> tensor<15x15xf32>
    %5 = stablehlo.dot_general %3, %4, contracting_dims = [2] x [1] : (tensor<12x15x15xf32>, tensor<15x15xf32>) -> tensor<12x15x15xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 1, 0] : (tensor<12x15x15xf32>) -> tensor<15x15x12xf32>
    %7 = stablehlo.dynamic_update_slice %arg0, %6, %c, %c, %c : (tensor<18x15x12xf32>, tensor<15x15x12xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<18x15x12xf32>
    return %7, %arg1 : tensor<18x15x12xf32>, tensor<18x18xf32>
  }
}

// CHECK: func.func @kernel_doitgen(%arg0: tensor<18x15x12xf32>, %arg1: tensor<18x18xf32>) -> (tensor<18x15x12xf32>, tensor<18x18xf32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg1 [0:15, 0:15] : (tensor<18x18xf32>) -> tensor<15x15xf32>
// CHECK-NEXT:   %1 = stablehlo.slice %arg0 [0:15, 0:15, 0:12] : (tensor<18x15x12xf32>) -> tensor<15x15x12xf32>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<15x15xf32>) -> tensor<15x15xf32>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<15x15xf32>) -> tensor<15x15xf32>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<15x15x12xf32>) -> tensor<15x15x12xf32>
// CHECK-NEXT:   %5 = stablehlo.dot_general %4, %3, contracting_dims = [0] x [1] : (tensor<15x15x12xf32>, tensor<15x15xf32>) -> tensor<15x12x15xf32>
// CHECK-NEXT:   %6 = stablehlo.transpose %5, dims = [1, 0, 2] : (tensor<15x12x15xf32>) -> tensor<12x15x15xf32>
// CHECK-NEXT:   %7 = stablehlo.broadcast_in_dim %6, dims = [2, 1, 0] : (tensor<12x15x15xf32>) -> tensor<15x15x12xf32>
// CHECK-NEXT:   %8 = stablehlo.dynamic_update_slice %arg0, %7, %c, %c, %c : (tensor<18x15x12xf32>, tensor<15x15x12xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<18x15x12xf32>
// CHECK-NEXT:   return %8, %arg1 : tensor<18x15x12xf32>, tensor<18x18xf32>
// CHECK-NEXT: }

// FULL: func.func @kernel_doitgen(%arg0: tensor<18x15x12xf32>, %arg1: tensor<18x18xf32>) -> (tensor<18x15x12xf32>, tensor<18x18xf32>) {
// FULL-NEXT:   %0 = stablehlo.slice %arg1 [0:15, 0:15] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<18x18xf32>) -> tensor<15x15xf32>
// FULL-NEXT:   %1 = stablehlo.slice %arg0 [0:15, 0:15, 0:12] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<18x15x12xf32>) -> tensor<15x15x12xf32>
// FULL-NEXT:   %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] : (tensor<15x15xf32>, tensor<15x15x12xf32>) -> tensor<15x15x12xf32>
// FULL-NEXT:   %3 = stablehlo.slice %arg0 [15:18, 0:15, 0:12] : (tensor<18x15x12xf32>) -> tensor<3x15x12xf32>
// FULL-NEXT:   %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<15x15x12xf32>, tensor<3x15x12xf32>) -> tensor<18x15x12xf32>
// FULL-NEXT:   return %4, %arg1 : tensor<18x15x12xf32>, tensor<18x18xf32>
// FULL-NEXT: }
