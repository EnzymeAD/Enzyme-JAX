// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// CHECK-LABEL: @slice_of_binop
// CHECK-SAME: %[[ARG0:.+]]: tensor<42xf32>, %[[ARG1:.+]]: tensor<42xf32>
func.func @slice_of_binop(%arg0: tensor<42xf32>, %arg1: tensor<42xf32>) -> tensor<2xf32> {
  // CHECK: %[[S0:.+]] = stablehlo.slice %[[ARG0]] [1:3]
  // CHECK: %[[S1:.+]] = stablehlo.slice %[[ARG1]] [1:3]
  // CHECK: %[[R:.+]] = stablehlo.add %[[S0]], %[[S1]] : tensor<2xf32>
  %0 = stablehlo.add %arg0, %arg1 : tensor<42xf32>
  %1 = stablehlo.slice %0 [1:3] : (tensor<42xf32>) -> tensor<2xf32>
  // CHECK-NOT: stablehlo.slice
  // CHECK: return %[[R]]
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: @slice_of_binop2
// CHECK-SAME: %[[ARG0:.+]]: tensor<42xf32>, %[[ARG1:.+]]: tensor<42xf32>
func.func @slice_of_binop2(%arg0: tensor<42xf32>, %arg1: tensor<42xf32>) -> tensor<2xf32> {
  // CHECK: %[[S0:.+]] = stablehlo.slice %[[ARG0]] [1:3]
  // CHECK: %[[S1:.+]] = stablehlo.slice %[[ARG1]] [1:3]
  // CHECK: %[[R:.+]] = stablehlo.multiply %[[S0]], %[[S1]] : tensor<2xf32>
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<42xf32>
  %1 = stablehlo.slice %0 [1:3] : (tensor<42xf32>) -> tensor<2xf32>
  // CHECK-NOT: stablehlo.slice
  // CHECK: return %[[R]]
  return %1 : tensor<2xf32>
}
