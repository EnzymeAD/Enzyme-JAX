// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main() -> tensor<1x3072xi32> {
    %174 = stablehlo.constant dense<0> : tensor<1x3072x1xi32>
    %200 = stablehlo.constant dense<[0, 256128]> : tensor<2xi32>
    %249 = "stablehlo.gather"(%200, %174) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x3072x1xi32>) -> tensor<1x3072xi32>
    return %249 : tensor<1x3072xi32>
  }

  func.func @main2() -> tensor<1x3072xi32> {
    %174 = stablehlo.constant dense<0> : tensor<1x3072x1xi32>
    %201 = stablehlo.constant dense<256128> : tensor<2xi32>
    %250 = "stablehlo.gather"(%201, %174) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x3072x1xi32>) -> tensor<1x3072xi32>
    return %250 : tensor<1x3072xi32>
  }

  func.func @main3() -> tensor<1x3072xi32> {
    %172 = stablehlo.constant dense<1> : tensor<1x3072x1xi32>
    %200 = stablehlo.constant dense<[0, 256128]> : tensor<2xi32>
    %268 = "stablehlo.gather"(%200, %172) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x3072x1xi32>) -> tensor<1x3072xi32>
    return %268 : tensor<1x3072xi32>
  }

  func.func @main4() -> tensor<1x3072xi32> {
    %172 = stablehlo.constant dense<1> : tensor<1x3072x1xi32>
    %201 = stablehlo.constant dense<256128> : tensor<2xi32>
    %269 = "stablehlo.gather"(%201, %172) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x3072x1xi32>) -> tensor<1x3072xi32>
    return %269 : tensor<1x3072xi32>
  }
}

// CHECK:   func.func @main() -> tensor<1x3072xi32> {
// CHECK-NEXT:     %[[i0:.+]] = stablehlo.constant dense<0> : tensor<1x3072xi32>
// CHECK-NEXT:     return %[[i0]] : tensor<1x3072xi32>
// CHECK-NEXT:   }
// CHECK:   func.func @main2() -> tensor<1x3072xi32> {
// CHECK-NEXT:     %[[i0:.+]] = stablehlo.constant dense<256128> : tensor<1x3072xi32>
// CHECK-NEXT:     return %[[i0]] : tensor<1x3072xi32>
// CHECK-NEXT:   }
// CHECK:   func.func @main3() -> tensor<1x3072xi32> {
// CHECK-NEXT:     %[[i0:.+]] = stablehlo.constant dense<256128> : tensor<1x3072xi32>
// CHECK-NEXT:     return %[[i0]] : tensor<1x3072xi32>
// CHECK-NEXT:   }
// CHECK:   func.func @main4() -> tensor<1x3072xi32> {
// CHECK-NEXT:     %[[i0:.+]] = stablehlo.constant dense<256128> : tensor<1x3072xi32>
// CHECK-NEXT:     return %[[i0]] : tensor<1x3072xi32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
