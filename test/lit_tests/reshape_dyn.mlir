// RUN: enzymexlamlir-opt %s  '--enzyme-hlo-generate-td=patterns=reshape_dynamic_slice(1)' --transform-interpreter --enzyme-hlo-remove-transform --canonicalize | FileCheck %s

module {
  func.func @main(%arg0: tensor<?x1x1x128xf64>) -> tensor<1x1x128xf64> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1238 = stablehlo.dynamic_slice %arg0, %c, %c, %c, %c, sizes = [1, 1, 1, 128] : (tensor<?x1x1x128xf64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x1x128xf64>
    %1239 = stablehlo.reshape %1238 : (tensor<1x1x1x128xf64>) -> tensor<1x1x128xf64>
    return %1239 : tensor<1x1x128xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<?x1x1x128xf64>) -> tensor<1x1x128xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<[0, 1, 128]> : tensor<3xi32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x1x1x128xf64>) -> tensor<i32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %c_0, %1, %c : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32>
// CHECK-NEXT:    %3 = stablehlo.dynamic_reshape %arg0, %2 : (tensor<?x1x1x128xf64>, tensor<3xi32>) -> tensor<?x1x128xf64>
// CHECK-NEXT:    %4 = stablehlo.dynamic_slice %3, %c_1, %c_1, %c_1, sizes = [1, 1, 128] : (tensor<?x1x128xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x128xf64>
// CHECK-NEXT:    return %4 : tensor<1x1x128xf64>
// CHECK-NEXT:  }
