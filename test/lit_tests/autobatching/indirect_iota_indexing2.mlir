// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" --enzyme-hlo-generate-td="patterns=reshape_elementwise(1)" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<6xf64> {
    %c = stablehlo.constant dense<6> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<6xf64>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %c_3 = stablehlo.constant dense<[2, 3, 4, 5, 6, 7]> : tensor<6xi32>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_4 = %cst) : tensor<i64>, tensor<6xf64>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_2 : tensor<i64>
      %2 = stablehlo.dynamic_slice %c_3, %iterArg, sizes = [1] : (tensor<6xi32>, tensor<i64>) -> tensor<1xi32>
      %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
      %4 = stablehlo.dynamic_slice %arg0, %3, sizes = [1] : (tensor<10xf64>, tensor<i32>) -> tensor<1xf64>
      %5 = stablehlo.dynamic_slice %arg1, %3, sizes = [1] : (tensor<10xf64>, tensor<i32>) -> tensor<1xf64>
      %6 = stablehlo.add %4, %5 : tensor<1xf64>
      %7 = stablehlo.maximum %4, %5 : tensor<1xf64>
      %8 = stablehlo.add %6, %7 : tensor<1xf64>
      %9 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %10 = stablehlo.subtract %9, %c_1 : tensor<i32>
      %11 = stablehlo.dynamic_update_slice %iterArg_4, %8, %10 : (tensor<6xf64>, tensor<1xf64>, tensor<i32>) -> tensor<6xf64>
      stablehlo.return %1, %11 : tensor<i64>, tensor<6xf64>
    }
    return %0#1 : tensor<6xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<6xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [2:8] : (tensor<10xf64>) -> tensor<6xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg1 [2:8] : (tensor<10xf64>) -> tensor<6xf64>
// CHECK-NEXT:   %2 = stablehlo.add %0, %1 : tensor<6xf64>
// CHECK-NEXT:   %3 = stablehlo.maximum %0, %1 : tensor<6xf64>
// CHECK-NEXT:   %4 = stablehlo.add %2, %3 : tensor<6xf64>
// CHECK-NEXT:   return %4 : tensor<6xf64>
// CHECK-NEXT: }
