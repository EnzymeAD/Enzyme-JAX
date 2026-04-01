// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reduce_licm(0)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5x4x3x2xf32>) -> tensor<5x24x2xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<5x24x2xf32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<24> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %c_4 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<5x4x3x2xf32>
    %1:2 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %cst_0) : tensor<i64>, tensor<5x24x2xf32>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_1 : tensor<i32>
      %5 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2, 1] : (tensor<5x4x3x2xf32>, tensor<f32>) -> tensor<5x2xf32>
      %6 = stablehlo.reshape %5 : (tensor<5x2xf32>) -> tensor<5x1x2xf32>
      %7 = stablehlo.dynamic_update_slice %iterArg_5, %6, %c, %4, %c : (tensor<5x24x2xf32>, tensor<5x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x24x2xf32>
      stablehlo.return %2, %7 : tensor<i64>, tensor<5x24x2xf32>
    }
    return %1#1 : tensor<5x24x2xf32>
  }
}

// CHECK:  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2, 1] : (tensor<5x4x3x2xf32>, tensor<f32>) -> tensor<5x2xf32>
// CHECK-NEXT:  %2:2 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %cst_0) : tensor<i64>, tensor<5x24x2xf32>
// CHECK-NEXT:  cond {
// CHECK-NEXT:    %3 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:    stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:  } do {
// CHECK-NEXT:    %3 = stablehlo.add %c_3, %iterArg : tensor<i64>
// CHECK-NEXT:    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:    %5 = stablehlo.subtract %4, %c_1 : tensor<i32>
// CHECK-NEXT:    %6 = stablehlo.reshape %1 : (tensor<5x2xf32>) -> tensor<5x1x2xf32>
// CHECK-NEXT:    %7 = stablehlo.dynamic_update_slice %iterArg_5, %6, %c, %5, %c : (tensor<5x24x2xf32>, tensor<5x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x24x2xf32>
// CHECK-NEXT:    stablehlo.return %3, %7 : tensor<i64>, tensor<5x24x2xf32>
// CHECK-NEXT:  }
