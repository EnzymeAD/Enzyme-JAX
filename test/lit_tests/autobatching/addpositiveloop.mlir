// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c_2 = stablehlo.constant dense<10> : tensor<i64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
    %0 = stablehlo.compare  GT, %arg0, %cst_3 : (tensor<10xf64>, tensor<10xf64>) -> tensor<10xi1>
    %1:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %cst) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_0, %iterArg {enzymexla.bounds = [[1, 10]]} : tensor<i64>
      %3 = stablehlo.convert %2 {enzymexla.bounds = [[1, 10]]} : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c {enzymexla.bounds = [[0, 9]]} : tensor<i32>
      %5 = stablehlo.dynamic_slice %arg0, %4, sizes = [1] : (tensor<10xf64>, tensor<i32>) -> tensor<1xf64>
      %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
      %7 = stablehlo.dynamic_slice %0, %iterArg, sizes = [1] : (tensor<10xi1>, tensor<i64>) -> tensor<1xi1>
      %8 = stablehlo.reshape %7 : (tensor<1xi1>) -> tensor<i1>
      %9 = "stablehlo.if"(%8) ({
        %10 = stablehlo.add %iterArg_4, %6 : tensor<f64>
        stablehlo.return %10 : tensor<f64>
      }, {
        stablehlo.return %iterArg_4 : tensor<f64>
      }) : (tensor<i1>) -> tensor<f64>
      stablehlo.return %2, %9 : tensor<i64>, tensor<f64>
    }
    return %1#1 : tensor<f64>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK-NEXT:     %0 = stablehlo.maximum %arg0, %cst_0 : tensor<10xf64>
// CHECK-NEXT:     %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<10xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }
