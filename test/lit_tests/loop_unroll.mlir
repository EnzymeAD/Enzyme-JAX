// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=enzyme_hlo_unroll(0)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=CHECK-ZERO
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=enzyme_hlo_unroll(5)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=CHECK-FIVE

module {
  func.func @main(%arg0: tensor<5x4x3x2xf32>) -> tensor<5x24x1xf32> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<5x24x1xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<5x24x1xf32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %c_4 = stablehlo.constant dense<3> : tensor<i64>
    %0 = stablehlo.reshape %arg0 : (tensor<5x4x3x2xf32>) -> tensor<5x24x1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<5x24x1xf32>
    %2:2 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %cst_0) : tensor<i64>, tensor<5x24x1xf32>
    cond {
      %3 = stablehlo.compare  LT, %iterArg, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_1 : tensor<i32>
      %6 = stablehlo.dynamic_update_slice %iterArg_4, %1, %c, %c, %5 : (tensor<5x24x1xf32>, tensor<5x24x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x24x1xf32>
      stablehlo.return %3, %6 : tensor<i64>, tensor<5x24x1xf32>
    }
    return %2#1 : tensor<5x24x1xf32>
  }
}

// CHECK-ZERO: stablehlo.while(

// CHECK-FIVE: stablehlo.multiply
// CHECK-FIVE: stablehlo.dynamic_update_slice
// CHECK-FIVE: stablehlo.dynamic_update_slice
// CHECK-FIVE: stablehlo.dynamic_update_slice
// CHECK-FIVE: return
