// RUN: enzymexlamlir-opt %s --enzyme-batch --enzyme-hlo-opt | FileCheck %s

module {
  func.func private @pad(%arg0: tensor<16xf64>) -> (tensor<19xf64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.pad %arg0, %cst, low = [2], high = [1], interior = [0] : (tensor<16xf64>, tensor<f64>) -> tensor<19xf64>
    return %0 : tensor<19xf64>
  }
  func.func @main(%arg0: tensor<4x16xf64>) -> (tensor<4x19xf64>) {
    %1 = enzyme.batch @pad(%arg0) {batch_shape = array<i64: 4>} : (tensor<4x16xf64>) -> (tensor<4x19xf64>)
    return %1 : tensor<4x19xf64>
  }
}

// CHECK: func.func private @batched_pad(%arg0: tensor<4x16xf64>) -> tensor<4x19xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 2], high = [0, 1], interior = [0, 0] : (tensor<4x16xf64>, tensor<f64>) -> tensor<4x19xf64>
// CHECK-NEXT:     return %0 : tensor<4x19xf64>
// CHECK-NEXT: }

module {
  func.func private @pad2(%arg0: tensor<16xf64>, %cst: tensor<f64>) -> (tensor<19xf64>) {
    %0 = stablehlo.pad %arg0, %cst, low = [2], high = [1], interior = [0] : (tensor<16xf64>, tensor<f64>) -> tensor<19xf64>
    return %0 : tensor<19xf64>
  }
  func.func @main(%arg0: tensor<4x16xf64>, %arg1: tensor<4xf64>) -> (tensor<4x19xf64>) {
    %1 = enzyme.batch @pad2(%arg0, %arg1) {batch_shape = array<i64: 4>} : (tensor<4x16xf64>, tensor<4xf64>) -> (tensor<4x19xf64>)
    return %1 : tensor<4x19xf64>
  }
}

// CHECK: func.func private @batched_pad2(%arg0: tensor<4x16xf64>, %arg1: tensor<4xf64>) -> tensor<4x19xf64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : tensor<4x19xf64>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %cst) : tensor<i64>, tensor<4x19xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %iterArg, %c {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.dynamic_slice %arg0, %iterArg, %c_1, sizes = [1, 16] : (tensor<4x16xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
// CHECK-NEXT:       %3 = stablehlo.dynamic_slice %arg1, %iterArg, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:       %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %5 = stablehlo.pad %2, %4, low = [0, 2], high = [0, 1], interior = [0, 0] : (tensor<1x16xf64>, tensor<f64>) -> tensor<1x19xf64>
// CHECK-NEXT:       %6 = stablehlo.dynamic_update_slice %iterArg_2, %5, %iterArg, %c_1 : (tensor<4x19xf64>, tensor<1x19xf64>, tensor<i64>, tensor<i64>) -> tensor<4x19xf64>
// CHECK-NEXT:       stablehlo.return %1, %6 : tensor<i64>, tensor<4x19xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<4x19xf64>
// CHECK-NEXT: }
