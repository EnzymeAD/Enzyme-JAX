// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=while_elementwise_reduction_to_reduce" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=elementwise_licm(1);dynamic_slice_licm(0);reshape_licm(0)" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-generate-td="patterns=while_elementwise_reduction_to_reduce" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s


module {
  func.func @main(%arg0: tensor<12x32xf32> {enzymexla.memory_effects = []}, %arg1: tensor<i64> {enzymexla.memory_effects = []}) -> tensor<12x32xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<6x1xf32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %c_0 = stablehlo.constant dense<6> : tensor<i64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<12x32xf32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<12x32xf32>) -> tensor<32x12xf32>
    %1 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.subtract %1, %c_2 : tensor<i32>
    %3:2 = stablehlo.while(%iterArg = %c_3, %iterArg_5 = %cst) : tensor<i64>, tensor<6x1xf32> attributes {enzyme.disable_mincut}
    cond {
      %5 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    } do {
      %5 = stablehlo.add %iterArg, %c_4 {enzymexla.bounds = [[1, 6]]} : tensor<i64>
      %6 = stablehlo.dynamic_slice %0, %2, %c, sizes = [1, 6] : (tensor<32x12xf32>, tensor<i32>, tensor<i32>) -> tensor<1x6xf32>
      %7 = stablehlo.multiply %6, %6 : tensor<1x6xf32>
      %8 = stablehlo.reshape %7 : (tensor<1x6xf32>) -> tensor<6x1xf32>
      %9 = stablehlo.add %iterArg_5, %8 : tensor<6x1xf32>
      stablehlo.return %5, %9 : tensor<i64>, tensor<6x1xf32>
    }
    %4 = stablehlo.dynamic_update_slice %cst_1, %3#1, %c, %2 : (tensor<12x32xf32>, tensor<6x1xf32>, tensor<i32>, tensor<i32>) -> tensor<12x32xf32>
    return %4 : tensor<12x32xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<12x32xf32> {enzymexla.memory_effects = []}, %arg1: tensor<i64> {enzymexla.memory_effects = []}) -> tensor<12x32xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<6x1xf32>
// CHECK-NEXT:   %c = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<12x32xf32>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<12x32xf32>) -> tensor<32x12xf32>
// CHECK-NEXT:   %1 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %2 = stablehlo.subtract %1, %c_2 : tensor<i32>
// CHECK-NEXT:   %3 = stablehlo.dynamic_slice %0, %2, %c, sizes = [1, 6] : (tensor<32x12xf32>, tensor<i32>, tensor<i32>) -> tensor<1x6xf32>
// CHECK-NEXT:   %4 = stablehlo.multiply %3, %3 : tensor<1x6xf32>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<1x6xf32>) -> tensor<6x1xf32>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %5, dims = [1, 2] : (tensor<6x1xf32>) -> tensor<6x6x1xf32>
// CHECK-NEXT:   %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x6x1xf32>, tensor<f32>) -> tensor<6x1xf32>
// CHECK-NEXT:   %8 = stablehlo.add %7, %cst_0 : tensor<6x1xf32>
// CHECK-NEXT:   %9 = stablehlo.dynamic_update_slice %cst_1, %8, %c, %2 : (tensor<12x32xf32>, tensor<6x1xf32>, tensor<i32>, tensor<i32>) -> tensor<12x32xf32>
// CHECK-NEXT:   return %9 : tensor<12x32xf32>
// CHECK-NEXT: }
