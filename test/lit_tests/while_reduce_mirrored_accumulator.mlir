// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=while_elementwise_reduction_to_reduce" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A raised kernel carries its accumulator twice: the body reads one copy and
// adds to it, the other is only yielded, and that copy is what the code after
// the loop reads. The loop is still the reduction of what it adds.
module {
  func.func @main(%arg0: tensor<12x32xf32>, %arg1: tensor<i64>) -> tensor<6x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<6x1xf32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %c_0 = stablehlo.constant dense<6> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<12x32xf32>) -> tensor<32x12xf32>
    %1 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.subtract %1, %c_2 : tensor<i32>
    %3:3 = stablehlo.while(%iterArg = %c_3, %iterArg_5 = %cst, %iterArg_6 = %cst) : tensor<i64>, tensor<6x1xf32>, tensor<6x1xf32>
    cond {
      %5 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    } do {
      %5 = stablehlo.add %iterArg, %c_4 : tensor<i64>
      %6 = stablehlo.dynamic_slice %0, %2, %c, sizes = [1, 6] : (tensor<32x12xf32>, tensor<i32>, tensor<i32>) -> tensor<1x6xf32>
      %7 = stablehlo.multiply %6, %6 : tensor<1x6xf32>
      %8 = stablehlo.reshape %7 : (tensor<1x6xf32>) -> tensor<6x1xf32>
      %9 = stablehlo.add %iterArg_5, %8 : tensor<6x1xf32>
      stablehlo.return %5, %9, %9 : tensor<i64>, tensor<6x1xf32>, tensor<6x1xf32>
    }
    return %3#2 : tensor<6x1xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<12x32xf32>, %arg1: tensor<i64>) -> tensor<6x1xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<6x1xf32>
// CHECK-NEXT:   %c = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<12x32xf32>) -> tensor<32x12xf32>
// CHECK-NEXT:   %1 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %2 = stablehlo.subtract %1, %c_1 : tensor<i32>
// CHECK-NEXT:   %3 = stablehlo.dynamic_slice %0, %2, %c, sizes = [1, 6] : (tensor<32x12xf32>, tensor<i32>, tensor<i32>) -> tensor<1x6xf32>
// CHECK-NEXT:   %4 = stablehlo.multiply %3, %3 : tensor<1x6xf32>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<1x6xf32>) -> tensor<6x1xf32>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %5, dims = [1, 2] : (tensor<6x1xf32>) -> tensor<6x6x1xf32>
// CHECK-NEXT:   %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x6x1xf32>, tensor<f32>) -> tensor<6x1xf32>
// CHECK-NEXT:   %8 = stablehlo.add %7, %cst_0 : tensor<6x1xf32>
// CHECK-NEXT:   return %8 : tensor<6x1xf32>
// CHECK-NEXT: }
