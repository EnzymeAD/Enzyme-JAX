// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_if;if_op_lift_common_ops" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt %s | FileCheck %s

module @reactant_conditi... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<10x2xf64> {tf.aliasing_output = 3 : i32}) -> (tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xf64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c = stablehlo.constant dense<1> : tensor<2x10xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<10x2xf64>) -> tensor<2x10xf64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<2x10xf64>) -> tensor<2x10xf64>
    %2 = stablehlo.convert %c : (tensor<2x10xi64>) -> tensor<2x10xf64>
    %3 = stablehlo.add %1, %2 : tensor<2x10xf64>
    %4 = stablehlo.convert %3 : tensor<2x10xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x10xf64>) -> tensor<2x10xf64>
    %6 = stablehlo.convert %5 : tensor<2x10xf64>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x10xf64>, tensor<f64>) -> tensor<f64>
    %8 = stablehlo.compare  GT, %7, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %9:5 = "stablehlo.if"(%8) ({
      %14 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x10xf64>) -> tensor<2x10xf64>
      %15 = stablehlo.convert %c : (tensor<2x10xi64>) -> tensor<2x10xf64>
      %16 = stablehlo.subtract %14, %15 : tensor<2x10xf64>
      %17 = stablehlo.convert %16 : tensor<2x10xf64>
      stablehlo.return %4, %4, %4, %17, %4 : tensor<2x10xf64>, tensor<2x10xf64>, tensor<2x10xf64>, tensor<2x10xf64>, tensor<2x10xf64>
    }, {
      %14 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x10xf64>) -> tensor<2x10xf64>
      %15 = stablehlo.negate %14 : tensor<2x10xf64>
      %16 = stablehlo.convert %15 : tensor<2x10xf64>
      %17 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x10xf64>) -> tensor<2x10xf64>
      %18 = stablehlo.convert %c : (tensor<2x10xi64>) -> tensor<2x10xf64>
      %19 = stablehlo.add %17, %18 : tensor<2x10xf64>
      %20 = stablehlo.convert %19 : tensor<2x10xf64>
      stablehlo.return %4, %4, %16, %4, %20 : tensor<2x10xf64>, tensor<2x10xf64>, tensor<2x10xf64>, tensor<2x10xf64>, tensor<2x10xf64>
    }) : (tensor<i1>) -> (tensor<2x10xf64>, tensor<2x10xf64>, tensor<2x10xf64>, tensor<2x10xf64>, tensor<2x10xf64>)
    %10 = stablehlo.transpose %9#2, dims = [1, 0] : (tensor<2x10xf64>) -> tensor<10x2xf64>
    %11 = stablehlo.transpose %9#3, dims = [1, 0] : (tensor<2x10xf64>) -> tensor<10x2xf64>
    %12 = stablehlo.transpose %9#4, dims = [1, 0] : (tensor<2x10xf64>) -> tensor<10x2xf64>
    %13 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x10xf64>) -> tensor<10x2xf64>
    return %10, %11, %12, %13 : tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x2xf64> {tf.aliasing_output = 3 : i32}) -> (tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xf64>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<10x2xf64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<10x2xf64>
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %cst_1 : tensor<10x2xf64>
// CHECK-NEXT:     %1 = stablehlo.negate %0 : tensor<10x2xf64>
// CHECK-NEXT:     %2 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.add across dimensions = [0, 1] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<10x2xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     %3 = stablehlo.compare  GT, %2, %cst_0 : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT:     %4 = stablehlo.select %3, %arg0, %0 : tensor<i1>, tensor<10x2xf64>
// CHECK-NEXT:     %5 = stablehlo.select %3, %0, %1 : tensor<i1>, tensor<10x2xf64>
// CHECK-NEXT:     %6 = "stablehlo.if"(%3) ({
// CHECK-NEXT:       stablehlo.return %0 : tensor<10x2xf64>
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %7 = stablehlo.add %arg0, %cst : tensor<10x2xf64>
// CHECK-NEXT:       stablehlo.return %7 : tensor<10x2xf64>
// CHECK-NEXT:     }) : (tensor<i1>) -> tensor<10x2xf64>
// CHECK-NEXT:     return %5, %4, %6, %arg0 : tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xf64>
// CHECK-NEXT:   }
