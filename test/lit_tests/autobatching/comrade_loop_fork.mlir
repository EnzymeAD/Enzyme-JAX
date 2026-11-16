// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt=enable_auto_batching_passes=true | FileCheck %s

module @reactant_write_acc attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<64xf64> {enzymexla.memory_effects = []}) -> tensor<f64> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<64> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<64xf64>
    %0:3 = stablehlo.while(%iterArg = %c_0, %iterArg_4 = %cst_3, %iterArg_5 = %cst) : tensor<i64>, tensor<64xf64>, tensor<f64> attributes {enzyme.disable_mincut}
    cond {
      %3 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_2, %iterArg {enzymexla.bounds = [[1, 64]]} : tensor<i64>
      %4 = stablehlo.convert %3 {enzymexla.bounds = [[1, 64]]} : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c {enzymexla.bounds = [[0, 63]]} : tensor<i32>
      %6 = stablehlo.dynamic_slice %arg0, %5, sizes = [1] : (tensor<64xf64>, tensor<i32>) -> tensor<1xf64>

      %7 = stablehlo.exponential %6 : tensor<1xf64>
      %8 = stablehlo.dynamic_update_slice %iterArg_4, %7, %5 : (tensor<64xf64>, tensor<1xf64>, tensor<i32>) -> tensor<64xf64>

      %9 = stablehlo.multiply %6, %6 : tensor<1xf64>
      %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
      %11 = stablehlo.add %iterArg_5, %10 : tensor<f64>
      stablehlo.return %3, %8, %11 : tensor<i64>, tensor<64xf64>, tensor<f64>
    }
    %1 = stablehlo.reduce(%0#1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64xf64>, tensor<f64>) -> tensor<f64>
    %2 = stablehlo.add %1, %0#2 : tensor<f64>
    return %2 : tensor<f64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<64xf64> {enzymexla.memory_effects = []}) -> tensor<f64> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.exponential %arg0 : tensor<64xf64>
// CHECK-NEXT:    %1 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
// CHECK-NEXT:    %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %3 = stablehlo.add %2, %1 : tensor<f64>
// CHECK-NEXT:    return %3 : tensor<f64>
// CHECK-NEXT:  }
