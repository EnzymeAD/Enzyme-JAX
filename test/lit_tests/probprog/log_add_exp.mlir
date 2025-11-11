// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  // CPU:  func.func @test(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
  // CPU-NEXT:    %0 = stablehlo.maximum %arg0, %arg1 : tensor<10xf64>
  // CPU-NEXT:    %1 = stablehlo.subtract %arg0, %arg1 : tensor<10xf64>
  // CPU-NEXT:    %2 = stablehlo.compare NE, %1, %1 : (tensor<10xf64>, tensor<10xf64>) -> tensor<10xi1>
  // CPU-NEXT:    %3 = stablehlo.add %arg0, %arg1 : tensor<10xf64>
  // CPU-NEXT:    %4 = stablehlo.abs %1 : tensor<10xf64>
  // CPU-NEXT:    %5 = stablehlo.negate %4 : tensor<10xf64>
  // CPU-NEXT:    %6 = stablehlo.exponential %5 : tensor<10xf64>
  // CPU-NEXT:    %7 = stablehlo.log_plus_one %6 : tensor<10xf64>
  // CPU-NEXT:    %8 = stablehlo.add %0, %7 : tensor<10xf64>
  // CPU-NEXT:    %9 = stablehlo.select %2, %3, %8 : tensor<10xi1>, tensor<10xf64>
  // CPU-NEXT:    return %9 : tensor<10xf64>
  // CPU-NEXT:  }
  func.func @test(%lhs: tensor<10xf64>, %rhs: tensor<10xf64>) -> tensor<10xf64> {
    %result = enzyme.log_add_exp %lhs, %rhs : (tensor<10xf64>, tensor<10xf64>) -> tensor<10xf64>
    return %result : tensor<10xf64>
  }
}
