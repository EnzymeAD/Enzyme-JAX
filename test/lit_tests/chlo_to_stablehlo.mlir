// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt)" | FileCheck %s

module {
  func.func @lgamma_f32() -> tensor<f32> {  
    %arg = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = chlo.lgamma %arg : tensor<f32> -> tensor<f32>
    func.return %1 : tensor<f32>
  }
}

// CHECK: func.func @lgamma_f32() -> tensor<f32> {
// CHECK-NEXT: %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK-NEXT: %cst_0 = stablehlo.constant dense<0.918938517> : tensor<f32>
// CHECK-NEXT: %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT: %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK-NEXT: %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT: %cst_4 = stablehlo.constant dense<263.38324> : tensor<f32>
// CHECK-NEXT: %cst_5 = stablehlo.constant dense<7.500000e+00> : tensor<f32>
// CHECK-NEXT: %cst_6 = stablehlo.constant dense<2.01490307> : tensor<f32>
// CHECK-NEXT: %0 = stablehlo.log_plus_one %cst_3 : tensor<f32>
// CHECK-NEXT: %1 = stablehlo.add %cst_6, %0 : tensor<f32>
// CHECK-NEXT: %2 = stablehlo.divide %cst_5, %1 : tensor<f32>
// CHECK-NEXT: %3 = stablehlo.subtract %cst_2, %2 : tensor<f32>
// CHECK-NEXT: %4 = stablehlo.multiply %3, %1 : tensor<f32>
// CHECK-NEXT: %5 = stablehlo.log %cst_4 : tensor<f32>
// CHECK-NEXT: %6 = stablehlo.add %cst_0, %4 : tensor<f32>
// CHECK-NEXT: %7 = stablehlo.add %6, %5 : tensor<f32>
// CHECK-NEXT: %8 = chlo.is_inf %cst_1 : tensor<f32> -> tensor<i1>
// CHECK-NEXT: %9 = stablehlo.select %8, %cst, %7 : tensor<i1>, tensor<f32>
// CHECK-NEXT: return %9 : tensor<f32>
// CHECK-NEXT: }
