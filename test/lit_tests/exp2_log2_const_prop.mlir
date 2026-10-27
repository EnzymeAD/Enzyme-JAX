// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=exp2_const_prop;log2_const_prop},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @exp2_const_prop() -> tensor<4xf32> {
    %arg = stablehlo.constant dense<[0.000000e+00, 1.000000e+00, -1.000000e+00, 3.000000e+00]> : tensor<4xf32>
    %result = stablehlo.exp2 %arg : tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @log2_const_prop() -> tensor<4xf64> {
    %arg = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 8.000000e+00, 5.000000e-01]> : tensor<4xf64>
    %result = stablehlo.log2 %arg : tensor<4xf64>
    return %result : tensor<4xf64>
  }

  func.func @exp2_splat_const_prop(%x : tensor<2x2xf32>) -> tensor<2x2xf32> {
    %arg = stablehlo.constant dense<2.000000e+00> : tensor<2x2xf32>
    %e = stablehlo.exp2 %arg : tensor<2x2xf32>
    %result = stablehlo.add %x, %e : tensor<2x2xf32>
    return %result : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @exp2_const_prop
// CHECK-NOT: stablehlo.exp2
// CHECK: %cst = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 5.000000e-01, 8.000000e+00]> : tensor<4xf32>
// CHECK-NEXT: return %cst : tensor<4xf32>

// CHECK-LABEL: func.func @log2_const_prop
// CHECK-NOT: stablehlo.log2
// CHECK: %cst = stablehlo.constant dense<[0.000000e+00, 1.000000e+00, 3.000000e+00, -1.000000e+00]> : tensor<4xf64>
// CHECK-NEXT: return %cst : tensor<4xf64>

// CHECK-LABEL: func.func @exp2_splat_const_prop
// CHECK-NOT: stablehlo.exp2
// CHECK: %cst = stablehlo.constant dense<4.000000e+00> : tensor<2x2xf32>
// CHECK-NEXT: %[[R:.+]] = stablehlo.add %arg0, %cst : tensor<2x2xf32>
// CHECK-NEXT: return %[[R]] : tensor<2x2xf32>
