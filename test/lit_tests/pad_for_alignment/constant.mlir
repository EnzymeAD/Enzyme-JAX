// RUN: enzymexlamlir-opt --pad-for-alignment --allow-unregistered-dialect %s | FileCheck %s

func.func @test_constant() {
  %0 = stablehlo.constant dense<0.0> : tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_constant() {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_constant_ret() -> tensor<4x760x1533xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_constant_ret() -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x768x1536xf32>
// CHECK-NEXT:     %0 = stablehlo.slice %cst [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %0 : tensor<4x760x1533xf32>
// CHECK-NEXT: }
