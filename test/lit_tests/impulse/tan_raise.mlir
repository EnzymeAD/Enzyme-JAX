// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

// CHECK-LABEL: func.func @tan_raise
func.func @tan_raise(%x: tensor<f64>) -> tensor<f64> {
  // CHECK: %[[T:.+]] = stablehlo.tan %arg0 : tensor<f64>
  // CHECK-NOT: math.tan
  %0 = math.tan %x : tensor<f64>
  return %0 : tensor<f64>
}
