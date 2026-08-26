// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s

func.func @main(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  return %arg0 : tensor<2xf64>
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: stablehlo.convert
// CHECK: return %arg0
