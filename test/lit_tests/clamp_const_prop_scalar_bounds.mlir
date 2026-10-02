// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=clamp_const_prop" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The bounds are scalars the operand is clamped against elementwise. Folding
// used to hand them to the reference clamp as they were, which indexes every
// bound as it does the operand and aborts on the shape.
func.func @main() -> tensor<4xf64> {
  %lo = stablehlo.constant dense<0.0> : tensor<f64>
  %hi = stablehlo.constant dense<2.0> : tensor<f64>
  %x = stablehlo.constant dense<[-1.0, 0.5, 1.5, 3.0]> : tensor<4xf64>
  %r = stablehlo.clamp %lo, %x, %hi : (tensor<f64>, tensor<4xf64>, tensor<f64>) -> tensor<4xf64>
  return %r : tensor<4xf64>
}

// CHECK:  func.func @main() -> tensor<4xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<[0.000000e+00, 5.000000e-01, 1.500000e+00, 2.000000e+00]> : tensor<4xf64>
// CHECK-NEXT:    return %cst : tensor<4xf64>
// CHECK-NEXT:  }
